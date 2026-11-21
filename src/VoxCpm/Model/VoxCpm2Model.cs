// Ported from VoxCPM (https://github.com/OpenBMB/VoxCPM, Apache License 2.0, Copyright OpenBMB): src/voxcpm/model/voxcpm2.py (inference)
// Modified: re-implemented in C# on TorchSharp (inference only). See NOTICE.
using TorchSharp;
using VoxCpm.Codec;
using VoxCpm.IO;
using VoxCpm.Text;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace VoxCpm.Model;

/// <summary>
/// Port of <c>VoxCPM2Model</c> (model/voxcpm2.py), inference only: the text-semantic LM (<c>base_lm</c>), the
/// residual acoustic LM, the local encoder / DiT (CFM decoder), the FSQ bottleneck, the stop predictor and the
/// AudioVAE. One instance is not thread-safe (it owns the KV caches); <see cref="VoxCpmRuntime"/> serializes calls.
/// </summary>
public sealed class VoxCpm2Model : IDisposable
{
    public const int AudioStartToken = 101;
    public const int AudioEndToken = 102;
    public const int RefAudioStartToken = 103;
    public const int RefAudioEndToken = 104;

    private readonly MiniCpm4Model _baseLm;
    private readonly MiniCpm4Model _residualLm;
    private readonly LocalEncoder _featEncoder;
    private readonly UnifiedCfm _featDecoder;
    private readonly LocalDit _dit;
    private readonly ScalarQuantization _fsq;
    private readonly Lin _encToLm, _lmToDit, _resToDit, _fusionConcat, _stopProj, _stopHead;

    public VoxCpmConfig Config { get; }
    public VoxCpmTextTokenizer Tokenizer { get; }
    public AudioVae AudioVae { get; }
    public Device Device { get; }
    /// <summary>The dtype of the LMs / encoder / DiT (the AudioVAE is always float32).</summary>
    public ScalarType DType { get; }
    public int PatchSize => Config.PatchSize;
    public int FeatDim => Config.FeatDim;
    /// <summary>Output sample rate (48 kHz).</summary>
    public int SampleRate => AudioVae.OutSampleRate;
    /// <summary>Sample rate the reference / prompt audio is encoded at (16 kHz).</summary>
    public int EncodeSampleRate => AudioVae.SampleRate;

    private VoxCpm2Model(VoxCpmConfig cfg, VoxCpmTextTokenizer tokenizer, WeightSource w, WeightSource vae, Device device, ScalarType dtype, bool int8Lm)
    {
        Config = cfg;
        Tokenizer = tokenizer;
        Device = device;
        DType = dtype;
        _baseLm = new MiniCpm4Model(cfg.LmConfig, w.Sub("base_lm"), dtype, cfg.MaxLength, embedDevice: torch.CPU, int8: int8Lm);
        _residualLm = new MiniCpm4Model(cfg.ResidualConfig(), w.Sub("residual_lm"), dtype, cfg.MaxLength, int8: int8Lm);
        _featEncoder = new LocalEncoder(cfg.LocalConfig(cfg.EncoderConfig), w.Sub("feat_encoder"), dtype);
        _dit = new LocalDit(cfg.LocalConfig(cfg.DitConfig), w.Sub("feat_decoder.estimator"), dtype);
        _featDecoder = new UnifiedCfm(_dit, cfg.FeatDim, cfg.DitConfig.DitMeanMode);
        _fsq = new ScalarQuantization(w.Sub("fsq_layer"), dtype, cfg.ScalarQuantizationScale);
        _encToLm = new Lin(w, "enc_to_lm_proj", dtype);
        _lmToDit = new Lin(w, "lm_to_dit_proj", dtype);
        _resToDit = new Lin(w, "res_to_dit_proj", dtype);
        _fusionConcat = new Lin(w, "fusion_concat_proj", dtype);
        _stopProj = new Lin(w, "stop_proj", dtype);
        _stopHead = new Lin(w, "stop_head", dtype, bias: false);
        AudioVae = new AudioVae(cfg.AudioVaeConfig!, vae);
    }

    /// <summary>
    /// Load a checkpoint directory (<c>config.json</c>, <c>tokenizer.json</c>, <c>model.safetensors</c> and
    /// <c>audiovae.safetensors</c> or <c>audiovae.pth</c>).
    /// </summary>
    /// <param name="vaeDevice">Device of the AudioVAE (default: <paramref name="device"/>). The CPU frees ~0.4 GB of GPU memory.</param>
    /// <param name="int8Lm">Keep the linear weights of the two LMs as int8 (~1.6 GB less memory, slower LM steps).</param>
    public static VoxCpm2Model Load(string directory, Device device, ScalarType dtype, Action<string>? log = null, Device? vaeDevice = null,
        bool int8Lm = false)
    {
        var cfg = VoxCpmConfig.Load(Path.Combine(directory, "config.json"));
        var tokenizer = VoxCpmTextTokenizer.FromDirectory(directory);
        var modelPath = Path.Combine(directory, "model.safetensors");
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"Model file not found: {modelPath}");

        var (vae, vaeDict) = AudioVae.OpenWeights(directory, vaeDevice ?? device);
        log?.Invoke($"Loading VoxCPM2 from {directory} ({device}, {dtype})");
        using var w = WeightSource.FromSafeTensors(modelPath, device);
        try
        {
            var model = new VoxCpm2Model(cfg, tokenizer, w, vae, device, dtype, int8Lm);
            var unused = w.Unused().ToList();
            if (unused.Count > 0) log?.Invoke($"Unused model weights: {string.Join(", ", unused.Take(8))}{(unused.Count > 8 ? " ..." : "")}");
            var unusedVae = vae.Unused().ToList();
            if (unusedVae.Count > 0) log?.Invoke($"Unused AudioVAE weights: {string.Join(", ", unusedVae.Take(8))}");
            return model;
        }
        finally
        {
            vae.Dispose();
            if (vaeDict is not null) foreach (var t in vaeDict.Values) t.Dispose();
        }
    }

    // ------------------------------------------------------------------ //
    // Audio encoding
    // ------------------------------------------------------------------ //

    /// <summary>
    /// <c>_encode_wav</c> after loading: mono audio at <see cref="EncodeSampleRate"/> is padded to a whole number of
    /// patches (left padding for continuation prompts, right padding for references) and VAE-encoded.
    /// Returns (T, P, D) float32 on the CPU.
    /// </summary>
    public Tensor EncodeAudio(float[] audio16k, bool padLeft)
    {
        using var inference = torch.inference_mode();
        using var scope = torch.NewDisposeScope();
        int patchLen = PatchSize * AudioVae.ChunkSize;
        var audio = torch.tensor(audio16k, dtype: ScalarType.Float32).unsqueeze(0);
        long n = audio.shape[1];
        if (n % patchLen != 0)
        {
            long pad = patchLen - n % patchLen;
            audio = F.pad(audio, padLeft ? new[] { pad, 0L } : new[] { 0L, pad });
        }
        var feat = AudioVae.Encode(audio.to(AudioVae.Device)).cpu();
        return feat.view(AudioVae.LatentDim, -1, PatchSize).permute(1, 2, 0).contiguous().MoveToOuterDisposeScope();
    }

    // ------------------------------------------------------------------ //
    // Generation
    // ------------------------------------------------------------------ //

    /// <summary>
    /// <c>_generate_with_prompt_cache</c>: builds the input sequence for the prompt mode, runs the AR loop (with
    /// bad-case retries when not streaming) and decodes. Non-streaming: returns the whole waveform.
    /// Streaming: <paramref name="onChunk"/> receives each decoded patch (and the return value is the concatenation).
    /// </summary>
    public GenerationOutput Generate(string targetText, PromptCache? prompt, GenerationOptions o, Action<float[]>? onChunk = null,
        CancellationToken cancellationToken = default)
    {
        using var inference = torch.inference_mode();
        using var scope = torch.NewDisposeScope();
        bool streaming = onChunk is not null;
        bool retry = o.RetryBadCase && !streaming;

        var mode = prompt?.Mode ?? PromptMode.ZeroShot;
        string text = mode is PromptMode.Continuation or PromptMode.RefContinuation ? (prompt!.PromptText ?? "") + targetText : targetText;

        var textIds = Tokenizer.Encode(text);
        textIds.Add(AudioStartToken);
        int textLength = textIds.Count;
        int targetTextLength = Tokenizer.Encode(targetText).Count;

        var tokens = new List<long>();
        var tMask = new List<float>();
        var aMask = new List<float>();
        var feats = new List<Tensor>();
        var zeroPatch = torch.zeros(1, PatchSize, FeatDim, dtype: ScalarType.Float32);

        void AddText(IEnumerable<int> ids)
        {
            foreach (var id in ids)
            {
                tokens.Add(id);
                tMask.Add(1);
                aMask.Add(0);
            }
            feats.Add(zeroPatch.expand(textLength, PatchSize, FeatDim));
        }
        void AddAudio(Tensor feat)
        {
            long len = feat.shape[0];
            for (long i = 0; i < len; i++)
            {
                tokens.Add(0);
                tMask.Add(0);
                aMask.Add(1);
            }
            feats.Add(feat);
        }
        void AddRef(Tensor refFeat)
        {
            // [ref_audio_start, ref audio..., ref_audio_end]: the markers are text positions with zero features.
            tokens.Add(RefAudioStartToken); tMask.Add(1); aMask.Add(0); feats.Add(zeroPatch);
            AddAudio(refFeat);
            tokens.Add(RefAudioEndToken); tMask.Add(1); aMask.Add(0); feats.Add(zeroPatch);
        }

        switch (mode)
        {
            case PromptMode.ZeroShot:
                AddText(textIds);
                break;
            case PromptMode.Continuation:
                AddText(textIds);
                AddAudio(prompt!.AudioFeat!);
                break;
            case PromptMode.Reference:
                AddRef(prompt!.RefAudioFeat!);
                AddText(textIds);
                break;
            case PromptMode.RefContinuation:
                AddRef(prompt!.RefAudioFeat!);
                AddText(textIds);
                AddAudio(prompt!.AudioFeat!);
                break;
        }

        var textToken = torch.tensor(tokens.ToArray(), dtype: ScalarType.Int64).unsqueeze(0).to(Device);
        var textMask = torch.tensor(tMask.ToArray()).unsqueeze(0).to(DType, Device);
        var audioMask = torch.tensor(aMask.ToArray()).unsqueeze(0).to(DType, Device);
        var audioFeat = torch.cat(feats.ToArray(), 0).unsqueeze(0).to(DType, Device);

        int maxLen = Math.Min((int)(targetTextLength * o.RetryBadCaseRatioThreshold + 10), o.MaxLen);
        long seed = o.Seed ?? (long)(torch.random.seed() & 0xFFFFFFFF);
        int attempts = 0;
        InferenceResult? result = null;
        long lastSeed = seed;
        var chunks = new List<float[]>();
        int maxAttempts = Math.Max(1, o.RetryBadCaseMaxTimes);
        while (attempts < maxAttempts)
        {
            lastSeed = seed;
            ApplySeed(seed);
            if (streaming)
            {
                using var dec = AudioVae.StartStreaming();
                result = Inference(textToken, textMask, audioFeat, audioMask, o.MinLen, maxLen, o.InferenceTimesteps, o.CfgValue,
                    o.StreamingPrefixLen, patch =>
                    {
                        using var s = torch.NewDisposeScope();
                        var wav = dec.DecodeChunk(patch.to(ScalarType.Float32, AudioVae.Device)).reshape(-1).cpu();
                        var samples = wav.data<float>().ToArray();
                        chunks.Add(samples);
                        onChunk!(samples);
                    }, cancellationToken);
                break;
            }
            result = Inference(textToken, textMask, audioFeat, audioMask, o.MinLen, maxLen, o.InferenceTimesteps, o.CfgValue,
                o.StreamingPrefixLen, null, cancellationToken);
            if (retry && result.GeneratedPatches >= targetTextLength * o.RetryBadCaseRatioThreshold)
            {
                o.Log?.Invoke($"Badcase detected, audio_text_ratio={(double)result.GeneratedPatches / targetTextLength}, retrying...");
                attempts++;
                seed++;
                // After the last attempt Python leaves the loop and decodes that (bad) result.
                if (attempts >= maxAttempts) break;
                result.Dispose();
                result = null;
                continue;
            }
            break;
        }

        if (result is null) throw new InvalidOperationException("No generation attempt was made.");
        float[] audio;
        if (streaming)
        {
            audio = chunks.SelectMany(c => c).ToArray();
        }
        else
        {
            var wav = AudioVae.Decode(result.FeatPred.to(ScalarType.Float32, AudioVae.Device));
            int decodePatchLen = PatchSize * AudioVae.DecodeChunkSize;
            if (result.ContextLen > 0) wav = wav[.., .., (decodePatchLen * result.ContextLen)..];
            audio = wav.reshape(-1).cpu().data<float>().ToArray();
        }
        var generated = result.GeneratedFeat.MoveToOuterDisposeScope();
        result.FeatPred.Dispose();
        return new GenerationOutput(audio, SampleRate, lastSeed, generated);
    }

    /// <summary>Print per-step timings to stderr (<c>VOXCPM_PROFILE=1</c>).</summary>
    internal static bool Profile { get; } = Environment.GetEnvironmentVariable("VOXCPM_PROFILE") is "1" or "true";

    internal static void ApplySeed(long seed)
    {
        torch.manual_seed(seed);
        if (torch.cuda.is_available()) torch.cuda.manual_seed_all(seed);
    }

    internal sealed class InferenceResult : IDisposable
    {
        /// <summary>(B, D, T·P) latents including the context patches.</summary>
        public required Tensor FeatPred { get; init; }
        /// <summary>(T, P, D) generated patches (without context), CPU.</summary>
        public required Tensor GeneratedFeat { get; init; }
        public required int ContextLen { get; init; }
        public int GeneratedPatches => (int)GeneratedFeat.shape[0];

        public void Dispose()
        {
            FeatPred.Dispose();
            GeneratedFeat.Dispose();
        }
    }

    /// <summary>
    /// <c>_inference</c>: prefill both LMs, then per step: DiT sampling of one patch, local encoding of it, stop
    /// prediction, one step of each LM. <paramref name="onPatch"/> (streaming) receives each new patch as (B, D, P).
    /// </summary>
    internal InferenceResult Inference(Tensor text, Tensor textMask, Tensor feat, Tensor featMask, int minLen, int maxLen,
        int inferenceTimesteps, double cfgValue, int streamingPrefixLen, Action<Tensor>? onPatch, CancellationToken cancellationToken)
    {
        using var scope = torch.NewDisposeScope();
        long b = feat.shape[0];

        var featEmbed = _encToLm.Forward(_featEncoder.Forward(feat));
        double scaleEmb = Config.LmConfig.UseMup ? Config.LmConfig.ScaleEmb : 1.0;
        var emb = _baseLm.EmbedTokens!;
        var textEmbed = emb.index_select(0, text.reshape(-1).to(emb.device)).reshape(text.shape[0], text.shape[1], -1).to(Device);
        if (scaleEmb != 1.0) textEmbed = textEmbed * scaleEmb;
        var tm = textMask.unsqueeze(-1);
        var fm = featMask.unsqueeze(-1);
        var combined = tm * textEmbed + fm * featEmbed;

        var prefixFeatCond = feat[.., -1, .., ..];
        var predFeatSeq = new List<Tensor>();
        int contextLen = 0;
        var mask = featMask[0].to(ScalarType.Float32).cpu().data<float>().ToArray();
        bool hasContinuation = mask[^1] == 1f;
        if (hasContinuation)
        {
            var audioIdx = Enumerable.Range(0, mask.Length).Where(i => mask[i] == 1f).ToArray();
            contextLen = Math.Min(streamingPrefixLen - 1, audioIdx.Length);
            foreach (var i in audioIdx[^contextLen..]) predFeatSeq.Add(feat[.., i..(i + 1), .., ..]);
        }

        long prefixLen = combined.shape[1];
        var (encOutputs, kv) = _baseLm.Forward(combined, isCausal: true);
        using var baseCache = new KvCache(_baseLm.Config, (int)b, (int)prefixLen + maxLen + 1, DType, Device);
        baseCache.Fill(kv);

        encOutputs = _fsq.Forward(encOutputs) * fm + encOutputs * tm;
        var lmHidden = encOutputs[.., -1, ..];

        var residualIn = _fusionConcat.Forward(torch.cat(new[] { encOutputs, fm * featEmbed }, -1));
        var (residualOut, rkv) = _residualLm.Forward(residualIn, isCausal: true);
        using var residualCache = new KvCache(_residualLm.Config, (int)b, (int)prefixLen + maxLen + 1, DType, Device);
        residualCache.Fill(rkv);
        var residualHidden = residualOut[.., -1, ..];

        // Free the prefill intermediates before the loop.
        foreach (var (k, v) in kv) { k.Dispose(); v.Dispose(); }
        foreach (var (k, v) in rkv) { k.Dispose(); v.Dispose(); }

        var prof = Profile ? new double[6] : null;
        if (prof is not null) AudioVae.LayerMs = null;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        void Mark(int slot)
        {
            if (prof is null) return;
            if (Device.type == DeviceType.CUDA) torch.cuda.synchronize();
            prof[slot] += sw.Elapsed.TotalMilliseconds;
            sw.Restart();
        }
        int steps = 0;
        for (int i = 0; i < maxLen; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            using var step = torch.NewDisposeScope();
            steps++;
            Mark(5);
            var ditHidden = torch.cat(new[] { _lmToDit.Forward(lmHidden), _resToDit.Forward(residualHidden) }, -1);
            var predFeat = _featDecoder.Forward(ditHidden, inferenceTimesteps, PatchSize,
                prefixFeatCond.transpose(1, 2).contiguous(), cfgValue: cfgValue).transpose(1, 2); // (B, P, D)
            Mark(0);

            var currEmbed = _encToLm.Forward(_featEncoder.Forward(predFeat.unsqueeze(1))); // (B, 1, H)
            predFeatSeq.Add(predFeat.unsqueeze(1).MoveToOuterDisposeScope());
            prefixFeatCond = predFeat.MoveToOuterDisposeScope();
            Mark(1);

            if (onPatch is not null)
            {
                // (B, 1, P, D) -> (B, D, P)
                onPatch(predFeat.permute(0, 2, 1).contiguous());
                if (predFeatSeq.Count > streamingPrefixLen) predFeatSeq.RemoveRange(0, predFeatSeq.Count - streamingPrefixLen);
                Mark(2);
            }

            var stopLogits = _stopHead.Forward(F.silu(_stopProj.Forward(lmHidden)));
            long stopFlag = stopLogits.argmax(-1)[0].cpu().item<long>();
            if (i > minLen && stopFlag == 1) break;

            var ce = currEmbed[.., 0, ..];
            var lmNext = _baseLm.ForwardStep(ce, baseCache);
            lmNext = _fsq.Forward(lmNext);
            var resIn = _fusionConcat.Forward(torch.cat(new[] { lmNext, ce }, -1));
            var resNext = _residualLm.ForwardStep(resIn, residualCache);
            lmHidden = lmNext.MoveToOuterDisposeScope();
            residualHidden = resNext.MoveToOuterDisposeScope();
            Mark(3);
        }
        if (prof is not null)
        {
            Console.Error.WriteLine($"[profile] {steps} steps, ms/step: dit {prof[0] / steps:F1}, locenc {prof[1] / steps:F1}, " +
                $"vae(stream) {prof[2] / steps:F1}, stop+lms {prof[3] / steps:F1}, other {prof[5] / steps:F1}");
            if (AudioVae.LayerMs is { } lm) Console.Error.WriteLine($"[profile] vae layers ms/step: {string.Join(", ", lm.Select(v => (v / steps).ToString("F1")))}");
        }

        Tensor featPred, generated;
        if (onPatch is null)
        {
            var seq = torch.cat(predFeatSeq.ToArray(), 1); // (B, T, P, D)
            featPred = seq.permute(0, 3, 1, 2).reshape(b, FeatDim, -1);
            generated = seq[.., contextLen.., .., ..].squeeze(0).cpu();
        }
        else
        {
            var seq = torch.cat(predFeatSeq.ToArray(), 1);
            featPred = seq.permute(0, 3, 1, 2).reshape(b, FeatDim, -1);
            generated = seq.squeeze(0).cpu();
        }
        return new InferenceResult
        {
            FeatPred = featPred.MoveToOuterDisposeScope(),
            GeneratedFeat = generated.MoveToOuterDisposeScope(),
            ContextLen = contextLen,
        };
    }

    // Exposed for the parity tests.
    internal MiniCpm4Model BaseLm => _baseLm;
    internal MiniCpm4Model ResidualLm => _residualLm;
    internal LocalEncoder FeatEncoder => _featEncoder;
    internal UnifiedCfm FeatDecoder => _featDecoder;
    internal ScalarQuantization Fsq => _fsq;

    public void Dispose()
    {
        _baseLm.Dispose();
        _residualLm.Dispose();
        _featEncoder.Dispose();
        _featDecoder.Dispose();
        _dit.Dispose();
        _fsq.Dispose();
        foreach (var l in new[] { _encToLm, _lmToDit, _resToDit, _fusionConcat, _stopProj, _stopHead }) l.Dispose();
        AudioVae.Dispose();
    }
}

/// <summary>Prompt modes of <c>build_prompt_cache</c>.</summary>
public enum PromptMode
{
    ZeroShot,
    /// <summary>Reference audio only (voice cloning, structurally isolated).</summary>
    Reference,
    /// <summary>Prompt audio + its transcript (the model continues from it).</summary>
    Continuation,
    /// <summary>Reference + continuation.</summary>
    RefContinuation,
}

/// <summary><c>build_prompt_cache</c> result: encoded reference / prompt latents (CPU, float32).</summary>
public sealed class PromptCache : IDisposable
{
    public PromptMode Mode { get; init; }
    public string? PromptText { get; init; }
    /// <summary>(T, P, D) prompt latents (continuation modes).</summary>
    public Tensor? AudioFeat { get; init; }
    /// <summary>(T, P, D) reference latents (reference modes).</summary>
    public Tensor? RefAudioFeat { get; init; }

    public void Dispose()
    {
        AudioFeat?.Dispose();
        RefAudioFeat?.Dispose();
    }
}

/// <summary>Generation parameters (<c>VoxCPM.generate</c> keyword arguments).</summary>
public sealed class GenerationOptions
{
    public double CfgValue { get; set; } = 2.0;
    public int InferenceTimesteps { get; set; } = 10;
    public int MinLen { get; set; } = 2;
    public int MaxLen { get; set; } = 4096;
    public bool RetryBadCase { get; set; } = true;
    public int RetryBadCaseMaxTimes { get; set; } = 3;
    public double RetryBadCaseRatioThreshold { get; set; } = 6.0;
    public int StreamingPrefixLen { get; set; } = 4;
    public long? Seed { get; set; }
    public Action<string>? Log { get; set; }
}

/// <summary>A generated waveform.</summary>
public sealed class GenerationOutput : IDisposable
{
    public GenerationOutput(float[] audio, int sampleRate, long seed, Tensor generatedFeat)
    {
        Audio = audio;
        SampleRate = sampleRate;
        Seed = seed;
        GeneratedFeat = generatedFeat;
    }

    /// <summary>Mono samples in [-1, 1].</summary>
    public float[] Audio { get; }
    public int SampleRate { get; }
    /// <summary>The seed of the successful attempt.</summary>
    public long Seed { get; }
    /// <summary>(T, P, D) generated latent patches (CPU).</summary>
    public Tensor GeneratedFeat { get; }

    public void Dispose() => GeneratedFeat.Dispose();
}
