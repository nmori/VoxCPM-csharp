// Ported from VoxCPM (https://github.com/OpenBMB/VoxCPM, Apache License 2.0, Copyright OpenBMB): src/voxcpm/core.py
// Modified: re-implemented in C# on TorchSharp (inference only). See NOTICE.
using System.Text.RegularExpressions;
using TorchSharp;
using VoxCpm.Audio;
using VoxCpm.IO;
using VoxCpm.Model;
using static TorchSharp.torch;

namespace VoxCpm;

/// <summary>Where and how to load a checkpoint.</summary>
public sealed class RuntimeOptions
{
    /// <summary>Checkpoint directory (config.json, tokenizer.json, model.safetensors, audiovae.*).</summary>
    public required string ModelDirectory { get; init; }
    /// <summary><c>cpu</c>, <c>cuda</c>, <c>cuda:N</c> or <c>auto</c> (CUDA when available).</summary>
    public string Device { get; init; } = "auto";
    /// <summary>
    /// <c>fp32</c>, <c>tf32</c> (CUDA only: fp32 weights, TF32 matmuls), <c>bf16</c> or <c>auto</c>
    /// (bf16 on CUDA as the checkpoint is stored, fp32 on the CPU).
    /// </summary>
    public string Precision { get; init; } = "auto";
    /// <summary>Device of the AudioVAE (null: the model device).</summary>
    public string? VaeDevice { get; init; }
    public Action<string>? Log { get; init; }
}

/// <summary>One synthesis request (<c>VoxCPM.generate</c>).</summary>
public sealed class SynthesisRequest
{
    /// <summary>Text to speak.</summary>
    public required string Text { get; init; }
    /// <summary>
    /// Voice design / style description. VoxCPM2 reads it from parentheses at the start of the text:
    /// the text becomes <c>"(Style)Text"</c>.
    /// </summary>
    public string? Style { get; init; }
    /// <summary>Reference audio for voice cloning (isolated by reference tokens).</summary>
    public AudioData? ReferenceAudio { get; init; }
    /// <summary>Prompt audio the generation continues from; requires <see cref="PromptText"/>.</summary>
    public AudioData? PromptAudio { get; init; }
    /// <summary>Transcript of <see cref="PromptAudio"/>.</summary>
    public string? PromptText { get; init; }
    /// <summary>A prepared prompt (overrides the audio fields), e.g. from <see cref="VoxCpmRuntime.BuildPromptCache"/>.</summary>
    public PromptCache? Prompt { get; init; }

    public double CfgValue { get; init; } = 2.0;
    public int InferenceTimesteps { get; init; } = 10;
    public int MinLen { get; init; } = 2;
    public int MaxLen { get; init; } = 4096;
    public bool RetryBadCase { get; init; } = true;
    public int RetryBadCaseMaxTimes { get; init; } = 3;
    public double RetryBadCaseRatioThreshold { get; init; } = 6.0;
    public long? Seed { get; init; }
}

/// <summary>
/// The VoxCPM2 inference runtime (Python <c>voxcpm.VoxCPM</c>). Thread-safe: requests are serialized.
/// Load once and reuse.
/// </summary>
public sealed class VoxCpmRuntime : IDisposable
{
    public const string DefaultRepo = "openbmb/VoxCPM2";

    private readonly VoxCpm2Model _model;
    private readonly object _lock = new();
    private readonly Action<string>? _log;

    private VoxCpmRuntime(VoxCpm2Model model, Action<string>? log)
    {
        _model = model;
        _log = log;
    }

    public VoxCpm2Model Model => _model;
    /// <summary>Output sample rate (48 kHz).</summary>
    public int SampleRate => _model.SampleRate;
    public Device Device => _model.Device;
    public ScalarType DType => _model.DType;

    public static Device ResolveDevice(string device)
    {
        var d = device.Trim().ToLowerInvariant();
        if (d is "" or "auto") return torch.cuda.is_available() ? torch.CUDA : torch.CPU;
        if (d.StartsWith("cuda", StringComparison.Ordinal) && !torch.cuda.is_available())
        {
            throw new ArgumentException($"Requested device '{device}', but CUDA is not available.");
        }
        return torch.device(d);
    }

    public static ScalarType ResolveDType(string precision, Device device)
    {
        switch (precision.Trim().ToLowerInvariant())
        {
            case "" or "auto":
                return device.type == DeviceType.CUDA ? ScalarType.BFloat16 : ScalarType.Float32;
            case "fp32" or "float32":
                return ScalarType.Float32;
            case "tf32":
                if (device.type != DeviceType.CUDA) throw new ArgumentException("precision 'tf32' requires a CUDA device.");
                return ScalarType.Float32;
            case "bf16" or "bfloat16":
                return ScalarType.BFloat16;
            default:
                throw new ArgumentException($"Unsupported precision '{precision}'. Expected auto, fp32, tf32 or bf16.");
        }
    }

    /// <summary>Download (if needed) a Hugging Face checkpoint and load it.</summary>
    public static async Task<VoxCpmRuntime> FromHuggingFaceAsync(
        string repo = DefaultRepo,
        string device = "auto",
        string precision = "auto",
        IProgress<(string File, long Downloaded, long? Total)>? progress = null,
        Action<string>? log = null,
        CancellationToken cancellationToken = default)
    {
        // An unusable device / precision fails now, not after the download.
        ResolveDType(precision, ResolveDevice(device));
        var dir = await HuggingFaceHub.DownloadCheckpointAsync(repo, progress, cancellationToken).ConfigureAwait(false);
        return Load(new RuntimeOptions { ModelDirectory = dir, Device = device, Precision = precision, Log = log });
    }

    public static VoxCpmRuntime Load(RuntimeOptions options)
    {
        NativeAllocator.Tune();
        var device = ResolveDevice(options.Device);
        var dtype = ResolveDType(options.Precision, device);
        if (device.type == DeviceType.CUDA)
        {
            // Process-wide: the runtime loaded last decides.
            torch.backends.cuda.matmul.allow_tf32 = options.Precision.Trim().Equals("tf32", StringComparison.OrdinalIgnoreCase);
        }
        var vaeDevice = options.VaeDevice is { } vd ? ResolveDevice(vd) : device;
        var model = VoxCpm2Model.Load(options.ModelDirectory, device, dtype, options.Log, vaeDevice);
        return new VoxCpmRuntime(model, options.Log);
    }

    /// <summary>
    /// <c>build_prompt_cache</c>: encode a reference and / or a prompt (with its transcript). The audio is mixed
    /// to mono and resampled to 16 kHz. Reuse the result for every request with the same voice.
    /// </summary>
    public PromptCache BuildPromptCache(AudioData? referenceAudio, AudioData? promptAudio = null, string? promptText = null)
    {
        if ((promptAudio is null) != (promptText is null))
        {
            throw new ArgumentException("promptAudio and promptText must both be provided or both be null.");
        }
        if (promptAudio is null && referenceAudio is null)
        {
            throw new ArgumentException("At least one of referenceAudio or promptAudio must be provided.");
        }
        lock (_lock)
        {
            var refFeat = referenceAudio is null ? null : _model.EncodeAudio(To16k(referenceAudio), padLeft: false);
            var promptFeat = promptAudio is null ? null : _model.EncodeAudio(To16k(promptAudio), padLeft: true);
            var mode = refFeat is not null && promptFeat is not null ? PromptMode.RefContinuation
                : refFeat is not null ? PromptMode.Reference : PromptMode.Continuation;
            return new PromptCache { Mode = mode, PromptText = promptText, AudioFeat = promptFeat, RefAudioFeat = refFeat };
        }
    }

    private float[] To16k(AudioData audio)
    {
        var mono = audio.ToMono();
        return audio.SampleRate == _model.EncodeSampleRate ? mono : AudioDsp.Resample(mono, audio.SampleRate, _model.EncodeSampleRate);
    }

    /// <summary>Synthesize the whole utterance.</summary>
    public GenerationOutput Synthesize(SynthesisRequest request, CancellationToken cancellationToken = default) =>
        Run(request, null, cancellationToken);

    /// <summary>
    /// Synthesize with streaming: <paramref name="onChunk"/> receives each decoded patch (160 ms at 48 kHz) as soon as it
    /// is generated (bad-case retries are disabled, as in Python).
    /// </summary>
    public GenerationOutput SynthesizeStreaming(SynthesisRequest request, Action<float[]> onChunk, CancellationToken cancellationToken = default) =>
        Run(request, onChunk ?? throw new ArgumentNullException(nameof(onChunk)), cancellationToken);

    private GenerationOutput Run(SynthesisRequest r, Action<float[]>? onChunk, CancellationToken cancellationToken)
    {
        var text = PrepareText(r.Text, r.Style);
        var o = new GenerationOptions
        {
            CfgValue = r.CfgValue,
            InferenceTimesteps = r.InferenceTimesteps,
            MinLen = r.MinLen,
            MaxLen = r.MaxLen,
            RetryBadCase = r.RetryBadCase,
            RetryBadCaseMaxTimes = r.RetryBadCaseMaxTimes,
            RetryBadCaseRatioThreshold = r.RetryBadCaseRatioThreshold,
            Seed = r.Seed,
            Log = _log,
        };
        lock (_lock)
        {
            PromptCache? owned = null;
            try
            {
                var prompt = r.Prompt;
                if (prompt is null && (r.ReferenceAudio is not null || r.PromptAudio is not null))
                {
                    if ((r.PromptAudio is null) != (r.PromptText is null))
                    {
                        throw new ArgumentException("PromptAudio and PromptText must both be provided or both be null.");
                    }
                    owned = prompt = BuildPromptCache(r.ReferenceAudio, r.PromptAudio, r.PromptText);
                }
                return _model.Generate(text, prompt, o, onChunk, cancellationToken);
            }
            finally
            {
                owned?.Dispose();
            }
        }
    }

    private static readonly Regex Whitespace = new(@"\s+", RegexOptions.Compiled);

    /// <summary>
    /// <c>VoxCPM._generate</c> text handling: newlines become spaces, whitespace runs collapse to one space; a style
    /// description is put in parentheses in front.
    /// </summary>
    public static string PrepareText(string text, string? style = null)
    {
        if (string.IsNullOrWhiteSpace(text)) throw new ArgumentException("target text must be a non-empty string");
        text = Whitespace.Replace(text.Replace("\n", " "), " ");
        if (!string.IsNullOrWhiteSpace(style)) text = $"({style.Trim()}){text}";
        return text;
    }

    public void Dispose() => _model.Dispose();
}
