// Ported from VoxCPM (https://github.com/OpenBMB/VoxCPM, Apache License 2.0, Copyright OpenBMB): src/voxcpm/modules/audiovae/audio_vae_v2.py
// Modified: re-implemented in C# on TorchSharp (inference only). See NOTICE.
using TorchSharp;
using VoxCpm.Model;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace VoxCpm.Codec;

/// <summary>
/// Port of <c>voxcpm/modules/audiovae/audio_vae_v2.py</c> (AudioVAE V2): a causal convolutional VAE that encodes
/// 16 kHz audio to 64-dim latents at 25 Hz and decodes them to 48 kHz with sample-rate conditioning.
/// Weight-normalized convolutions are folded into plain weights at load time. Always float32 (as in Python).
/// </summary>
public sealed class AudioVae : IDisposable
{
    private readonly Seq _encoderBlock;
    private readonly Conv _fcMu;
    private readonly List<IOp> _decoder;
    private readonly List<SampleRateCondition?> _srCond;
    private readonly int[]? _srBoundaries;

    public AudioVaeConfig Config { get; }
    public int SampleRate => Config.SampleRate;
    public int OutSampleRate => Config.OutSampleRate;
    public int LatentDim => Config.LatentDim;
    /// <summary>Input samples per latent frame.</summary>
    public int ChunkSize => Config.ChunkSize;
    /// <summary>Output samples per latent frame.</summary>
    public int DecodeChunkSize => Config.DecodeChunkSize;
    public Device Device { get; }

    internal AudioVae(AudioVaeConfig cfg, WeightSource w)
    {
        Config = cfg;
        Device = w.Device;
        if (cfg.UseNoiseBlock) throw new NotSupportedException("AudioVAE noise blocks are not supported.");
        if (cfg.CondType is not ("scale_bias" or "scale_bias_init") || cfg.CondOutLayer)
        {
            throw new NotSupportedException($"AudioVAE cond_type '{cfg.CondType}' (out_layer={cfg.CondOutLayer}) is not supported.");
        }

        // Encoder
        var enc = w.Sub("encoder");
        var blocks = new List<IOp> { Conv.Causal(enc.Sub("block.0"), padding: 3) };
        int d = cfg.EncoderDim;
        for (int i = 0; i < cfg.EncoderRates.Length; i++)
        {
            int stride = cfg.EncoderRates[i];
            d *= 2;
            int groups = cfg.Depthwise ? d / 2 : 1;
            var b = enc.Sub($"block.{i + 1}.block");
            blocks.Add(new Seq(
                ResidualUnit(b.Sub("0"), 1, groups),
                ResidualUnit(b.Sub("1"), 3, groups),
                ResidualUnit(b.Sub("2"), 9, groups),
                new Snake(b.Sub("3")),
                Conv.Causal(b.Sub("4"), stride: stride, padding: (stride + 1) / 2, outputPadding: stride % 2)));
        }
        _encoderBlock = new Seq(blocks.ToArray());
        _fcMu = Conv.Causal(enc.Sub("fc_mu"), padding: 1);
        // fc_logvar is not needed for inference (encode returns mu); it is taken so it is not reported unused.
        enc.Sub("fc_logvar").Take("weight_g", ScalarType.Float32).Dispose();
        enc.Sub("fc_logvar").Take("weight_v", ScalarType.Float32).Dispose();
        enc.Sub("fc_logvar").Take("bias", ScalarType.Float32).Dispose();

        // Decoder
        var dec = w.Sub("decoder");
        _decoder = new List<IOp>();
        int idx = 0;
        if (cfg.Depthwise)
        {
            _decoder.Add(Conv.Causal(dec.Sub($"model.{idx++}"), padding: 3, groups: cfg.LatentDim));
            _decoder.Add(Conv.Causal(dec.Sub($"model.{idx++}")));
        }
        else
        {
            _decoder.Add(Conv.Causal(dec.Sub($"model.{idx++}"), padding: 3));
        }
        var blockInputs = new List<(int Index, int Channels)>();
        int outputDim = cfg.DecoderDim;
        for (int i = 0; i < cfg.DecoderRates.Length; i++)
        {
            int stride = cfg.DecoderRates[i];
            int inputDim = cfg.DecoderDim >> i;
            outputDim = cfg.DecoderDim >> (i + 1);
            int groups = cfg.Depthwise ? outputDim : 1;
            var b = dec.Sub($"model.{idx}.block");
            blockInputs.Add((idx, inputDim));
            _decoder.Add(new Seq(
                new Snake(b.Sub("0")),
                Conv.CausalTranspose(b.Sub("1"), stride: stride, padding: (stride + 1) / 2, outputPadding: stride % 2),
                ResidualUnit(b.Sub("2"), 1, groups),
                ResidualUnit(b.Sub("3"), 3, groups),
                ResidualUnit(b.Sub("4"), 9, groups)));
            idx++;
        }
        _decoder.Add(new Snake(dec.Sub($"model.{idx++}")));
        _decoder.Add(Conv.Causal(dec.Sub($"model.{idx++}"), padding: 3));
        _decoder.Add(new Tanh());

        _srCond = new List<SampleRateCondition?>(new SampleRateCondition?[_decoder.Count]);
        if (cfg.SrBinBoundaries is { } boundaries)
        {
            _srBoundaries = boundaries;
            if (dec.Contains("sr_bin_boundaries")) dec.Take("sr_bin_boundaries", ScalarType.Int32).Dispose();
            foreach (var (i, _) in blockInputs) _srCond[i] = new SampleRateCondition(dec.Sub($"sr_cond_model.{i}"));
        }
    }

    /// <summary>Load only the AudioVAE of a checkpoint directory (config.json + audiovae.safetensors / .pth).</summary>
    public static AudioVae Load(string directory, Device device)
    {
        var cfg = VoxCpmConfig.Load(Path.Combine(directory, "config.json"));
        var (w, dict) = OpenWeights(directory, device);
        try
        {
            return new AudioVae(cfg.AudioVaeConfig!, w);
        }
        finally
        {
            w.Dispose();
            if (dict is not null) foreach (var t in dict.Values) t.Dispose();
        }
    }

    /// <summary>The AudioVAE weights: <c>audiovae.safetensors</c>, else the <c>state_dict</c> of <c>audiovae.pth</c>.</summary>
    internal static (WeightSource Source, Dictionary<string, Tensor>? Dict) OpenWeights(string directory, Device device)
    {
        var st = Path.Combine(directory, "audiovae.safetensors");
        var pth = Path.Combine(directory, "audiovae.pth");
        if (File.Exists(st)) return (WeightSource.FromSafeTensors(st, device), null);
        if (!File.Exists(pth)) throw new FileNotFoundException($"AudioVAE checkpoint not found. Expected {st} or {pth}");
        var root = VoxCpm.IO.TorchPickle.Load(pth);
        if (root is Dictionary<object, object?> d && d.TryGetValue("state_dict", out var inner)) root = inner;
        var dict = VoxCpm.IO.TorchPickle.AsStateDict(root);
        return (WeightSource.FromStateDict(dict, device), dict);
    }

    private static IOp ResidualUnit(WeightSource w, int dilation, int groups) =>
        new Residual(new Seq(
            new Snake(w.Sub("block.0")),
            Conv.Causal(w.Sub("block.1"), dilation: dilation, padding: 3 * dilation, groups: groups),
            new Snake(w.Sub("block.2")),
            Conv.Causal(w.Sub("block.3"))));

    /// <summary>
    /// <c>encode</c>: audio (B, T) or (B, 1, T) at <see cref="SampleRate"/> → mu (B, D, T / chunk). The input is
    /// right-padded to a multiple of the hop length.
    /// </summary>
    public Tensor Encode(Tensor audio)
    {
        if (audio.dim() == 2) audio = audio.unsqueeze(1);
        long length = audio.shape[^1];
        long rightPad = (length + ChunkSize - 1) / ChunkSize * ChunkSize - length;
        if (rightPad > 0) audio = F.pad(audio, new[] { 0L, rightPad });
        long chunk = (long)EncodeChunkFrames * ChunkSize;
        if (chunk <= 0 || audio.shape[^1] <= chunk) return _fcMu.Forward(_encoderBlock.Forward(audio, null), null);

        // Every encoder convolution is causal and each strided one has kernel 2·stride with stride samples of
        // causal padding, so chunks of whole latent frames that carry the convolution state give exactly the
        // output of one pass.
        using var state = new StreamState();
        var parts = new List<Tensor>();
        for (long start = 0; start < audio.shape[^1]; start += chunk)
        {
            using var scope = torch.NewDisposeScope();
            var piece = audio.narrow(-1, start, Math.Min(chunk, audio.shape[^1] - start));
            parts.Add(_fcMu.Forward(_encoderBlock.Forward(piece, state), state).MoveToOuterDisposeScope());
        }
        return torch.cat(parts.ToArray(), -1);
    }

    /// <summary>
    /// Latent frames per pass of <see cref="Encode"/> (50 = 2 s at 16 kHz). Longer clips are encoded in passes that
    /// carry the causal-convolution state (the same output as one pass): the first layers hold every input sample
    /// in 64+ channels, and a reference clip of several seconds in one pass ran out of memory on a 6 GB GPU next to
    /// the model. 0 encodes in one pass.
    /// </summary>
    public int EncodeChunkFrames { get; set; } = 50;

    /// <summary>
    /// Latent frames per pass of <see cref="Decode(Tensor)"/>. Longer inputs are decoded in passes that carry the
    /// causal-convolution state (exactly the same output as one pass), which bounds the activation memory: at
    /// 48 kHz the last layers hold 1920 samples per frame, and a full-utterance pass can exhaust a small GPU.
    /// 0 decodes in one pass.
    /// </summary>
    public int DecodeChunkFrames { get; set; } = 32;

    /// <summary><c>decode</c>: z (B, D, T) → audio (B, 1, T · decode_chunk) at <see cref="OutSampleRate"/>.</summary>
    public Tensor Decode(Tensor z)
    {
        long frames = z.shape[^1];
        int chunk = DecodeChunkFrames;
        if (chunk <= 0 || frames <= chunk) return Decode(z, null);
        using var state = new StreamState();
        var parts = new List<Tensor>();
        for (long start = 0; start < frames; start += chunk)
        {
            long len = Math.Min(chunk, frames - start);
            parts.Add(Decode(z.narrow(-1, start, len), state));
        }
        return torch.cat(parts.ToArray(), -1);
    }

    /// <summary>A stateful decoder that decodes consecutive latent chunks without overlap (<c>StreamingVAEDecoder</c>).</summary>
    public StreamingDecoder StartStreaming() => new(this);

    /// <summary>Per decoder layer timings in ms (only with VOXCPM_PROFILE=1).</summary>
    internal double[]? LayerMs;

    private Tensor Decode(Tensor z, StreamState? state)
    {
        var sw = VoxCpm2Model.Profile ? System.Diagnostics.Stopwatch.StartNew() : null;
        if (sw is not null) LayerMs ??= new double[_decoder.Count];
        var x = z;
        long? bucket = null;
        if (_srBoundaries is not null)
        {
            // torch.bucketize(sr, boundaries) (right=False): number of boundaries strictly below sr.
            bucket = _srBoundaries.Count(b => b < OutSampleRate);
        }
        for (int i = 0; i < _decoder.Count; i++)
        {
            if (_srCond[i] is { } cond) x = cond.Forward(x, bucket!.Value);
            x = _decoder[i].Forward(x, state);
            if (sw is not null)
            {
                if (Device.type == DeviceType.CUDA) torch.cuda.synchronize();
                LayerMs![i] += sw.Elapsed.TotalMilliseconds;
                sw.Restart();
            }
        }
        return x;
    }

    public void Dispose()
    {
        _encoderBlock.Dispose();
        _fcMu.Dispose();
        foreach (var op in _decoder) op.Dispose();
        foreach (var c in _srCond) c?.Dispose();
    }

    /// <summary>
    /// Stateful chunk decoding: every causal convolution keeps its last inputs (its left context) and every
    /// transposed convolution its last input frame, so <see cref="DecodeChunk"/> on consecutive chunks gives the
    /// same audio as one <see cref="Decode(Tensor)"/> of the concatenation.
    /// </summary>
    public sealed class StreamingDecoder : IDisposable
    {
        private readonly AudioVae _vae;
        private readonly StreamState _state = new();

        internal StreamingDecoder(AudioVae vae) => _vae = vae;

        /// <summary>z: (B, D, T) → audio (B, 1, T · decode_chunk).</summary>
        public Tensor DecodeChunk(Tensor z) => _vae.Decode(z, _state);

        public void Dispose() => _state.Dispose();
    }

    internal sealed class StreamState : IDisposable
    {
        private readonly Dictionary<object, Tensor> _buffers = new(ReferenceEqualityComparer.Instance);

        public bool TryGet(object key, out Tensor value) => _buffers.TryGetValue(key, out value!);

        public void Set(object key, Tensor value)
        {
            if (_buffers.TryGetValue(key, out var old)) old.Dispose();
            _buffers[key] = value.DetachFromDisposeScope();
        }

        public void Dispose()
        {
            foreach (var t in _buffers.Values) t.Dispose();
            _buffers.Clear();
        }
    }

    private interface IOp : IDisposable
    {
        Tensor Forward(Tensor x, StreamState? state);
    }

    private sealed class Seq : IOp
    {
        private readonly IOp[] _ops;
        public Seq(params IOp[] ops) => _ops = ops;

        public Tensor Forward(Tensor x, StreamState? state)
        {
            foreach (var op in _ops) x = op.Forward(x, state);
            return x;
        }

        public void Dispose()
        {
            foreach (var op in _ops) op.Dispose();
        }
    }

    private sealed class Residual : IOp
    {
        private readonly IOp _block;
        public Residual(IOp block) => _block = block;
        public Tensor Forward(Tensor x, StreamState? state) => x + _block.Forward(x, state);
        public void Dispose() => _block.Dispose();
    }

    private sealed class Tanh : IOp
    {
        public Tensor Forward(Tensor x, StreamState? state) => torch.tanh(x);
        public void Dispose() { }
    }

    /// <summary><c>Snake1d</c>: x + 1 / (alpha + 1e-9) · sin(alpha · x)².</summary>
    private sealed class Snake : IOp
    {
        private readonly Tensor _alpha;
        private readonly Tensor _invAlpha;

        public Snake(WeightSource w)
        {
            _alpha = w.Take("alpha", ScalarType.Float32);
            using var _ = torch.no_grad();
            _invAlpha = (_alpha + 1e-9).reciprocal().DetachFromDisposeScope();
        }

        public Tensor Forward(Tensor x, StreamState? state)
        {
            var shape = x.shape;
            var h = x.reshape(shape[0], shape[1], -1);
            h = h + _invAlpha * torch.sin(_alpha * h).pow(2);
            return h.reshape(shape);
        }

        public void Dispose()
        {
            _alpha.Dispose();
            _invAlpha.Dispose();
        }
    }

    /// <summary>
    /// A weight-normalized causal (transposed) 1-D convolution. <c>CausalConv1d</c> left-pads by
    /// <c>2·padding − output_padding</c>; <c>CausalTransposeConv1d</c> trims that many samples from the end.
    /// </summary>
    private sealed class Conv : IOp
    {
        private readonly Tensor _weight;
        private readonly Tensor? _bias;
        private readonly bool _transpose;
        private readonly long _stride, _dilation, _groups, _causalPad;
        private readonly long _kernel;

        private Conv(WeightSource w, bool transpose, long stride, long dilation, long groups, long padding, long outputPadding)
        {
            _transpose = transpose;
            _stride = stride;
            _dilation = dilation;
            _groups = groups;
            _causalPad = padding * 2 - outputPadding;
            using (torch.no_grad())
            {
                if (w.Contains("weight_v"))
                {
                    // weight_norm(dim=0): w = g · v / ‖v‖, the norm over every dim but the first.
                    using var g = w.Take("weight_g", ScalarType.Float32);
                    using var v = w.Take("weight_v", ScalarType.Float32);
                    // torch._weight_norm: v * (g / norm_except_dim(v, 2, 0)), the norm of v.view(C, -1) along dim 1.
                    var normShape = new long[v.dim()];
                    normShape[0] = v.shape[0];
                    for (int i = 1; i < normShape.Length; i++) normShape[i] = 1;
                    using var norm = v.contiguous().view(v.shape[0], -1).norm(1, true, 2).view(normShape);
                    _weight = (v * (g / norm)).DetachFromDisposeScope();
                }
                else
                {
                    _weight = w.Take("weight", ScalarType.Float32);
                }
            }
            _bias = w.TryTake("bias", ScalarType.Float32);
            _kernel = _weight.shape[^1];
        }

        public static Conv Causal(WeightSource w, long stride = 1, long dilation = 1, long groups = 1, long padding = 0, long outputPadding = 0) =>
            new(w, false, stride, dilation, groups, padding, outputPadding);

        public static Conv CausalTranspose(WeightSource w, long stride = 1, long padding = 0, long outputPadding = 0) =>
            new(w, true, stride, 1, 1, padding, outputPadding);

        public Tensor Forward(Tensor x, StreamState? state)
        {
            if (!_transpose)
            {
                Tensor padded;
                if (_causalPad <= 0)
                {
                    padded = x;
                }
                else if (state is null)
                {
                    padded = F.pad(x, new[] { _causalPad, 0L });
                }
                else
                {
                    // Python _patch_causal_conv: prepend the previous inputs (zeros at the start) and keep the last
                    // causalPad samples of the (previous + current) input for the next chunk.
                    var prev = state.TryGet(this, out var p) ? p : torch.zeros(x.shape[0], x.shape[1], _causalPad, dtype: x.dtype, device: x.device);
                    padded = torch.cat(new[] { prev, x }, -1);
                    state.Set(this, padded[.., .., (int)(padded.shape[^1] - _causalPad)..].clone());
                }
                return F.conv1d(padded, _weight, _bias, stride: _stride, padding: 0, dilation: _dilation, groups: _groups);
            }
            else
            {
                if (state is null)
                {
                    var y = F.conv_transpose1d(x, _weight, _bias, stride: _stride);
                    return _causalPad > 0 ? y[.., .., ..(int)(y.shape[^1] - _causalPad)] : y;
                }
                // Python _patch_transpose_conv: prepend ctx previous input frames (zeros at the start), drop the
                // ctx·stride samples they produce, and trim the causal padding from the end.
                long ctx = (_kernel - 1) / _stride;
                if (ctx <= 0)
                {
                    var y0 = F.conv_transpose1d(x, _weight, _bias, stride: _stride);
                    return _causalPad > 0 ? y0[.., .., ..(int)(y0.shape[^1] - _causalPad)] : y0;
                }
                var prev = state.TryGet(this, out var p) ? p : torch.zeros(x.shape[0], x.shape[1], ctx, dtype: x.dtype, device: x.device);
                var full = torch.cat(new[] { prev, x }, -1);
                state.Set(this, x[.., .., (int)(x.shape[^1] - ctx)..].clone());
                var o = F.conv_transpose1d(full, _weight, _bias, stride: _stride);
                long left = ctx * _stride;
                long end = _causalPad > 0 ? o.shape[^1] - _causalPad : o.shape[^1];
                return o[.., .., (int)left..(int)end];
            }
        }

        public void Dispose()
        {
            _weight.Dispose();
            _bias?.Dispose();
        }
    }

    /// <summary><c>SampleRateConditionLayer</c> (scale_bias): x · scale[sr] + bias[sr].</summary>
    private sealed class SampleRateCondition : IDisposable
    {
        private readonly Tensor _scale, _bias;

        public SampleRateCondition(WeightSource w)
        {
            _scale = w.Take("scale_embed.weight", ScalarType.Float32);
            _bias = w.Take("bias_embed.weight", ScalarType.Float32);
        }

        public Tensor Forward(Tensor x, long bucket) =>
            x * _scale[bucket].view(1, -1, 1) + _bias[bucket].view(1, -1, 1);

        public void Dispose()
        {
            _scale.Dispose();
            _bias.Dispose();
        }
    }
}
