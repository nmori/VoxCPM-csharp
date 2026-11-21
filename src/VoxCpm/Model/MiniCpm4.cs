// Ported from VoxCPM (https://github.com/OpenBMB/VoxCPM, Apache License 2.0, Copyright OpenBMB): src/voxcpm/modules/minicpm4/model.py, cache.py
// Modified: re-implemented in C# on TorchSharp (inference only). See NOTICE.
using TorchSharp;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace VoxCpm.Model;

/// <summary>
/// Port of <c>voxcpm/modules/minicpm4/model.py</c>: the MiniCPM-4 transformer (RMSNorm, GQA self-attention with
/// LongRoPE, SiLU-gated MLP, optional muP residual scaling), used for the base LM, the residual LM, the local
/// encoder and the local DiT. <see cref="Forward"/> is <c>MiniCPMModel.forward</c> (prefill / full sequence) and
/// <see cref="ForwardStep"/> is <c>forward_step</c> (one token against a <see cref="KvCache"/>).
/// </summary>
internal sealed class MiniCpm4Model : IDisposable
{
    private readonly MiniCpm4Config _cfg;
    private readonly Layer[] _layers;
    private readonly Tensor _norm;
    private readonly LongRope? _rope;
    private readonly double _residualScale;

    public Tensor? EmbedTokens { get; }
    public MiniCpm4Config Config => _cfg;
    public ScalarType DType { get; }

    /// <param name="maxPositions">Positions the RoPE table must cover (the table is cut to this length).</param>
    /// <param name="embedDevice">Where the token embedding table lives (it is only read at prefill, so the CPU
    /// saves GPU memory).</param>
    /// <param name="int8">Keep the linear weights as int8 (see <see cref="Weight"/>): half the memory, a little slower.</param>
    public MiniCpm4Model(MiniCpm4Config cfg, WeightSource w, ScalarType dtype, int maxPositions, Device? embedDevice = null, bool int8 = false)
    {
        _cfg = cfg;
        DType = dtype;
        if (cfg.VocabSize > 0) EmbedTokens = w.Take("embed_tokens.weight", dtype, embedDevice ?? w.Device);
        _layers = new Layer[cfg.NumHiddenLayers];
        for (int i = 0; i < _layers.Length; i++) _layers[i] = new Layer(cfg, w.Sub($"layers.{i}"), dtype, int8);
        _norm = w.Take("norm.weight", dtype);
        _rope = cfg.NoRope ? null : new LongRope(cfg, w.Device, Math.Min(maxPositions, cfg.MaxPositionEmbeddings));
        _residualScale = cfg.UseMup ? cfg.ScaleDepth / Math.Sqrt(cfg.NumHiddenLayers) : 1.0;
    }

    public int NumLayers => _layers.Length;

    /// <summary>
    /// x: (B, T, H). Returns the normed hidden states and, per layer, the (B, KVH, T, D) keys / values
    /// (after RoPE), as <c>next_decoder_cache</c>.
    /// </summary>
    public (Tensor Hidden, List<(Tensor K, Tensor V)> Cache) Forward(Tensor x, bool isCausal)
    {
        (Tensor Cos, Tensor Sin)? pos = _rope?.Get(0, x.shape[1]);
        var cache = new List<(Tensor, Tensor)>(_layers.Length);
        var h = x;
        foreach (var layer in _layers)
        {
            Tensor next;
            using (torch.NewDisposeScope())
            {
                (next, var kv) = layer.Forward(h, pos, isCausal, _residualScale);
                next.MoveToOuterDisposeScope();
                cache.Add((kv.Item1.MoveToOuterDisposeScope(), kv.Item2.MoveToOuterDisposeScope()));
            }
            h = Advance(h, next, x);
        }
        return (RmsNorm(h, _norm, _cfg.RmsNormEps), cache);
    }

    /// <summary>
    /// The next hidden state; the previous one is freed unless it is the caller's input. Every layer runs in its
    /// own dispose scope: TorchSharp frees a tensor only when its scope ends, so without this the activations of
    /// all layers (and, in the callers, of a whole generation step) would stay allocated — Python frees each one
    /// right after use. On a small GPU that difference pushes the working set into shared system memory.
    /// </summary>
    private static Tensor Advance(Tensor previous, Tensor next, Tensor input)
    {
        if (!ReferenceEquals(previous, input)) previous.Dispose();
        return next;
    }

    /// <summary>One layer in its own dispose scope (see <see cref="Advance"/>).</summary>
    private static Tensor Scoped(Func<Tensor> layer)
    {
        using var _ = torch.NewDisposeScope();
        return layer().MoveToOuterDisposeScope();
    }

    /// <summary>
    /// x: (B, T, H) through the bidirectional model (the local encoder / DiT): <see cref="Forward"/> without
    /// collecting the keys / values, which saves two copies per layer.
    /// </summary>
    public Tensor Encode(Tensor x)
    {
        (Tensor Cos, Tensor Sin)? pos = _rope?.Get(0, x.shape[1]);
        var h = x;
        foreach (var layer in _layers)
        {
            var input = h;
            h = Advance(h, Scoped(() => layer.Encode(input, pos, _residualScale)), x);
        }
        return RmsNorm(h, _norm, _cfg.RmsNormEps);
    }

    /// <summary>
    /// x: (B, H): one token at the next cache position (<c>forward_step(x, kv_cache.step())</c>); writes its keys /
    /// values into the cache.
    /// </summary>
    public Tensor ForwardStep(Tensor x, KvCache cache)
    {
        int position = cache.Step();
        (Tensor Cos, Tensor Sin)? pos = _rope?.Get(position, 1);
        var h = x;
        for (int i = 0; i < _layers.Length; i++)
        {
            var (input, layer) = (h, _layers[i]);
            h = Advance(h, Scoped(() => layer.ForwardStep(input, pos, position, cache.Keys[i], cache.Values[i], _residualScale)), x);
        }
        return RmsNorm(h, _norm, _cfg.RmsNormEps);
    }

    /// <summary><c>rms_layernorm</c>: variance in float32, scaled input cast back, then times the weight.</summary>
    internal static Tensor RmsNorm(Tensor x, Tensor weight, double eps)
    {
        var dtype = x.dtype;
        var variance = x.to(ScalarType.Float32).pow(2).mean(new long[] { -1 }, keepdim: true);
        var h = (x * torch.rsqrt(variance + eps)).to(dtype);
        return h * weight;
    }

    public void Dispose()
    {
        EmbedTokens?.Dispose();
        foreach (var l in _layers) l.Dispose();
        _norm.Dispose();
        _rope?.Dispose();
    }

    private sealed class Layer : IDisposable
    {
        private readonly Tensor _inNorm, _postNorm;
        // q / k / v and gate / up are concatenated into one matmul each (fewer kernel launches per token).
        private readonly Weight _wqkv, _wo, _wGateUp, _wDown;
        private readonly int _heads, _kvHeads, _headDim, _intermediate;
        private readonly double _eps;

        public Layer(MiniCpm4Config cfg, WeightSource w, ScalarType dtype, bool int8)
        {
            _inNorm = w.Take("input_layernorm.weight", dtype);
            _postNorm = w.Take("post_attention_layernorm.weight", dtype);
            _wqkv = Weight.Load(w, dtype, int8, "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight");
            _wo = Weight.Load(w, dtype, int8, "self_attn.o_proj.weight");
            _wGateUp = Weight.Load(w, dtype, int8, "mlp.gate_proj.weight", "mlp.up_proj.weight");
            _wDown = Weight.Load(w, dtype, int8, "mlp.down_proj.weight");
            _heads = cfg.NumAttentionHeads;
            _kvHeads = cfg.NumKeyValueHeads;
            _headDim = cfg.HeadDim;
            _intermediate = cfg.IntermediateSize;
            _eps = cfg.RmsNormEps;
        }

        /// <summary>(B, T, H + 2·KVH, D) projections → q (B, T, H, D), k / v (B, T, KVH, D), RoPE applied to q and k.</summary>
        private (Tensor Q, Tensor K, Tensor V) Project(Tensor h, long b, long t, (Tensor Cos, Tensor Sin)? pos)
        {
            var qkv = _wqkv.Linear(h).view(b, t, _heads + 2 * _kvHeads, _headDim);
            var qk = qkv.narrow(2, 0, _heads + _kvHeads);
            if (pos is { } p) qk = ApplyRotary(qk, p.Cos.unsqueeze(1), p.Sin.unsqueeze(1));
            return (qk.narrow(2, 0, _heads), qk.narrow(2, _heads, _kvHeads), qkv.narrow(2, _heads + _kvHeads, _kvHeads));
        }

        private Tensor Residual(Tensor x, Tensor y, double scale) => scale == 1.0 ? x + y : x + y * scale;

        public (Tensor, (Tensor, Tensor)) Forward(Tensor x, (Tensor Cos, Tensor Sin)? pos, bool isCausal, double residualScale)
        {
            long b = x.shape[0], t = x.shape[1];
            var (q0, k0, v0) = Project(RmsNorm(x, _inNorm, _eps), b, t, pos);
            var q = q0.transpose(1, 2).contiguous();
            var k = k0.transpose(1, 2).contiguous();
            var v = v0.transpose(1, 2).contiguous();
            // enable_gqa=True: query head i uses key/value head i / (heads / kvHeads).
            int groups = _heads / _kvHeads;
            var kk = groups > 1 ? k.repeat_interleave(groups, 1) : k;
            var vv = groups > 1 ? v.repeat_interleave(groups, 1) : v;
            var attn = F.scaled_dot_product_attention(q, kk, vv, null, 0.0, isCausal);
            attn = attn.transpose(1, 2).contiguous().reshape(b, t, _heads * _headDim);
            var y = Residual(x, _wo.Linear(attn), residualScale);
            return (Residual(y, Mlp(RmsNorm(y, _postNorm, _eps)), residualScale), (k, v));
        }

        public Tensor Encode(Tensor x, (Tensor Cos, Tensor Sin)? pos, double residualScale)
        {
            long b = x.shape[0], t = x.shape[1];
            var (q0, k0, v0) = Project(RmsNorm(x, _inNorm, _eps), b, t, pos);
            var q = q0.transpose(1, 2).contiguous();
            // repeat_interleave writes new (contiguous) tensors: the same keys / values as Forward without the copies.
            int groups = _heads / _kvHeads;
            var kk = groups > 1 ? k0.transpose(1, 2).repeat_interleave(groups, 1) : k0.transpose(1, 2).contiguous();
            var vv = groups > 1 ? v0.transpose(1, 2).repeat_interleave(groups, 1) : v0.transpose(1, 2).contiguous();
            var attn = F.scaled_dot_product_attention(q, kk, vv, null, 0.0, false);
            attn = attn.transpose(1, 2).contiguous().reshape(b, t, _heads * _headDim);
            var y = Residual(x, _wo.Linear(attn), residualScale);
            return Residual(y, Mlp(RmsNorm(y, _postNorm, _eps)), residualScale);
        }

        public Tensor ForwardStep(Tensor x, (Tensor Cos, Tensor Sin)? pos, int position, Tensor keyCache, Tensor valueCache, double residualScale)
        {
            long b = x.shape[0];
            var (q, k, v) = Project(RmsNorm(x, _inNorm, _eps), b, 1, pos);
            // key_cache[:, :, position_id, :] = key_states
            keyCache[.., .., position, ..] = k[.., 0];
            valueCache[.., .., position, ..] = v[.., 0];
            // Attend to positions 0..position (Python masks the rest of the static cache, which is equivalent).
            // The query heads of one key/value head are laid out as the "query length" of a (B, KVH, G, D) SDPA.
            int groups = _heads / _kvHeads;
            var keys = keyCache[.., .., ..(position + 1), ..];
            var values = valueCache[.., .., ..(position + 1), ..];
            var qg = q.reshape(b, _kvHeads, groups, _headDim);
            var attn = F.scaled_dot_product_attention(qg, keys, values, null, 0.0, false);
            attn = attn.reshape(b, _heads * _headDim);
            var y = Residual(x, _wo.Linear(attn), residualScale);
            return Residual(y, Mlp(RmsNorm(y, _postNorm, _eps)), residualScale);
        }

        private Tensor Mlp(Tensor x)
        {
            var gu = _wGateUp.Linear(x);
            return _wDown.Linear(F.silu(gu.narrow(-1, 0, _intermediate)) * gu.narrow(-1, _intermediate, _intermediate));
        }

        public void Dispose()
        {
            _inNorm.Dispose();
            _postNorm.Dispose();
            foreach (var w in new[] { _wqkv, _wo, _wGateUp, _wDown }) w.Dispose();
        }
    }

    /// <summary>
    /// <c>apply_rotary_pos_emb</c> in float32, cast back: x·cos + rotate_half(x)·sin, where
    /// rotate_half(x)·sin = roll(x, D/2)·[−sin₁, sin₂] (the same products, so the same values).
    /// </summary>
    private static Tensor ApplyRotary(Tensor x, Tensor cos, Tensor signedSin)
    {
        var dtype = x.dtype;
        var xf = x.to(ScalarType.Float32);
        var r = xf * cos + xf.roll(xf.shape[^1] / 2, -1) * signedSin;
        return r.to(dtype);
    }
}

/// <summary>
/// <c>MiniCPMLongRoPE</c>: cos / sin tables (float32) for <c>max_position_embeddings</c> positions, built once with
/// the long or short factors and the LongRoPE scaling factor.
/// </summary>
internal sealed class LongRope : IDisposable
{
    private readonly Tensor _cos, _sin;

    /// <param name="length">Rows to build (each position is computed independently, so a shorter table holds the
    /// same values as Python's full one).</param>
    public LongRope(MiniCpm4Config cfg, Device device, int length)
    {
        using var scope = torch.NewDisposeScope();
        int dim = cfg.HeadDim;
        int maxPos = cfg.MaxPositionEmbeddings;
        int original = cfg.RopeScaling.OriginalMaxPositionEmbeddings;
        double scale = (double)maxPos / original;
        double scalingFactor = Math.Sqrt(1 + Math.Log(scale) / Math.Log(original));
        var invFreq = 1.0 / torch.pow(cfg.RopeTheta, torch.arange(0, dim, 2, dtype: ScalarType.Int64).to(ScalarType.Float32) / dim);
        var t = torch.arange(length, dtype: ScalarType.Float32);
        var factors = maxPos > original ? cfg.RopeScaling.LongFactor : cfg.RopeScaling.ShortFactor;
        var ext = torch.tensor(factors.Select(f => (float)f).ToArray(), dtype: ScalarType.Float32);
        var freqs = torch.outer(t, 1.0 / ext) * invFreq;
        var emb = torch.cat(new[] { freqs, freqs }, -1);
        _cos = (emb.cos() * scalingFactor).to(device).DetachFromDisposeScope();
        var sin = emb.sin() * scalingFactor;
        // For rotate_half: the first half negated.
        var half = dim / 2;
        _sin = torch.cat(new[] { -sin.narrow(1, 0, half), sin.narrow(1, half, half) }, 1).to(device).DetachFromDisposeScope();
    }

    /// <summary>
    /// (len, D) cos and signed sin (first half negated, for <c>rotate_half</c>) for positions
    /// <paramref name="start"/> .. start + len - 1.
    /// </summary>
    public (Tensor Cos, Tensor Sin) Get(long start, long length) =>
        (_cos[TensorIndex.Slice(start, start + length)], _sin[TensorIndex.Slice(start, start + length)]);

    public void Dispose()
    {
        _cos.Dispose();
        _sin.Dispose();
    }
}

/// <summary>
/// <c>StaticKVCache</c>: per layer (B, KVH, capacity, D) keys and values, filled by the prefill
/// (<see cref="Fill"/>) and then one position per <see cref="MiniCpm4Model.ForwardStep"/>.
/// </summary>
internal sealed class KvCache : IDisposable
{
    public Tensor[] Keys { get; }
    public Tensor[] Values { get; }
    public int Length { get; private set; }
    public int Capacity { get; }

    public KvCache(MiniCpm4Config cfg, int batch, int capacity, ScalarType dtype, Device device)
    {
        Capacity = capacity;
        Keys = new Tensor[cfg.NumHiddenLayers];
        Values = new Tensor[cfg.NumHiddenLayers];
        for (int i = 0; i < Keys.Length; i++)
        {
            Keys[i] = torch.zeros(batch, cfg.NumKeyValueHeads, capacity, cfg.HeadDim, dtype: dtype, device: device).DetachFromDisposeScope();
            Values[i] = torch.zeros(batch, cfg.NumKeyValueHeads, capacity, cfg.HeadDim, dtype: dtype, device: device).DetachFromDisposeScope();
        }
    }

    /// <summary><c>fill_caches</c>.</summary>
    public void Fill(List<(Tensor K, Tensor V)> kv)
    {
        int len = (int)kv[0].K.shape[2];
        if (len > Capacity) throw new InvalidOperationException("KV cache is full");
        using var _ = torch.no_grad();
        for (int i = 0; i < Keys.Length; i++)
        {
            Keys[i][.., .., ..len, ..] = kv[i].K;
            Values[i][.., .., ..len, ..] = kv[i].V;
        }
        Length = len;
    }

    /// <summary><c>step</c>: the position of the next token.</summary>
    public int Step()
    {
        if (Length >= Capacity) throw new InvalidOperationException("KV cache is full");
        return Length++;
    }

    public void Dispose()
    {
        foreach (var t in Keys) t.Dispose();
        foreach (var t in Values) t.Dispose();
    }
}
