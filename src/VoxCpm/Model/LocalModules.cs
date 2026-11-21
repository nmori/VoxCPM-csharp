// Ported from VoxCPM (https://github.com/OpenBMB/VoxCPM, Apache License 2.0, Copyright OpenBMB): src/voxcpm/modules/locenc/local_encoder.py, locdit/local_dit_v2.py, locdit/unified_cfm.py, layers/scalar_quantization_layer.py
// Modified: re-implemented in C# on TorchSharp (inference only). See NOTICE.
using TorchSharp;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace VoxCpm.Model;

/// <summary>
/// <c>VoxCPMLocEnc</c> (locenc/local_encoder.py): each patch of P latent frames is projected, prefixed with a
/// learned special token and run through a bidirectional MiniCPM-4; the special token's output is the patch embedding.
/// </summary>
internal sealed class LocalEncoder : IDisposable
{
    private readonly Tensor _specialToken;
    private readonly Lin _inProj;
    private readonly MiniCpm4Model _encoder;

    public LocalEncoder(MiniCpm4Config cfg, WeightSource w, ScalarType dtype)
    {
        _specialToken = w.Take("special_token", dtype);
        _inProj = new Lin(w, "in_proj", dtype);
        // One special token + the patch frames.
        _encoder = new MiniCpm4Model(cfg, w.Sub("encoder"), dtype, maxPositions: 64);
    }

    /// <summary>x: (B, T, P, D) → (B, T, H).</summary>
    public Tensor Forward(Tensor x)
    {
        long b = x.shape[0], t = x.shape[1];
        var h = _inProj.Forward(x);
        var special = _specialToken.expand(b, t, 1, -1);
        h = torch.cat(new[] { special, h }, 2);
        h = h.reshape(b * t, h.shape[2], h.shape[3]);
        var outputs = _encoder.Encode(h);
        var cls = outputs[.., 0, ..];
        return cls.reshape(b, t, -1);
    }

    public void Dispose()
    {
        _specialToken.Dispose();
        _inProj.Dispose();
        _encoder.Dispose();
    }
}

/// <summary>
/// <c>VoxCPMLocDiT</c> V2 (locdit/local_dit_v2.py): the velocity estimator. The sequence is
/// [mu tokens (2: LM + residual), time token, P condition frames, P noisy frames] through a bidirectional
/// MiniCPM-4; the last P outputs are projected back to the latent dimension.
/// </summary>
internal sealed class LocalDit : IDisposable
{
    private readonly Lin _inProj, _condProj, _outProj;
    private readonly Lin _time1, _time2, _delta1, _delta2;
    private readonly MiniCpm4Model _decoder;
    private readonly int _hidden;

    public LocalDit(MiniCpm4Config cfg, WeightSource w, ScalarType dtype)
    {
        _hidden = cfg.HiddenSize;
        _inProj = new Lin(w, "in_proj", dtype);
        _condProj = new Lin(w, "cond_proj", dtype);
        _outProj = new Lin(w, "out_proj", dtype);
        _time1 = new Lin(w, "time_mlp.linear_1", dtype);
        _time2 = new Lin(w, "time_mlp.linear_2", dtype);
        _delta1 = new Lin(w, "delta_time_mlp.linear_1", dtype);
        _delta2 = new Lin(w, "delta_time_mlp.linear_2", dtype);
        // mu tokens + time token + condition and noisy patches.
        _decoder = new MiniCpm4Model(cfg, w.Sub("decoder"), dtype, maxPositions: 64);
    }

    /// <summary>
    /// x: (N, C, T) noisy latents; mu: (N, k·H) conditioning; t, dt: (N,); cond: (N, C, T') prefix frames.
    /// Returns the velocity (N, C, T).
    /// </summary>
    public Tensor Forward(Tensor x, Tensor mu, Tensor t, Tensor cond, Tensor dt)
    {
        var (m, c) = Prepare(mu, cond);
        return Forward(x, m, c, TimeEmbedding(t, dt));
    }

    /// <summary>
    /// The parts of <see cref="Forward(Tensor, Tensor, Tensor, Tensor, Tensor)"/> that only depend on mu and the
    /// condition frames (the same for every Euler step of a patch): mu as (N, k, H) tokens and the projected
    /// condition (N, T', H).
    /// </summary>
    internal (Tensor Mu, Tensor Cond) Prepare(Tensor mu, Tensor cond)
    {
        var c = _condProj.Forward(cond.transpose(1, 2).contiguous());
        return (mu.view(mu.shape[0], -1, _hidden), c);
    }

    /// <summary>The time token (N, H) for t and dt (N,) (depends only on the step of the schedule).</summary>
    internal Tensor TimeEmbedding(Tensor t, Tensor dt)
    {
        var dtype = _inProj.W.dtype;
        var te = _time2.Forward(F.silu(_time1.Forward(SinusoidalPosEmb(t, _hidden).to(dtype))));
        var de = _delta2.Forward(F.silu(_delta1.Forward(SinusoidalPosEmb(dt, _hidden).to(dtype))));
        return te + de;
    }

    /// <summary>x: (N, C, T) with the prepared mu / condition tokens and time token → velocity (N, C, T).</summary>
    internal Tensor Forward(Tensor x, Tensor m, Tensor c, Tensor te)
    {
        var h = _inProj.Forward(x.transpose(1, 2).contiguous());
        var seq = torch.cat(new[] { m, te.unsqueeze(1), c, h }, 1);
        var hidden = _decoder.Encode(seq);
        hidden = hidden[.., (int)(c.shape[1] + m.shape[1] + 1).., ..];
        return _outProj.Forward(hidden).transpose(1, 2).contiguous();
    }

    /// <summary><c>SinusoidalPosEmb.forward</c> with scale 1000 (computed in the dtype of <paramref name="x"/>).</summary>
    internal static Tensor SinusoidalPosEmb(Tensor x, int dim, double scale = 1000)
    {
        if (x.dim() < 1) x = x.unsqueeze(0);
        int half = dim / 2;
        double e = Math.Log(10000) / (half - 1);
        var emb = torch.exp(torch.arange(half, dtype: x.dtype, device: x.device) * -e);
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0);
        return torch.cat(new[] { emb.sin(), emb.cos() }, -1);
    }

    public void Dispose()
    {
        foreach (var l in new[] { _inProj, _condProj, _outProj, _time1, _time2, _delta1, _delta2 }) l.Dispose();
        _decoder.Dispose();
    }
}

/// <summary>
/// <c>UnifiedCFM</c> inference (locdit/unified_cfm.py): Euler integration of the flow from t = 1 (noise) to 0 with a
/// sway-shifted schedule and classifier-free guidance with the CFG-Zero* optimized scale (and zero velocity for the
/// first 4% of the steps).
/// </summary>
internal sealed class UnifiedCfm : IDisposable
{
    private readonly LocalDit _estimator;
    private readonly int _inChannels;
    private readonly bool _meanMode;
    private readonly Dictionary<(long Steps, long Batch, ScalarType DType, string Device, double Sway), Schedule> _schedules = new();

    public UnifiedCfm(LocalDit estimator, int inChannels, bool meanMode)
    {
        _estimator = estimator;
        _inChannels = inChannels;
        _meanMode = meanMode;
    }

    public LocalDit Estimator => _estimator;

    /// <summary>
    /// mu: (B, k·H); cond: (B, C, P). Draws the initial noise from the global generator
    /// (<c>torch.randn</c>, like Python), returns (B, C, P).
    /// </summary>
    public Tensor Forward(Tensor mu, int nTimesteps, int patchSize, Tensor cond, double temperature = 1.0, double cfgValue = 1.0,
        double swaySamplingCoef = 1.0, bool useCfgZeroStar = true)
    {
        long b = mu.shape[0];
        var z = torch.randn(new long[] { b, _inChannels, patchSize }, dtype: mu.dtype, device: mu.device) * temperature;
        return SolveEuler(z, GetSchedule(nTimesteps, b, mu.dtype, mu.device, swaySamplingCoef), mu, cond, cfgValue, useCfgZeroStar);
    }

    /// <summary>
    /// The Euler steps' dt and time tokens. They only depend on the number of steps (not on the patch), so they
    /// are computed once — with the very operations of the per-patch loop, hence the same values.
    /// </summary>
    private sealed class Schedule : IDisposable
    {
        /// <summary>Per step (index 1..n-1): dt (0-d) and the [cond; uncond] time token (2B, H).</summary>
        public required Tensor[] Dt { get; init; }
        public required Tensor[] TimeEmbed { get; init; }

        public void Dispose()
        {
            foreach (var t in Dt) t?.Dispose();
            foreach (var t in TimeEmbed) t?.Dispose();
        }
    }

    private Schedule GetSchedule(int nTimesteps, long b, ScalarType dtype, Device device, double sway)
    {
        var key = ((long)nTimesteps, b, dtype, device.ToString(), sway);
        if (_schedules.TryGetValue(key, out var cached)) return cached;
        using var scope = torch.NewDisposeScope();
        var tSpan = torch.linspace(1, 0, nTimesteps + 1, dtype: dtype, device: device);
        tSpan = tSpan + sway * (torch.cos(Math.PI / 2 * tSpan) - 1 + tSpan);
        long n = tSpan.shape[0];
        var dts = new Tensor[n];
        var tes = new Tensor[n];
        var t = tSpan[0];
        var dt = tSpan[0] - tSpan[1];
        for (int step = 1; step < n; step++)
        {
            dts[step] = dt.clone().DetachFromDisposeScope();
            var tIn = t.unsqueeze(0).expand(2 * b).contiguous();
            var dtIn = _meanMode ? dt.unsqueeze(0).expand(2 * b).contiguous() : torch.zeros(2 * b, dtype: dtype, device: device);
            tes[step] = _estimator.TimeEmbedding(tIn, dtIn).DetachFromDisposeScope();
            t = t - dt;
            if (step < n - 1) dt = t - tSpan[step + 1];
        }
        var s = new Schedule { Dt = dts, TimeEmbed = tes };
        _schedules[key] = s;
        return s;
    }

    private Tensor SolveEuler(Tensor x, Schedule schedule, Tensor mu, Tensor cond, double cfgValue, bool useCfgZeroStar)
    {
        var input = x;
        long n = schedule.Dt.Length;
        int zeroInitSteps = Math.Max(1, (int)(n * 0.04));
        long b = x.shape[0];
        // Batch [conditional; unconditional (mu = 0)]; mu and the condition are the same for every step.
        var (muTok, condTok) = _estimator.Prepare(torch.cat(new[] { mu, torch.zeros_like(mu) }, 0), torch.cat(new[] { cond, cond }, 0));
        for (int step = 1; step < n; step++)
        {
            var dt = schedule.Dt[step];
            Tensor dphi;
            if (useCfgZeroStar && step <= zeroInitSteps)
            {
                // Zero velocity: x - dt · 0 is x.
                continue;
            }
            // Each step in its own dispose scope, so its DiT activations are freed before the next one.
            using var stepScope = torch.NewDisposeScope();
            {
                var xIn = torch.cat(new[] { x, x }, 0);
                var v = _estimator.Forward(xIn, muTok, condTok, schedule.TimeEmbed[step]);
                var pos = v[..(int)b];
                var neg = v[(int)b..];

                Tensor stStar;
                if (useCfgZeroStar)
                {
                    var pf = pos.reshape(b, -1);
                    var nf = neg.reshape(b, -1);
                    var dot = (pf * nf).sum(1, keepdim: true);
                    var sq = (nf * nf).sum(1, keepdim: true) + 1e-8;
                    stStar = (dot / sq).view(b, 1, 1);
                }
                else
                {
                    stStar = torch.ones(1, dtype: x.dtype, device: x.device);
                }
                dphi = neg * stStar + cfgValue * (pos - neg * stStar);
            }
            var next = (x - dt * dphi).MoveToOuterDisposeScope();
            if (!ReferenceEquals(x, input)) x.Dispose();
            x = next;
        }
        return x;
    }

    public void Dispose()
    {
        foreach (var s in _schedules.Values) s.Dispose();
        _schedules.Clear();
    }
}

/// <summary><c>ScalarQuantizationLayer</c> (inference): tanh, round to 1/scale steps, project back.</summary>
internal sealed class ScalarQuantization : IDisposable
{
    private readonly Lin _in, _out;
    private readonly int _scale;

    public ScalarQuantization(WeightSource w, ScalarType dtype, int scale)
    {
        _in = new Lin(w, "in_proj", dtype);
        _out = new Lin(w, "out_proj", dtype);
        _scale = scale;
    }

    public Tensor Forward(Tensor x)
    {
        var h = torch.tanh(_in.Forward(x));
        h = torch.round(h * _scale) / _scale;
        return _out.Forward(h);
    }

    public void Dispose()
    {
        _in.Dispose();
        _out.Dispose();
    }
}
