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
        var (outputs, _) = _encoder.Forward(h, isCausal: false);
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
        var h = _inProj.Forward(x.transpose(1, 2).contiguous());
        var c = _condProj.Forward(cond.transpose(1, 2).contiguous());
        long prefix = c.shape[1];

        var te = _time2.Forward(F.silu(_time1.Forward(SinusoidalPosEmb(t, _hidden).to(h.dtype))));
        var de = _delta2.Forward(F.silu(_delta1.Forward(SinusoidalPosEmb(dt, _hidden).to(h.dtype))));
        te = te + de;

        var m = mu.view(h.shape[0], -1, h.shape[2]);
        var seq = torch.cat(new[] { m, te.unsqueeze(1), c, h }, 1);
        var (hidden, _) = _decoder.Forward(seq, isCausal: false);
        hidden = hidden[.., (int)(prefix + m.shape[1] + 1).., ..];
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
internal sealed class UnifiedCfm
{
    private readonly LocalDit _estimator;
    private readonly int _inChannels;
    private readonly bool _meanMode;

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
        var tSpan = torch.linspace(1, 0, nTimesteps + 1, dtype: mu.dtype, device: mu.device);
        tSpan = tSpan + swaySamplingCoef * (torch.cos(Math.PI / 2 * tSpan) - 1 + tSpan);
        return SolveEuler(z, tSpan, mu, cond, cfgValue, useCfgZeroStar);
    }

    private Tensor SolveEuler(Tensor x, Tensor tSpan, Tensor mu, Tensor cond, double cfgValue, bool useCfgZeroStar)
    {
        var t = tSpan[0];
        var dt = tSpan[0] - tSpan[1];
        long n = tSpan.shape[0];
        int zeroInitSteps = Math.Max(1, (int)(n * 0.04));
        for (int step = 1; step < n; step++)
        {
            Tensor dphi;
            if (useCfgZeroStar && step <= zeroInitSteps)
            {
                dphi = torch.zeros_like(x);
            }
            else
            {
                long b = x.shape[0];
                // Batch [conditional; unconditional (mu = 0)].
                var xIn = torch.cat(new[] { x, x }, 0);
                var muIn = torch.cat(new[] { mu, torch.zeros_like(mu) }, 0);
                var tIn = t.unsqueeze(0).expand(2 * b).contiguous();
                var dtIn = _meanMode ? dt.unsqueeze(0).expand(2 * b).contiguous() : torch.zeros(2 * b, dtype: x.dtype, device: x.device);
                var condIn = torch.cat(new[] { cond, cond }, 0);

                var v = _estimator.Forward(xIn, muIn, tIn, condIn, dtIn);
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
            x = x - dt * dphi;
            t = t - dt;
            if (step < n - 1) dt = t - tSpan[step + 1];
        }
        return x;
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
