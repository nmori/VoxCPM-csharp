// Ported from VoxCPM (https://github.com/OpenBMB/VoxCPM, Apache License 2.0, Copyright OpenBMB): src/voxcpm/model/voxcpm2.py (checkpoint loading)
// Modified: re-implemented in C# on TorchSharp (inference only). See NOTICE.
using TorchSharp;
using VoxCpm.IO;
using static TorchSharp.torch;

namespace VoxCpm.Model;

/// <summary>
/// Named checkpoint tensors moved to the target device / dtype on demand. Every tensor handed out is owned by
/// the module that took it; <see cref="Unused"/> lists the checkpoint keys nobody took.
/// </summary>
internal sealed class WeightSource : IDisposable
{
    private readonly SafeTensorsFile? _file;
    private readonly Dictionary<string, Tensor>? _dict;
    private readonly string _prefix;
    private readonly HashSet<string> _taken;
    private readonly Device _device;

    private WeightSource(SafeTensorsFile? file, Dictionary<string, Tensor>? dict, string prefix, HashSet<string> taken, Device device)
    {
        _file = file;
        _dict = dict;
        _prefix = prefix;
        _taken = taken;
        _device = device;
    }

    public static WeightSource FromSafeTensors(string path, Device device) =>
        new(new SafeTensorsFile(path), null, "", new HashSet<string>(StringComparer.Ordinal), device);

    public static WeightSource FromStateDict(Dictionary<string, Tensor> dict, Device device) =>
        new(null, dict, "", new HashSet<string>(StringComparer.Ordinal), device);

    public Device Device => _device;

    /// <summary>A view on the keys below <paramref name="prefix"/> (<c>"a.b."</c>).</summary>
    public WeightSource Sub(string prefix) => new(_file, _dict, _prefix + prefix + ".", _taken, _device);

    public bool Contains(string name)
    {
        var key = _prefix + name;
        return _file?.Contains(key) ?? _dict!.ContainsKey(key);
    }

    /// <summary>The tensor <paramref name="name"/> on the device, converted to <paramref name="dtype"/>.</summary>
    public Tensor Take(string name, ScalarType dtype) => Take(name, dtype, _device);

    /// <summary>The tensor <paramref name="name"/> on <paramref name="device"/>, converted to <paramref name="dtype"/>.</summary>
    public Tensor Take(string name, ScalarType dtype, Device device)
    {
        var key = _prefix + name;
        Tensor src;
        if (_file is not null)
        {
            if (!_file.Contains(key)) throw new InvalidDataException($"Checkpoint is missing '{key}'.");
            src = _file.Read(key);
        }
        else
        {
            if (!_dict!.TryGetValue(key, out var t)) throw new InvalidDataException($"Checkpoint is missing '{key}'.");
            src = t;
        }
        _taken.Add(key);
        using var _ = torch.no_grad();
        var result = src.to(dtype, device).detach();
        if (_file is not null && !ReferenceEquals(result, src)) src.Dispose();
        return result.DetachFromDisposeScope();
    }

    public Tensor? TryTake(string name, ScalarType dtype) => Contains(name) ? Take(name, dtype) : null;

    public IEnumerable<string> Unused()
    {
        var keys = _file?.Keys ?? _dict!.Keys;
        return keys.Where(k => !_taken.Contains(k));
    }

    public void Dispose()
    {
        _file?.Dispose();
    }
}

/// <summary>
/// A bias-free linear weight (out, in), optionally the concatenation of several checkpoint weights along the
/// output dim. Either as is (in the model dtype) or int8 with one scale per <see cref="GroupSize"/> input columns
/// of each row (symmetric, absmax / 127): about half the memory of bf16; the weight is widened to the model dtype
/// for each matmul (there are no int8 matmul kernels in TorchSharp), which costs some speed.
/// </summary>
internal sealed class Weight : IDisposable
{
    private readonly Tensor _w;
    private readonly Tensor? _scale;
    private readonly ScalarType _dtype;

    private Weight(Tensor w, Tensor? scale, ScalarType dtype)
    {
        _w = w;
        _scale = scale;
        _dtype = dtype;
    }

    public bool IsInt8 => _scale is not null;

    /// <summary>Bytes on the device.</summary>
    public long Bytes => _w.NumberOfElements * _w.ElementSize + (_scale is null ? 0 : _scale.NumberOfElements * _scale.ElementSize);

    public static Weight Load(WeightSource w, ScalarType dtype, bool int8, params string[] names)
    {
        using var _ = torch.no_grad();
        if (!int8 && names.Length == 1) return new Weight(w.Take(names[0], dtype), null, dtype);
        // Concatenate / quantize on the CPU, then move once (no GPU temporaries).
        var parts = names.Select(n => w.Take(n, int8 ? ScalarType.Float32 : dtype, torch.CPU)).ToArray();
        try
        {
            using var scope = torch.NewDisposeScope();
            var full = parts.Length == 1 ? parts[0] : torch.cat(parts, 0);
            if (!int8) return new Weight(full.to(w.Device).DetachFromDisposeScope(), null, dtype);
            long rows = full.shape[0], cols = full.shape[1];
            long group = cols % GroupSize == 0 ? GroupSize : cols;
            var g = full.view(rows, cols / group, group);
            // The scale is rounded to the model dtype first, so the codes are those of the scale used at run time.
            var scale = (g.abs().amax(new long[] { 2 }, keepdim: true) / 127.0).clamp_min(1e-12).to(dtype).to(ScalarType.Float32);
            var q = torch.round(g / scale).clamp(-127, 127).to(ScalarType.Int8);
            return new Weight(q.to(w.Device).DetachFromDisposeScope(), scale.to(dtype, w.Device).DetachFromDisposeScope(), dtype);
        }
        finally
        {
            foreach (var p in parts) p.Dispose();
        }
    }

    /// <summary>Input columns per int8 scale.</summary>
    internal static int GroupSize { get; set; } = 128;

    /// <summary>x · Wᵀ; int8: W = Q · scale per group, in one kernel (int8 × bf16 gives bf16).</summary>
    public Tensor Linear(Tensor x)
    {
        if (_scale is null) return nn.functional.linear(x, _w);
        using var w = (_w * _scale).view(_w.shape[0], -1);
        return nn.functional.linear(x, w);
    }

    public void Dispose()
    {
        _w.Dispose();
        _scale?.Dispose();
    }
}

/// <summary><c>nn.Linear</c> with weights taken from a checkpoint.</summary>
internal sealed class Lin : IDisposable
{
    public Tensor W { get; private set; }
    public Tensor? B { get; private set; }

    public Lin(WeightSource w, string name, ScalarType dtype, bool bias = true)
    {
        var s = w.Sub(name);
        W = s.Take("weight", dtype);
        B = bias ? s.Take("bias", dtype) : null;
    }

    public Tensor Forward(Tensor x) => nn.functional.linear(x, W, B);

    public void Dispose()
    {
        W.Dispose();
        B?.Dispose();
    }
}
