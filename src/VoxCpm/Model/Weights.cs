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
