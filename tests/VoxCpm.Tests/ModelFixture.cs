using TorchSharp;
using VoxCpm.IO;
using VoxCpm.Model;
using static TorchSharp.torch;

namespace VoxCpm.Tests;

/// <summary>The VoxCPM2 checkpoint loaded once (CPU, float32) plus the Python reference tensors.</summary>
public sealed class ModelFixture : IDisposable
{
    public VoxCpm2Model? Model { get; }
    public SafeTensorsFile? Ref { get; }

    public ModelFixture()
    {
        var ckpt = TestPaths.Checkpoint;
        var refPath = TestPaths.Data("model_ref.safetensors");
        if (ckpt is null || !File.Exists(refPath)) return;
        torch.set_num_threads(Math.Max(1, Environment.ProcessorCount / 2));
        Model = VoxCpm2Model.Load(ckpt, torch.CPU, ScalarType.Float32);
        Ref = new SafeTensorsFile(refPath);
    }

    public Tensor R(string key) => Ref!.Read(key);

    public void Dispose()
    {
        Model?.Dispose();
        Ref?.Dispose();
    }
}

[CollectionDefinition("model")]
public sealed class ModelCollection : ICollectionFixture<ModelFixture>;

internal static class TensorAssert
{
    public static double MaxAbsDiff(Tensor a, Tensor b)
    {
        Assert.Equal(b.shape, a.shape);
        return (a.to(ScalarType.Float64) - b.to(ScalarType.Float64)).abs().max().item<double>();
    }

    public static void Close(Tensor actual, Tensor expected, double tol, string what)
    {
        double d = MaxAbsDiff(actual.cpu(), expected.cpu());
        TestContext.Current.TestOutputHelper?.WriteLine($"{what}: max |diff| = {d:E2}");
        Console.Error.WriteLine($"[parity] {what}: max |diff| = {d:E2}");
        Assert.True(d <= tol, $"{what}: max |diff| = {d:E3} > {tol:E1}");
    }
}
