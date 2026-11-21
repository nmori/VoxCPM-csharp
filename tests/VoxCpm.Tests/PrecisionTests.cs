using TorchSharp;
using static TorchSharp.torch;

namespace VoxCpm.Tests;

public class PrecisionTests
{
    [Theory]
    [InlineData("tf32", "fp32")]
    [InlineData("bf16", "fp32")]
    [InlineData(" BF16 ", "fp32")]
    [InlineData("fp32", "fp32")]
    [InlineData("auto", "auto")]
    public void GpuPrecisionsFallBackToFp32OnTheCpu(string requested, string expected)
    {
        var log = new List<string>();
        string p = VoxCpmRuntime.FitPrecisionToDevice(requested, torch.CPU, log.Add);
        Assert.Equal(expected, p);
        Assert.Equal(expected != requested.Trim() ? 1 : 0, log.Count);
        Assert.Equal(ScalarType.Float32, VoxCpmRuntime.ResolveDType(p, torch.CPU));
    }

    [Fact]
    public void CudaKeepsTheRequestedPrecision()
    {
        // A Device value only (torch.CUDA would initialise CUDA, which the CPU test build cannot).
        var cuda = new Device(DeviceType.CUDA, 0);
        Assert.Equal("tf32", VoxCpmRuntime.FitPrecisionToDevice("tf32", cuda, null));
        Assert.Equal("bf16", VoxCpmRuntime.FitPrecisionToDevice("bf16", cuda, null));
    }
}
