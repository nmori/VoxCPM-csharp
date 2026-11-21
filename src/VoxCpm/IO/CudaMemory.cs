using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;

namespace VoxCpm.IO;

/// <summary>
/// Free / total memory of a CUDA device, counting every process (TorchSharp has no memory API). Read from
/// <c>nvidia-smi</c> (installed with the NVIDIA driver); else <c>cudaMemGetInfo</c> of the CUDA runtime that comes
/// with libtorch — which on Windows (WDDM) does not see the memory other processes hold, so it is only a fallback.
/// </summary>
public static class CudaMemory
{
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int SetDeviceFn(int device);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int MemGetInfoFn(out nuint free, out nuint total);

    private static readonly string[] RuntimeNames = { "cudart64_12", "cudart64_13", "libcudart.so.12", "libcudart.so.13", "libcudart.so" };

    /// <summary>(free, total) bytes of <paramref name="device"/>, or null when neither source is available.</summary>
    public static (long Free, long Total)? TryGetInfo(int device = 0) => FromNvidiaSmi(Math.Max(0, device)) ?? FromCudaRuntime(Math.Max(0, device));

    private static (long Free, long Total)? FromNvidiaSmi(int device)
    {
        try
        {
            var psi = new ProcessStartInfo("nvidia-smi", $"--query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i {device}")
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            using var p = Process.Start(psi);
            if (p is null) return null;
            string output = p.StandardOutput.ReadToEnd();
            if (!p.WaitForExit(5000) || p.ExitCode != 0) return null;
            var parts = output.Trim().Split(',');
            if (parts.Length != 2
                || !long.TryParse(parts[0].Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var usedMiB)
                || !long.TryParse(parts[1].Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var totalMiB))
            {
                return null;
            }
            return ((totalMiB - usedMiB) << 20, totalMiB << 20);
        }
        catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or InvalidOperationException or IOException)
        {
            return null;
        }
    }

    private static (long Free, long Total)? FromCudaRuntime(int device)
    {
        try
        {
            foreach (var name in RuntimeNames)
            {
                // Already loaded with libtorch, so the name resolves to that module.
                if (!NativeLibrary.TryLoad(name, typeof(CudaMemory).Assembly, null, out var lib)) continue;
                if (!NativeLibrary.TryGetExport(lib, "cudaSetDevice", out var setDevice)
                    || !NativeLibrary.TryGetExport(lib, "cudaMemGetInfo", out var memGetInfo))
                {
                    continue;
                }
                if (Marshal.GetDelegateForFunctionPointer<SetDeviceFn>(setDevice)(device) != 0) return null;
                if (Marshal.GetDelegateForFunctionPointer<MemGetInfoFn>(memGetInfo)(out var free, out var total) != 0) return null;
                return ((long)free, (long)total);
            }
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException)
        {
        }
        return null;
    }
}
