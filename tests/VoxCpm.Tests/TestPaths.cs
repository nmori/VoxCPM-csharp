namespace VoxCpm.Tests;

internal static class TestPaths
{
    public static string Data(string name) => Path.Combine(AppContext.BaseDirectory, "Data", name);

    /// <summary>Local VoxCPM2 checkpoint (VOXCPM_CKPT, else the C# Hugging Face cache), or null when absent.</summary>
    public static string? Checkpoint
    {
        get
        {
            var env = Environment.GetEnvironmentVariable("VOXCPM_CKPT");
            var dir = !string.IsNullOrEmpty(env) ? env : Path.Combine(VoxCpm.IO.HuggingFaceHub.CacheRoot, "openbmb", "VoxCPM2", "main");
            return File.Exists(Path.Combine(dir, "config.json")) ? dir : null;
        }
    }
}
