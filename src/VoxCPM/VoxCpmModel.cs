using System.Text.Json;

namespace VoxCPM;

/// <summary>Input/output contract for a native or ONNX VoxCPM runtime.</summary>
public interface IVoxCpmRuntime
{
    float[] Generate(string text, GenerationOptions options);
    IEnumerable<float[]> GenerateStreaming(string text, GenerationOptions options);
}

public sealed class VoxCPMModel : IDisposable
{
    private readonly IVoxCpmRuntime? runtime;

    private VoxCPMModel(string modelPath, IVoxCpmRuntime? runtime)
    {
        ModelPath = Path.GetFullPath(modelPath);
        this.runtime = runtime;

        var configPath = Path.Combine(ModelPath, "config.json");
        if (!File.Exists(configPath))
            throw new FileNotFoundException("VoxCPM model assets must contain config.json.", configPath);

        using var stream = File.OpenRead(configPath);
        using var document = JsonDocument.Parse(stream);
        Architecture = document.RootElement.TryGetProperty("architecture", out var architecture)
            ? architecture.GetString()?.Trim().ToLowerInvariant() ?? "voxcpm"
            : "voxcpm";

        if (Architecture is not ("voxcpm" or "voxcpm2"))
            throw new ArgumentException($"Unsupported VoxCPM architecture: {Architecture}", nameof(modelPath));
    }

    public string ModelPath { get; }
    public string Architecture { get; }
    public int SampleRate => Architecture == "voxcpm2" ? 48000 : 16000;

    public static VoxCPMModel FromPretrained(
        string modelId,
        bool loadDenoiser = true,
        string? cacheDir = null,
        bool localFilesOnly = false,
        bool optimize = true,
        string? device = null,
        IVoxCpmRuntime? runtime = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);

        if (!Directory.Exists(modelId))
        {
            throw new NotSupportedException(
                "Remote Hugging Face model downloads are not implemented by this C# port. " +
                "Download the model snapshot locally and pass its directory path.");
        }

        return new VoxCPMModel(modelId, runtime);
    }

    public float[] Generate(string text, GenerationOptions? options = null)
    {
        var normalized = ValidateAndNormalize(text, options ?? new GenerationOptions());
        return runtime?.Generate(normalized.Text, normalized.Options)
            ?? throw new NotSupportedException(
                "No VoxCPM runtime is configured. Implement IVoxCpmRuntime and pass it to FromPretrained.");
    }

    public IEnumerable<float[]> GenerateStreaming(string text, GenerationOptions? options = null)
    {
        var normalized = ValidateAndNormalize(text, options ?? new GenerationOptions());
        if (runtime is null)
            throw new NotSupportedException(
                "No VoxCPM runtime is configured. Implement IVoxCpmRuntime and pass it to FromPretrained.");
        return runtime.GenerateStreaming(normalized.Text, normalized.Options);
    }

    private (string Text, GenerationOptions Options) ValidateAndNormalize(string text, GenerationOptions options)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(text);
        if (options.CfgValue < 0)
            throw new ArgumentOutOfRangeException(nameof(options.CfgValue));
        if (options.InferenceTimesteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(options.InferenceTimesteps));
        if (options.MinLen < 0)
            throw new ArgumentOutOfRangeException(nameof(options.MinLen));
        if (options.MaxLen <= 0)
            throw new ArgumentOutOfRangeException(nameof(options.MaxLen));

        if (options.PromptWavPath is not null && !File.Exists(options.PromptWavPath))
            throw new FileNotFoundException("Prompt WAV does not exist.", options.PromptWavPath);
        if (options.ReferenceWavPath is not null && !File.Exists(options.ReferenceWavPath))
            throw new FileNotFoundException("Reference WAV does not exist.", options.ReferenceWavPath);
        if ((options.PromptWavPath is null) != (options.PromptText is null))
            throw new ArgumentException("PromptWavPath and PromptText must both be provided or both be null.");
        if (options.ReferenceWavPath is not null && Architecture != "voxcpm2")
            throw new ArgumentException("ReferenceWavPath is only supported with VoxCPM2 models.");

        var normalized = string.Join(' ', text.Replace("\r", " ").Replace("\n", " ").Split(
            (char[]?)null, StringSplitOptions.RemoveEmptyEntries));
        if (options.Normalize)
            normalized = normalized.Trim();
        return (normalized, options);
    }

    public void Dispose() { }
}

/// <summary>Python-compatible short name for the VoxCPM pipeline.</summary>
public sealed class VoxCPM : IDisposable
{
    private readonly VoxCPMModel model;

    private VoxCPM(VoxCPMModel model)
    {
        this.model = model;
    }

    public string ModelPath => model.ModelPath;
    public string Architecture => model.Architecture;
    public int SampleRate => model.SampleRate;

    public static VoxCPM FromPretrained(
        string modelId,
        bool loadDenoiser = true,
        string? cacheDir = null,
        bool localFilesOnly = false,
        bool optimize = true,
        string? device = null,
        IVoxCpmRuntime? runtime = null)
    {
        var model = VoxCPMModel.FromPretrained(
            modelId, loadDenoiser, cacheDir, localFilesOnly, optimize, device, runtime);
        return new VoxCPM(model);
    }

    public float[] Generate(string text, GenerationOptions? options = null)
        => model.Generate(text, options);

    public IEnumerable<float[]> GenerateStreaming(string text, GenerationOptions? options = null)
        => model.GenerateStreaming(text, options);

    public void Dispose() => model.Dispose();
}
