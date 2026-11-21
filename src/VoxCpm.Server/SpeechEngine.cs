using VoxCpm.Audio;
using VoxCpm.Model;

namespace VoxCpm.Server;

/// <summary>What to synthesize: the text plus the voice's audio and the sampling options.</summary>
public sealed record EngineRequest
{
    public required string Text { get; init; }
    public string? Style { get; init; }
    public string? RefWav { get; init; }
    public string? PromptWav { get; init; }
    public string? PromptText { get; init; }
    public double CfgValue { get; init; } = 2.0;
    public int InferenceTimesteps { get; init; } = 10;
    public int MinLen { get; init; } = 2;
    public int MaxLen { get; init; } = 4096;
    public bool RetryBadCase { get; init; } = true;
    public int RetryBadCaseMaxTimes { get; init; } = 3;
    public double RetryBadCaseRatioThreshold { get; init; } = 6.0;
    public long? Seed { get; init; }
}

public sealed record EngineResult(float[] Audio, int SampleRate, long Seed);

/// <summary>The synthesis engine behind the server (VoxCPM, or a fake in tests). Implementations serialize calls.</summary>
public interface ISpeechEngine
{
    string Description { get; }
    int SampleRate { get; }

    /// <summary>Synthesize; with <paramref name="onChunk"/> the audio is handed out while it is generated.</summary>
    EngineResult Synthesize(EngineRequest request, Action<float[]>? onChunk, CancellationToken cancellationToken);

    /// <summary>Encode a voice's clips ahead of time, so its first request does not pay for it.</summary>
    void Prepare(EngineRequest request) { }
}

/// <summary>
/// <see cref="ISpeechEngine"/> on <see cref="VoxCpmRuntime"/>. Encoded reference / prompt clips are kept per file
/// (path + size + time stamp), so a voice is encoded once.
/// </summary>
public sealed class VoxCpmEngine : ISpeechEngine, IDisposable
{
    private readonly VoxCpmRuntime _tts;
    private readonly Dictionary<string, PromptCache> _prompts = new(StringComparer.OrdinalIgnoreCase);
    private readonly object _lock = new();
    private const int MaxCachedPrompts = 64;

    public VoxCpmEngine(VoxCpmRuntime tts) => _tts = tts;

    public VoxCpmRuntime Runtime => _tts;
    public string Description => $"voxcpm2 torch {_tts.Device} {_tts.DType}";
    public int SampleRate => _tts.SampleRate;

    public EngineResult Synthesize(EngineRequest r, Action<float[]>? onChunk, CancellationToken cancellationToken)
    {
        lock (_lock)
        {
            var prompt = PromptFor(r);
            var request = new SynthesisRequest
            {
                Text = r.Text,
                Style = r.Style,
                Prompt = prompt,
                CfgValue = r.CfgValue,
                InferenceTimesteps = r.InferenceTimesteps,
                MinLen = r.MinLen,
                MaxLen = r.MaxLen,
                RetryBadCase = r.RetryBadCase,
                RetryBadCaseMaxTimes = r.RetryBadCaseMaxTimes,
                RetryBadCaseRatioThreshold = r.RetryBadCaseRatioThreshold,
                Seed = r.Seed,
            };
            using var result = onChunk is null
                ? _tts.Synthesize(request, cancellationToken)
                : _tts.SynthesizeStreaming(request, onChunk, cancellationToken);
            return new EngineResult(result.Audio, result.SampleRate, result.Seed);
        }
    }

    public void Prepare(EngineRequest r)
    {
        lock (_lock) PromptFor(r);
    }

    private PromptCache? PromptFor(EngineRequest r)
    {
        if (r.RefWav is null && r.PromptWav is null) return null;
        if ((r.PromptWav is null) != (r.PromptText is null))
        {
            throw new ArgumentException("prompt_wav and prompt_text must be given together.");
        }
        string key = $"{Stamp(r.RefWav)}|{Stamp(r.PromptWav)}|{r.PromptText}";
        if (_prompts.TryGetValue(key, out var cached)) return cached;
        var prompt = _tts.BuildPromptCache(r.RefWav is null ? null : WavFile.Read(r.RefWav),
            r.PromptWav is null ? null : WavFile.Read(r.PromptWav), r.PromptText);
        if (_prompts.Count >= MaxCachedPrompts)
        {
            foreach (var p in _prompts.Values) p.Dispose();
            _prompts.Clear();
        }
        _prompts[key] = prompt;
        return prompt;
    }

    private static string Stamp(string? path)
    {
        if (path is null) return "";
        var fi = new FileInfo(path);
        if (!fi.Exists) throw new FileNotFoundException($"Audio file not found: {path}");
        return $"{fi.FullName}:{fi.Length}:{fi.LastWriteTimeUtc.Ticks}";
    }

    public void Dispose()
    {
        foreach (var p in _prompts.Values) p.Dispose();
        _prompts.Clear();
        _tts.Dispose();
    }
}
