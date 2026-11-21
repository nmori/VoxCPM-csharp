// Derived from Irodori-TTS-csharp (MIT License, Copyright (c) 2026 Aratako). See NOTICE.
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace VoxCpm.Server;

public sealed class SpeechOptionException(string message, string? param = null) : Exception(message)
{
    public string? Param { get; } = param;
}

/// <summary>
/// Options of a <c>/v1/audio/speech</c> request. Each one is looked up in the request's <c>voxcpm</c> object, then
/// its <c>irodori</c> object (what Speech.dll sends for a voice), then the top level of the body, then the voice's
/// <c>.json</c>, then the server defaults. Values may be JSON numbers / booleans or strings.
/// </summary>
public sealed class SpeechOptions
{
    private readonly JsonObject?[] _sources;

    public SpeechOptions(JsonObject body, JsonObject? voiceOptions, JsonObject? serverDefaults)
    {
        _sources = new[] { body["voxcpm"] as JsonObject, body["irodori"] as JsonObject, body, voiceOptions, serverDefaults };
    }

    /// <summary>The first source that sets one of <paramref name="names"/> (an empty string or null means unset).</summary>
    public JsonNode? Get(params string[] names)
    {
        foreach (var src in _sources)
        {
            if (src is null) continue;
            foreach (var n in names)
            {
                var v = src[n];
                if (v is null) continue;
                if (v is JsonValue jv && jv.TryGetValue<string>(out var s) && s.Trim().Length == 0) continue;
                return v;
            }
        }
        return null;
    }

    public string? String(params string[] names) => Get(names) is { } v ? v.GetValueKind() == JsonValueKind.String ? v.GetValue<string>() : v.ToJsonString() : null;

    public double? Double(params string[] names)
    {
        var v = Get(names);
        if (v is null) return null;
        try
        {
            return v.GetValueKind() == JsonValueKind.String
                ? double.Parse(v.GetValue<string>(), NumberStyles.Float, CultureInfo.InvariantCulture)
                : v.GetValue<double>();
        }
        catch (Exception ex) when (ex is FormatException or InvalidOperationException)
        {
            throw new SpeechOptionException($"'{names[0]}' must be a number.", names[0]);
        }
    }

    public long? Long(params string[] names)
    {
        var d = Double(names);
        if (d is null) return null;
        if (d != Math.Floor(d.Value)) throw new SpeechOptionException($"'{names[0]}' must be an integer.", names[0]);
        return (long)d.Value;
    }

    public int? Int(params string[] names) => Long(names) is long l ? checked((int)l) : null;

    public bool? Bool(params string[] names)
    {
        var v = Get(names);
        if (v is null) return null;
        switch (v.GetValueKind())
        {
            case JsonValueKind.True: return true;
            case JsonValueKind.False: return false;
            case JsonValueKind.String:
                switch (v.GetValue<string>().Trim().ToLowerInvariant())
                {
                    case "true" or "1" or "yes" or "on": return true;
                    case "false" or "0" or "no" or "off": return false;
                }
                break;
            case JsonValueKind.Number:
                return v.GetValue<double>() != 0;
        }
        throw new SpeechOptionException($"'{names[0]}' must be a boolean.", names[0]);
    }

    /// <summary>
    /// The engine request for <paramref name="input"/> with <paramref name="voice"/> (null: references given in the
    /// request itself, or zero-shot). Paths in the request are relative to the voices folder.
    /// </summary>
    public EngineRequest Build(string input, VoiceSpec? voice, VoiceStore store)
    {
        string? refWav = String("ref_wav", "reference_wav", "reference_audio") is string r ? store.ResolvePath(r) : voice?.RefWav;
        string? promptWav = String("prompt_wav", "prompt_audio") is string p ? store.ResolvePath(p) : voice?.PromptWav;
        string? promptText = String("prompt_text") ?? voice?.PromptText;
        if (Bool("no_ref") == true) refWav = promptWav = promptText = null;
        if ((promptWav is null) != (promptText is null))
        {
            throw new SpeechOptionException("prompt_wav and prompt_text must be given together.", "prompt_text");
        }
        foreach (var f in new[] { refWav, promptWav })
        {
            if (f is not null && !File.Exists(f)) throw new SpeechOptionException($"Audio file not found: {f}", "ref_wav");
        }

        var req = new EngineRequest
        {
            Text = input,
            // "caption" / "control": the Irodori and Python CLI names of a voice description.
            Style = String("style", "control", "caption", "instruct"),
            RefWav = refWav,
            PromptWav = promptWav,
            PromptText = promptText,
            CfgValue = Double("cfg_value", "cfg_scale") ?? 2.0,
            InferenceTimesteps = Int("inference_timesteps", "num_steps") ?? 10,
            MinLen = Int("min_len") ?? 2,
            MaxLen = Int("max_len") ?? 4096,
            RetryBadCase = Bool("retry_badcase") ?? true,
            RetryBadCaseMaxTimes = Int("retry_badcase_max_times") ?? 3,
            RetryBadCaseRatioThreshold = Double("retry_badcase_ratio_threshold") ?? 6.0,
            Seed = Long("seed"),
        };
        if (req.CfgValue is <= 0 or > 20) throw new SpeechOptionException("cfg_value must be in (0, 20].", "cfg_value");
        if (req.InferenceTimesteps is < 1 or > 100) throw new SpeechOptionException("inference_timesteps must be between 1 and 100.", "inference_timesteps");
        if (req.MinLen < 0 || req.MaxLen < 1 || req.MaxLen < req.MinLen) throw new SpeechOptionException("Invalid min_len / max_len.", "max_len");
        return req;
    }
}
