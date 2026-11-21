// Derived from Irodori-TTS-csharp (MIT License, Copyright (c) 2026 Aratako). See NOTICE.
using System.Text.Json.Nodes;

namespace VoxCpm.Server;

/// <summary>
/// A voice of the voices folder. <see cref="RefWav"/>: clip for voice cloning; <see cref="PromptWav"/> +
/// <see cref="PromptText"/>: clip and transcript the generation continues from; <see cref="Options"/>: request
/// defaults (style, cfg_value, seed, ...). No audio at all: zero-shot (with an optional style = voice design).
/// </summary>
public sealed record VoiceSpec(string Id, string Type, string? RefWav, string? PromptWav, string? PromptText, JsonObject Options)
{
    public bool NoRef => RefWav is null && PromptWav is null;

    /// <summary>The files behind the voice, for the cache key and the listing.</summary>
    public IEnumerable<string> Files =>
        new[] { RefWav, PromptWav }.Where(f => f is not null).Distinct(StringComparer.OrdinalIgnoreCase)!;
}

/// <summary>
/// Voices from the voices folder, rescanned on every lookup so files can be added while the server runs:
/// <list type="bullet">
/// <item><c>name.wav</c>: voice cloning from the clip (VoxCPM2 reference mode).</item>
/// <item><c>name.wav</c> + <c>name.txt</c> (its exact transcript): "ultimate cloning" — the clip is both the
/// reference and the prompt the generation continues from.</item>
/// <item><c>name.json</c>: <c>{"style": "...", "seed": 42, "cfg_value": 2.0, "inference_timesteps": 10,
/// "ref_wav": "a.wav", "prompt_wav": "b.wav", "prompt_text": "..."}</c>. Without audio it is a designed voice
/// (zero-shot with a style); next to <c>name.wav</c> it adds options (e.g. a style for controllable cloning).</item>
/// <item><c>none</c> (when allowed): zero-shot.</item>
/// </list>
/// </summary>
public sealed class VoiceStore
{
    public const string NoneVoice = "none";
    public static readonly string[] NoRefIds = ["none", "no_ref", "no-ref", "null", "text-only"];

    private readonly string _dir;
    private readonly bool _allowNoRef;

    public VoiceStore(string directory, bool allowNoRef)
    {
        _dir = Path.GetFullPath(directory);
        _allowNoRef = allowNoRef;
        Directory.CreateDirectory(_dir);
    }

    public string DirectoryPath => _dir;

    public static bool IsSupportedFile(string file) =>
        file.EndsWith(".wav", StringComparison.OrdinalIgnoreCase) || file.EndsWith(".json", StringComparison.OrdinalIgnoreCase);

    public string ResolvePath(string p) => Path.IsPathRooted(p) ? p : Path.GetFullPath(Path.Combine(_dir, p));

    public IReadOnlyList<VoiceSpec> List(out List<string> warnings)
    {
        warnings = new List<string>();
        var result = new Dictionary<string, VoiceSpec>(StringComparer.OrdinalIgnoreCase);
        var files = Directory.EnumerateFiles(_dir).ToList();
        foreach (var wav in files.Where(f => f.EndsWith(".wav", StringComparison.OrdinalIgnoreCase)))
        {
            var id = Path.GetFileNameWithoutExtension(wav);
            var txt = Path.ChangeExtension(wav, ".txt");
            string? transcript = File.Exists(txt) ? File.ReadAllText(txt).Trim() : null;
            result[id] = string.IsNullOrEmpty(transcript)
                ? new VoiceSpec(id, "reference", wav, null, null, new JsonObject())
                : new VoiceSpec(id, "continuation", wav, wav, transcript, new JsonObject());
        }
        foreach (var json in files.Where(f => f.EndsWith(".json", StringComparison.OrdinalIgnoreCase)))
        {
            var id = Path.GetFileNameWithoutExtension(json);
            if (id.Equals("voices", StringComparison.OrdinalIgnoreCase)) continue;
            try
            {
                var o = JsonNode.Parse(File.ReadAllText(json)) as JsonObject ?? throw new FormatException("not a JSON object");
                result.TryGetValue(id, out var wavVoice);
                string? refWav = o["ref_wav"]?.GetValue<string>() is string r ? ResolvePath(r) : wavVoice?.RefWav;
                string? promptWav = o["prompt_wav"]?.GetValue<string>() is string p ? ResolvePath(p) : wavVoice?.PromptWav;
                string? promptText = o["prompt_text"]?.GetValue<string>() ?? wavVoice?.PromptText;
                if (o["no_ref"]?.GetValue<bool>() == true) refWav = promptWav = promptText = null;
                if ((promptWav is null) != (promptText is null))
                {
                    warnings.Add($"{Path.GetFileName(json)}: prompt_wav and prompt_text must be given together.");
                    continue;
                }
                var options = new JsonObject();
                foreach (var (k, v) in o)
                {
                    if (k is "ref_wav" or "prompt_wav" or "prompt_text" or "no_ref") continue;
                    options[k] = v?.DeepClone();
                }
                string type = refWav is null && promptWav is null ? "design"
                    : promptWav is not null ? "continuation" : "reference";
                result[id] = new VoiceSpec(id, type, refWav, promptWav, promptText, options);
            }
            catch (Exception ex)
            {
                warnings.Add($"{Path.GetFileName(json)} could not be read: {ex.Message}");
            }
        }
        if (_allowNoRef) result.TryAdd(NoneVoice, new VoiceSpec(NoneVoice, "none", null, null, null, new JsonObject()));
        return result.Values.OrderBy(v => v.Id, StringComparer.OrdinalIgnoreCase).ToList();
    }

    public VoiceSpec? Find(string id)
    {
        id = id.Trim();
        var found = List(out _).FirstOrDefault(v => v.Id.Equals(id, StringComparison.OrdinalIgnoreCase));
        if (found is null && _allowNoRef && NoRefIds.Contains(id, StringComparer.OrdinalIgnoreCase))
        {
            return new VoiceSpec(id, "none", null, null, null, new JsonObject());
        }
        return found;
    }

    /// <summary>Save an uploaded <c>.wav</c> (and optionally its transcript) as <paramref name="id"/>.</summary>
    public void Save(string id, string originalName, Stream content, string? transcript)
    {
        if (string.IsNullOrWhiteSpace(id) || id.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0 || id is "." or "..")
        {
            throw new ArgumentException($"Invalid voice_id: {id}");
        }
        string ext = Path.GetExtension(originalName).ToLowerInvariant();
        if (ext is not (".wav" or ".json")) throw new ArgumentException("Only .wav and .json voice files are supported.");
        if (ext == ".wav") DeleteFile(Path.Combine(_dir, id + ".txt"));
        var path = Path.Combine(_dir, id + ext);
        using (var fs = File.Create(path)) content.CopyTo(fs);
        if (ext == ".wav" && !string.IsNullOrWhiteSpace(transcript)) File.WriteAllText(Path.Combine(_dir, id + ".txt"), transcript.Trim());
    }

    /// <summary>Delete <c>id.wav</c>, <c>id.txt</c> and <c>id.json</c>. Returns false when nothing was deleted.</summary>
    public bool Delete(string id)
    {
        bool any = false;
        foreach (var ext in new[] { ".wav", ".txt", ".json" }) any |= DeleteFile(Path.Combine(_dir, id + ext));
        return any;
    }

    private static bool DeleteFile(string path)
    {
        if (!File.Exists(path)) return false;
        File.Delete(path);
        return true;
    }
}
