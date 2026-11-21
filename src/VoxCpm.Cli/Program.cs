// Ported from VoxCPM (https://github.com/OpenBMB/VoxCPM, Apache License 2.0, Copyright OpenBMB): src/voxcpm/cli.py
// Modified: re-implemented in C# on TorchSharp (inference only). See NOTICE.
using System.Diagnostics;
using System.Globalization;
using TorchSharp;
using VoxCpm;
using VoxCpm.Audio;

// voxcpm-tts: a C# port of the VoxCPM CLI (voxcpm/cli.py), single-utterance synthesis.
var a = Args.Parse(args);
if (a.Has("help") || a.Has("h") || args.Length == 0)
{
    Console.WriteLine("""
        Usage: voxcpm-tts --text TEXT --output OUT.wav [options]

          --text, -t TEXT            Text to synthesize
          --control TEXT             Voice design / style instruction (put in parentheses before the text)
          --reference-audio WAV      Reference audio for voice cloning
          --prompt-audio WAV         Prompt audio to continue from (requires --prompt-text or --prompt-file)
          --prompt-text, -pt TEXT    Transcript of the prompt audio
          --prompt-file FILE         File with the transcript of the prompt audio
          --output, -o WAV           Output WAV (48 kHz, 16-bit)
          --cfg-value F              CFG guidance scale (default 2.0)
          --inference-timesteps N    Diffusion steps (default 10)
          --seed N                   Random seed
          --max-len N                Maximum patches (default 4096)
          --no-retry                 Disable bad-case retries
          --streaming                Generate with streaming (prints the time to the first chunk)
          --model-path DIR           Local checkpoint directory
          --hf-model-id ID           Hugging Face repo id (default openbmb/VoxCPM2)
          --device D                 auto, cpu, cuda, cuda:N (default auto)
          --precision P              auto, fp32, tf32, bf16 (default auto: bf16 on CUDA, fp32 on CPU)
          --vae-device D             Device of the AudioVAE (default: same as --device)
          --quantize auto|none|int8  int8: LM weights as int8 (~1.6 GB less memory, slower); auto (default): int8 only when GPU memory is short
          --threads N                CPU threads for libtorch
          --repeat N                 Synthesize N times (timing), keep the last
        """);
    return 0;
}

string text = a.Get("text") ?? a.Get("t") ?? throw new ArgumentException("--text is required");
string output = a.Get("output") ?? a.Get("o") ?? throw new ArgumentException("--output is required");
string? promptText = a.Get("prompt-text") ?? a.Get("pt");
if (a.Get("prompt-file") is { } promptFile) promptText = File.ReadAllText(promptFile).Trim();
string? promptAudio = a.Get("prompt-audio");
if ((promptAudio is null) != (promptText is null))
{
    Console.Error.WriteLine("--prompt-audio and --prompt-text/--prompt-file must be given together.");
    return 2;
}
if (a.Int("threads") is int threads) torch.set_num_threads(threads);

var sw = Stopwatch.StartNew();
Action<string> log = m => Console.Error.WriteLine(m);
string? modelPath = a.Get("model-path");
if (modelPath is null)
{
    long lastReport = 0;
    var progress = new Progress<(string File, long Downloaded, long? Total)>(p =>
    {
        if (p.Downloaded - lastReport < (64L << 20) && p.Downloaded != p.Total) return;
        lastReport = p.Downloaded;
        Console.Error.WriteLine(p.Total is long t ? $"{p.File}: {p.Downloaded >> 20} / {t >> 20} MB" : $"{p.File}: {p.Downloaded >> 20} MB");
    });
    modelPath = await VoxCpm.IO.HuggingFaceHub.DownloadCheckpointAsync(a.Get("hf-model-id") ?? VoxCpmRuntime.DefaultRepo, progress);
}
var tts = VoxCpmRuntime.Load(new RuntimeOptions
{
    ModelDirectory = modelPath,
    Device = a.Get("device") ?? "auto",
    Precision = a.Get("precision") ?? "auto",
    VaeDevice = a.Get("vae-device"),
    Quantize = a.Get("quantize") ?? "auto",
    Log = log,
});
using var _ = tts;
Console.Error.WriteLine($"Loaded in {sw.Elapsed.TotalSeconds:F1} s ({tts.Device}, {tts.DType}{(tts.Int8 ? ", int8 LM weights" : "")})");

var request = new SynthesisRequest
{
    Text = text,
    Style = a.Get("control"),
    ReferenceAudio = a.Get("reference-audio") is { } r ? WavFile.Read(r) : null,
    PromptAudio = promptAudio is null ? null : WavFile.Read(promptAudio),
    PromptText = promptText,
    CfgValue = a.Double("cfg-value") ?? 2.0,
    InferenceTimesteps = a.Int("inference-timesteps") ?? 10,
    MaxLen = a.Int("max-len") ?? 4096,
    RetryBadCase = !a.Has("no-retry"),
    Seed = a.Get("seed") is { } s ? long.Parse(s, CultureInfo.InvariantCulture) : null,
};


VoxCpm.Model.GenerationOutput result;
int repeat = Math.Max(1, a.Int("repeat") ?? 1);
for (int run = 0; ; run++)
{
    sw.Restart();
    if (a.Has("streaming"))
    {
        double? first = null;
        result = tts.SynthesizeStreaming(request, _ => first ??= sw.Elapsed.TotalSeconds);
        Console.Error.WriteLine($"First chunk after {first:F2} s");
    }
    else
    {
        result = tts.Synthesize(request);
    }
    using (result)
    {
        double seconds = result.Audio.Length / (double)result.SampleRate;
        double elapsed = sw.Elapsed.TotalSeconds;
        string rtf = $"{seconds:F2} s audio in {elapsed:F2} s (RTF {elapsed / Math.Max(seconds, 1e-6):F2}), seed {result.Seed}";
        if (run + 1 < repeat)
        {
            Console.Error.WriteLine($"Run {run + 1}: {rtf}");
            continue;
        }
        WavFile.WritePcm16(output, new[] { result.Audio }, result.SampleRate);
        Console.Error.WriteLine($"Saved {output}: {rtf}");
        break;
    }
}
return 0;

/// <summary>Minimal <c>--name value</c> / <c>--flag</c> parser.</summary>
internal sealed class Args
{
    private readonly Dictionary<string, string?> _values = new(StringComparer.Ordinal);

    public static Args Parse(string[] args)
    {
        var r = new Args();
        for (int i = 0; i < args.Length; i++)
        {
            var arg = args[i];
            if (!arg.StartsWith('-')) throw new ArgumentException($"Unexpected argument '{arg}'.");
            var name = arg.TrimStart('-');
            string? value = null;
            int eq = name.IndexOf('=');
            if (eq >= 0)
            {
                value = name[(eq + 1)..];
                name = name[..eq];
            }
            else if (i + 1 < args.Length && !IsOption(args[i + 1]))
            {
                value = args[++i];
            }
            r._values[name] = value;
        }
        return r;
    }

    // "-5" (a negative number) is a value, "--x" / "-x" an option.
    private static bool IsOption(string s) => s.Length > 1 && s[0] == '-' && !char.IsDigit(s[1]);

    public bool Has(string name) => _values.ContainsKey(name);
    public string? Get(string name) => _values.TryGetValue(name, out var v) ? v : null;
    public int? Int(string name) => Get(name) is { } v ? int.Parse(v, CultureInfo.InvariantCulture) : null;
    public double? Double(string name) => Get(name) is { } v ? double.Parse(v, CultureInfo.InvariantCulture) : null;
}
