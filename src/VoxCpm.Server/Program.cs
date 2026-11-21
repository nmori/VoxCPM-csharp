// Derived from Irodori-TTS-csharp (MIT License, Copyright (c) 2026 Aratako). See NOTICE.
using System.Globalization;
using System.Text.Json.Nodes;
using TorchSharp;
using VoxCpm;
using VoxCpm.Server;

var opts = Args.Parse(args);
if (opts.ContainsKey("help") || opts.ContainsKey("h"))
{
    Console.WriteLine(Args.Usage);
    return 0;
}
string? Env(string name) => Environment.GetEnvironmentVariable(name) is { Length: > 0 } e ? e.Trim() : null;
string? Opt(string key, string env) => Args.Str(opts, key) ?? Env(env);

// Started without a console (e.g. by Speech.dll): write the log to a file instead.
if (Args.Str(opts, "log-file") is string logFile)
{
    var dir = Path.GetDirectoryName(Path.GetFullPath(logFile));
    if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
    var writer = new StreamWriter(new FileStream(logFile, FileMode.Create, FileAccess.Write, FileShare.ReadWrite)) { AutoFlush = true };
    Console.SetOut(writer);
    Console.SetError(writer);
    Console.WriteLine($"[server] started {DateTime.Now:yyyy-MM-dd HH:mm:ss} pid={Environment.ProcessId} args={string.Join(' ', args)}");
}

// Exit together with the process that launched us, so the model does not keep holding (GPU) memory.
using var parentExited = new CancellationTokenSource();
if (Args.Str(opts, "parent-pid") is string parentPidText)
{
    try
    {
        var parent = System.Diagnostics.Process.GetProcessById(int.Parse(parentPidText, CultureInfo.InvariantCulture));
        _ = Task.Run(async () =>
        {
            await parent.WaitForExitAsync();
            Console.WriteLine($"[server] parent process {parentPidText} exited; shutting down.");
            parentExited.Cancel();
        });
    }
    catch (ArgumentException)
    {
        Console.Error.WriteLine($"error: parent process {parentPidText} is not running.");
        return 1;
    }
}

ServerOptions options;
try
{
    options = new ServerOptions
    {
        Host = Opt("host", "VOXCPM_HOST") ?? "127.0.0.1",
        Port = int.Parse(Opt("port", "VOXCPM_PORT") ?? "8090", CultureInfo.InvariantCulture),
        ModelName = Opt("model-name", "VOXCPM_MODEL_NAME") ?? "voxcpm-tts",
        ApiKey = Opt("api-key", "VOXCPM_API_KEY"),
        DefaultVoice = Opt("default-voice", "VOXCPM_DEFAULT_VOICE"),
        AllowNoRef = !(Opt("allow-no-ref-voice", "VOXCPM_ALLOW_NO_REF_VOICE") ?? "true").Equals("false", StringComparison.OrdinalIgnoreCase),
        VoicesDir = Opt("voices-dir", "VOXCPM_VOICES_DIR") ?? Path.Combine(AppContext.BaseDirectory, "voices"),
        ChunkingEnabled = !(Opt("chunking", "VOXCPM_CHUNKING") ?? "true").Equals("false", StringComparison.OrdinalIgnoreCase),
        ChunkMinChars = int.Parse(Opt("chunk-min-chars", "VOXCPM_CHUNK_MIN_CHARS") ?? "80", CultureInfo.InvariantCulture),
        FirstSentenceChunkMinChars = Opt("first-sentence-chunk-min-chars", "VOXCPM_FIRST_SENTENCE_CHUNK_MIN_CHARS") is string fs
            ? int.Parse(fs, CultureInfo.InvariantCulture) : null,
        Defaults = DefaultsFrom(opts),
    };
}
catch (FormatException ex)
{
    Console.Error.WriteLine($"error: {ex.Message}");
    return 1;
}

// The port opens only after the model has loaded. When another server already holds it, exit at once with the
// code the launcher recognizes (it then starts again on a free port) instead of loading the model for nothing.
if (!PortIsFree(options.Host, options.Port))
{
    Console.Error.WriteLine($"error: port {options.Port} on {options.Host} is already in use.");
    return 3;
}

// --model-device / --model-precision: the names the Irodori launcher (Speech.dll) passes. The command line comes
// before the environment (VOXCPM_DEVICE / VOXCPM_PRECISION, which PlayEngine sets from the user's choice), so a
// --device written in the launcher's serverArgs still wins.
string device = Args.Str(opts, "device") ?? Args.Str(opts, "model-device") ?? Env("VOXCPM_DEVICE") ?? "auto";
string precision = Args.Str(opts, "precision") ?? Args.Str(opts, "model-precision") ?? Env("VOXCPM_PRECISION") ?? "auto";

// The distributed server carries only the CPU build of LibTorch (libtorch-cpu\); on an NVIDIA PC the CUDA build
// is downloaded once (%VOXCPM_CACHE%\runtime or %LOCALAPPDATA%\VoxCpm\runtime). Either way it has to be loaded
// before the first TorchSharp call. Builds that keep LibTorch next to the exe are left alone.
try
{
    int lastRuntimePercent = -1;
    string? libtorch = await VoxCpm.IO.LibTorchRuntime.Prepare(
        Args.Str(opts, "cuda-runtime-dir"),
        allowDownload: !opts.ContainsKey("no-cuda-download"),
        wantCuda: !device.StartsWith("cpu", StringComparison.OrdinalIgnoreCase),
        Console.WriteLine,
        new Progress<(string File, long Downloaded, long? Total)>(p =>
        {
            if (p.Total is not long total || total <= 0) return;
            int percent = (int)(p.Downloaded * 100 / total);
            if (percent / 10 == lastRuntimePercent / 10 && percent != 100) return;
            lastRuntimePercent = percent;
            Console.WriteLine($"[download] {p.File}: {percent}% ({p.Downloaded >> 20} / {total >> 20} MB)");
        }),
        parentExited.Token);
    if (libtorch is not null) Console.WriteLine($"[runtime] LibTorch: {libtorch}");
}
catch (OperationCanceledException) when (parentExited.IsCancellationRequested)
{
    return 0;
}

if (Args.Str(opts, "threads") is string threads) torch.set_num_threads(int.Parse(threads, CultureInfo.InvariantCulture));
string? vaeDevice = Opt("vae-device", "VOXCPM_VAE_DEVICE");
string? modelPath = Opt("model-path", "VOXCPM_MODEL_PATH") ?? Args.Str(opts, "checkpoint");
string repo = Opt("hf-model-id", "VOXCPM_HF_MODEL_ID") ?? Args.Str(opts, "hf-checkpoint") ?? VoxCpmRuntime.DefaultRepo;

VoxCpmRuntime tts;
try
{
    var watch = System.Diagnostics.Stopwatch.StartNew();
    if (modelPath is null)
    {
        Console.WriteLine($"[model] {repo} (cache: {VoxCpm.IO.HuggingFaceHub.CacheRoot})");
        int lastPercent = -1;
        var progress = new Progress<(string File, long Downloaded, long? Total)>(p =>
        {
            if (p.Total is not long total || total <= 0) return;
            int percent = (int)(p.Downloaded * 100 / total);
            if (percent / 10 == lastPercent / 10 && percent != 100) return;
            lastPercent = percent;
            Console.WriteLine($"[download] {p.File}: {percent}% ({p.Downloaded >> 20} / {total >> 20} MB)");
        });
        modelPath = await VoxCpm.IO.HuggingFaceHub.DownloadCheckpointAsync(repo, progress, parentExited.Token);
    }
    tts = VoxCpmRuntime.Load(new RuntimeOptions
    {
        ModelDirectory = modelPath,
        Device = device,
        Precision = precision,
        VaeDevice = vaeDevice,
        Log = Console.WriteLine,
    });
    Console.WriteLine($"[model] loaded in {watch.Elapsed.TotalSeconds:F1} s ({tts.Device}, {tts.DType})");
}
catch (OperationCanceledException) when (parentExited.IsCancellationRequested)
{
    return 0;
}
catch (Exception ex) when (ex is ArgumentException or FileNotFoundException or InvalidDataException or NotSupportedException or HttpRequestException)
{
    Console.Error.WriteLine($"error: {ex.Message}");
    return 1;
}
if (parentExited.IsCancellationRequested) return 0;

using var engine = new VoxCpmEngine(tts);
using var stopping = CancellationTokenSource.CreateLinkedTokenSource(parentExited.Token);
var app = ServerApp.Build(options, engine);
app.Lifetime.ApplicationStopping.Register(stopping.Cancel);
Console.WriteLine($"[server] VoxCPM2 C# server listening on http://{options.Host}:{options.Port} " +
                  $"(model='{options.ModelName}', voices='{Path.GetFullPath(options.VoicesDir)}', backend={engine.Description})");
// In the background so the port opens at once; a request arriving meanwhile waits (synthesis is serialized).
var warmup = !opts.ContainsKey("no-warmup") && tts.Device.type == DeviceType.CUDA ? Task.Run(() => Warmup(engine)) : Task.CompletedTask;
var prepare = warmup.ContinueWith(_ => ServerApp.PrepareVoices(engine, new VoiceStore(options.VoicesDir, options.AllowNoRef), Console.WriteLine, stopping.Token),
    TaskScheduler.Default);
await app.RunAsync(parentExited.Token);
stopping.Cancel();
await warmup;
await prepare;
return 0;

static void Warmup(ISpeechEngine engine)
{
    var watch = System.Diagnostics.Stopwatch.StartNew();
    try
    {
        engine.Synthesize(new EngineRequest { Text = "こんにちは。", Seed = 0, MaxLen = 8, RetryBadCase = false }, _ => { }, CancellationToken.None);
        Console.WriteLine($"[warmup] done in {watch.Elapsed.TotalSeconds:F1} s");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"[warmup] skipped: {ex.Message}");
    }
}

static bool PortIsFree(string host, int port)
{
    try
    {
        var address = host.Equals("localhost", StringComparison.OrdinalIgnoreCase)
            ? System.Net.IPAddress.Loopback
            : System.Net.IPAddress.TryParse(host, out var parsed) ? parsed : System.Net.IPAddress.Any;
        var listener = new System.Net.Sockets.TcpListener(address, port) { ExclusiveAddressUse = true };
        listener.Start();
        listener.Stop();
        return true;
    }
    catch (System.Net.Sockets.SocketException)
    {
        return false;
    }
}

// Server-wide defaults of speech options: --cfg-value, --inference-timesteps, --seed, --style, --response-format.
static JsonObject? DefaultsFrom(Dictionary<string, List<string>> o)
{
    var d = new JsonObject();
    foreach (var (arg, key) in new[] { ("cfg-value", "cfg_value"), ("inference-timesteps", "inference_timesteps"), ("seed", "seed"),
                                       ("style", "style"), ("response-format", "response_format"), ("max-len", "max_len") })
    {
        if (Args.Str(o, arg) is string v) d[key] = v;
    }
    return d.Count > 0 ? d : null;
}

internal static class Args
{
    public const string Usage = """
        Usage: voxcpm-tts-server [options]

        Server:
          --host <H>                 VOXCPM_HOST, default 127.0.0.1
          --port <P>                 VOXCPM_PORT, default 8090
          --voices-dir <DIR>         VOXCPM_VOICES_DIR, default <exe>\voices
                                     name.wav (clone) / name.wav + name.txt (transcript: ultimate cloning) /
                                     name.json ({"style": ..., "seed": ..., "cfg_value": ..., "ref_wav": ...})
          --model-name <NAME>        VOXCPM_MODEL_NAME, default voxcpm-tts (the "model" of /v1/audio/speech)
          --api-key <KEY>            VOXCPM_API_KEY (Bearer token for /v1/*)
          --default-voice <ID>       VOXCPM_DEFAULT_VOICE
          --allow-no-ref-voice <B>   default true (voice "none": zero-shot)
          --chunking <B>             default true: split long input at punctuation
          --chunk-min-chars <N>      default 80 (non-space characters per piece)
          --first-sentence-chunk-min-chars <N>
          --log-file <PATH>          Write the log to a file (for launches without a console)
          --parent-pid <PID>         Exit when that process exits
          --no-warmup                Skip the warm-up synthesis on a GPU

        Speech defaults (a request or a voice's .json overrides them):
          --cfg-value <F>  --inference-timesteps <N>  --seed <N>  --style <TEXT>  --response-format wav|pcm|flac  --max-len <N>

        Model:
          --model-path <DIR>         Local checkpoint directory (alias --checkpoint)
          --hf-model-id <ID>         default openbmb/VoxCPM2 (alias --hf-checkpoint); cache: VOXCPM_CACHE or %LOCALAPPDATA%\VoxCpm\hf
          --device auto|cpu|cuda     (alias --model-device; VOXCPM_DEVICE)
          --precision auto|fp32|tf32|bf16   (alias --model-precision; VOXCPM_PRECISION; auto = bf16 on CUDA, fp32 on the CPU)
          --no-cuda-download         Do not download the CUDA build of LibTorch on an NVIDIA PC (use the CPU)
          --cuda-runtime-dir <DIR>   Keep / find the CUDA build of LibTorch in DIR. Default: downloaded once, when an
                                     NVIDIA driver is present, to %VOXCPM_CACHE%\runtime or else %LOCALAPPDATA%\VoxCpm\runtime
          --vae-device <D>           AudioVAE device (default: --device)
          --threads <N>              CPU threads for libtorch
        """;

    public static Dictionary<string, List<string>> Parse(string[] argv)
    {
        var result = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        string? current = null;
        foreach (var a in argv)
        {
            if (a.StartsWith("--", StringComparison.Ordinal))
            {
                current = a[2..];
                int eq = current.IndexOf('=');
                if (eq >= 0)
                {
                    result[current[..eq]] = new List<string> { current[(eq + 1)..] };
                    current = null;
                    continue;
                }
                result[current] = new List<string>();
            }
            else if (current is not null)
            {
                result[current].Add(a);
            }
        }
        return result;
    }

    /// <summary>The value of <c>--key</c>; "auto" counts as unset for the engine options.</summary>
    public static string? Str(Dictionary<string, List<string>> o, string key) =>
        o.TryGetValue(key, out var v) && v.Count > 0 ? v[0] : null;
}
