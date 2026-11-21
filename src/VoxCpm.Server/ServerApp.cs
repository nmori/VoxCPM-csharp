// Derived from Irodori-TTS-csharp (MIT License, Copyright (c) 2026 Aratako). See NOTICE.
using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;
using System.Threading.Channels;
using VoxCpm.Audio;

namespace VoxCpm.Server;

public sealed class ServerOptions
{
    public string Host { get; init; } = "127.0.0.1";
    public int Port { get; init; } = 8090;
    public string ModelName { get; init; } = "voxcpm-tts";
    public string? ApiKey { get; init; }
    public string? DefaultVoice { get; init; }
    public bool AllowNoRef { get; init; } = true;
    public string VoicesDir { get; init; } = "voices";
    public Action<string> Log { get; init; } = Console.WriteLine;
    /// <summary>Split long input into pieces at punctuation (a request can override it).</summary>
    public bool ChunkingEnabled { get; init; } = true;
    /// <summary>Non-space characters a piece needs before it may end at a punctuation mark.</summary>
    public int ChunkMinChars { get; init; } = 80;
    /// <summary>The same for the first boundary only (null: <see cref="ChunkMinChars"/>).</summary>
    public int? FirstSentenceChunkMinChars { get; init; }
    /// <summary>Server defaults of the speech options, used when neither the request nor the voice sets them.</summary>
    public JsonObject? Defaults { get; init; }
}

/// <summary>
/// OpenAI-compatible HTTP API with the endpoints and fields of the Irodori-TTS C# server, so Speech.dll can drive
/// it the same way: /health, /v1/models, /v1/audio/voices, /v1/audio/speech.
/// </summary>
public static class ServerApp
{
    public const int MaxInputChars = 4096;

    /// <summary>Encode every voice's clips (one at a time), so no voice is slow on its first request.</summary>
    public static void PrepareVoices(ISpeechEngine engine, VoiceStore store, Action<string> log, CancellationToken cancellationToken = default)
    {
        var watch = Stopwatch.StartNew();
        int count = 0;
        var all = store.List(out var warnings);
        foreach (var voice in all)
        {
            if (cancellationToken.IsCancellationRequested) return;
            if (PrepareVoice(engine, store, voice, log)) count++;
        }
        foreach (var w in warnings) log($"[voices] {w}");
        if (count > 0) log($"[voices] prepared {count} voice(s) in {watch.Elapsed.TotalSeconds:F1} s");
    }

    private static bool PrepareVoice(ISpeechEngine engine, VoiceStore store, VoiceSpec voice, Action<string> log)
    {
        if (voice.NoRef) return false;
        try
        {
            engine.Prepare(new SpeechOptions(new JsonObject(), voice.Options, null).Build("-", voice, store));
            return true;
        }
        catch (Exception ex)
        {
            log($"[voices] could not prepare '{voice.Id}': {ex.Message}");
            return false;
        }
    }

    private static List<string> SpeechPieces(ServerOptions o, SpeechOptions opts, string input)
    {
        bool enabled = opts.Bool("chunking_enabled", "chunking") ?? o.ChunkingEnabled;
        if (!enabled) return [input];
        int minChars = opts.Int("chunk_min_chars") ?? o.ChunkMinChars;
        if (minChars <= 0) throw new SpeechOptionException("chunk_min_chars must be greater than 0.", "chunk_min_chars");
        int? firstMin = opts.Int("first_sentence_chunk_min_chars") ?? o.FirstSentenceChunkMinChars;
        if (firstMin is <= 0) throw new SpeechOptionException("first_sentence_chunk_min_chars must be greater than 0.", "first_sentence_chunk_min_chars");
        return SpeechChunker.Split(input, minChars, firstMin);
    }

    public static WebApplication Build(ServerOptions o, ISpeechEngine engine)
    {
        var voices = new VoiceStore(o.VoicesDir, o.AllowNoRef);
        var builder = WebApplication.CreateSlimBuilder();
        builder.WebHost.UseUrls($"http://{o.Host}:{o.Port}");
        builder.Logging.SetMinimumLevel(LogLevel.Warning);
        var app = builder.Build();

        IResult Error(int status, string message, string type = "invalid_request_error", string? param = null) =>
            Results.Json(new JsonObject
            {
                ["error"] = new JsonObject { ["message"] = message, ["type"] = type, ["param"] = param, ["code"] = null },
            }, statusCode: status);

        app.Use(async (ctx, next) =>
        {
            try
            {
                await next(ctx);
            }
            catch (Exception ex) when (!ctx.Response.HasStarted && !ctx.RequestAborted.IsCancellationRequested)
            {
                o.Log($"[server] unhandled error: {ex}");
                await Error(500, ex.Message, "server_error").ExecuteAsync(ctx);
            }
        });

        app.Use(async (ctx, next) =>
        {
            if (o.ApiKey is not null && ctx.Request.Path.StartsWithSegments("/v1")
                && ctx.Request.Headers.Authorization.ToString() != $"Bearer {o.ApiKey}")
            {
                await Error(401, "Invalid or missing API key.", "authentication_error").ExecuteAsync(ctx);
                return;
            }
            await next(ctx);
        });

        app.MapGet("/", () => Results.Text(
            $"VoxCPM2 C# server. POST /v1/audio/speech {{\"model\":\"{o.ModelName}\",\"input\":\"...\",\"voice\":\"none\"}}; GET /v1/audio/voices; GET /health.\n"));

        app.MapGet("/health", () => Results.Json(new JsonObject
        {
            ["status"] = "ok",
            ["model"] = o.ModelName,
            ["backend"] = engine.Description,
            ["sample_rate"] = engine.SampleRate,
            ["voices_dir"] = voices.DirectoryPath,
            ["default_voice"] = o.DefaultVoice,
            ["allow_no_ref_voice"] = o.AllowNoRef,
            ["voices"] = new JsonObject
            {
                ["dir"] = voices.DirectoryPath,
                ["dir_exists"] = Directory.Exists(voices.DirectoryPath),
                ["files"] = Directory.Exists(voices.DirectoryPath) ? Directory.EnumerateFiles(voices.DirectoryPath).Count(VoiceStore.IsSupportedFile) : 0,
            },
            ["defaults"] = new JsonObject
            {
                ["response_format"] = o.Defaults?["response_format"]?.DeepClone() ?? "wav",
                ["chunking_enabled"] = o.ChunkingEnabled,
                ["chunk_min_chars"] = o.ChunkMinChars,
                ["first_sentence_chunk_min_chars"] = o.FirstSentenceChunkMinChars,
            },
        }));

        app.MapGet("/v1/models", () => Results.Json(new JsonObject
        {
            ["object"] = "list",
            ["data"] = new JsonArray(new JsonObject { ["id"] = o.ModelName, ["object"] = "model", ["created"] = 0, ["owned_by"] = "openbmb" }),
        }));

        JsonObject VoiceJson(VoiceSpec v) => new()
        {
            ["id"] = v.Id,
            ["object"] = "voice",
            ["type"] = v.Type,
            ["files"] = new JsonArray(v.Files.Select(f => (JsonNode)Path.GetFileName(f)).ToArray()),
            ["ref_wav"] = v.RefWav,
            ["prompt_wav"] = v.PromptWav,
            ["prompt_text"] = v.PromptText,
            ["style"] = v.Options["style"]?.DeepClone(),
            ["no_ref"] = v.NoRef,
        };

        app.MapGet("/v1/audio/voices", () =>
        {
            var list = voices.List(out var warnings);
            foreach (var w in warnings) o.Log($"[voices] {w}");
            return Results.Json(new JsonObject
            {
                ["object"] = "list",
                ["data"] = new JsonArray(list.Select(v => (JsonNode)VoiceJson(v)).ToArray()),
            });
        });

        app.MapGet("/v1/audio/voices/{id}", (string id) =>
            voices.Find(id) is { } v ? Results.Json(VoiceJson(v)) : Error(404, $"Voice '{id}' not found.", param: "voice_id"));

        async Task<IResult> Upload(HttpRequest request, string? fixedId)
        {
            if (!request.HasFormContentType) return Error(400, "Expected multipart/form-data with a 'file' field.");
            var form = await request.ReadFormAsync();
            var file = form.Files["file"];
            if (file is null) return Error(400, "Missing 'file' field.", param: "file");
            string id = fixedId ?? (form["voice_id"].ToString() is { Length: > 0 } given ? given : Path.GetFileNameWithoutExtension(file.FileName));
            try
            {
                using var s = file.OpenReadStream();
                voices.Save(id, file.FileName, s, form["prompt_text"].ToString());
            }
            catch (ArgumentException ex)
            {
                return Error(400, ex.Message, param: "file");
            }
            if (voices.Find(id) is not { } saved) return Error(500, "Voice was saved but could not be resolved.", "server_error");
            _ = Task.Run(() => PrepareVoice(engine, voices, saved, o.Log));
            return Results.Json(VoiceJson(saved));
        }

        app.MapPost("/v1/audio/voices", (HttpRequest r) => Upload(r, null)).DisableAntiforgery();
        app.MapPut("/v1/audio/voices/{id}", (HttpRequest r, string id) => Upload(r, id)).DisableAntiforgery();
        app.MapDelete("/v1/audio/voices/{id}", (string id) =>
            voices.Delete(id) ? Results.Json(new JsonObject { ["id"] = id, ["object"] = "voice", ["deleted"] = true })
                : Error(404, $"Voice '{id}' not found.", param: "voice_id"));

        app.MapPost("/v1/audio/speech", async (HttpRequest http) =>
        {
            try
            {
                return await Speech(http);
            }
            catch (Exception ex) when (ex is InvalidOperationException or FormatException or System.Text.Json.JsonException or OverflowException)
            {
                return Error(400, $"Invalid request: {ex.Message}");
            }
        });

        async Task<IResult> Speech(HttpRequest http)
        {
            JsonObject body;
            try
            {
                body = (await JsonNode.ParseAsync(http.Body))?.AsObject() ?? throw new FormatException();
            }
            catch (Exception)
            {
                return Error(400, "Request body must be a JSON object.");
            }

            string? model = body["model"]?.GetValue<string>();
            if (model is null) return Error(400, "'model' is required.", param: "model");
            if (!model.Equals(o.ModelName, StringComparison.Ordinal)) return Error(404, $"Model '{model}' not found. Use '{o.ModelName}'.", param: "model");
            string? input = body["input"]?.GetValue<string>();
            if (string.IsNullOrWhiteSpace(input)) return Error(400, "'input' is required.", param: "input");
            if (input.Length > MaxInputChars) return Error(400, $"'input' must be at most {MaxInputChars} characters.", param: "input");

            string format = (body["response_format"] ?? o.Defaults?["response_format"])?.GetValue<string>()?.Trim().ToLowerInvariant() ?? "wav";
            if (!AudioFormats.IsSupported(format))
            {
                return Error(400, $"response_format '{format}' is not supported (use {AudioFormats.SupportedList}).", param: "response_format");
            }
            double speed = body["speed"]?.GetValue<double>() ?? 1.0;
            if (speed is < 0.25 or > 4.0) return Error(400, "'speed' must be between 0.25 and 4.0.", param: "speed");

            string? voiceId = body["voice"] switch
            {
                JsonValue v => v.GetValue<string>(),
                JsonObject vo => vo["id"]?.GetValue<string>(),
                _ => null,
            };
            if (string.IsNullOrWhiteSpace(voiceId)) voiceId = o.DefaultVoice;

            VoiceSpec? voice = null;
            var probe = new SpeechOptions(body, null, null);
            bool requestHasAudio = probe.String("ref_wav", "reference_wav", "reference_audio", "prompt_wav", "prompt_audio") is not null;
            if (!requestHasAudio)
            {
                if (voiceId is null) return Error(400, "'voice' is required (no default voice is configured).", param: "voice");
                voice = voices.Find(voiceId);
                if (voice is null) return Error(404, $"Voice '{voiceId}' not found.", param: "voice");
            }

            var opts = new SpeechOptions(body, voice?.Options, o.Defaults);
            EngineRequest req;
            List<string> pieces;
            bool stream;
            try
            {
                req = opts.Build(input, voice, voices);
                pieces = SpeechPieces(o, opts, input);
                // Off by default: the response is then a complete WAV with exact sizes, which every player (and
                // Speech.dll, which saves and plays the file) reads. "stream": true sends each 160 ms patch as soon as
                // it is generated, under a WAV header with unknown sizes (the length is known only at the end).
                stream = (opts.Bool("stream") ?? false) && AudioFormats.IsProgressive(format);
            }
            catch (SpeechOptionException ex)
            {
                return Error(400, ex.Message, param: ex.Param);
            }
            // One seed for every piece, so the seed header reproduces the whole response.
            if (pieces.Count > 1 && req.Seed is null) req = req with { Seed = Random.Shared.NextInt64(uint.MaxValue) };

            var response = http.HttpContext.Response;
            var aborted = http.HttpContext.RequestAborted;
            var sw = Stopwatch.StartNew();

            // Generation runs on a worker; with streaming every 160 ms patch is written as soon as it is decoded.
            var chunks = Channel.CreateUnbounded<float[]>(new UnboundedChannelOptions { SingleReader = true });
            var work = Task.Run(() =>
            {
                var all = new List<float>();
                EngineResult? first = null;
                foreach (var piece in pieces)
                {
                    var r = engine.Synthesize(req with { Text = piece }, stream ? c => chunks.Writer.TryWrite(c) : null, aborted);
                    first ??= r;
                    all.AddRange(r.Audio);
                }
                return new EngineResult(all.ToArray(), first!.SampleRate, first.Seed);
            });
            _ = work.ContinueWith(t => { _ = t.Exception; chunks.Writer.TryComplete(); }, TaskScheduler.Default);

            void SetHeaders(long seed, int sampleRate)
            {
                response.Headers["X-VoxCPM-Seed"] = seed.ToString(CultureInfo.InvariantCulture);
                response.Headers["X-VoxCPM-Sample-Rate"] = sampleRate.ToString(CultureInfo.InvariantCulture);
                response.Headers.ContentDisposition = $"attachment; filename=\"speech.{format}\"";
            }

            void LogSpeech(long samples, int sampleRate, long? seed) =>
                o.Log($"[speech] voice={voice?.Id ?? "(request)"} chars={input.Length} pieces={pieces.Count} " +
                      $"audio={samples / (double)sampleRate:F2}s took={sw.Elapsed.TotalSeconds:F2}s seed={seed?.ToString(CultureInfo.InvariantCulture) ?? "?"}" +
                      $"{(speed != 1.0 ? $" speed={speed.ToString(CultureInfo.InvariantCulture)} (ignored)" : "")}");

            if (!await chunks.Reader.WaitToReadAsync(CancellationToken.None))
            {
                EngineResult result;
                try
                {
                    result = await work;
                }
                catch (Exception ex) when (ex is ArgumentException or FileNotFoundException or InvalidDataException or NotSupportedException)
                {
                    return Error(400, ex.Message);
                }
                catch (OperationCanceledException) when (aborted.IsCancellationRequested)
                {
                    o.Log($"[speech] client disconnected after {sw.Elapsed.TotalSeconds:F2}s");
                    return Results.Empty;
                }
                LogSpeech(result.Audio.Length, result.SampleRate, result.Seed);
                SetHeaders(result.Seed, result.SampleRate);
                return Results.Bytes(AudioFormats.Encode(result.Audio, result.SampleRate, format), AudioFormats.ContentType(format));
            }

            // Streaming: the seed is the request's (or drawn by the engine and reported in the log only).
            int sr = engine.SampleRate;
            if (req.Seed is long fixedSeed) SetHeaders(fixedSeed, sr);
            else response.Headers["X-VoxCPM-Sample-Rate"] = sr.ToString(CultureInfo.InvariantCulture);
            response.ContentType = AudioFormats.ContentType(format);
            long sent = 0;
            try
            {
                if (format == "wav") await response.Body.WriteAsync(WavFile.Pcm16StreamingHeader(1, sr), aborted);
                await foreach (var chunk in chunks.Reader.ReadAllAsync(aborted))
                {
                    await response.Body.WriteAsync(WavFile.ToPcm16Bytes(chunk), aborted);
                    await response.Body.FlushAsync(aborted);
                    sent += chunk.Length;
                }
                var done = await work;
                LogSpeech(sent, sr, done.Seed);
            }
            catch (OperationCanceledException) when (aborted.IsCancellationRequested)
            {
                o.Log($"[speech] client disconnected after {sw.Elapsed.TotalSeconds:F2}s");
            }
            catch (Exception ex)
            {
                o.Log($"[speech] failed while streaming: {ex.Message}");
                http.HttpContext.Abort();
            }
            return Results.Empty;
        }

        app.MapFallback(() => Error(404, "Not found."));
        return app;
    }
}
