using System.Text;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;
using VoxCpm.Audio;

namespace VoxCpm.Server.Tests;

/// <summary>Records requests and returns a fixed tone instead of running the model.</summary>
public sealed class FakeEngine : ISpeechEngine
{
    public const int Rate = 48000;

    public List<EngineRequest> Requests { get; } = new();
    public List<EngineRequest> Prepared { get; } = new();
    public string Description => "fake";
    public int SampleRate => Rate;
    public double Seconds { get; set; } = 0.5;
    /// <summary>Pieces handed to the chunk callback when streaming.</summary>
    public int StreamPieces { get; set; } = 3;
    public Exception? Throw { get; set; }

    public EngineResult Synthesize(EngineRequest request, Action<float[]>? onChunk, CancellationToken cancellationToken)
    {
        lock (Requests) Requests.Add(request);
        if (Throw is not null) throw Throw;
        var audio = new float[(int)(Rate * Seconds)];
        for (int i = 0; i < audio.Length; i++) audio[i] = 0.25f * MathF.Sin(2 * MathF.PI * 440 * i / Rate);
        if (onChunk is not null)
        {
            int size = (audio.Length + StreamPieces - 1) / StreamPieces;
            for (int start = 0; start < audio.Length; start += size) onChunk(audio[start..Math.Min(audio.Length, start + size)]);
        }
        return new EngineResult(audio, Rate, request.Seed ?? 1234);
    }

    public void Prepare(EngineRequest request)
    {
        lock (Prepared) Prepared.Add(request);
    }
}

/// <summary>Runs <see cref="ServerApp"/> on a free loopback port with a temporary voices directory.</summary>
public sealed class TestServer : IAsyncDisposable
{
    private readonly WebApplication _app;

    public FakeEngine Engine { get; }
    public string VoicesDir { get; }
    public HttpClient Client { get; }

    private TestServer(WebApplication app, string voicesDir, string baseUrl, FakeEngine engine)
    {
        _app = app;
        VoicesDir = voicesDir;
        Engine = engine;
        Client = new HttpClient { BaseAddress = new Uri(baseUrl) };
    }

    public static async Task<TestServer> StartAsync(Action<string>? prepareVoices = null, string? apiKey = null, string? defaultVoice = null,
        bool allowNoRef = true, JsonObject? defaults = null)
    {
        var dir = Path.Combine(Path.GetTempPath(), "voxcpm-server-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        prepareVoices?.Invoke(dir);
        var engine = new FakeEngine();
        var app = ServerApp.Build(new ServerOptions
        {
            Host = "127.0.0.1",
            Port = 0,
            ApiKey = apiKey,
            DefaultVoice = defaultVoice,
            AllowNoRef = allowNoRef,
            VoicesDir = dir,
            Log = _ => { },
            Defaults = defaults,
        }, engine);
        await app.StartAsync();
        var address = app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First();
        return new TestServer(app, dir, address, engine);
    }

    public Task<HttpResponseMessage> SpeechAsync(JsonObject body, string? bearer = null)
    {
        var req = new HttpRequestMessage(HttpMethod.Post, "/v1/audio/speech")
        {
            Content = new StringContent(body.ToJsonString(), Encoding.UTF8, "application/json"),
        };
        if (bearer is not null) req.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", bearer);
        return Client.SendAsync(req);
    }

    public async ValueTask DisposeAsync()
    {
        Client.Dispose();
        await _app.StopAsync();
        await _app.DisposeAsync();
        try { Directory.Delete(VoicesDir, recursive: true); } catch (IOException) { }
    }

    public static void WriteWav(string path, double seconds = 0.2)
    {
        var samples = new float[(int)(16000 * seconds)];
        WavFile.WritePcm16(path, new[] { samples }, 16000);
    }
}
