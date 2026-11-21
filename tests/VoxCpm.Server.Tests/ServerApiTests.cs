using System.Net;
using System.Text.Json.Nodes;
using VoxCpm.Audio;

namespace VoxCpm.Server.Tests;

public class ServerApiTests
{
    private static JsonObject Speech(string voice = "none", string input = "こんにちは。") =>
        new() { ["model"] = "voxcpm-tts", ["input"] = input, ["voice"] = voice, ["response_format"] = "wav" };

    [Fact]
    public async Task HealthReportsVoicesDir()
    {
        await using var s = await TestServer.StartAsync();
        var json = JsonNode.Parse(await s.Client.GetStringAsync("/health"))!;
        Assert.Equal("ok", (string?)json["status"]);
        // Speech.dll's launcher recognizes its own server by this path.
        Assert.Equal(Path.GetFullPath(s.VoicesDir), (string?)json["voices_dir"]);
        Assert.Equal("voxcpm-tts", (string?)json["model"]);
    }

    [Fact]
    public async Task ListsVoicesByFileConvention()
    {
        await using var s = await TestServer.StartAsync(dir =>
        {
            TestServer.WriteWav(Path.Combine(dir, "alice.wav"));
            TestServer.WriteWav(Path.Combine(dir, "bob.wav"));
            File.WriteAllText(Path.Combine(dir, "bob.txt"), "ボブの書き起こし");
            File.WriteAllText(Path.Combine(dir, "carol.json"), """{"style": "落ち着いた女性", "seed": 7}""");
            File.WriteAllText(Path.Combine(dir, "alice.json"), """{"style": "楽しそうに"}""");
        });
        var data = JsonNode.Parse(await s.Client.GetStringAsync("/v1/audio/voices"))!["data"]!.AsArray();
        var byId = data.ToDictionary(v => (string)v!["id"]!, v => v!);
        Assert.Equal(new[] { "alice", "bob", "carol", "none" }, byId.Keys.Order().ToArray());
        Assert.Equal("reference", (string?)byId["alice"]["type"]);
        Assert.Equal("楽しそうに", (string?)byId["alice"]["style"]);
        Assert.Equal("continuation", (string?)byId["bob"]["type"]);
        Assert.Equal("ボブの書き起こし", (string?)byId["bob"]["prompt_text"]);
        Assert.Equal("design", (string?)byId["carol"]["type"]);
        Assert.True((bool)byId["none"]["no_ref"]!);
    }

    [Fact]
    public async Task VoiceOptionsAndRequestOptionsReachTheEngine()
    {
        await using var s = await TestServer.StartAsync(dir =>
        {
            TestServer.WriteWav(Path.Combine(dir, "bob.wav"));
            File.WriteAllText(Path.Combine(dir, "bob.txt"), "書き起こし");
            File.WriteAllText(Path.Combine(dir, "bob.json"), """{"style": "優しく", "seed": 7, "cfg_value": 1.5}""");
        });
        var body = Speech("bob");
        // What Speech.dll sends: the voice definition's options under "irodori".
        body["irodori"] = new JsonObject { ["seed"] = 11, ["num_steps"] = 6 };
        body["voxcpm"] = new JsonObject { ["cfg_value"] = "2.5" };
        var res = await s.SpeechAsync(body);
        Assert.Equal(HttpStatusCode.OK, res.StatusCode);
        var r = Assert.Single(s.Engine.Requests);
        Assert.Equal("優しく", r.Style);
        Assert.Equal(11, r.Seed);
        Assert.Equal(6, r.InferenceTimesteps);
        Assert.Equal(2.5, r.CfgValue);
        Assert.EndsWith("bob.wav", r.RefWav);
        Assert.Equal(r.RefWav, r.PromptWav);
        Assert.Equal("書き起こし", r.PromptText);
    }

    [Fact]
    public async Task StreamsWavAndAudioIsComplete()
    {
        await using var s = await TestServer.StartAsync();
        var body = Speech();
        body["voxcpm"] = new JsonObject { ["stream"] = true };
        var res = await s.SpeechAsync(body);
        Assert.Equal(HttpStatusCode.OK, res.StatusCode);
        Assert.Equal("audio/wav", res.Content.Headers.ContentType?.MediaType);
        var bytes = await res.Content.ReadAsByteArrayAsync();
        Assert.Equal(44 + FakeEngine.Rate / 2 * 2, bytes.Length);
        Assert.NotNull(s.Engine.Requests.Single());
    }

    [Theory]
    [InlineData("flac", "audio/flac")]
    [InlineData("pcm", "audio/pcm")]
    public async Task OtherFormats(string format, string contentType)
    {
        await using var s = await TestServer.StartAsync();
        var body = Speech();
        body["response_format"] = format;
        var res = await s.SpeechAsync(body);
        Assert.Equal(HttpStatusCode.OK, res.StatusCode);
        Assert.Equal(contentType, res.Content.Headers.ContentType?.MediaType);
        if (format == "pcm") Assert.Equal(FakeEngine.Rate / 2 * 2, (await res.Content.ReadAsByteArrayAsync()).Length);
    }

    [Fact]
    public async Task NonStreamingWavHasExactHeaderAndSeed()
    {
        await using var s = await TestServer.StartAsync();
        var body = Speech();
        body["voxcpm"] = new JsonObject { ["seed"] = 99 };
        var res = await s.SpeechAsync(body);
        Assert.Equal("99", res.Headers.GetValues("X-VoxCPM-Seed").Single());
        var audio = WavFile.Read(await res.Content.ReadAsStreamAsync());
        Assert.Equal(FakeEngine.Rate, audio.SampleRate);
        Assert.Equal(FakeEngine.Rate / 2, audio.Length);
        Assert.Null(s.Engine.Requests.Single().RefWav);
    }

    [Fact]
    public async Task LongInputIsChunkedWithOneSeed()
    {
        await using var s = await TestServer.StartAsync();
        var text = string.Concat(Enumerable.Repeat("これはとても長い文章のテストです、句読点で区切られます。", 6));
        var res = await s.SpeechAsync(Speech(input: text));
        Assert.Equal(HttpStatusCode.OK, res.StatusCode);
        Assert.True(s.Engine.Requests.Count > 1);
        Assert.Single(s.Engine.Requests.Select(r => r.Seed).Distinct());
        Assert.Equal(text, string.Concat(s.Engine.Requests.Select(r => r.Text)));
    }

    [Fact]
    public async Task Errors()
    {
        await using var s = await TestServer.StartAsync(apiKey: null);
        Assert.Equal(HttpStatusCode.NotFound, (await s.SpeechAsync(Speech("missing"))).StatusCode);
        var bad = Speech();
        bad["model"] = "other";
        Assert.Equal(HttpStatusCode.NotFound, (await s.SpeechAsync(bad)).StatusCode);
        var fmt = Speech();
        fmt["response_format"] = "mp3";
        Assert.Equal(HttpStatusCode.BadRequest, (await s.SpeechAsync(fmt)).StatusCode);
        var opt = Speech();
        opt["voxcpm"] = new JsonObject { ["cfg_value"] = "abc" };
        var res = await s.SpeechAsync(opt);
        Assert.Equal(HttpStatusCode.BadRequest, res.StatusCode);
        Assert.Equal("cfg_value", (string?)JsonNode.Parse(await res.Content.ReadAsStringAsync())!["error"]!["param"]);
        Assert.Empty(s.Engine.Requests);
    }

    [Fact]
    public async Task ApiKeyIsRequiredForV1()
    {
        await using var s = await TestServer.StartAsync(apiKey: "secret");
        Assert.Equal(HttpStatusCode.Unauthorized, (await s.SpeechAsync(Speech())).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await s.SpeechAsync(Speech(), "secret")).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await s.Client.GetAsync("/health")).StatusCode);
    }

    [Fact]
    public async Task UploadAndDeleteVoice()
    {
        await using var s = await TestServer.StartAsync();
        var wavPath = Path.Combine(Path.GetTempPath(), $"voxcpm-up-{Guid.NewGuid():N}.wav");
        TestServer.WriteWav(wavPath);
        try
        {
            using var form = new MultipartFormDataContent();
            form.Add(new ByteArrayContent(await File.ReadAllBytesAsync(wavPath)), "file", "dave.wav");
            form.Add(new StringContent("デイブの書き起こし"), "prompt_text");
            var res = await s.Client.PostAsync("/v1/audio/voices", form);
            Assert.Equal(HttpStatusCode.OK, res.StatusCode);
            var v = JsonNode.Parse(await res.Content.ReadAsStringAsync())!;
            Assert.Equal("dave", (string?)v["id"]);
            Assert.Equal("continuation", (string?)v["type"]);
            Assert.True(File.Exists(Path.Combine(s.VoicesDir, "dave.txt")));
            Assert.Equal(HttpStatusCode.OK, (await s.Client.DeleteAsync("/v1/audio/voices/dave")).StatusCode);
            Assert.False(File.Exists(Path.Combine(s.VoicesDir, "dave.wav")));
        }
        finally
        {
            File.Delete(wavPath);
        }
    }

    [Fact]
    public void PrepareVoicesEncodesEveryVoiceWithAudio()
    {
        var dir = Path.Combine(Path.GetTempPath(), "voxcpm-server-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            TestServer.WriteWav(Path.Combine(dir, "a.wav"));
            File.WriteAllText(Path.Combine(dir, "b.json"), """{"style": "x"}""");
            var engine = new FakeEngine();
            ServerApp.PrepareVoices(engine, new VoiceStore(dir, true), _ => { });
            var p = Assert.Single(engine.Prepared);
            Assert.EndsWith("a.wav", p.RefWav);
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }
}
