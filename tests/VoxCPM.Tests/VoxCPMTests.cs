namespace VoxCPM.Tests;

public sealed class VoxCPMTests
{
    private sealed class TestRuntime : IVoxCpmRuntime
    {
        public float[] Generate(string text, GenerationOptions options)
            => new[] { 0.25f, -0.25f };

        public IEnumerable<float[]> GenerateStreaming(string text, GenerationOptions options)
            => new[] { new[] { 0.25f }, new[] { -0.25f } };
    }

    [Fact]
    public void FromPretrained_ReadsArchitectureAndSampleRate()
    {
        var directory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            File.WriteAllText(Path.Combine(directory, "config.json"),
                """{"architecture":"voxcpm2"}""");

            using var model = VoxCPMModel.FromPretrained(directory);

            Assert.Equal("voxcpm2", model.Architecture);
            Assert.Equal(48000, model.SampleRate);
        }
        finally
        {
            Directory.Delete(directory, true);
        }
    }

    [Fact]
    public void Generate_RequiresRuntimeAndValidatesText()
    {
        var directory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            File.WriteAllText(Path.Combine(directory, "config.json"),
                """{"architecture":"voxcpm"}""");
            using var model = VoxCPMModel.FromPretrained(directory);

            Assert.Throws<ArgumentException>(() => model.Generate(" "));
            Assert.Throws<NotSupportedException>(() => model.Generate("Hello"));
        }
        finally
        {
            Directory.Delete(directory, true);
        }
    }

    [Fact]
    public void Generate_NormalizesTextAndDelegatesToRuntime()
    {
        var directory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            File.WriteAllText(Path.Combine(directory, "config.json"),
                """{"architecture":"voxcpm","sample_rate":44100}""");
            using var model = VoxCPMModel.FromPretrained(directory, runtime: new TestRuntime());

            var samples = model.Generate("  Hello\r\n world  ");

            Assert.Equal(44100, model.SampleRate);
            Assert.Equal(new[] { 0.25f, -0.25f }, samples);
        }
        finally
        {
            Directory.Delete(directory, true);
        }
    }

    [Fact]
    public void WavFile_RoundTripsMonoSamples()
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".wav");
        try
        {
            WavFile.Write(path, new[] { -1f, 0f, 1f }, 16000);
            var samples = WavFile.Read(path);

            Assert.Equal(3, samples.Length);
            Assert.True(samples[0] <= -0.99f);
            Assert.Equal(0f, samples[1], 3);
            Assert.True(samples[2] >= 0.99f);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
