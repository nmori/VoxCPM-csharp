using System.Text.Json;
using VoxCpm.Text;

namespace VoxCpm.Tests;

public class TextTokenizerTests
{
    [Fact]
    public void MatchesPythonTokenizer()
    {
        var ckpt = TestPaths.Checkpoint;
        Assert.SkipWhen(ckpt is null, "VoxCPM2 checkpoint not found");
        var tok = VoxCpmTextTokenizer.FromDirectory(ckpt!);
        using var doc = JsonDocument.Parse(File.ReadAllBytes(TestPaths.Data("text_ref.json")));
        foreach (var c in doc.RootElement.EnumerateArray())
        {
            var text = c.GetProperty("text").GetString()!;
            var expected = c.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            var got = tok.Encode(text);
            Assert.True(expected.SequenceEqual(got), $"ids differ for: {text}\n got {string.Join(",", got)}\n exp {string.Join(",", expected)}");
        }
    }
}
