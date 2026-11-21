// Derived from Irodori-TTS-csharp (MIT License, Copyright (c) 2026 Aratako). See NOTICE.
using System.Text.Json;

namespace VoxCpm.Text;

/// <summary>A Hugging Face <c>tokenizer.json</c> model encoded without special tokens.</summary>
public interface ISubwordTokenizer
{
    int? BosTokenId { get; }
    int? EosTokenId { get; }
    int? PadTokenId { get; }
    int VocabSize { get; }

    /// <summary>Encode text without special tokens (equivalent to <c>add_special_tokens=False</c>).</summary>
    List<int> Encode(string text);
}

public static class SubwordTokenizer
{
    /// <summary>
    /// Load <c>tokenizer.json</c> (and <c>tokenizer_config.json</c>) from <paramref name="directory"/>:
    /// Only BPE models (Llama / MiniCPM) are supported.
    /// </summary>
    public static ISubwordTokenizer FromDirectory(string directory)
    {
        var tokPath = Path.Combine(directory, "tokenizer.json");
        if (!File.Exists(tokPath))
        {
            throw new FileNotFoundException($"tokenizer.json not found in {directory}");
        }
        return ReadModelType(tokPath) switch
        {
            "BPE" => BpeTokenizer.FromDirectory(directory),
            var t => throw new NotSupportedException($"Unsupported tokenizer model type: {t}"),
        };
    }

    private static string? ReadModelType(string path)
    {
        var reader = new Utf8JsonReader(File.ReadAllBytes(path), new JsonReaderOptions { AllowTrailingCommas = true });
        int depth = 0;
        bool inModel = false;
        while (reader.Read())
        {
            switch (reader.TokenType)
            {
                case JsonTokenType.StartObject or JsonTokenType.StartArray:
                    depth++;
                    break;
                case JsonTokenType.EndObject or JsonTokenType.EndArray:
                    depth--;
                    if (inModel && depth == 1) inModel = false;
                    break;
                case JsonTokenType.PropertyName when depth == 1 && reader.ValueTextEquals("model"):
                    inModel = true;
                    break;
                case JsonTokenType.PropertyName when depth == 2 && inModel && reader.ValueTextEquals("type"):
                    reader.Read();
                    return reader.GetString();
            }
        }
        return null;
    }
}
