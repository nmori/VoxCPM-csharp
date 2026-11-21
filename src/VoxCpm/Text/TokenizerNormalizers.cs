// Derived from Irodori-TTS-csharp (MIT License, Copyright (c) 2026 Aratako). See NOTICE.
using System.Text;
using System.Text.Json;

namespace VoxCpm.Text;

/// <summary>The subset of Hugging Face <c>tokenizers</c> normalizers used by the supported tokenizers.</summary>
internal static class TokenizerNormalizers
{
    internal static Func<string, string> BuildNormalizer(JsonElement n)
    {
        if (n.ValueKind != JsonValueKind.Object)
        {
            return s => s;
        }
        var type = n.GetProperty("type").GetString();
        switch (type)
        {
            case "Sequence":
            {
                var steps = n.GetProperty("normalizers").EnumerateArray().Select(BuildNormalizer).ToArray();
                return s =>
                {
                    foreach (var step in steps) s = step(s);
                    return s;
                };
            }
            case "NFC": return s => s.Normalize(NormalizationForm.FormC);
            case "NFD": return s => s.Normalize(NormalizationForm.FormD);
            case "NFKC": return s => s.Normalize(NormalizationForm.FormKC);
            case "NFKD": return s => s.Normalize(NormalizationForm.FormKD);
            case "Prepend":
            {
                var prefix = n.GetProperty("prepend").GetString()!;
                return s => s.Length == 0 ? s : prefix + s;
            }
            case "Replace":
            {
                var pattern = n.GetProperty("pattern");
                if (!pattern.TryGetProperty("String", out var str))
                {
                    throw new NotSupportedException("Regex Replace normalizers are not supported.");
                }
                var from = str.GetString()!;
                var to = n.GetProperty("content").GetString()!;
                return s => s.Replace(from, to, StringComparison.Ordinal);
            }
            default:
                throw new NotSupportedException($"Unsupported tokenizer normalizer: {type}");
        }
}
}
