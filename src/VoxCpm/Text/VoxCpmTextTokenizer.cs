// Ported from VoxCPM (https://github.com/OpenBMB/VoxCPM, Apache License 2.0, Copyright OpenBMB): src/voxcpm/model/utils.py (mask_multichar_chinese_tokens)
// Modified: re-implemented in C# on TorchSharp (inference only). See NOTICE.
namespace VoxCpm.Text;

/// <summary>
/// The VoxCPM text tokenizer: the checkpoint's <c>LlamaTokenizerFast</c> (<c>tokenizer.json</c>, BPE) wrapped like
/// <c>voxcpm.model.utils.mask_multichar_chinese_tokens</c>: a token whose text without the <c>▁</c> marker is a
/// multi-character token made only of CJK Unified Ideographs (U+4E00–U+9FFF) is replaced by its single characters.
/// No special tokens are added (the Python wrapper calls <c>tokenize</c> + <c>convert_tokens_to_ids</c>).
/// <para>
/// transformers 5 (the version VoxCPM pins) rebuilds <c>LlamaTokenizerFast</c> instead of using the
/// <c>tokenizer.json</c> normalizer (<c>Prepend("▁")</c> + <c>Replace(" ", "▁")</c>): it uses a
/// <c>Metaspace(replacement="▁", prepend_scheme="always", split=false)</c> pre-tokenizer, which does not prepend
/// <c>▁</c> to a segment that already starts with one (text with leading spaces). That behavior is reproduced here.
/// </para>
/// <para>
/// The added (special) tokens are those of <c>tokenizer_config.json</c> <c>added_tokens_decoder</c>, as transformers 5
/// registers them: a token whose text is in the vocabulary keeps its vocabulary id, the others get new ids from the
/// vocabulary size up, in file order (so <c>&lt;|im_end|&gt;</c> is 73442, not the 73440 written in the file).
/// </para>
/// </summary>
public sealed class VoxCpmTextTokenizer
{
    private readonly BpeTokenizer _bpe;
    private readonly string?[] _idToToken;
    private readonly HashSet<string> _multichar;

    public VoxCpmTextTokenizer(BpeTokenizer bpe, IReadOnlyList<string>? addedTokens = null)
    {
        _bpe = bpe;
        _bpe.OverrideNormalizer(Metaspace);
        if (addedTokens is not null)
        {
            int next = bpe.Vocab.Count;
            var resolved = new List<(string, int)>();
            var seen = new HashSet<string>(StringComparer.Ordinal);
            foreach (var t in addedTokens)
            {
                if (!seen.Add(t)) continue;
                resolved.Add((t, bpe.Vocab.TryGetValue(t, out var id) ? id : next++));
            }
            _bpe.OverrideAddedTokens(resolved);
        }
        _idToToken = new string?[bpe.VocabSize];
        _multichar = new HashSet<string>(StringComparer.Ordinal);
        foreach (var (token, id) in bpe.Vocab)
        {
            if (id >= 0 && id < _idToToken.Length) _idToToken[id] = token;
            if (token.Length >= 2 && IsAllCjk(token)) _multichar.Add(token);
        }
    }

    public static VoxCpmTextTokenizer FromDirectory(string directory)
    {
        var bpe = BpeTokenizer.FromDirectory(directory);
        List<string>? added = null;
        var cfgPath = Path.Combine(directory, "tokenizer_config.json");
        if (File.Exists(cfgPath))
        {
            using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllBytes(cfgPath));
            if (doc.RootElement.TryGetProperty("added_tokens_decoder", out var dec) && dec.ValueKind == System.Text.Json.JsonValueKind.Object)
            {
                added = new List<string>();
                foreach (var p in dec.EnumerateObject())
                {
                    if (p.Value.TryGetProperty("content", out var c) && c.GetString() is { Length: > 0 } content) added.Add(content);
                }
            }
        }
        return new VoxCpmTextTokenizer(bpe, added);
    }

    public int VocabSize => _bpe.VocabSize;

    /// <summary>Token ids of <paramref name="text"/> (Python: <c>self.text_tokenizer(text)</c>).</summary>
    public List<int> Encode(string text)
    {
        var baseIds = _bpe.Encode(text);
        var ids = new List<int>(baseIds.Count);
        foreach (var id in baseIds)
        {
            var token = id >= 0 && id < _idToToken.Length ? _idToToken[id] : null;
            if (token is not null)
            {
                var clean = token.Replace("▁", "", StringComparison.Ordinal);
                if (_multichar.Contains(clean))
                {
                    // Every char is in the BMP (U+4E00–U+9FFF), so one UTF-16 unit per character.
                    foreach (var c in clean)
                    {
                        ids.Add(_bpe.Vocab.TryGetValue(c.ToString(), out var cid) ? cid : _bpe.UnkId);
                    }
                    continue;
                }
            }
            ids.Add(id);
        }
        return ids;
    }

    /// <summary><c>Metaspace(replacement="▁", prepend_scheme="always", split=false)</c>.</summary>
    private static string Metaspace(string segment)
    {
        var s = segment.Replace(' ', '▁');
        return s.Length > 0 && s[0] != '▁' ? "▁" + s : s;
    }

    private static bool IsAllCjk(string s)
    {
        foreach (var c in s)
        {
            if (c < '一' || c > '鿿') return false;
        }
        return true;
    }
}
