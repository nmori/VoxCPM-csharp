// Ported from VoxCPM (https://github.com/OpenBMB/VoxCPM, Apache License 2.0, Copyright OpenBMB): src/voxcpm/model/voxcpm2.py, modules/minicpm4/config.py (configuration)
// Modified: re-implemented in C# on TorchSharp (inference only). See NOTICE.
using System.Text.Json;
using System.Text.Json.Serialization;

namespace VoxCpm.Model;

/// <summary><c>RopeScalingConfig</c> (minicpm4/config.py).</summary>
public sealed class RopeScalingConfig
{
    [JsonPropertyName("type")] public string Type { get; set; } = "longrope";
    [JsonPropertyName("long_factor")] public double[] LongFactor { get; set; } = Array.Empty<double>();
    [JsonPropertyName("short_factor")] public double[] ShortFactor { get; set; } = Array.Empty<double>();
    [JsonPropertyName("original_max_position_embeddings")] public int OriginalMaxPositionEmbeddings { get; set; }
}

/// <summary><c>MiniCPM4Config</c> (minicpm4/config.py).</summary>
public sealed class MiniCpm4Config
{
    [JsonPropertyName("bos_token_id")] public int BosTokenId { get; set; }
    [JsonPropertyName("eos_token_id")] public int EosTokenId { get; set; }
    [JsonPropertyName("hidden_size")] public int HiddenSize { get; set; }
    [JsonPropertyName("intermediate_size")] public int IntermediateSize { get; set; }
    [JsonPropertyName("max_position_embeddings")] public int MaxPositionEmbeddings { get; set; }
    [JsonPropertyName("num_attention_heads")] public int NumAttentionHeads { get; set; }
    [JsonPropertyName("num_hidden_layers")] public int NumHiddenLayers { get; set; }
    [JsonPropertyName("num_key_value_heads")] public int NumKeyValueHeads { get; set; }
    [JsonPropertyName("rms_norm_eps")] public double RmsNormEps { get; set; }
    [JsonPropertyName("rope_scaling")] public RopeScalingConfig RopeScaling { get; set; } = new();
    [JsonPropertyName("vocab_size")] public int VocabSize { get; set; }
    [JsonPropertyName("use_mup")] public bool UseMup { get; set; } = true;
    [JsonPropertyName("scale_emb")] public double ScaleEmb { get; set; }
    [JsonPropertyName("dim_model_base")] public int DimModelBase { get; set; }
    [JsonPropertyName("scale_depth")] public double ScaleDepth { get; set; }
    [JsonPropertyName("rope_theta")] public double RopeTheta { get; set; }
    [JsonPropertyName("kv_channels")] public int? KvChannels { get; set; }
    [JsonPropertyName("no_rope")] public bool NoRope { get; set; }

    /// <summary>Per-head dimension (<c>kv_channels</c>, else <c>hidden_size / num_attention_heads</c>).</summary>
    [JsonIgnore] public int HeadDim => KvChannels is int kv and > 0 ? kv : HiddenSize / NumAttentionHeads;

    public MiniCpm4Config Clone() => (MiniCpm4Config)MemberwiseClone();
}

/// <summary><c>VoxCPMEncoderConfig</c> / the transformer part of <c>VoxCPMDitConfig</c>.</summary>
public sealed class LocalTransformerConfig
{
    [JsonPropertyName("hidden_dim")] public int HiddenDim { get; set; } = 1024;
    [JsonPropertyName("ffn_dim")] public int FfnDim { get; set; } = 4096;
    [JsonPropertyName("num_heads")] public int NumHeads { get; set; } = 16;
    [JsonPropertyName("num_layers")] public int NumLayers { get; set; } = 4;
    [JsonPropertyName("kv_channels")] public int? KvChannels { get; set; }
    // pydantic ignores the unknown "mean_mode" key of the checkpoint config; "dit_mean_mode" is the real field.
    [JsonPropertyName("dit_mean_mode")] public bool DitMeanMode { get; set; }
    [JsonPropertyName("cfm_config")] public CfmConfig? CfmConfig { get; set; }
}

/// <summary><c>CfmConfig</c> (locdit/unified_cfm.py), the inference-relevant fields.</summary>
public sealed class CfmConfig
{
    [JsonPropertyName("sigma_min")] public double SigmaMin { get; set; } = 1e-6;
    [JsonPropertyName("solver")] public string Solver { get; set; } = "euler";
    [JsonPropertyName("t_scheduler")] public string TScheduler { get; set; } = "log-norm";
    [JsonPropertyName("inference_cfg_rate")] public double InferenceCfgRate { get; set; } = 1.0;
}

/// <summary><c>AudioVAEConfig</c> (audiovae/audio_vae_v2.py).</summary>
public sealed class AudioVaeConfig
{
    [JsonPropertyName("encoder_dim")] public int EncoderDim { get; set; } = 128;
    [JsonPropertyName("encoder_rates")] public int[] EncoderRates { get; set; } = { 2, 5, 8, 8 };
    [JsonPropertyName("latent_dim")] public int LatentDim { get; set; } = 64;
    [JsonPropertyName("decoder_dim")] public int DecoderDim { get; set; } = 2048;
    [JsonPropertyName("decoder_rates")] public int[] DecoderRates { get; set; } = { 8, 6, 5, 2, 2, 2 };
    [JsonPropertyName("depthwise")] public bool Depthwise { get; set; } = true;
    [JsonPropertyName("sample_rate")] public int SampleRate { get; set; } = 16000;
    [JsonPropertyName("out_sample_rate")] public int OutSampleRate { get; set; } = 48000;
    [JsonPropertyName("use_noise_block")] public bool UseNoiseBlock { get; set; }
    [JsonPropertyName("sr_bin_boundaries")] public int[]? SrBinBoundaries { get; set; } = { 20000, 30000, 40000 };
    [JsonPropertyName("cond_type")] public string CondType { get; set; } = "scale_bias";
    [JsonPropertyName("cond_dim")] public int CondDim { get; set; } = 128;
    [JsonPropertyName("cond_out_layer")] public bool CondOutLayer { get; set; }

    /// <summary>Input samples per latent frame (<c>chunk_size</c> = prod(encoder_rates)).</summary>
    [JsonIgnore] public int ChunkSize => EncoderRates.Aggregate(1, (a, b) => a * b);

    /// <summary>Output samples per latent frame (<c>decode_chunk_size</c> = prod(decoder_rates)).</summary>
    [JsonIgnore] public int DecodeChunkSize => DecoderRates.Aggregate(1, (a, b) => a * b);
}

/// <summary><c>VoxCPMConfig</c> (model/voxcpm2.py), read from the checkpoint's <c>config.json</c>.</summary>
public sealed class VoxCpmConfig
{
    [JsonPropertyName("architecture")] public string? Architecture { get; set; }
    [JsonPropertyName("lm_config")] public MiniCpm4Config LmConfig { get; set; } = new();
    [JsonPropertyName("patch_size")] public int PatchSize { get; set; } = 4;
    [JsonPropertyName("feat_dim")] public int FeatDim { get; set; } = 64;
    [JsonPropertyName("residual_lm_num_layers")] public int ResidualLmNumLayers { get; set; } = 8;
    [JsonPropertyName("residual_lm_no_rope")] public bool ResidualLmNoRope { get; set; }
    [JsonPropertyName("scalar_quantization_latent_dim")] public int ScalarQuantizationLatentDim { get; set; } = 512;
    [JsonPropertyName("scalar_quantization_scale")] public int ScalarQuantizationScale { get; set; } = 9;
    [JsonPropertyName("encoder_config")] public LocalTransformerConfig EncoderConfig { get; set; } = new();
    [JsonPropertyName("dit_config")] public LocalTransformerConfig DitConfig { get; set; } = new();
    [JsonPropertyName("audio_vae_config")] public AudioVaeConfig? AudioVaeConfig { get; set; }
    [JsonPropertyName("max_length")] public int MaxLength { get; set; } = 8192;
    [JsonPropertyName("device")] public string Device { get; set; } = "cuda";
    [JsonPropertyName("dtype")] public string DType { get; set; } = "bfloat16";

    public static VoxCpmConfig Load(string path)
    {
        var cfg = JsonSerializer.Deserialize<VoxCpmConfig>(File.ReadAllBytes(path),
            new JsonSerializerOptions { AllowTrailingCommas = true, ReadCommentHandling = JsonCommentHandling.Skip })
            ?? throw new InvalidDataException($"Invalid config: {path}");
        if (cfg.Architecture is not null && cfg.Architecture != "voxcpm2")
        {
            throw new NotSupportedException($"Only VoxCPM2 checkpoints are supported (architecture '{cfg.Architecture}').");
        }
        cfg.AudioVaeConfig ??= new AudioVaeConfig();
        return cfg;
    }

    /// <summary>The base LM config with the transformer shape of a local module (Python: <c>lm_config.model_copy</c>).</summary>
    internal MiniCpm4Config LocalConfig(LocalTransformerConfig local)
    {
        var c = LmConfig.Clone();
        c.HiddenSize = local.HiddenDim;
        c.IntermediateSize = local.FfnDim;
        c.NumAttentionHeads = local.NumHeads;
        c.NumHiddenLayers = local.NumLayers;
        c.KvChannels = local.KvChannels;
        c.VocabSize = 0;
        return c;
    }

    internal MiniCpm4Config ResidualConfig()
    {
        var c = LmConfig.Clone();
        c.NumHiddenLayers = ResidualLmNumLayers;
        c.VocabSize = 0;
        c.NoRope = ResidualLmNoRope;
        return c;
    }
}
