using TorchSharp;
using VoxCpm.Model;
using static TorchSharp.torch;

namespace VoxCpm.Tests;

[Collection("model")]
public class ModelParityTests(ModelFixture f)
{
    private void SkipIfMissing() => Assert.SkipWhen(f.Model is null, "VoxCPM2 checkpoint or model_ref.safetensors not found");

    [Fact]
    public void LanguageModels()
    {
        SkipIfMissing();
        using var _ = torch.inference_mode();
        using var scope = torch.NewDisposeScope();
        var x = f.R("lm.in");
        foreach (var (name, lm) in new[] { ("base", f.Model!.BaseLm), ("res", f.Model!.ResidualLm) })
        {
            var (h, kv) = lm.Forward(x, isCausal: true);
            TensorAssert.Close(h, f.R($"lm.{name}.prefill"), 2e-4, $"{name} prefill");
            using var cache = new KvCache(lm.Config, 1, 16, ScalarType.Float32, torch.CPU);
            cache.Fill(kv);
            for (int s = 0; s < 2; s++)
            {
                var y = lm.ForwardStep(f.R($"lm.{name}.step{s}.in"), cache);
                TensorAssert.Close(y, f.R($"lm.{name}.step{s}"), 2e-4, $"{name} step{s}");
            }
        }
    }

    /// <summary>
    /// int8 LM weights (<c>--quantize int8</c>): the prefill error against the Python fp32 reference, next to that of
    /// the default bf16 weights (measured: base 6.5e-2 vs 3.5e-2, residual 3.5e-2 vs 1.8e-2). A regression guard.
    /// </summary>
    [Fact]
    public void Int8LanguageModelsStayWithinBf16Error()
    {
        SkipIfMissing();
        using var _ = torch.inference_mode();
        using var scope = torch.NewDisposeScope();
        var cfg = f.Model!.Config;
        using var w = WeightSource.FromSafeTensors(Path.Combine(TestPaths.Checkpoint!, "model.safetensors"), torch.CPU);
        var x = f.R("lm.in");
        foreach (var (name, lmCfg) in new[] { ("base", cfg.LmConfig), ("res", cfg.ResidualConfig()) })
        {
            var expected = f.R($"lm.{name}.prefill");
            double RelErr(Tensor h) => ((h.to(ScalarType.Float64) - expected.to(ScalarType.Float64)).norm() / expected.to(ScalarType.Float64).norm()).item<double>();
            double bf16, int8;
            using (var lm = new MiniCpm4Model(lmCfg, w.Sub(name == "base" ? "base_lm" : "residual_lm"), ScalarType.BFloat16, 64))
            {
                bf16 = RelErr(lm.Forward(x.to(ScalarType.BFloat16), isCausal: true).Hidden.to(ScalarType.Float32));
            }
            using (var lm = new MiniCpm4Model(lmCfg, w.Sub(name == "base" ? "base_lm" : "residual_lm"), ScalarType.BFloat16, 64, int8: true))
            {
                int8 = RelErr(lm.Forward(x.to(ScalarType.BFloat16), isCausal: true).Hidden.to(ScalarType.Float32));
            }
            Console.Error.WriteLine($"[parity] {name} prefill relative error: bf16 {bf16:E2}, bf16+int8 {int8:E2}");
            TestContext.Current.TestOutputHelper?.WriteLine($"{name}: bf16 {bf16:E2}, int8 {int8:E2}");
            Assert.True(int8 <= 2.5 * bf16, $"{name}: int8 error {int8:E2} > 2.5 x bf16 error {bf16:E2}");
        }
    }

    [Fact]
    public void LocalEncoderDitFsq()
    {
        SkipIfMissing();
        using var _ = torch.inference_mode();
        using var scope = torch.NewDisposeScope();
        TensorAssert.Close(f.Model!.FeatEncoder.Forward(f.R("enc.in")), f.R("enc.out"), 1e-4, "local encoder");
        var dit = f.Model.FeatDecoder.Estimator.Forward(f.R("dit.x"), f.R("dit.mu"), f.R("dit.t"), f.R("dit.cond"), torch.zeros(2));
        TensorAssert.Close(dit, f.R("dit.out"), 1e-4, "dit");
        TensorAssert.Close(f.Model.Fsq.Forward(f.R("fsq.in")), f.R("fsq.out"), 1e-4, "fsq");
    }

    [Fact]
    public void CfmSampler()
    {
        SkipIfMissing();
        using var _ = torch.inference_mode();
        using var scope = torch.NewDisposeScope();
        VoxCpm2Model.ApplySeed(123);
        var y = f.Model!.FeatDecoder.Forward(f.R("cfm.mu"), 10, 4, f.R("cfm.cond"), cfgValue: 2.0);
        TensorAssert.Close(y, f.R("cfm.out"), 1e-4, "cfm");
    }

    [Fact]
    public void AudioVae()
    {
        SkipIfMissing();
        using var _ = torch.inference_mode();
        using var scope = torch.NewDisposeScope();
        var audio = f.R("vae.audio").data<float>().ToArray();
        TensorAssert.Close(f.Model!.EncodeAudio(audio, padLeft: false), f.R("vae.ref_feat"), 1e-4, "vae encode (right pad)");
        TensorAssert.Close(f.Model.EncodeAudio(audio, padLeft: true), f.R("vae.prompt_feat"), 1e-4, "vae encode (left pad)");
        // Chunked encoding (the state carried across passes) gives the one-pass result.
        int chunkFrames = f.Model.AudioVae.EncodeChunkFrames;
        try
        {
            f.Model.AudioVae.EncodeChunkFrames = 1;
            TensorAssert.Close(f.Model.EncodeAudio(audio, padLeft: false), f.R("vae.ref_feat"), 1e-4, "vae chunked encode (right pad)");
            f.Model.AudioVae.EncodeChunkFrames = 3;
            TensorAssert.Close(f.Model.EncodeAudio(audio, padLeft: true), f.R("vae.prompt_feat"), 1e-4, "vae chunked encode (left pad)");
        }
        finally
        {
            f.Model.AudioVae.EncodeChunkFrames = chunkFrames;
        }
        var z = f.R("vae.ref_feat").permute(2, 0, 1).reshape(1, 64, -1);
        TensorAssert.Close(f.Model.AudioVae.Decode(z), f.R("vae.decoded"), 1e-4, "vae decode");
        using var dec = f.Model.AudioVae.StartStreaming();
        var chunks = new List<Tensor>();
        for (int i = 0; i < z.shape[2]; i += 4) chunks.Add(dec.DecodeChunk(z[.., .., i..(i + 4)]));
        TensorAssert.Close(torch.cat(chunks.ToArray(), -1), f.R("vae.streamed"), 1e-4, "vae streaming decode");
    }

    [Theory]
    [InlineData("zs")]
    [InlineData("ref")]
    [InlineData("cont")]
    public void EndToEnd(string key)
    {
        SkipIfMissing();
        using var scope = torch.NewDisposeScope();
        var refFeat = f.R("vae.ref_feat");
        var promptFeat = f.R("vae.prompt_feat");
        var (text, cache) = key switch
        {
            "zs" => ("Hello world.", (PromptCache?)null),
            "ref" => ("Nice to meet you.", new PromptCache { Mode = PromptMode.Reference, RefAudioFeat = refFeat }),
            _ => ("and then we left.", new PromptCache { Mode = PromptMode.Continuation, PromptText = "We went to the park,", AudioFeat = promptFeat }),
        };
        var o = new GenerationOptions { MinLen = 2, MaxLen = 6, InferenceTimesteps = 10, CfgValue = 2.0, RetryBadCase = false, Seed = 42 };
        using var result = f.Model!.Generate(text, cache, o);
        var expFeat = f.R($"e2e.{key}.feat");
        Assert.Equal(expFeat.shape, result.GeneratedFeat.shape);
        for (int i = 0; i < expFeat.shape[0]; i++)
        {
            double d = TensorAssert.MaxAbsDiff(result.GeneratedFeat[i], expFeat[i]);
            TestContext.Current.TestOutputHelper?.WriteLine($"{key} patch {i}: {d:E2}");
        }
        // The first patches must agree closely. Later ones accumulate rounding differences through the AR loop;
        // the continuation case (a synthetic tone as prompt) amplifies them: Python itself drifts by 1.9e-2 at
        // patch 5 when only its decode-step attention is computed over the used cache length instead of masking.
        TensorAssert.Close(result.GeneratedFeat[..2], expFeat[..2], 1e-4, $"{key} first latents");
        double tol = key == "cont" ? 5e-2 : 2e-3;
        TensorAssert.Close(result.GeneratedFeat, expFeat, tol, $"{key} latents");
        var wav = torch.tensor(result.Audio).unsqueeze(0);
        TensorAssert.Close(wav, f.R($"e2e.{key}.wav"), key == "cont" ? 0.1 : 2e-3, $"{key} waveform");
    }
}
