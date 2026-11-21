"""Golden tensors of the VoxCPM2 modules and of short end-to-end generations (CPU, float32).

Writes tests/VoxCpm.Tests/Data/model_ref.safetensors. Inputs are stored next to the outputs so the C# tests feed the
exact same values. Random noise of the CFM sampler comes from torch.manual_seed, which the C# port reproduces.
"""
import os

import torch
from safetensors.torch import save_file

import _common


def main():
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    model = _common.load_model("float32")
    out = {}

    with torch.inference_mode():
        g = torch.Generator().manual_seed(0)

        # base / residual LM: prefill + 2 steps
        x = torch.randn(1, 9, 2048, generator=g) * 0.5
        out["lm.in"] = x
        for name, lm in (("base", model.base_lm), ("res", model.residual_lm)):
            h, kv = lm(x, is_causal=True)
            out[f"lm.{name}.prefill"] = h
            lm.kv_cache.fill_caches(kv)
            for s in range(2):
                xs = torch.randn(1, 2048, generator=g) * 0.5
                out[f"lm.{name}.step{s}.in"] = xs
                pos = torch.tensor([lm.kv_cache.step()])
                out[f"lm.{name}.step{s}"] = lm.forward_step(xs, pos).clone()

        # local encoder
        feat = torch.randn(1, 3, 4, 64, generator=g)
        out["enc.in"] = feat
        out["enc.out"] = model.feat_encoder(feat)

        # DiT estimator
        dx = torch.randn(2, 64, 4, generator=g)
        dmu = torch.randn(2, 2048, generator=g)
        dt_ = torch.tensor([0.7, 0.7])
        dcond = torch.randn(2, 64, 4, generator=g)
        out.update({"dit.x": dx, "dit.mu": dmu, "dit.t": dt_, "dit.cond": dcond})
        out["dit.out"] = model.feat_decoder.estimator(dx, dmu, dt_, dcond, torch.zeros(2))

        # CFM sampler (global RNG)
        cmu = torch.randn(1, 2048, generator=g)
        ccond = torch.randn(1, 64, 4, generator=g)
        out.update({"cfm.mu": cmu, "cfm.cond": ccond})
        torch.manual_seed(123)
        out["cfm.out"] = model.feat_decoder(mu=cmu, patch_size=4, cond=ccond, n_timesteps=10, cfg_value=2.0)

        # FSQ
        fx = torch.randn(1, 5, 2048, generator=g)
        out["fsq.in"] = fx
        out["fsq.out"] = model.fsq_layer(fx)

        # AudioVAE encode / decode
        tone = torch.from_numpy(_common.test_tone(1.3))
        out["vae.audio"] = tone
        ref_feat = _common.encode_audio(model, tone.numpy(), "right")
        out["vae.ref_feat"] = ref_feat
        prompt_feat = _common.encode_audio(model, tone.numpy(), "left")
        out["vae.prompt_feat"] = prompt_feat
        z = ref_feat.permute(2, 0, 1).reshape(1, 64, -1)
        out["vae.decoded"] = model.audio_vae.decode(z)
        with model.audio_vae.streaming_decode() as dec:
            chunks = [dec.decode_chunk(z[..., i : i + 4]) for i in range(0, z.shape[-1], 4)]
        out["vae.streamed"] = torch.cat(chunks, dim=-1)

        # end-to-end (short): zero-shot, reference, continuation
        cases = {
            "zs": ("Hello world.", None),
            "ref": ("Nice to meet you.", {"mode": "reference", "ref_audio_feat": ref_feat}),
            "cont": ("and then we left.", {"mode": "continuation", "prompt_text": "We went to the park,", "audio_feat": prompt_feat}),
        }
        for key, (text, cache) in cases.items():
            wav, _, feat_gen = next(
                model._generate_with_prompt_cache(
                    target_text=text,
                    prompt_cache=cache,
                    min_len=2,
                    max_len=6,
                    inference_timesteps=10,
                    cfg_value=2.0,
                    retry_badcase=False,
                    seed=42,
                )
            )
            out[f"e2e.{key}.wav"] = wav
            out[f"e2e.{key}.feat"] = feat_gen

    out = {k: v.detach().contiguous().to(torch.float32) for k, v in out.items()}
    path = os.path.join(_common.DATA_DIR, "model_ref.safetensors")
    save_file(out, path)
    print("wrote", path, {k: tuple(v.shape) for k, v in out.items()})


if __name__ == "__main__":
    main()
