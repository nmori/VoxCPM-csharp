"""Generate one utterance with the Python VoxCPM2 (CPU) and compare it with a WAV written by the C# CLI.

usage: compare_e2e.py TEXT SEED CSHARP_WAV [--dtype float32]
"""
import argparse
import os

import numpy as np
import soundfile as sf
import torch

import _common


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text")
    p.add_argument("seed", type=int)
    p.add_argument("csharp_wav")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--save", default=None)
    a = p.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    model = _common.load_model(a.dtype)
    with torch.inference_mode():
        wav, _, feat = next(model._generate_with_prompt_cache(target_text=a.text, prompt_cache=None, min_len=2, max_len=4096,
                                                             inference_timesteps=10, cfg_value=2.0, retry_badcase=True, seed=a.seed))
    py = wav.squeeze(0).float().numpy()
    if a.save:
        sf.write(a.save, py, 48000)
    cs, sr = sf.read(a.csharp_wav, dtype="float32")
    print(f"python: {len(py) / 48000:.2f} s, csharp: {len(cs) / sr:.2f} s, patches: {feat.shape[0]}")
    n = min(len(py), len(cs))
    err = py[:n] - cs[:n]
    snr = 10 * np.log10(np.sum(py[:n] ** 2) / max(np.sum(err ** 2), 1e-20))
    print(f"max |diff| {np.abs(err).max():.3e} (16-bit quantization ~1.5e-5), SNR {snr:.1f} dB")


if __name__ == "__main__":
    main()
