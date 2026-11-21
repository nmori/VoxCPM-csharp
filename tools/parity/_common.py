"""Shared helpers for the parity scripts: import the original VoxCPM sources and locate the checkpoint."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VOXCPM_SRC = os.environ.get("VOXCPM_SRC", os.path.normpath(os.path.join(HERE, "..", "..", "..", "VoxCPM", "src")))
if VOXCPM_SRC not in sys.path:
    sys.path.insert(0, VOXCPM_SRC)

DEFAULT_CKPT = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "VoxCpm", "hf", "openbmb", "VoxCPM2", "main")
CKPT = os.environ.get("VOXCPM_CKPT", DEFAULT_CKPT)
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "tests", "VoxCpm.Tests", "Data"))


def load_model(dtype: str = "float32", device: str = "cpu"):
    """VoxCPM2Model.from_local with the checkpoint dtype overridden (the checkpoint says bfloat16)."""
    from voxcpm.model.voxcpm2 import VoxCPM2Model, VoxCPMConfig

    orig = VoxCPMConfig.model_validate_json.__func__

    def patched(cls, data, *args, **kwargs):
        cfg = orig(cls, data, *args, **kwargs)
        cfg.dtype = dtype
        return cfg

    VoxCPMConfig.model_validate_json = classmethod(patched)
    try:
        model = VoxCPM2Model.from_local(CKPT, optimize=False, device=device)
    finally:
        VoxCPMConfig.model_validate_json = classmethod(orig)
    return model


def encode_audio(model, audio_16k, padding_mode="right"):
    """VoxCPM2Model._encode_wav without the file loading / resampling (audio is already 16 kHz mono float32)."""
    import torch

    audio = torch.as_tensor(audio_16k, dtype=torch.float32).unsqueeze(0)
    patch_len = model.patch_size * model.chunk_size
    if audio.size(1) % patch_len != 0:
        padding_size = patch_len - audio.size(1) % patch_len
        pad = (padding_size, 0) if padding_mode == "left" else (0, padding_size)
        audio = torch.nn.functional.pad(audio, pad)
    feat = model.audio_vae.encode(audio.to(model.device), model._encode_sample_rate).cpu()
    return feat.view(model.audio_vae.latent_dim, -1, model.patch_size).permute(1, 2, 0)


def test_tone(seconds: float, sr: int = 16000):
    """A deterministic voiced-like test signal (harmonics with vibrato and an envelope)."""
    import numpy as np

    t = np.arange(int(seconds * sr)) / sr
    f0 = 140 + 30 * np.sin(2 * np.pi * 3 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    sig = sum(np.sin(k * phase) / k for k in range(1, 8))
    env = np.minimum(1, t / 0.05) * np.minimum(1, (seconds - t) / 0.05)
    return (0.3 * sig * env).astype(np.float32)
