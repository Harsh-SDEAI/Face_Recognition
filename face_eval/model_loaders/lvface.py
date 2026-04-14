"""LVFace (ByteDance, ICCV 2025) loader - PyTorch backend.

The ViT backbone code is vendored under `lvface_backbones/` so no cloning
or env-var setup is required.  Weights (.pt) are auto-downloaded from
Hugging Face Hub on first run.

Mirrors bytedance/LVFace inference.py:
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    # 112x112 BGR, (img/255 - 0.5) / 0.5 normalization
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn.functional import normalize

import config

from .lvface_backbones import get_model


def _download_weights_if_missing() -> Path:
    target = Path(config.LVFACE_WEIGHTS)
    if target.exists():
        return target
    from huggingface_hub import hf_hub_download
    cached = hf_hub_download(
        repo_id=config.LVFACE_HF_REPO,
        filename=config.LVFACE_HF_FILENAME,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(cached, target)
    return target


class LVFaceEmbedder:
    INPUT_SIZE = 112
    EMBED_DIM = 512

    def __init__(self, device: torch.device):
        self.device = device
        self.model = get_model(config.LVFACE_MODEL_NAME, fp16=False)

        weights = _download_weights_if_missing()
        state = torch.load(weights, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)
        self.model.eval().to(device)

    @torch.no_grad()
    def embed(self, aligned_rgb: np.ndarray) -> np.ndarray:
        img = cv2.resize(aligned_rgb, (self.INPUT_SIZE, self.INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)     # LVFace inference.py uses BGR
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        t = (t / 255.0 - 0.5) / 0.5
        e = self.model(t)
        if isinstance(e, (tuple, list)):
            e = e[0]
        e = normalize(e, p=2, dim=1)
        return e.detach().cpu().numpy().reshape(-1).astype(np.float32)


def load(device: torch.device) -> LVFaceEmbedder:
    return LVFaceEmbedder(device)
