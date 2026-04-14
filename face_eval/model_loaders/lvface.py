"""LVFace (ByteDance, ICCV 2025) loader - PyTorch backend.

LVFace's official inference.py loads weights like this:

    from backbones import get_model
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()

We replicate that here.  Rather than copy the backbones/ folder into this
repo, we ask the user to clone https://github.com/bytedance/LVFace once
and point env var LVFACE_REPO_DIR at the clone.  The loader adds that
directory to sys.path so `from backbones import get_model` resolves.

Weights (.pt) are auto-downloaded from Hugging Face Hub on first run.

Input (matches LVFace inference.py):
    - 112x112 BGR
    - (pixel / 255 - 0.5) / 0.5  -> [-1, 1]
    - NCHW float32
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn.functional import normalize

import config


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


def _import_get_model():
    """Add LVFACE_REPO_DIR to sys.path and import backbones.get_model."""
    repo_dir = config.LVFACE_REPO_DIR
    if not repo_dir:
        raise RuntimeError(
            "LVFACE_REPO_DIR is not set.  Clone https://github.com/bytedance/LVFace "
            "and set LVFACE_REPO_DIR in .env to that path so the loader can "
            "import backbones.get_model."
        )
    repo_path = Path(repo_dir).expanduser().resolve()
    if not (repo_path / "backbones" / "__init__.py").exists():
        raise RuntimeError(
            f"LVFACE_REPO_DIR={repo_path} does not contain backbones/__init__.py. "
            "Did you clone bytedance/LVFace to this path?"
        )
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    from backbones import get_model  # type: ignore
    return get_model


class LVFaceEmbedder:
    INPUT_SIZE = 112
    EMBED_DIM = 512

    def __init__(self, device: torch.device):
        self.device = device
        get_model = _import_get_model()
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
