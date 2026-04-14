"""LVFace (ByteDance, ICCV 2025) loader.

Depends on `lvface_model.py` - copy the model definition from
https://github.com/bytedance/LVFace.  The exact class name / factory
function in that file varies by release; this loader tries a few common
names.  Override via config if needed.

Weights are auto-downloaded from Hugging Face Hub on first use:
    repo:     bytedance-research/LVFace
    filename: LVFace-B_WebFace4M.pt  (override via LVFACE_HF_FILENAME env)

Input and normalization: assumed 112x112 RGB with ImageNet mean/std, 512-d
output.  If the LVFace repo README specifies something different, adjust
INPUT_SIZE / mean / std constants below.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn.functional import normalize

import config

try:
    from . import lvface_model  # type: ignore
except Exception as exc:  # noqa: BLE001
    lvface_model = None
    _IMPORT_ERR = exc
else:
    _IMPORT_ERR = None


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _download_weights_if_missing() -> Path:
    target = Path(config.LVFACE_WEIGHTS)
    if target.exists():
        return target
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install huggingface_hub") from exc
    cached = hf_hub_download(
        repo_id=config.LVFACE_HF_REPO,
        filename=config.LVFACE_HF_FILENAME,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    # Copy cached -> models/lvface.pt so the path matches config.
    import shutil
    shutil.copy(cached, target)
    return target


def _build_model():
    """LVFace repos have used different entry points across releases.

    Try them in order and raise with a helpful message if none exist.
    """
    candidates = ["build_model", "LVFace", "lvface_base", "get_model"]
    for name in candidates:
        fn = getattr(lvface_model, name, None)
        if callable(fn):
            try:
                return fn()
            except TypeError:
                # Some factories require args - try defaults
                try:
                    return fn("base")
                except Exception:
                    continue
    raise RuntimeError(
        f"Could not find a model constructor in lvface_model.py. "
        f"Tried: {candidates}.  Inspect the LVFace repo's entry point "
        f"and add it to the candidates list."
    )


class LVFaceEmbedder:
    INPUT_SIZE = 112
    EMBED_DIM = 512

    def __init__(self, device: torch.device):
        if lvface_model is None:
            raise ImportError(
                "lvface_model.py not found in model_loaders/.  Copy the "
                "model definition file from bytedance/LVFace.  Original: "
                f"{_IMPORT_ERR}"
            )
        self.device = device
        self.model = _build_model()
        weights = _download_weights_if_missing()
        state = torch.load(weights, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)
        self.model.eval().to(device)

    @torch.no_grad()
    def embed(self, aligned_rgb: np.ndarray) -> np.ndarray:
        img = cv2.resize(aligned_rgb, (self.INPUT_SIZE, self.INPUT_SIZE))
        arr = img.astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        e = self.model(t)
        # Some LVFace forwards return (feature, logits) - take the feature only.
        if isinstance(e, (tuple, list)):
            e = e[0]
        e = normalize(e, p=2, dim=1)
        return e.detach().cpu().numpy().reshape(-1).astype(np.float32)


def load(device: torch.device) -> LVFaceEmbedder:
    return LVFaceEmbedder(device)
