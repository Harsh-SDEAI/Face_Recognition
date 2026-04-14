"""ArcFace (IR-100 / Glint360K) loader.

Uses the InsightFace arcface_torch backbones installed via pip from the
GitHub source, e.g.:
    pip install git+https://github.com/deepinsight/insightface.git#subdirectory=recognition/arcface_torch

We try several import paths because the arcface_torch package exposes
`iresnet100` under different names depending on how it was installed.
No Python file needs to be copied into model_loaders/ for this model.

Input: 112x112 BGR, normalized to [-1, 1] via (img/255 - 0.5) / 0.5.
Output: 512-d, L2-normalized so cosine similarity == dot product.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.nn.functional import normalize

import config


def _import_iresnet100():
    """Try known arcface_torch import paths; return iresnet100 or None."""
    errors = []
    for path in (
        "backbones.iresnet",             # arcface_torch installed as package
        "arcface_torch.backbones.iresnet",
        "insightface.recognition.arcface_torch.backbones.iresnet",
    ):
        try:
            mod = __import__(path, fromlist=["iresnet100"])
            return getattr(mod, "iresnet100")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path}: {exc}")
    # Fallback: a locally copied file (optional - only if someone dropped it in).
    try:
        from . import arcface_iresnet  # type: ignore
        return getattr(arcface_iresnet, "iresnet100")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"local arcface_iresnet: {exc}")
    raise ImportError(
        "Could not import iresnet100 from arcface_torch.  Tried:\n  - "
        + "\n  - ".join(errors)
    )


class ArcFaceEmbedder:
    INPUT_SIZE = 112
    EMBED_DIM = 512

    def __init__(self, device: torch.device, weights_path=None):
        self.device = device
        iresnet100 = _import_iresnet100()
        self.model = iresnet100(num_features=self.EMBED_DIM)
        state = torch.load(weights_path or config.ARCFACE_WEIGHTS, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)
        self.model.eval().to(device)

    @torch.no_grad()
    def embed(self, aligned_rgb: np.ndarray) -> np.ndarray:
        img = cv2.resize(aligned_rgb, (self.INPUT_SIZE, self.INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)       # ArcFace trained on BGR
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        t = (t / 255.0 - 0.5) / 0.5
        e = self.model(t)
        e = normalize(e, p=2, dim=1)
        return e.detach().cpu().numpy().reshape(-1).astype(np.float32)


def load(device: torch.device) -> ArcFaceEmbedder:
    return ArcFaceEmbedder(device)
