"""ArcFace (IR-100 / Glint360K) loader.

Depends on `arcface_iresnet.py` which is the IResNet implementation copied
from https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch.

Input: 112x112 BGR, normalized to [-1, 1] via (img/255 - 0.5) / 0.5.
Output: 512-d, L2-normalized here so cosine similarity == dot product.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.nn.functional import normalize

import config

try:
    from .arcface_iresnet import iresnet100  # type: ignore
except Exception as exc:  # noqa: BLE001
    iresnet100 = None
    _IMPORT_ERR = exc
else:
    _IMPORT_ERR = None


class ArcFaceEmbedder:
    INPUT_SIZE = 112
    EMBED_DIM = 512

    def __init__(self, device: torch.device, weights_path=None):
        if iresnet100 is None:
            raise ImportError(
                "arcface_iresnet.py not found in model_loaders/.  Copy iresnet.py "
                "from the InsightFace arcface_torch repo.  Original error: "
                f"{_IMPORT_ERR}"
            )
        self.device = device
        self.model = iresnet100(num_features=self.EMBED_DIM)
        state = torch.load(weights_path or config.ARCFACE_WEIGHTS, map_location="cpu")
        # Weights may be a plain state_dict or wrapped.
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
