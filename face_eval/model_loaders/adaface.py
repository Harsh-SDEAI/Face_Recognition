"""AdaFace (IR-101 / WebFace12M) loader.

Depends on `adaface_net.py` which is `net.py` from the root of
https://github.com/mk-minchul/AdaFace.

AdaFace takes BGR [-1, 1] input and returns (embedding, norm) where `norm`
is the raw feature magnitude used as a quality proxy.  We expose the norm
directly so the Streamlit UI can display "quality" for each face.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.nn.functional import normalize

import config

try:
    from . import adaface_net  # type: ignore
except Exception as exc:  # noqa: BLE001
    adaface_net = None
    _IMPORT_ERR = exc
else:
    _IMPORT_ERR = None


class AdaFaceEmbedder:
    INPUT_SIZE = 112
    EMBED_DIM = 512

    def __init__(self, device: torch.device, weights_path=None):
        if adaface_net is None:
            raise ImportError(
                "adaface_net.py not found in model_loaders/.  Copy net.py from "
                "the AdaFace repo root.  Original error: "
                f"{_IMPORT_ERR}"
            )
        self.device = device
        # build_model is the public factory in AdaFace/net.py
        self.model = adaface_net.build_model("ir_101")
        ckpt = torch.load(weights_path or config.ADAFACE_WEIGHTS, map_location="cpu")
        # AdaFace .ckpt is a pytorch-lightning checkpoint: model weights live
        # in 'state_dict' with 'model.' prefix; strip it.
        state = ckpt.get("state_dict", ckpt)
        state = {k.replace("model.", "", 1): v for k, v in state.items() if k.startswith("model.")}
        self.model.load_state_dict(state, strict=False)
        self.model.eval().to(device)

    @torch.no_grad()
    def embed_with_norm(self, aligned_rgb: np.ndarray) -> tuple[np.ndarray, float]:
        img = cv2.resize(aligned_rgb, (self.INPUT_SIZE, self.INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)       # AdaFace uses BGR
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        t = (t / 255.0 - 0.5) / 0.5
        feature, norm = self.model(t)
        feature = normalize(feature, p=2, dim=1)
        norm_val = float(norm.detach().cpu().numpy().reshape(-1)[0])
        return (feature.detach().cpu().numpy().reshape(-1).astype(np.float32), norm_val)

    def embed(self, aligned_rgb: np.ndarray) -> np.ndarray:
        return self.embed_with_norm(aligned_rgb)[0]


def load(device: torch.device) -> AdaFaceEmbedder:
    return AdaFaceEmbedder(device)
