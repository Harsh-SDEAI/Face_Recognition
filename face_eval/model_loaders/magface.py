"""MagFace (IR-100) loader.

Depends on `magface_iresnet.py` which is `models/iresnet.py` from
https://github.com/IrvingMeng/MagFace.

MagFace shares ArcFace's input pipeline: 112x112 BGR, [-1, 1] normalization,
512-d output.  The magnitude of the un-normalized embedding is itself a
quality signal in MagFace (larger norm == higher confidence), but we return
L2-normalized features here so cosine similarity is comparable to the other
models.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.nn.functional import normalize

import config

try:
    from . import magface_iresnet  # type: ignore
except Exception as exc:  # noqa: BLE001
    magface_iresnet = None
    _IMPORT_ERR = exc
else:
    _IMPORT_ERR = None


class MagFaceEmbedder:
    INPUT_SIZE = 112
    EMBED_DIM = 512

    def __init__(self, device: torch.device, weights_path=None):
        if magface_iresnet is None:
            raise ImportError(
                "magface_iresnet.py not found in model_loaders/.  Copy "
                "models/iresnet.py from the MagFace repo.  Original error: "
                f"{_IMPORT_ERR}"
            )
        self.device = device
        # MagFace's iresnet.py exposes iresnet100 as the public constructor.
        self.model = magface_iresnet.iresnet100(num_classes=self.EMBED_DIM)
        ckpt = torch.load(weights_path or config.MAGFACE_WEIGHTS, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        # The released checkpoint wraps:
        #   features.*   -> IResNet backbone (what we want)
        #   fc.weight    -> identity classifier head from training (discard)
        # DataParallel may also prefix with "module.".  Keep only the backbone.
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        # Two checkpoint layouts exist:
        #   (a) features.* = backbone, fc.* = training head  (official release)
        #   (b) backbone keys at top level, fc.* = training head (epoch_*.pth)
        if any(k.startswith("features.") for k in state):
            state = {k[len("features."):]: v for k, v in state.items()
                     if k.startswith("features.")}
        else:
            # Drop the classifier head (fc.weight has shape [num_ids, 512]).
            state = {k: v for k, v in state.items() if not k.startswith("fc.")}
        self.model.load_state_dict(state, strict=False)
        self.model.eval().to(device)

    @torch.no_grad()
    def embed(self, aligned_rgb: np.ndarray) -> np.ndarray:
        img = cv2.resize(aligned_rgb, (self.INPUT_SIZE, self.INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        t = (t / 255.0 - 0.5) / 0.5
        e = self.model(t)
        e = normalize(e, p=2, dim=1)
        return e.detach().cpu().numpy().reshape(-1).astype(np.float32)


def load(device: torch.device) -> MagFaceEmbedder:
    return MagFaceEmbedder(device)
