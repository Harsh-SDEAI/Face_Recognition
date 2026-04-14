"""FaceNet baseline loader.

Uses the exact same model as production (`InceptionResnetV1` pretrained on
VGGFace2).  Input 160x160 RGB, scaled to [0,1].  Output is L2-normalized here
so cosine similarity == dot product.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from torch.nn.functional import normalize


class FaceNetEmbedder:
    INPUT_SIZE = 160
    EMBED_DIM = 512

    def __init__(self, device: torch.device):
        self.device = device
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    @torch.no_grad()
    def embed(self, aligned_rgb: np.ndarray) -> np.ndarray:
        # aligned_rgb: HxWx3 uint8, RGB channel order (PIL default)
        img = cv2.resize(aligned_rgb, (self.INPUT_SIZE, self.INPUT_SIZE))
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        e = self.model(t)
        e = normalize(e, p=2, dim=1)
        return e.detach().cpu().numpy().reshape(-1).astype(np.float32)


def load(device: torch.device) -> FaceNetEmbedder:
    return FaceNetEmbedder(device)
