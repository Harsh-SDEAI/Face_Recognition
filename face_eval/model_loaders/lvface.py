"""LVFace (ByteDance, ICCV 2025) loader - ONNX Runtime backend.

The official LVFace release (github.com/bytedance/LVFace) ships an
`inference_onnx.py` using `onnxruntime` and publishes ONNX weights on the
Hugging Face repo `bytedance-research/LVFace`.  We use the ONNX path so no
Python model-definition file needs to be copied - the ONNX graph carries
the full architecture.

Weights are auto-downloaded to `models/lvface.onnx` on first run.

Input and normalization match the official inference_onnx.py:
    - 112x112 RGB
    - scaled to [-1, 1] via (img/255 - 0.5) / 0.5
    - NCHW float32
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

import config

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    ort = None
    _IMPORT_ERR = exc
else:
    _IMPORT_ERR = None


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

    def __init__(self, device):
        if ort is None:
            raise ImportError(f"onnxruntime not installed: {_IMPORT_ERR}")
        weights = _download_weights_if_missing()

        # Prefer CUDA provider when available; fall back to CPU.
        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(str(weights), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def embed(self, aligned_rgb: np.ndarray) -> np.ndarray:
        img = cv2.resize(aligned_rgb, (self.INPUT_SIZE, self.INPUT_SIZE))
        arr = img.astype(np.float32)
        arr = (arr / 255.0 - 0.5) / 0.5           # [-1, 1]
        arr = arr.transpose(2, 0, 1)[None, ...]   # NCHW, 1x3x112x112
        out = self.session.run(None, {self.input_name: arr})[0]
        e = out.reshape(-1).astype(np.float32)
        # L2 normalize so cosine = dot product, matching the other loaders.
        norm = float(np.linalg.norm(e))
        if norm > 0:
            e = e / norm
        return e


def load(device) -> LVFaceEmbedder:
    return LVFaceEmbedder(device)
