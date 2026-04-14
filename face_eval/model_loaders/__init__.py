"""Each loader exposes:
    - class FooEmbedder with .embed(aligned_crop_np: np.ndarray) -> np.ndarray (512-d, L2-normalized)
    - load() factory that returns a ready-to-use embedder

AdaFace additionally returns a (embedding, quality_norm) tuple.
"""

from . import adaface, arcface, facenet, lvface, magface  # noqa: F401
