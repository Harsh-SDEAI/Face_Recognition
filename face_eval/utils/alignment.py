"""Eye-landmark-based rotation alignment.

Ported from production AIPhotoMatch.py:72 (`alignment_procedure`) unchanged
so evaluation matches production's alignment exactly.  All 5 models consume
the same aligned crop, so only the embedding stage varies between models.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def find_euclidean_distance(src: np.ndarray, dst: np.ndarray) -> float:
    """Euclidean distance used for alignment triangle sides."""
    return float(np.linalg.norm(src - dst))


def alignment_procedure(img: Image.Image, left_eye, right_eye) -> np.ndarray:
    """Rotate a cropped face so the eyes are horizontal.

    Takes a PIL image and two (x, y) eye coordinates; returns a numpy array.
    Direct port of production function so evaluation alignment == production.
    """
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # clockwise
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # counter-clockwise

    a = find_euclidean_distance(np.array(left_eye), np.array(point_3rd))
    b = find_euclidean_distance(np.array(right_eye), np.array(point_3rd))
    c = find_euclidean_distance(np.array(left_eye), np.array(right_eye))

    if b != 0 and c != 0:
        cos_a = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
        angle = np.degrees(np.arccos(cos_a))
        if direction == -1:
            angle = 90 - angle
        img = img.rotate(direction * angle, resample=Image.BICUBIC)

    return np.array(img)
