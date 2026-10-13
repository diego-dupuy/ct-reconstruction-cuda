"""Synthetic 2D pelvis phantom (256 x 256) built from ellipses."""

import numpy as np
from skimage.draw import ellipse

# (centre_row, centre_col, radius_row, radius_col, attenuation)
PELVIS_SHAPES = [
    (128, 128, 85, 115, 0.20),  # outer contour: fat
    (130, 128, 70, 100, 0.30),  # muscle / soft organs
    (195, 128, 15, 18, 1.00),  # vertebral body (bone)
    (197, 128, 5, 5, 0.20),  # spinal canal
    (140, 50, 45, 15, 1.00),  # left ilium (bone)
    (140, 206, 45, 15, 1.00),  # right ilium (bone)
    (90, 128, 25, 35, 0.45),  # bladder (fluid)
    (130, 95, 12, 12, 0.00),  # bowel gas
    (150, 160, 10, 15, 0.00),  # bowel gas
]


def pelvis_phantom(size=256):
    """Return the pelvis phantom as a float32 image. Later shapes overwrite earlier ones."""
    image = np.zeros((size, size), dtype=np.float32)
    for cy, cx, ry, rx, value in PELVIS_SHAPES:
        rr, cc = ellipse(cy, cx, ry, rx, shape=image.shape)
        image[rr, cc] = value
    return image
