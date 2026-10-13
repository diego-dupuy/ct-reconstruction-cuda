"""Image quality against the original image: RMSE, PSNR and SSIM."""

import numpy as np
from skimage.metrics import structural_similarity


def fov_mask(shape):
    """Circle inscribed in the image: the region every projection angle actually sees."""
    rows, cols = shape[0], shape[1]
    y, x = np.mgrid[:rows, :cols]
    radius = min(rows, cols) / 2
    return (y - (rows - 1) / 2) ** 2 + (x - (cols - 1) / 2) ** 2 <= radius**2


def rmse(reference, image, mask=None):
    diff = image.astype(np.float64) - reference.astype(np.float64)
    if mask is not None:
        diff = diff[mask]
    return float(np.sqrt(np.mean(diff**2)))


def psnr(reference, image, mask=None):
    """Peak signal-to-noise ratio in dB (higher = better)."""
    data_range = float(reference.max() - reference.min())
    return float(20 * np.log10(data_range / rmse(reference, image, mask)))


def ssim(reference, image):
    """Structural similarity (1 = identical). For 3D volumes: mean over axial slices."""
    data_range = float(reference.max() - reference.min())
    if reference.ndim == 2:
        return float(structural_similarity(reference, image, data_range=data_range))
    values = [
        structural_similarity(reference[:, :, z], image[:, :, z], data_range=data_range)
        for z in range(reference.shape[2])
    ]
    return float(np.mean(values))


def evaluate(reference, images, use_fov=True):
    """Metrics for a dict {method: image}. Returns {method: {"RMSE":, "PSNR":, "SSIM":}}."""
    mask = fov_mask(reference.shape) if use_fov and reference.ndim == 2 else None
    return {
        name: {
            "RMSE": rmse(reference, img, mask),
            "PSNR": psnr(reference, img, mask),
            "SSIM": ssim(reference, img),
        }
        for name, img in images.items()
    }
