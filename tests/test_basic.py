import astra
import numpy as np

from src.metrics import fov_mask, psnr, rmse, ssim
from src.phantom import pelvis_phantom
from src.reconstruction import reconstruct_2d


def setup_parallel_2d(image):
    vol_geom = astra.create_vol_geom(*image.shape)
    angles = np.linspace(0, np.pi, 180, endpoint=False)
    proj_geom = astra.create_proj_geom("parallel", 1.0, image.shape[1], angles)
    proj_id = astra.create_projector("linear", proj_geom, vol_geom)
    sino_id, _ = astra.create_sino(image, proj_id)
    return vol_geom, proj_id, sino_id


def test_phantom_values():
    p = pelvis_phantom()
    assert p.shape == (256, 256)
    assert p[197, 128] == np.float32(0.2)  # spinal canal
    assert p[140, 50] == np.float32(1.0)  # ilium
    assert p[130, 95] == 0.0  # gas


def test_metrics_perfect_image():
    p = pelvis_phantom()
    assert rmse(p, p) == 0.0
    assert ssim(p, p) == 1.0


def test_fbp_reconstructs_phantom():
    p = pelvis_phantom()
    vol_geom, proj_id, sino_id = setup_parallel_2d(p)
    image, _ = reconstruct_2d("FBP", sino_id, vol_geom, proj_id)
    astra.data2d.delete(sino_id)
    astra.projector.delete(proj_id)
    assert psnr(p, image, fov_mask(p.shape)) > 30


def test_sart_needs_iterations_per_angle():
    """1 ASTRA iteration of SART = 1 angle: 180 iterations is only ONE pass."""
    p = pelvis_phantom()
    vol_geom, proj_id, sino_id = setup_parallel_2d(p)
    random_order = {"ProjectionOrder": "random"}
    one_pass, _ = reconstruct_2d("SART", sino_id, vol_geom, proj_id, 180, random_order)
    ten_passes, _ = reconstruct_2d("SART", sino_id, vol_geom, proj_id, 1800, random_order)
    astra.data2d.delete(sino_id)
    astra.projector.delete(proj_id)
    assert rmse(p, ten_passes) < rmse(p, one_pass)
