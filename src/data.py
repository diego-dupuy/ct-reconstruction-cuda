"""Load the thoracic CT volume (datos_ct.mat)."""

import numpy as np
import scipy.io as sio


def load_ct_volume(path, key="C002_images_full_att"):
    """Return the volume (rows, cols, slices) as float32, normalised to [0, 1]."""
    volume = sio.loadmat(path)[key].astype(np.float32)
    return volume / volume.max()
