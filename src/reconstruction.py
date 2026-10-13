"""Reconstruction with ASTRA Toolbox: 2D (CPU or GPU), fan beam slice by slice, cone beam 3D.

Every function deletes the ASTRA objects it creates (inside try/finally), so running
hundreds of slices in Colab does not fill the GPU/RAM.
"""

import time

import astra
import numpy as np


def reconstruct_2d(method, sinogram_id, vol_geom, proj_id=None, iterations=1, options=None):
    """Run one ASTRA 2D algorithm. Returns (image, seconds).

    method: "FBP", "SIRT", "SART", "ART", "CGLS", "BP" (CPU, needs proj_id)
            or the same name + "_CUDA" (GPU).
    iterations: for SIRT/CGLS one iteration = all angles; for SART one iteration = ONE
            angle; for ART one iteration = ONE ray (see README).
    """
    rec_id = astra.data2d.create("-vol", vol_geom)
    cfg = astra.astra_dict(method)
    cfg["ReconstructionDataId"] = rec_id
    cfg["ProjectionDataId"] = sinogram_id
    if proj_id is not None:
        cfg["ProjectorId"] = proj_id
    if options:
        cfg["option"] = options
    alg_id = astra.algorithm.create(cfg)
    try:
        start = time.perf_counter()
        astra.algorithm.run(alg_id, iterations)
        seconds = time.perf_counter() - start
        image = astra.data2d.get(rec_id)
    finally:
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
    return image, seconds


def fanbeam_slice_by_slice(volume, methods, options=None, n_angles=360):
    """Reconstruct a (rows, cols, slices) volume slice by slice on the GPU (fan beam).

    methods: dict {"FBP": 1, "SIRT": 50, ...} -> method name and iterations.
    options: optional dict {"SART": {...}} with ASTRA options per method.
    Returns (dict of reconstructed volumes, dict of total GPU seconds).
    """
    rows, cols, n_slices = volume.shape
    assert rows == cols, "slices must be square"
    size = rows
    vol_geom = astra.create_vol_geom(size, size)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    # Source at 3*size and detector at 2*size from the centre -> magnification 5/3.
    # A detector 2.5*size wide covers the whole magnified slice (needs ~2.43*size).
    proj_geom = astra.create_proj_geom(
        "fanflat", 1.0, int(2.5 * size), angles, 3.0 * size, 2.0 * size
    )
    proj_id = astra.create_projector("cuda", proj_geom, vol_geom)
    options = options or {}
    results = {m: np.zeros_like(volume) for m in methods}
    times = {m: 0.0 for m in methods}
    try:
        for z in range(n_slices):
            if z % 20 == 0:
                print(f"  slice {z}/{n_slices}")
            sino_id, _ = astra.create_sino(volume[:, :, z], proj_id)
            try:
                for method, iterations in methods.items():
                    image, seconds = reconstruct_2d(
                        method + "_CUDA", sino_id, vol_geom,
                        iterations=iterations, options=options.get(method),
                    )
                    results[method][:, :, z] = image
                    times[method] += seconds
            finally:
                astra.data2d.delete(sino_id)
    finally:
        astra.projector.delete(proj_id)
    return results, times


def conebeam_3d(volume, methods, n_angles=360):
    """Reconstruct the whole volume at once with a cone beam on the GPU.

    methods: dict {"FDK": 1, "SIRT": 20, "CGLS": 10, "BP": 1}.
    Note: ASTRA has no 3D SART ("SART3D_CUDA" raises "Unknown algorithm type").
    Returns (dict of reconstructed volumes, dict of GPU seconds).
    """
    rows, cols, n_slices = volume.shape
    size = max(rows, cols)
    dist_source, dist_detector = 3.0 * size, 2.0 * size
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    # The cone also magnifies vertically. Worst case = a voxel at the top of the volume
    # on the side closest to the source (distance dist_source - size/sqrt(2)).
    # With one detector row per slice the outer slices fall off the detector.
    magnification = (dist_source + dist_detector) / (dist_source - size / np.sqrt(2))
    det_rows = int(np.ceil(n_slices * magnification))

    vol_geom = astra.create_vol_geom(rows, cols, n_slices)
    proj_geom = astra.create_proj_geom(
        "cone", 1.0, 1.0, det_rows, int(2.5 * size), angles, dist_source, dist_detector
    )
    # ASTRA 3D stores volumes as (slices, rows, cols)
    vol_astra = np.ascontiguousarray(np.transpose(volume, (2, 0, 1)))

    # 1) Simulated acquisition (forward projection)
    vol_id = astra.data3d.create("-vol", vol_geom, data=vol_astra)
    sino_id = astra.data3d.create("-sino", proj_geom)
    cfg = astra.astra_dict("FP3D_CUDA")
    cfg["VolumeDataId"] = vol_id
    cfg["ProjectionDataId"] = sino_id
    fp_id = astra.algorithm.create(cfg)
    astra.algorithm.run(fp_id)
    astra.algorithm.delete(fp_id)
    astra.data3d.delete(vol_id)  # free RAM: the original volume is not needed in ASTRA

    # 2) Reconstruction with each method
    results, times = {}, {}
    try:
        for method, iterations in methods.items():
            name = "FDK_CUDA" if method == "FDK" else method + "3D_CUDA"
            rec_id = astra.data3d.create("-vol", vol_geom)
            cfg = astra.astra_dict(name)
            cfg["ReconstructionDataId"] = rec_id
            cfg["ProjectionDataId"] = sino_id
            alg_id = astra.algorithm.create(cfg)
            try:
                start = time.perf_counter()
                astra.algorithm.run(alg_id, iterations)
                times[method] = time.perf_counter() - start
                results[method] = np.transpose(astra.data3d.get(rec_id), (1, 2, 0))
            finally:
                astra.algorithm.delete(alg_id)
                astra.data3d.delete(rec_id)
    finally:
        astra.data3d.delete(sino_id)
    return results, times
