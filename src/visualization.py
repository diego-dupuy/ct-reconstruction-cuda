"""Figures: side-by-side comparison, error maps and 3D bone isosurface."""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure


def show_comparison(reference, images, metrics=None, times=None, title=None, ncols=4):
    """Original + reconstructions with the SAME grey scale and one shared colour bar.

    BP is not normalised by ASTRA, so it is rescaled to [0, max(reference)] and labelled.
    """
    panels = [("Original", reference)] + list(images.items())
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.5 * nrows), squeeze=False)
    vmax = float(reference.max())
    for ax, (name, img) in zip(axes.flat, panels):
        label = name
        if name == "BP":
            img = img / img.max() * vmax
            label = "BP (rescaled)"
        if metrics and name in metrics:
            label += f"\nPSNR {metrics[name]['PSNR']:.1f} dB · SSIM {metrics[name]['SSIM']:.3f}"
        if times and name in times:
            label += f" · {times[name]:.2f} s"
        im = ax.imshow(img, cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(label, fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.7, label="attenuation")
    if title:
        fig.suptitle(title, fontweight="bold")
    return fig


def show_error_maps(reference, images, limit=0.2, ncols=3):
    """Reconstruction minus original. White = no error, red = too high, blue = too low."""
    names = [n for n in images if n != "BP"]
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.3 * nrows), squeeze=False)
    for ax, name in zip(axes.flat, names):
        im = ax.imshow(images[name] - reference, cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_title(name, fontsize=10)
    for ax in axes.flat:
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.7, label="error")
    return fig


def show_isosurface(volume, level, step_size=2, title="Bone isosurface"):
    """Marching cubes at `level` (bone threshold). step_size=2 -> ~4x fewer triangles,
    so the mesh fits in Colab's RAM."""
    verts, faces, _, _ = measure.marching_cubes(volume, level=level, step_size=step_size)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(verts[faces], alpha=1.0, linewidths=0)
    mesh.set_facecolor((0.85, 0.82, 0.75))
    ax.add_collection3d(mesh)
    ax.set_xlim(0, volume.shape[0])
    ax.set_ylim(0, volume.shape[1])
    ax.set_zlim(0, volume.shape[2])
    ax.view_init(elev=15, azim=-60)
    ax.set_title(title, fontweight="bold")
    print(f"{len(faces):,} triangles")
    return fig
