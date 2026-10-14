"""Colormap and matrix->RGB conversion.

Parity reference: MATLAB mat2rgb behavior used by cvnlookupimages.m.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import colormaps


def _cmapsign4_table(n: int = 64) -> np.ndarray:
    """Port of knkutils/colormap/cmapsign4.m."""
    n = int(n)
    if n <= 1:
        n = 2
    colors = np.array(
        [
            [0.8, 1.0, 1.0],  # cyan-white
            [0.0, 1.0, 1.0],  # cyan
            [0.0, 0.0, 1.0],  # blue
            [0.0, 0.0, 0.0],  # black
            [1.0, 0.0, 0.0],  # red
            [1.0, 1.0, 0.0],  # yellow
            [1.0, 1.0, 0.8],  # yellow-white
        ],
        dtype=float,
    )
    xp = np.linspace(0.0, 1.0, colors.shape[0], dtype=float)
    x = np.linspace(0.0, 1.0, n, dtype=float)
    out = np.empty((n, 3), dtype=float)
    for c in range(3):
        out[:, c] = np.interp(x, xp, colors[:, c])
    return out


def _table_mapper(arr: np.ndarray) -> Any:
    def _table_map(x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
        x = np.clip(x, 0.0, 1.0)
        ix = np.rint(x * (arr.shape[0] - 1)).astype(int)
        return arr[ix]

    return _table_map


def _resolve_cmap(cmap: Any) -> Any:
    if cmap is None:
        return _table_mapper(_cmapsign4_table())
    if isinstance(cmap, str):
        if cmap.lower() == "cmapsign4":
            return _table_mapper(_cmapsign4_table())
        return colormaps.get_cmap(cmap)
    if callable(cmap):
        return cmap
    arr = np.asarray(cmap)
    if arr.ndim == 2 and arr.shape[1] in (3, 4):
        return _table_mapper(arr)
    raise ValueError("Unsupported cmap type")


def mat_to_rgb(mappedvals: np.ndarray, **kwargs: object) -> np.ndarray:
    """Convert mapped values to RGB.

    Supports minimal options:
    - cmap/colormap
    - clim=(min,max)
    - rgbnan (scalar or 3-vector)
    """
    vals = np.asarray(mappedvals, dtype=float)
    clim = kwargs.get("clim", None)
    cmap_in = kwargs.get("cmap", kwargs.get("colormap", None))
    rgbnan = kwargs.get("rgbnan", 1.0)

    if clim is None:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
    else:
        clim_arr = np.asarray(clim, dtype=float)
        vmin = float(clim_arr[0])
        vmax = float(clim_arr[1])
        if not np.isfinite(vmin):
            vmin = float(np.nanmin(vals))
        if not np.isfinite(vmax):
            vmax = float(np.nanmax(vals))

    if vmax == vmin:
        vmax = vmin + 1.0

    norm = (vals - vmin) / (vmax - vmin)
    cmap = _resolve_cmap(cmap_in)
    rgba = cmap(np.clip(norm, 0.0, 1.0))
    rgb = np.asarray(rgba[..., :3], dtype=float)

    nan_mask = ~np.isfinite(vals)
    if np.any(nan_mask):
        rgbnan_arr = np.asarray(rgbnan, dtype=float)
        if rgbnan_arr.size == 1:
            rgb[nan_mask] = float(rgbnan_arr)
        elif rgbnan_arr.size == 3:
            rgb[nan_mask] = rgbnan_arr.reshape(1, 3)
        else:
            raise ValueError("rgbnan must be scalar or RGB triplet")

    return np.clip(rgb, 0.0, 1.0)
