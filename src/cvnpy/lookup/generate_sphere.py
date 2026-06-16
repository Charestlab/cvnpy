"""Sphere lookup generation.

Parity target:
- /Users/iancharest/Documents/codex/cvncode/kjutils/spherelookup_generate.m
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import cKDTree


def _as_img_shape(img_n: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(img_n, int):
        return img_n, img_n
    if len(img_n) != 2:
        raise ValueError("img_n must be int or (height,width)")
    return int(img_n[0]), int(img_n[1])


def _rotation_about_z(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta), np.sin(theta), 0.0, 0.0],
            [-np.sin(theta), np.cos(theta), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _apply_affine(tx: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    homog = np.c_[xyz, np.ones((xyz.shape[0], 1), dtype=float)]
    out = (tx @ homog.T).T
    return out[:, :3]


def generate_sphere_lookup(
    vertsph: np.ndarray,
    azimuth: float,
    elevation: float,
    tilt: float,
    xyextent: tuple[float, float],
    img_n: int | tuple[int, int],
    **kwargs: Any,
) -> dict:
    """Generate forward/reverse lookup tables for spherical surfaces.

    Implemented behavior follows MATLAB logic at a practical level for:
    - forward nearest-neighbor image lookup
    - extrapolation mask
    - reverse lookup to image pixels
    """
    reversevert = kwargs.get("reversevert")

    vertsph = np.asarray(vertsph, dtype=float)
    if vertsph.ndim != 2 or vertsph.shape[1] != 3:
        raise ValueError("vertsph must be Nx3")

    h, w = _as_img_shape(img_n)
    if not xyextent:
        xyextent = (0.6, 0.6)
    if len(xyextent) == 1:
        xyextent = (abs(xyextent[0]), abs(xyextent[0]))

    # map az/el to a view direction, matching MATLAB conventions.
    view_az = np.deg2rad(float(azimuth)) - np.pi / 2
    view_el = np.deg2rad(float(elevation))
    view_tilt = np.deg2rad(float(tilt)) + np.pi

    v3 = np.array(
        [
            np.cos(view_el) * np.cos(view_az),
            np.cos(view_el) * np.sin(view_az),
            np.sin(view_el),
        ],
        dtype=float,
    )
    v3 = v3 / np.linalg.norm(v3)

    az_tilt = 0.0
    if np.sqrt(np.dot(v3[:2], v3[:2])) < 1e-3:
        v2 = np.cross(v3, np.array([0.0, 1.0, 0.0]))
        v2 = v2 / np.linalg.norm(v2)
        v1 = -np.cross(v3, v2)
        v1 = v1 / np.linalg.norm(v1)
        if v3[2] < 0:
            az_tilt = -np.deg2rad(float(azimuth))
            if abs(az_tilt) < 1e-3:
                az_tilt += np.pi
        else:
            az_tilt = np.deg2rad(float(azimuth))
    else:
        v2 = np.cross(v3, np.array([0.0, 0.0, 1.0]))
        v2 = v2 / np.linalg.norm(v2)
        v1 = np.cross(v2, v3)
        v1 = v1 / np.linalg.norm(v1)

    # recenter and rescale to approximate unit sphere.
    c = np.mean(vertsph, axis=0)
    r = np.mean(np.linalg.norm(vertsph - c, axis=1))
    msph = np.array(
        [
            [1.0 / r, 0.0, 0.0, -c[0] / r],
            [0.0, 1.0 / r, 0.0, -c[1] / r],
            [0.0, 0.0, 1.0 / r, -c[2] / r],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    basis = np.c_[v1, v2, v3]
    m = np.eye(4, dtype=float)
    m[:3, :3] = np.linalg.inv(basis)
    mtilt = _rotation_about_z(view_tilt + az_tilt)
    txview = mtilt @ m @ msph

    viewvert = _apply_affine(txview, vertsph)

    xview = np.array([-xyextent[0], xyextent[0]], dtype=float)
    yview = np.array([-xyextent[1], xyextent[1]], dtype=float)
    zview = np.array([0.0, np.inf], dtype=float)

    xview_padded = xview * 1.1
    yview_padded = yview * 1.1

    imgy, imgx = np.meshgrid(
        np.linspace(yview[0], yview[1], w),
        np.linspace(xview[0], xview[1], h),
    )

    viewmask = (
        (viewvert[:, 0] >= xview[0])
        & (viewvert[:, 0] <= xview[1])
        & (viewvert[:, 1] >= yview[0])
        & (viewvert[:, 1] <= yview[1])
        & (viewvert[:, 2] >= zview[0])
        & (viewvert[:, 2] <= zview[1])
    )
    viewmask_padded = (
        (viewvert[:, 0] >= xview_padded[0])
        & (viewvert[:, 0] <= xview_padded[1])
        & (viewvert[:, 1] >= yview_padded[0])
        & (viewvert[:, 1] <= yview_padded[1])
        & (viewvert[:, 2] >= zview[0])
        & (viewvert[:, 2] <= zview[1])
    )

    vertidx = np.arange(vertsph.shape[0], dtype=np.int64)

    if np.count_nonzero(viewmask_padded) == 0:
        imglookup = np.zeros((h, w), dtype=np.int64)
        extrapmask = np.ones((h, w), dtype=bool)
    else:
        interp = NearestNDInterpolator(
            viewvert[viewmask_padded, :2], vertidx[viewmask_padded]
        )
        mapped = interp(imgx, imgy)
        extrapmask = ~np.isfinite(mapped)
        mapped[extrapmask] = 0
        imglookup = mapped.astype(np.int64)

    lookupmasks = np.zeros((vertsph.shape[0],), dtype=bool)
    lookupmasks[np.unique(imglookup)] = True

    reverselookup = np.full((vertsph.shape[0],), -1, dtype=np.int64)
    pixflat = np.arange(h * w, dtype=np.int64)
    valid_pix = ~extrapmask.ravel()
    reverselookup[imglookup.ravel()[valid_pix]] = pixflat[valid_pix]

    # Fill missing reverse lookups for in-view vertices using nearest already-mapped vertices.
    missing = viewmask_padded & (reverselookup < 0)
    have = viewmask & (reverselookup >= 0)
    if np.any(missing) and np.any(have):
        if reversevert is not None:
            reversevert = np.asarray(reversevert, dtype=float)
            if reversevert.shape != vertsph.shape:
                raise ValueError("reversevert must have same shape as vertsph")
            src = reversevert[have]
            qry = reversevert[missing]
        else:
            src = viewvert[have, :2]
            qry = viewvert[missing, :2]

        tree = cKDTree(src)
        _dist, nn = tree.query(qry, k=1)
        have_idx = np.flatnonzero(have)
        missing_idx = np.flatnonzero(missing)
        reverselookup[missing_idx] = reverselookup[have_idx[nn]]

    txpix = np.array(
        [
            [(h - 1) / (xview[1] - xview[0]), 0.0, 0.0, (h - 1) * (-xview[0]) / (xview[1] - xview[0]) + 1.0],
            [0.0, (w - 1) / (yview[1] - yview[0]), 0.0, (w - 1) * (-yview[0]) / (yview[1] - yview[0]) + 1.0],
            [0.0, 0.0, (h + w) / (xview[1] - xview[0] + yview[1] - yview[0]), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    ) @ txview

    return {
        "imglookup": imglookup,
        "vertmasks": viewmask,
        "lookupmasks": lookupmasks,
        "reverselookup": reverselookup,
        "extrapmask": extrapmask,
        "is_extrapolated": bool(np.any(extrapmask)),
        "azimuth": float(azimuth),
        "elevation": float(elevation),
        "tilt": float(tilt),
        "imgN": w,
        "imgsize": (h, w),
        "vertsN": int(vertsph.shape[0]),
        "TXview": txview,
        "xyextent": tuple(float(v) for v in xyextent),
        "xview": tuple(float(v) for v in xview),
        "yview": tuple(float(v) for v in yview),
        "zview": tuple(float(v) for v in zview),
        "TXpix": txpix,
        "input2surf": None,
    }
