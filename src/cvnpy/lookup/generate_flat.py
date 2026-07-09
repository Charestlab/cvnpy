"""Flat-patch lookup generation.

Parity target:
- /Users/iancharest/Documents/codex/cvncode/kjutils/spherelookup_generate_flat.m
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import cKDTree


def _as_img_shape(img_n: int | tuple[int, int], aspect: float) -> tuple[int, int]:
    if isinstance(img_n, int):
        h = int(img_n)
        w = max(1, int(round(h * aspect)))
        return h, w
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


def _patchedgeloop_longest(faces: np.ndarray) -> np.ndarray:
    if faces.shape[0] == 0:
        return np.array([], dtype=np.int64)

    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [0, 2]], faces[:, [1, 2]]], axis=0
    )
    edges = np.sort(edges, axis=1)
    uniq, inv = np.unique(edges, axis=0, return_inverse=True)
    counts = np.bincount(inv, minlength=uniq.shape[0])
    loopedges = uniq[counts == 1, :]
    if loopedges.shape[0] == 0:
        return np.array([], dtype=np.int64)

    oldloop = loopedges.copy()
    o = int(loopedges[0, 0])
    newloop = np.zeros((oldloop.size,), dtype=float)
    c = 0
    newloop[c] = o

    for _ in range(loopedges.shape[0]):
        r = np.where(np.any(oldloop == o, axis=1))[0]
        if r.size == 0:
            remain = oldloop[oldloop >= 0]
            if remain.size == 0:
                break
            c += 1
            newloop[c] = np.nan
            o = int(remain[0])
            continue

        rr = int(r[0])
        m = oldloop[rr, :]
        nxt = m[m != o]
        if nxt.size == 0:
            break
        o = int(nxt[0])
        oldloop[rr, :] = -1
        c += 1
        newloop[c] = o

    newloop = newloop[: c + 1]
    if not np.any(np.isnan(newloop)):
        return newloop.astype(np.int64)

    nan_idx = np.where(np.isnan(newloop))[0]
    starts = np.r_[0, nan_idx + 1]
    ends = np.r_[nan_idx - 1, newloop.size - 1]
    lengths = ends - starts + 1
    k = int(np.argmax(lengths))
    seg = newloop[starts[k] : ends[k] + 1]
    return seg[~np.isnan(seg)].astype(np.int64)


def _clean_flat_patch_polygon(
    faces: np.ndarray, vertices: np.ndarray, max_iter: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    f = faces.copy()
    nverts = vertices.shape[0]

    for _ in range(max_iter):
        pmask = np.zeros((nverts,), dtype=bool)
        if f.shape[0] > 0:
            pmask[np.unique(f.ravel())] = True

        nF = f.shape[0]
        nP = int(np.sum(pmask))

        loop = _patchedgeloop_longest(f)
        if loop.size < 3:
            break

        poly = vertices[loop, :2]
        path = MplPath(poly, closed=True)
        inside = np.zeros((nverts,), dtype=bool)
        if np.any(pmask):
            inside[pmask] = path.contains_points(vertices[pmask, :2], radius=1e-9)

        pmask = pmask & inside
        if f.shape[0] > 0:
            f = f[np.all(pmask[f], axis=1), :]

        if f.shape[0] == nF and int(np.sum(pmask)) == nP:
            break

    patchmask = np.zeros((nverts,), dtype=bool)
    if f.shape[0] > 0:
        patchmask[np.unique(f.ravel())] = True
    return f, patchmask


def generate_flat_lookup(
    verts: np.ndarray,
    faces: np.ndarray,
    azimuth: float,
    elevation: float,
    tilt: float,
    xyextent: tuple[float, float],
    img_n: int | tuple[int, int],
    **kwargs: Any,
) -> dict:
    """Generate lookup tables for flat patch surfaces."""
    reversevert = kwargs.get("reversevert")
    margin_frac = float(kwargs.get("margin_frac", 0.04))

    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("verts must be Nx3")

    # MATLAB uses tilt-90deg for flat patches.
    view_tilt = np.deg2rad(float(tilt)) - np.pi / 2
    txview = _rotation_about_z(view_tilt)
    viewvert = _apply_affine(txview, verts)

    # MATLAB parity: clean patch polygon immediately in the current rotated view,
    # then derive flat mask and extents from the cleaned faces.
    faces_valid = np.asarray(faces, dtype=np.int64)
    if faces_valid.shape[0] > 0:
        faces_valid, _ = _clean_flat_patch_polygon(faces_valid, viewvert, max_iter=10)

    flatmask = np.zeros((verts.shape[0],), dtype=bool)
    if faces_valid.size == 0:
        flatmask[:] = True
    else:
        flatmask[np.unique(faces_valid.ravel())] = True

    flatverts = viewvert[flatmask]
    flat_mid = np.mean(flatverts, axis=0)
    flat_lim_min = np.min(flatverts, axis=0)
    flat_lim_max = np.max(flatverts, axis=0)
    flat_size = np.abs(flat_lim_max - flat_lim_min)
    if flat_size[0] == 0:
        flat_size[0] = 1.0
    if flat_size[1] == 0:
        flat_size[1] = 1.0

    h, w = _as_img_shape(img_n, aspect=float(flat_size[1] / flat_size[0]))

    if not xyextent:
        xyextent = (0.6, 0.6)
    if len(xyextent) == 1:
        xyextent = (abs(xyextent[0]), abs(xyextent[0]))
    if tuple(xyextent) == (1, 1):
        xyextent = (round(float(flat_size[0]) * 100.0) / 100.0, 0.0)

    xyextent_mm = (float(xyextent[0]), float(xyextent[1]))
    xyextent_frac_x = xyextent_mm[0] / float(flat_size[0])
    xyextent_frac_y = xyextent_frac_x * (w / h) / (flat_size[1] / flat_size[0])

    xoffset_mm = float(azimuth)
    yoffset_mm = float(elevation)

    xview = (flat_lim_min[0] - flat_mid[0]) * xyextent_frac_x + flat_mid[0] - xoffset_mm, (
        flat_lim_max[0] - flat_mid[0]
    ) * xyextent_frac_x + flat_mid[0] - xoffset_mm
    yview = (flat_lim_min[1] - flat_mid[1]) * xyextent_frac_y + flat_mid[1] - yoffset_mm, (
        flat_lim_max[1] - flat_mid[1]
    ) * xyextent_frac_y + flat_mid[1] - yoffset_mm
    # Add a small fractional margin so flatmaps do not touch image borders.
    margin_frac = max(0.0, margin_frac)
    if margin_frac > 0:
        xspan = xview[1] - xview[0]
        yspan = yview[1] - yview[0]
        xview = (xview[0] - margin_frac * xspan, xview[1] + margin_frac * xspan)
        yview = (yview[0] - margin_frac * yspan, yview[1] + margin_frac * yspan)

    xpad = (
        (flat_lim_min[0] - flat_mid[0]) * xyextent_frac_x * 1.1 + flat_mid[0] - xoffset_mm,
        (flat_lim_max[0] - flat_mid[0]) * xyextent_frac_x * 1.1 + flat_mid[0] - xoffset_mm,
    )
    ypad = (
        (flat_lim_min[1] - flat_mid[1]) * xyextent_frac_y * 1.1 + flat_mid[1] - yoffset_mm,
        (flat_lim_max[1] - flat_mid[1]) * xyextent_frac_y * 1.1 + flat_mid[1] - yoffset_mm,
    )

    zview = (-np.inf, np.inf)

    imgy, imgx = np.meshgrid(
        np.linspace(yview[0], yview[1], w),
        np.linspace(xview[0], xview[1], h),
    )

    viewmask = (
        (viewvert[:, 0] >= xview[0])
        & (viewvert[:, 0] <= xview[1])
        & (viewvert[:, 1] >= yview[0])
        & (viewvert[:, 1] <= yview[1])
        & flatmask
    )
    viewmask_padded = (
        (viewvert[:, 0] >= xpad[0])
        & (viewvert[:, 0] <= xpad[1])
        & (viewvert[:, 1] >= ypad[0])
        & (viewvert[:, 1] <= ypad[1])
        & flatmask
    )

    vertidx = np.arange(verts.shape[0], dtype=np.int64)

    if faces_valid.shape[0] > 0:
        flatmask_interp = np.zeros_like(flatmask)
        flatmask_interp[np.unique(faces_valid.ravel())] = True
    else:
        flatmask_interp = flatmask.copy()

    viewmask = (
        (viewvert[:, 0] >= xview[0])
        & (viewvert[:, 0] <= xview[1])
        & (viewvert[:, 1] >= yview[0])
        & (viewvert[:, 1] <= yview[1])
        & flatmask_interp
    )
    viewmask_padded = (
        (viewvert[:, 0] >= xpad[0])
        & (viewvert[:, 0] <= xpad[1])
        & (viewvert[:, 1] >= ypad[0])
        & (viewvert[:, 1] <= ypad[1])
        & flatmask_interp
    )
    interp_mask = viewmask_padded.copy()

    if np.count_nonzero(interp_mask) == 0:
        imglookup = np.zeros((h, w), dtype=np.int64)
        extrapmask = np.ones((h, w), dtype=bool)
    else:
        interp = NearestNDInterpolator(
            viewvert[interp_mask, :2], vertidx[interp_mask]
        )
        mapped = interp(imgx, imgy)
        extrapmask = ~np.isfinite(mapped)
        mapped[extrapmask] = 0
        imglookup = mapped.astype(np.int64)

    edge_loop = _patchedgeloop_longest(faces_valid)
    if edge_loop.size >= 3:
        poly = viewvert[edge_loop, :2]
        inside_poly = MplPath(poly, closed=True).contains_points(
            np.c_[imgx.ravel(), imgy.ravel()], radius=1e-9
        ).reshape((h, w))
        inside_mask = inside_poly
    else:
        inside_mask = ~extrapmask

    extrapmask = extrapmask | (~inside_mask)
    imglookup[extrapmask] = 0

    lookupmasks = np.zeros((verts.shape[0],), dtype=bool)
    if np.any(~extrapmask):
        lookupmasks[np.unique(imglookup[~extrapmask])] = True

    reverselookup = np.full((verts.shape[0],), -1, dtype=np.int64)
    pixflat = np.arange(h * w, dtype=np.int64)
    valid_pix = ~extrapmask.ravel()
    reverselookup[imglookup.ravel()[valid_pix]] = pixflat[valid_pix]

    missing = viewmask_padded & (reverselookup < 0)
    have = viewmask & (reverselookup >= 0)
    if np.any(missing) and np.any(have):
        if reversevert is not None:
            reversevert = np.asarray(reversevert, dtype=float)
            if reversevert.shape != verts.shape:
                raise ValueError("reversevert must have same shape as verts")
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

    pixels_per_mm = float(h / xyextent_mm[0]) if xyextent_mm[0] != 0 else float("inf")

    sx = (h - 1) / (xview[1] - xview[0])
    sy = (w - 1) / (yview[1] - yview[0])
    sz = (sx + sy) / 2.0
    tx = -xview[0]
    ty = -yview[0]
    mpix = np.array(
        [
            [sx, 0.0, 0.0, sx * tx + 1.0],
            [0.0, sy, 0.0, sy * ty + 1.0],
            [0.0, 0.0, sz, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    txpix = mpix @ txview

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
        "vertsN": int(verts.shape[0]),
        "TXview": txview,
        "xyextent": tuple(float(v) for v in xyextent),
        "xyextent_mm": xyextent_mm,
        "pixels_per_mm": pixels_per_mm,
        "xview": tuple(float(v) for v in xview),
        "yview": tuple(float(v) for v in yview),
        "zview": zview,
        "TXpix": txpix,
        "input2surf": None,
    }
