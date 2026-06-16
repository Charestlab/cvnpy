"""Inflated/non-sphere lookup generation.

Parity target:
- /Users/iancharest/Documents/codex/cvncode/kjutils/spherelookup_generate_notsphere.m
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree


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


def _barycentric(
    px: np.ndarray,
    py: np.ndarray,
    tri2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x1, y1 = tri2d[0]
    x2, y2 = tri2d[1]
    x3, y3 = tri2d[2]
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(denom) < 1e-12:
        z = np.zeros_like(px, dtype=float)
        return z, z, z
    w1 = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denom
    w2 = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denom
    w3 = 1.0 - w1 - w2
    return w1, w2, w3


def generate_surface_lookup(
    vertices: np.ndarray,
    faces: np.ndarray,
    azimuth: float,
    elevation: float,
    tilt: float,
    imgheight: int | None = None,
    **kwargs: Any,
) -> dict:
    """Generate lookup + shading for non-spherical surfaces.

    Practical implementation with z-buffer rasterization:
    - projects mesh to view plane
    - rasterizes visible faces
    - builds imglookup, extrapmask, reverselookup
    - computes simple headlight shading map
    """
    _ = kwargs
    verts = np.asarray(vertices, dtype=float)
    tri = np.asarray(faces, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("vertices must be Nx3")
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("faces must be Mx3")

    h = int(imgheight if imgheight is not None else 500)
    w = h

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

    az_tilt = np.pi / 2
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

    basis = np.c_[v1, v2, v3]
    m = np.eye(4, dtype=float)
    m[:3, :3] = np.linalg.inv(basis)
    mtilt = _rotation_about_z(view_tilt + az_tilt)
    txview = mtilt @ m

    viewvert = _apply_affine(txview, verts)

    vmin = np.min(viewvert[:, :2], axis=0)
    vmax = np.max(viewvert[:, :2], axis=0)
    vmid = (vmin + vmax) / 2.0
    vsize = vmax - vmin
    side = max(vsize[0], vsize[1]) * 1.1
    if side == 0:
        side = 1.0
    xview = (vmid[0] - side / 2.0, vmid[0] + side / 2.0)
    yview = (vmid[1] - side / 2.0, vmid[1] + side / 2.0)
    zview = (-np.inf, np.inf)

    sx = (w - 1) / (xview[1] - xview[0])
    sy = (h - 1) / (yview[1] - yview[0])
    px = (viewvert[:, 0] - xview[0]) * sx
    py = (viewvert[:, 1] - yview[0]) * sy
    pz = viewvert[:, 2]

    # flip y so row 0 is top
    py = (h - 1) - py

    zbuf = np.full((h, w), -np.inf, dtype=float)
    facebuf = np.full((h, w), -1, dtype=np.int64)
    imglookup = np.zeros((h, w), dtype=np.int64)
    shading = np.zeros((h, w), dtype=float)

    # Precompute face normals in view space for shading.
    fv = viewvert[tri]
    fn = np.cross(fv[:, 1, :] - fv[:, 0, :], fv[:, 2, :] - fv[:, 0, :])
    fn_norm = np.linalg.norm(fn, axis=1)
    valid_norm = fn_norm > 0
    fn[valid_norm] = fn[valid_norm] / fn_norm[valid_norm, None]
    face_light = np.clip(fn[:, 2], 0.0, 1.0)
    face_light = 0.35 + 0.65 * face_light

    for fi, f in enumerate(tri):
        tri_px = np.array(
            [[px[f[0]], py[f[0]]], [px[f[1]], py[f[1]]], [px[f[2]], py[f[2]]]],
            dtype=float,
        )
        xmin = max(0, int(np.floor(np.min(tri_px[:, 0]))))
        xmax = min(w - 1, int(np.ceil(np.max(tri_px[:, 0]))))
        ymin = max(0, int(np.floor(np.min(tri_px[:, 1]))))
        ymax = min(h - 1, int(np.ceil(np.max(tri_px[:, 1]))))
        if xmin > xmax or ymin > ymax:
            continue

        gx, gy = np.meshgrid(
            np.arange(xmin, xmax + 1, dtype=float),
            np.arange(ymin, ymax + 1, dtype=float),
        )
        w1, w2, w3 = _barycentric(gx, gy, tri_px)
        inside = (w1 >= 0) & (w2 >= 0) & (w3 >= 0)
        if not np.any(inside):
            continue

        z = w1 * pz[f[0]] + w2 * pz[f[1]] + w3 * pz[f[2]]
        zz = zbuf[ymin : ymax + 1, xmin : xmax + 1]
        update = inside & (z > zz)
        if not np.any(update):
            continue

        # pick nearest triangle vertex per pixel by max barycentric weight.
        bary = np.stack([w1, w2, w3], axis=-1)
        vidx_local = np.argmax(bary, axis=-1)
        fverts = np.array([f[0], f[1], f[2]], dtype=np.int64)
        chosen_v = fverts[vidx_local]

        sub_face = facebuf[ymin : ymax + 1, xmin : xmax + 1]
        sub_lookup = imglookup[ymin : ymax + 1, xmin : xmax + 1]
        sub_sh = shading[ymin : ymax + 1, xmin : xmax + 1]

        zz[update] = z[update]
        sub_face[update] = fi
        sub_lookup[update] = chosen_v[update]
        sub_sh[update] = face_light[fi]

    imgmask = facebuf >= 0
    extrapmask = ~imgmask
    imglookup[extrapmask] = 0

    # Reverse lookup: first use direct visible assignments.
    verts_n = verts.shape[0]
    reverselookup = np.full((verts_n,), -1, dtype=np.int64)
    pixflat = np.arange(h * w, dtype=np.int64)
    flat_lookup = imglookup.ravel()
    flat_mask = imgmask.ravel()
    reverselookup[flat_lookup[flat_mask]] = pixflat[flat_mask]

    # Fill missing vertices with nearest visible vertex in view-space XY.
    missing = reverselookup < 0
    have = reverselookup >= 0
    if np.any(missing) and np.any(have):
        tree = cKDTree(viewvert[have, :2])
        _dist, nn = tree.query(viewvert[missing, :2], k=1)
        have_idx = np.flatnonzero(have)
        missing_idx = np.flatnonzero(missing)
        reverselookup[missing_idx] = reverselookup[have_idx[nn]]

    # Compute TXpix similar to other generators (row/col conventions).
    tx = -xview[0]
    ty = -yview[0]
    sxp = (w - 1) / (xview[1] - xview[0])
    syp = (h - 1) / (yview[1] - yview[0])
    szp = (sxp + syp) / 2.0
    mpix = np.array(
        [
            [sxp, 0.0, 0.0, sxp * tx + 1.0],
            [0.0, -syp, 0.0, -syp * ty + h],
            [0.0, 0.0, szp, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    mpix = mpix[[1, 0, 2, 3], :]
    txpix = mpix @ txview

    visible_vertices = np.unique(imglookup[imgmask]) if np.any(imgmask) else np.array([], dtype=np.int64)
    vertmasks = np.zeros((verts_n,), dtype=bool)
    lookupmasks = np.zeros((verts_n,), dtype=bool)
    vertmasks[visible_vertices] = True
    lookupmasks[visible_vertices] = True

    return {
        "imglookup": imglookup,
        "vertmasks": vertmasks,
        "lookupmasks": lookupmasks,
        "reverselookup": reverselookup,
        "extrapmask": extrapmask,
        "is_extrapolated": bool(np.any(extrapmask)),
        "azimuth": float(azimuth),
        "elevation": float(elevation),
        "tilt": float(tilt),
        "imgN": w,
        "imgsize": (h, w),
        "vertsN": int(verts_n),
        "xyextent": (1.0, 1.0),
        "xview": tuple(float(v) for v in xview),
        "yview": tuple(float(v) for v in yview),
        "zview": zview,
        "TXview": txview,
        "TXpix": txpix,
        "shading": np.clip(shading, 0.0, 1.0),
        "input2surf": None,
    }
