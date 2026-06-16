"""Flat-patch lookup generation.

Parity target:
- /Users/iancharest/Documents/codex/cvncode/kjutils/spherelookup_generate_flat.m
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.path import Path as MplPath
from scipy import ndimage as ndi
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import cKDTree
from skimage.draw import polygon


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


def _fill_small_extrap_holes(
    extrapmask: np.ndarray,
    imglookup: np.ndarray,
    max_hole_px: int,
    bary_face: np.ndarray | None = None,
    bary_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    bad = np.asarray(extrapmask, dtype=bool)
    if not np.any(bad):
        return extrapmask, imglookup, bary_face, bary_weights

    # Robust/topological fill: close all enclosed invalid islands.
    valid = ~bad
    enclosed = ndi.binary_fill_holes(valid) & bad
    if not np.any(enclosed):
        return extrapmask, imglookup, bary_face, bary_weights

    if max_hole_px > 0:
        labels, nlab = ndi.label(enclosed)
        counts = np.bincount(labels.ravel(), minlength=nlab + 1)
        keep = np.zeros((nlab + 1,), dtype=bool)
        keep[1:] = counts[1:] <= max_hole_px
        fillmask = keep[labels] & enclosed
    else:
        fillmask = enclosed

    validmask = ~bad
    vy, vx = np.nonzero(validmask)
    hy, hx = np.nonzero(fillmask)
    if hy.size == 0 or vy.size == 0:
        return extrapmask, imglookup, bary_face, bary_weights

    tree = cKDTree(np.c_[vy.astype(float), vx.astype(float)])
    _dist, nn = tree.query(np.c_[hy.astype(float), hx.astype(float)], k=1)
    srcy = vy[nn]
    srcx = vx[nn]

    imglookup[hy, hx] = imglookup[srcy, srcx]
    extrapmask[hy, hx] = False
    if bary_face is not None:
        bary_face[hy, hx] = bary_face[srcy, srcx]
    if bary_weights is not None:
        bary_weights[hy, hx, :] = bary_weights[srcy, srcx, :]

    return extrapmask, imglookup, bary_face, bary_weights


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
    apply_support_mask = bool(kwargs.get("apply_support_mask", True))
    support_ratio = float(kwargs.get("support_ratio", 2.5))
    prune_artifact_faces = bool(kwargs.get("prune_artifact_faces", False))
    prune_quantile = float(kwargs.get("prune_quantile", 99.0))
    prune_factor = float(kwargs.get("prune_factor", 1.5))
    mask_high_area_pixels = bool(kwargs.get("mask_high_area_pixels", True))
    filter_sparse_vertices = bool(kwargs.get("filter_sparse_vertices", False))
    sparse_vertex_quantile = float(kwargs.get("sparse_vertex_quantile", 95.0))
    sparse_vertex_factor = float(kwargs.get("sparse_vertex_factor", 1.2))
    margin_frac = float(kwargs.get("margin_frac", 0.04))
    fill_small_holes_px = int(kwargs.get("fill_small_holes_px", 0))

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

    # Optional pruning of outlier faces that create large fan-like artifacts.
    if prune_artifact_faces and faces_valid.shape[0] > 0:
        fv = viewvert[faces_valid]
        l12 = np.linalg.norm(fv[:, 0, :] - fv[:, 1, :], axis=1)
        l23 = np.linalg.norm(fv[:, 1, :] - fv[:, 2, :], axis=1)
        l31 = np.linalg.norm(fv[:, 2, :] - fv[:, 0, :], axis=1)
        max_edge = np.maximum(l12, np.maximum(l23, l31))
        area2 = np.abs(
            (fv[:, 1, 0] - fv[:, 0, 0]) * (fv[:, 2, 1] - fv[:, 0, 1])
            - (fv[:, 1, 1] - fv[:, 0, 1]) * (fv[:, 2, 0] - fv[:, 0, 0])
        ) / 2.0

        q = np.clip(prune_quantile, 80.0, 100.0)
        e_thr = np.percentile(max_edge, q) * max(prune_factor, 1.0)
        a_thr = np.percentile(area2, q) * max(prune_factor, 1.0)
        keep = (max_edge <= e_thr) & (area2 <= a_thr)
        # Keep pruning conservative: if too much gets removed, skip pruning.
        if np.mean(keep) >= 0.5:
            faces_valid = faces_valid[keep]

    if faces_valid.shape[0] > 0:
        # Re-clean again after optional pruning.
        faces_valid, _ = _clean_flat_patch_polygon(faces_valid, viewvert, max_iter=10)
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
    if filter_sparse_vertices and np.count_nonzero(flatmask_interp) > 10:
        pts_all = viewvert[flatmask_interp, :2]
        tree_v = cKDTree(pts_all)
        k = min(7, pts_all.shape[0])
        d, _ = tree_v.query(pts_all, k=k)
        # use distance to kth neighbor as local density proxy.
        dk = d[:, -1] if d.ndim > 1 else d
        q = np.clip(sparse_vertex_quantile, 50.0, 100.0)
        thr = np.percentile(dk, q) * max(sparse_vertex_factor, 1.0)
        reliable_local = dk <= thr
        reliable_vertices = np.zeros_like(flatmask_interp)
        reliable_vertices[np.where(flatmask_interp)[0][reliable_local]] = True
        interp_mask = interp_mask & reliable_vertices

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

    # Restrict valid pixels to areas actually covered by patch triangles.
    # This avoids nearest-neighbor "bleeding" outside patch boundaries.
    sx = (h - 1) / (xview[1] - xview[0])
    sy = (w - 1) / (yview[1] - yview[0])
    px = (viewvert[:, 0] - xview[0]) * sx
    py = (viewvert[:, 1] - yview[0]) * sy
    py = (w - 1) - py  # row index coordinates

    inside_mask = np.zeros((h, w), dtype=bool)
    face_map = np.full((h, w), -1, dtype=np.int64)
    bary_weights = np.zeros((h, w, 3), dtype=np.float32)
    for fi_local, f in enumerate(faces_valid):
        tri_px = np.array(
            [[px[f[0]], py[f[0]]], [px[f[1]], py[f[1]]], [px[f[2]], py[f[2]]]],
            dtype=float,
        )
        rr, cc = polygon(
            tri_px[:, 1],
            tri_px[:, 0],
            shape=(h, w),
        )
        if rr.size == 0:
            continue
        w1, w2, w3 = _barycentric(cc.astype(float), rr.astype(float), tri_px)
        valid = (w1 >= -1e-5) & (w2 >= -1e-5) & (w3 >= -1e-5)
        if not np.any(valid):
            continue
        rr = rr[valid]
        cc = cc[valid]
        w1 = w1[valid]
        w2 = w2[valid]
        w3 = w3[valid]
        # Keep first assignment to avoid boundary overwrite ambiguity.
        unassigned = face_map[rr, cc] < 0
        if not np.any(unassigned):
            continue
        rr = rr[unassigned]
        cc = cc[unassigned]
        w1 = w1[unassigned]
        w2 = w2[unassigned]
        w3 = w3[unassigned]
        inside_mask[rr, cc] = True
        face_map[rr, cc] = fi_local
        bary_weights[rr, cc, 0] = w1.astype(np.float32)
        bary_weights[rr, cc, 1] = w2.astype(np.float32)
        bary_weights[rr, cc, 2] = w3.astype(np.float32)

    edge_loop = _patchedgeloop_longest(faces_valid)
    if edge_loop.size >= 3:
        poly = viewvert[edge_loop, :2]
        inside_poly = MplPath(poly, closed=True).contains_points(
            np.c_[imgx.ravel(), imgy.ravel()], radius=1e-9
        ).reshape((h, w))
        inside_mask = inside_poly
    else:
        inside_mask = ndi.binary_fill_holes(inside_mask)

    extrapmask = extrapmask | (~inside_mask)
    imglookup[extrapmask] = 0

    if mask_high_area_pixels and faces_valid.shape[0] > 0:
        fv = viewvert[faces_valid]
        area2 = np.abs(
            (fv[:, 1, 0] - fv[:, 0, 0]) * (fv[:, 2, 1] - fv[:, 0, 1])
            - (fv[:, 1, 1] - fv[:, 0, 1]) * (fv[:, 2, 0] - fv[:, 0, 0])
        ) / 2.0
        q = np.clip(prune_quantile, 80.0, 100.0)
        area_thr = np.percentile(area2, q) * max(prune_factor, 1.0)
        if np.isfinite(area_thr) and area_thr > 0:
            valid_face_pix = face_map >= 0
            bad_pix = np.zeros_like(extrapmask)
            bad_pix[valid_face_pix] = (
                area2[face_map[valid_face_pix]] > area_thr
            )
            extrapmask = extrapmask | bad_pix
            imglookup[extrapmask] = 0

    if apply_support_mask:
        # Mark pixels far from any patch vertex as extrapolated.
        # This suppresses large fan artifacts in sparse boundary regions.
        vert_px = np.c_[px[flatmask], py[flatmask]]
        if vert_px.shape[0] > 0:
            tree = cKDTree(vert_px)
            pix_y, pix_x = np.nonzero(inside_mask)
            if pix_x.size > 0:
                q = np.c_[pix_x.astype(float), pix_y.astype(float)]
                dist, _ = tree.query(q, k=1)
                med = float(np.median(dist))
                if med > 0:
                    far = dist > (support_ratio * med)
                    if np.any(far):
                        fy = pix_y[far]
                        fx = pix_x[far]
                        extrapmask[fy, fx] = True
                        imglookup[fy, fx] = 0

    extrapmask, imglookup, face_map, bary_weights = _fill_small_extrap_holes(
        extrapmask,
        imglookup,
        max_hole_px=fill_small_holes_px,
        bary_face=face_map,
        bary_weights=bary_weights,
    )

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
        "bary_face": face_map,
        "bary_faces": faces_valid.astype(np.int64),
        "bary_weights": bary_weights,
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
