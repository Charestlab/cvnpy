"""Image/vertex mapping helpers.

Parity targets:
- /Users/iancharest/Documents/codex/cvncode/kjutils/spherelookup_vert2image.m
- /Users/iancharest/Documents/codex/cvncode/kjutils/spherelookup_image2vert.m
- /Users/iancharest/Documents/codex/cvncode/kjutils/spherelookup_imagexy2vertidx.m
- /Users/iancharest/Documents/codex/cvncode/kjutils/spherelookup_vertidx2imagexy.m
"""

from __future__ import annotations

import numpy as np


def vert_to_image(vertvalues: np.ndarray, lookup: dict, badval: float = 0.0) -> np.ndarray:
    """Map vertex values to image space.

    Uses lookup['imglookup'] as nearest-neighbor vertex indices.
    """
    vals = np.asarray(vertvalues)
    was_1d = vals.ndim == 1
    if was_1d:
        vals = vals[:, None]

    input2surf = lookup.get("input2surf")
    input_n = lookup.get("inputN")
    if input2surf is not None and input_n is not None and vals.shape[0] == int(input_n):
        fullvals = vals[np.asarray(input2surf, dtype=np.int64), :]
    else:
        fullvals = vals
    if not np.issubdtype(fullvals.dtype, np.floating):
        fullvals = fullvals.astype(float, copy=False)

    imglookup = np.asarray(lookup["imglookup"], dtype=np.int64)
    bary_face = lookup.get("bary_face")
    bary_faces = lookup.get("bary_faces")
    bary_weights = lookup.get("bary_weights")
    use_barycentric = bool(lookup.get("use_barycentric", True))
    use_bary = (
        use_barycentric
        and vals.shape[1] == 1
        and bary_face is not None
        and bary_faces is not None
        and bary_weights is not None
    )

    if use_bary:
        face_idx = np.asarray(bary_face, dtype=np.int64)
        tri = np.asarray(bary_faces, dtype=np.int64)
        wts = np.asarray(bary_weights, dtype=float)
        nn_img = fullvals[imglookup, :]
        img = np.full(
            (imglookup.shape[0], imglookup.shape[1], fullvals.shape[1]),
            float(badval),
            dtype=fullvals.dtype if np.issubdtype(fullvals.dtype, np.floating) else float,
        )
        valid = face_idx >= 0
        if np.any(valid):
            f = tri[face_idx[valid]]
            wt = wts[valid]
            v0 = fullvals[f[:, 0], :]
            v1 = fullvals[f[:, 1], :]
            v2 = fullvals[f[:, 2], :]
            pix = v0 * wt[:, [0]] + v1 * wt[:, [1]] + v2 * wt[:, [2]]
            img[valid, :] = pix
        if np.any(~valid):
            img[~valid, :] = nn_img[~valid, :]
    else:
        img = fullvals[imglookup, :]

    extrapmask = np.asarray(lookup.get("extrapmask", np.zeros_like(imglookup, dtype=bool)))
    if bool(lookup.get("is_extrapolated", False)):
        img[extrapmask, :] = badval

    if was_1d:
        return img[:, :, 0]
    return img


def image_to_vert(img: np.ndarray, lookup: dict, badval: float = 0.0) -> np.ndarray:
    """Map image values back to vertex space using reverselookup."""
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError("image_to_vert currently supports 2D img only")

    imglookup = np.asarray(lookup["imglookup"], dtype=np.int64)
    if img.shape != imglookup.shape:
        raise ValueError("img shape must match lookup['imglookup'] shape")

    verts_n = int(lookup["vertsN"])
    vertvalues = np.full((verts_n,), badval, dtype=float)

    flat_img = img.ravel()
    flat_lookup = imglookup.ravel()
    vertvalues[flat_lookup] = flat_img

    reverselookup = np.asarray(lookup["reverselookup"], dtype=np.int64)
    valid = reverselookup >= 0
    vertvalues[valid] = flat_img[reverselookup[valid]]

    input2surf = lookup.get("input2surf")
    input_n = lookup.get("inputN")
    if input2surf is not None and input_n is not None:
        out = np.full((int(input_n),), badval, dtype=float)
        out[np.asarray(input2surf, dtype=np.int64)] = vertvalues
        return out

    return vertvalues


def imagexy_to_vertidx(imgxy: np.ndarray, lookup: dict) -> np.ndarray:
    """Map x,y image coordinates to vertex indices.

    Accepts coordinates in MATLAB-style 1-based pixel space.
    """
    pts = np.asarray(imgxy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("imgxy must be Nx2")

    imglookup = np.asarray(lookup["imglookup"], dtype=np.int64)
    h, w = imglookup.shape

    x = np.clip(np.rint(pts[:, 0]).astype(np.int64), 1, w) - 1
    y = np.clip(np.rint(pts[:, 1]).astype(np.int64), 1, h) - 1

    out = imglookup[y, x]
    input2surf = lookup.get("input2surf")
    if input2surf is not None:
        out = np.asarray(input2surf, dtype=np.int64)[out]
    return out


def vertidx_to_imagexy(vertidx: np.ndarray, lookup: dict) -> np.ndarray:
    """Map vertex indices to x,y image coordinates in 1-based space."""
    idx = np.asarray(vertidx, dtype=np.int64)

    input2surf = lookup.get("input2surf")
    if input2surf is not None:
        # Build inverse mapping from surface index -> input index.
        inv = np.full((int(lookup["vertsN"]),), -1, dtype=np.int64)
        input2surf_arr = np.asarray(input2surf, dtype=np.int64)
        inv[input2surf_arr] = np.arange(input2surf_arr.shape[0], dtype=np.int64)
        idx = inv[idx]

    rev = np.asarray(lookup["reverselookup"], dtype=np.int64)
    pix = rev[idx]
    h, w = np.asarray(lookup["imglookup"]).shape

    y = (pix // w) + 1
    x = (pix % w) + 1
    return np.c_[x, y]
