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

    Uses lookup['imglookup'] as nearest-neighbor vertex indices, matching
    Kendrick Kay's MATLAB spherelookup_vert2image.m.
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
    _h, w = np.asarray(lookup["imglookup"]).shape

    y = (pix // w) + 1
    x = (pix % w) + 1
    xy = np.c_[x, y].astype(float)
    xy[pix < 0] = np.nan
    return xy
