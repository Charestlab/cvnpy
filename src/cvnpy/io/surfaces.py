"""Surface loading and patch handling.

Parity target:
- /Users/iancharest/Documents/codex/cvncode/cvnreadsurface.m
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import nibabel.freesurfer.io as fsio
import numpy as np
from matplotlib.path import Path as MplPath

from ..types import Surface


def _resolve_surfdir(subject: str, surfdir: str | None = None) -> Path:
    if surfdir:
        p = Path(surfdir)
        if not p.is_dir():
            raise FileNotFoundError(f"surfdir does not exist: {p}")
        return p

    subjects_dir = os.environ.get("SUBJECTS_DIR")
    if not subjects_dir:
        raise ValueError("SUBJECTS_DIR is not set and no surfdir was provided")

    p = Path(subjects_dir) / subject / "surf"
    if not p.is_dir():
        raise FileNotFoundError(f"missing surf directory: {p}")
    return p


def _suffix_file(surfsuffix: str) -> str:
    return "" if surfsuffix == "orig" else surfsuffix


def _triangle_face_area(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    p1 = vertices[faces[:, 0], :]
    p2 = vertices[faces[:, 1], :]
    p3 = vertices[faces[:, 2], :]
    cross = np.cross(p2 - p1, p3 - p1)
    return 0.5 * np.linalg.norm(cross, axis=1)


def _patchedgeloop_longest(faces: np.ndarray) -> np.ndarray:
    """Port of MATLAB patchedgeloop(..., true)."""
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

    # return biggest loop segment separated by NaNs
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
    """Approximate MATLAB cleanflatpatch: keep only faces inside longest boundary loop."""
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


def _read_patch_binary(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as f:
        ver = np.fromfile(f, dtype=">i4", count=1)
        if ver.size != 1 or int(ver[0]) != -1:
            raise ValueError(f"Invalid binary patch version in {path}")
        npts = np.fromfile(f, dtype=">i4", count=1)
        if npts.size != 1:
            raise ValueError(f"Failed to read patch npts in {path}")
        n = int(npts[0])
        rec = np.fromfile(
            f,
            dtype=np.dtype(
                [("ind", ">i4"), ("x", ">f4"), ("y", ">f4"), ("z", ">f4")]
            ),
            count=n,
        )
        if rec.shape[0] != n:
            raise ValueError(f"Unexpected EOF in patch file {path}")

    ind = rec["ind"].astype(np.int64)
    vno = np.where(ind < 0, -ind - 1, ind - 1).astype(np.int64)
    edge = vno[ind < 0]
    return {
        "ind": ind,
        "vno": vno,
        "edge": edge,
        "x": rec["x"].astype(float),
        "y": rec["y"].astype(float),
        "z": rec["z"].astype(float),
    }


def _read_patch_ascii(path: Path) -> dict[str, np.ndarray]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if len(lines) < 3:
        raise ValueError(f"ASCII patch file too short: {path}")

    # Typical format:
    # line1 comment
    # line2 "<npts> <something>"
    # then pairs of lines:
    #   "<raw_index> vno=<...>"
    #   "<x> <y> <z>"
    if len(lines) >= 2 and lines[1].split():
        try:
            npts = int(lines[1].split()[0])
        except ValueError:
            npts = (len(lines) - 2) // 2
    else:
        npts = (len(lines) - 2) // 2

    inds: list[int] = []
    xyz: list[list[float]] = []
    cursor = 2
    while cursor + 1 < len(lines) and len(inds) < npts:
        raw_line = lines[cursor]
        xyz_line = lines[cursor + 1]
        cursor += 2

        raw_tok = raw_line.split()[0]
        inds.append(int(raw_tok))

        coords = [float(tok) for tok in xyz_line.split()[:3]]
        if len(coords) != 3:
            raise ValueError(f"Invalid XYZ line in {path}: {xyz_line}")
        xyz.append(coords)

    ind = np.asarray(inds, dtype=np.int64)
    xyz_arr = np.asarray(xyz, dtype=float)
    vno = np.where(ind < 0, -ind - 1, ind - 1).astype(np.int64)
    edge = vno[ind < 0]
    return {
        "ind": ind,
        "vno": vno,
        "edge": edge,
        "x": xyz_arr[:, 0],
        "y": xyz_arr[:, 1],
        "z": xyz_arr[:, 2],
    }


def _read_patch(path: Path) -> dict[str, np.ndarray]:
    if path.suffix.lower() == ".asc":
        return _read_patch_ascii(path)
    try:
        return _read_patch_binary(path)
    except Exception:
        # Some installs only provide ascii patch export; try sibling *.asc.
        alt = Path(str(path) + ".asc")
        if alt.exists():
            return _read_patch_ascii(alt)
        raise


def _read_one_surface(
    surfdir: Path,
    hemi: str,
    surftype: str,
    surfsuffix: str,
    justcount: bool,
    noscale: bool,
) -> Surface | int:
    suffix_file = _suffix_file(surfsuffix)
    is_patch = ".patch." in surftype

    patchfile: Path | None = None
    patchfile_asc: Path | None = None
    if is_patch:
        suffix2 = suffix_file
        if suffix2:
            suffix2 = f"{suffix2}."
        patchfile = surfdir / f"{hemi}.{suffix2}{surftype}"
        patchfile_asc = Path(str(patchfile) + ".asc")
        if not patchfile.exists() and not patchfile_asc.exists():
            raise FileNotFoundError(f"patch file not found: {patchfile}")
        surffile = surfdir / f"{hemi}.white{suffix_file}"
    else:
        surffile = surfdir / f"{hemi}.{surftype}{suffix_file}"

    if not surffile.exists():
        if is_patch and patchfile is not None and patchfile.exists():
            # Fallback for datasets where *.patch.* is already stored as a regular mesh file.
            try:
                if justcount:
                    verts, _faces = fsio.read_geometry(str(patchfile))
                    return int(verts.shape[0])
                verts, faces = fsio.read_geometry(str(patchfile))
                return Surface(
                    vertices=np.asarray(verts, dtype=float),
                    faces=np.asarray(faces, dtype=np.int64),
                    patchmask=None,
                )
            except Exception:
                pass
        raise FileNotFoundError(f"surface file not found: {surffile}")

    try:
        if justcount:
            verts, _faces = fsio.read_geometry(str(surffile))
            return int(verts.shape[0])

        verts, faces = fsio.read_geometry(str(surffile))
    except Exception as exc:
        if is_patch:
            raise NotImplementedError(
                "Failed to read base white surface for patch rendering."
            ) from exc
        raise

    # MATLAB uses 1-based faces; nibabel returns 0-based. Python Surface stays 0-based.
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    patchmask: np.ndarray | None = None
    if patchfile is not None:
        origverts = verts.copy()
        p = _read_patch(patchfile)

        patchmask = np.zeros((verts.shape[0],), dtype=bool)
        valid = (p["vno"] >= 0) & (p["vno"] < verts.shape[0])
        pv = p["vno"][valid]
        patchmask[pv] = True

        faces = faces[np.all(patchmask[faces], axis=1), :]
        verts[pv, :] = np.c_[p["x"][valid], p["y"][valid], p["z"][valid]]

        if ".flat." in surftype:
            # MATLAB-like cleanup pass.
            faces, patchmask = _clean_flat_patch_polygon(faces, verts, max_iter=10)

        if not noscale and np.any(patchmask) and faces.shape[0] > 0:
            a_orig = float(np.sum(_triangle_face_area(origverts, faces)))
            a_flat = float(np.sum(_triangle_face_area(verts, faces)))
            if a_orig > 0 and a_flat > 0:
                scale = float(np.sqrt(a_orig / a_flat))
                flatverts = verts[patchmask, :]
                flatmean = np.mean(flatverts, axis=0, keepdims=True)
                verts[patchmask, :] = (flatverts - flatmean) * scale + flatmean

    return Surface(vertices=verts, faces=faces, patchmask=patchmask)


def read_surface(
    subject: str,
    hemi: str | list[str],
    surftype: str,
    surfsuffix: str = "orig",
    **kwargs: Any,
) -> Surface | list[Surface] | int | tuple[int, int]:
    """Read a FreeSurfer surface or return vertex counts.

    Supported kwargs:
    - justcount: bool (default False)
    - surfdir: override subjects/<subject>/surf
    - noscale: accepted for API compatibility (currently unused)
    """
    justcount = bool(kwargs.get("justcount", False))
    surfdir_arg = kwargs.get("surfdir")
    noscale = bool(kwargs.get("noscale", False))

    surfdir = _resolve_surfdir(subject, surfdir_arg if isinstance(surfdir_arg, str) else None)

    hemis = [hemi] if isinstance(hemi, str) else list(hemi)
    if not hemis:
        hemis = ["lh", "rh"]

    results = [
        _read_one_surface(
            surfdir,
            h.lower(),
            surftype,
            surfsuffix,
            justcount,
            noscale,
        )
        for h in hemis
    ]

    if len(results) == 1:
        return results[0]

    if justcount:
        return int(results[0]), int(results[1])
    return results
