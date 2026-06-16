"""Main lookup/rendering engine.

Parity target:
- /Users/iancharest/Documents/codex/cvncode/cvnlookupimages.m
"""

from __future__ import annotations

import os
import pickle
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation

from ..io.metrics import read_surface_metric
from ..io.surfaces import read_surface
from ..lookup.generate_flat import generate_flat_lookup
from ..lookup.generate_surface import generate_surface_lookup
from ..lookup.generate_sphere import generate_sphere_lookup
from ..lookup.map import vert_to_image
from ..render.colormap import mat_to_rgb
from ..render.overlay import apply_overlays
from ..view.viewpoints import lookup_viewpoint

LOOKUP_CACHE_VERSION = 9


def _is_flat_surftype(surftype: str) -> bool:
    return ".flat.patch." in surftype.lower()


def _is_sphere_surftype(surftype: str) -> bool:
    return surftype.lower().startswith("sphere")


def _default_options() -> dict[str, Any]:
    return {
        "surftype": "sphere",
        "surfsuffix": "orig",
        "xyextent": (1.0, 1.0),
        "imageres": 1000,
        "clim": (-np.inf, np.inf),
        "cmap": "cmapsign4",
        "rgbnan": 1.0,
        "surfshading": True,
        "threshold": None,
        "absthreshold": None,
        "overlayrange": None,
        "overlayalpha": None,
        "inclusive": False,
        "background": "curv",
        "bg_cmap": "gray",
        "bg_clim": (-1.0, 2.0),
        "roimask": None,
        "roicolor": [0.0, 0.0, 0.0],
        "roiwidth": 1.0,
        "reset": False,
        "savelookup": True,
        "cache_dir": None,
        "flat_apply_support_mask": False,
        "flat_support_ratio": 2.5,
        "trim_extrapolation_px": 0,
        "prune_artifact_faces": False,
        "prune_quantile": 99.0,
        "prune_factor": 1.5,
        "flat_mask_high_area_pixels": False,
        "flat_filter_sparse_vertices": False,
        "flat_sparse_vertex_quantile": 95.0,
        "flat_sparse_vertex_factor": 1.2,
        "flat_edge_trim_steps": 0,
        "flat_margin_frac": 0.04,
        "flat_fill_small_holes_px": 0,
    }


def _merge_options(kwargs: dict[str, Any]) -> dict[str, Any]:
    out = _default_options()
    out.update(kwargs)
    if isinstance(out["xyextent"], (int, float)):
        out["xyextent"] = (float(out["xyextent"]), float(out["xyextent"]))
    return out


def _as_view_triplet(subject: str, hemi: str, view_in: Any, surftype: str) -> list[float]:
    if isinstance(view_in, str):
        viewpt, _flip, _viewhemi = lookup_viewpoint(subject, hemi, view_in, surftype)
        return [float(viewpt[0]), float(viewpt[1]), float(viewpt[2])]
    arr = np.asarray(view_in, dtype=float).ravel()
    if arr.size != 3:
        raise ValueError("view_az_el_tilt must be a view name or 3-element view triplet")
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _safe_tag(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _cache_dir_for_subject(subject: str, options: dict[str, Any]) -> Path:
    cache_dir_opt = options.get("cache_dir")
    if cache_dir_opt:
        p = Path(str(cache_dir_opt))
        p.mkdir(parents=True, exist_ok=True)
        return p

    surfdir_opt = options.get("surfdir")
    if isinstance(surfdir_opt, str) and surfdir_opt:
        p = Path(surfdir_opt) / "imglookup"
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p
        except PermissionError:
            fallback = (
                Path(tempfile.gettempdir()) / "cvnpy_lookup_cache" / _safe_tag(subject)
            )
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    subjects_dir = os.environ.get("SUBJECTS_DIR")
    if not subjects_dir:
        raise ValueError(
            "SUBJECTS_DIR is not set and no cache_dir was provided for lookup caching"
        )
    p = Path(subjects_dir) / subject / "surf" / "imglookup"
    try:
        p.mkdir(parents=True, exist_ok=True)
        return p
    except PermissionError:
        # Common for system-wide FreeSurfer installs under /Applications.
        fallback = (
            Path(tempfile.gettempdir()) / "cvnpy_lookup_cache" / _safe_tag(subject)
        )
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _cache_file_path(
    subject: str,
    hemi: str,
    view: list[float],
    options: dict[str, Any],
) -> Path:
    surftype = _safe_tag(str(options["surftype"]))
    surfsuffix = _safe_tag(str(options["surfsuffix"]))
    xy = tuple(options["xyextent"])
    res = options["imageres"]
    if isinstance(res, (list, tuple)):
        res_tag = f"{int(res[0])}x{int(res[1])}"
    else:
        res_tag = str(int(res))
    tag = (
        f"{surftype}_lookupv{LOOKUP_CACHE_VERSION}_{hemi}_"
        f"az{int(round(view[0]))}_el{int(round(view[1]))}_tilt{int(round(view[2]))}_"
        f"xy{xy[0]:.3f}x{xy[1]:.3f}_res{res_tag}_{surfsuffix}.pkl"
    )
    return _cache_dir_for_subject(subject, options) / _safe_tag(tag)


def _try_load_cached_lookup(
    subject: str,
    hemi: str,
    view: list[float],
    options: dict[str, Any],
) -> dict[str, Any] | None:
    if bool(options.get("reset", False)):
        return None
    path = _cache_file_path(subject, hemi, view, options)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, dict):
            return None
        return obj
    except Exception:
        return None


def _try_save_cached_lookup(
    subject: str,
    hemi: str,
    view: list[float],
    options: dict[str, Any],
    lookup: dict[str, Any],
) -> None:
    if not bool(options.get("savelookup", True)):
        return
    try:
        path = _cache_file_path(subject, hemi, view, options)
        with path.open("wb") as f:
            pickle.dump(lookup, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        # cache write failures should not block rendering
        return


def _generate_lookup(
    subject: str,
    hemi: str,
    view: list[float],
    options: dict[str, Any],
) -> dict:
    surftype = str(options["surftype"])
    surfsuffix = str(options["surfsuffix"])
    imageres = options["imageres"]
    xyextent = tuple(options["xyextent"])

    cached = _try_load_cached_lookup(subject, hemi, view, options)
    if cached is not None:
        lookup = cached
        lookup["hemi"] = hemi
        lookup["surftype"] = surftype
        lookup["surfsuffix"] = surfsuffix
        return lookup

    surfdir_opt = options.get("surfdir")
    surf_kwargs: dict[str, Any] = {}
    if isinstance(surfdir_opt, str) and surfdir_opt:
        surf_kwargs["surfdir"] = surfdir_opt
    if _is_flat_surftype(surftype):
        surf_kwargs["flat_edge_trim_steps"] = int(
            options.get("flat_edge_trim_steps", 0)
        )

    surf = read_surface(subject, hemi, surftype, surfsuffix, **surf_kwargs)

    if _is_sphere_surftype(surftype):
        lookup = generate_sphere_lookup(
            surf.vertices,
            view[0],
            view[1],
            view[2],
            xyextent,
            imageres,
        )
    elif _is_flat_surftype(surftype):
        lookup = generate_flat_lookup(
            surf.vertices,
            surf.faces,
            view[0],
            view[1],
            view[2],
            xyextent,
            imageres,
            apply_support_mask=bool(options.get("flat_apply_support_mask", False)),
            support_ratio=float(options.get("flat_support_ratio", 2.5)),
            prune_artifact_faces=bool(options.get("prune_artifact_faces", False)),
            prune_quantile=float(options.get("prune_quantile", 99.0)),
            prune_factor=float(options.get("prune_factor", 1.5)),
            mask_high_area_pixels=bool(
                options.get("flat_mask_high_area_pixels", False)
            ),
            filter_sparse_vertices=bool(
                options.get("flat_filter_sparse_vertices", False)
            ),
            sparse_vertex_quantile=float(
                options.get("flat_sparse_vertex_quantile", 95.0)
            ),
            sparse_vertex_factor=float(
                options.get("flat_sparse_vertex_factor", 1.2)
            ),
            margin_frac=float(options.get("flat_margin_frac", 0.04)),
            fill_small_holes_px=int(options.get("flat_fill_small_holes_px", 0)),
        )
    else:
        lookup = generate_surface_lookup(
            surf.vertices,
            surf.faces,
            view[0],
            view[1],
            view[2],
            imgheight=int(imageres) if isinstance(imageres, int) else int(imageres[0]),
        )

    lookup["hemi"] = hemi
    lookup["surftype"] = surftype
    lookup["surfsuffix"] = surfsuffix
    _try_save_cached_lookup(subject, hemi, view, options, lookup)
    return lookup


def _lookup_one_hemi(
    subject: str,
    vals: np.ndarray,
    hemi: str,
    view_az_el_tilt: Any,
    lookup: dict | None,
    options: dict[str, Any],
) -> tuple[np.ndarray, dict, np.ndarray]:
    view = _as_view_triplet(subject, hemi, view_az_el_tilt, str(options["surftype"]))
    lk = lookup if lookup is not None else _generate_lookup(subject, hemi, view, options)

    local_options = dict(options)
    alpha_opt = local_options.get("overlayalpha")
    if alpha_opt is not None:
        alpha_arr = np.asarray(alpha_opt)
        if alpha_arr.ndim == 1 and alpha_arr.shape[0] == vals.shape[0]:
            local_options["overlayalpha"] = vert_to_image(alpha_arr, lk, badval=0.0)

    mapped = vert_to_image(vals, lk, badval=np.nan)
    trim_px = int(local_options.get("trim_extrapolation_px", 0))
    if trim_px > 0 and "extrapmask" in lk:
        ex = np.asarray(lk["extrapmask"], dtype=bool)
        if ex.shape == mapped.shape:
            ex = binary_dilation(ex, iterations=trim_px)
            mapped = mapped.copy()
            mapped[ex] = np.nan

    rgb = mat_to_rgb(
        mapped,
        clim=local_options.get("clim"),
        cmap=local_options.get("cmap"),
        rgbnan=local_options.get("rgbnan", 1.0),
    )
    rgb = apply_overlays(
        mapped,
        rgb,
        lk,
        local_options,
        vals,
        subject,
        hemi,
        read_surface_metric,
        vert_to_image,
    )
    if local_options.get("surfshading", True) and "shading" in lk:
        rgb = rgb * np.asarray(lk["shading"], dtype=float)[:, :, None]
    return mapped, lk, rgb


def lookup_images(
    subject: str,
    vals: Any,
    hemi: str | list[str],
    view_az_el_tilt: Any,
    lookup: Any | None = None,
    **kwargs: Any,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Generate mapped values, lookup object, RGB image, and resolved options.

    Milestone 3 scope:
    - sphere, flat, and non-sphere (inflated/white/pial) support
    - one or two hemispheres
    - mapped values + simple RGB output
    """
    options = _merge_options(kwargs)

    vals_arr = np.asarray(vals)

    if isinstance(hemi, list):
        if vals_arr.ndim != 1:
            raise ValueError("For multi-hemi mode, vals must be 1D concatenated LH+RH array")

        hemis = [h.lower() for h in hemi]
        if set(hemis) != {"lh", "rh"}:
            raise ValueError("Multi-hemi mode currently expects ['lh','rh'] in any order")

        # Determine split by querying vertex counts.
        surf_kwargs: dict[str, Any] = {}
        surfdir_opt = options.get("surfdir")
        if isinstance(surfdir_opt, str) and surfdir_opt:
            surf_kwargs["surfdir"] = surfdir_opt
        if _is_flat_surftype(str(options["surftype"])):
            surf_kwargs["flat_edge_trim_steps"] = int(
                options.get("flat_edge_trim_steps", 0)
            )

        n_lh = int(
            read_surface(
                subject,
                "lh",
                str(options["surftype"]),
                str(options["surfsuffix"]),
                justcount=True,
                **surf_kwargs,
            )
        )
        n_rh = int(
            read_surface(
                subject,
                "rh",
                str(options["surftype"]),
                str(options["surfsuffix"]),
                justcount=True,
                **surf_kwargs,
            )
        )
        if vals_arr.shape[0] != n_lh + n_rh:
            raise ValueError("vals length does not match lh+rh vertex counts")

        lvals = vals_arr[:n_lh]
        rvals = vals_arr[n_lh : n_lh + n_rh]

        lookup_list = lookup if isinstance(lookup, list) else [None, None]
        if len(lookup_list) < 2:
            lookup_list = [None, None]

        view_in = view_az_el_tilt
        if isinstance(view_in, list) and len(view_in) == 2 and all(
            isinstance(v, (list, tuple, np.ndarray)) for v in view_in
        ):
            v_lh = view_in[0]
            v_rh = view_in[1]
        else:
            v_lh = view_in
            v_rh = view_in

        mapped_lh, lk_lh, rgb_lh = _lookup_one_hemi(
            subject, lvals, "lh", v_lh, lookup_list[0], options
        )
        mapped_rh, lk_rh, rgb_rh = _lookup_one_hemi(
            subject, rvals, "rh", v_rh, lookup_list[1], options
        )

        mapped = np.concatenate([mapped_lh, mapped_rh], axis=1)
        rgb = np.concatenate([rgb_lh, rgb_rh], axis=1)
        return mapped, [lk_lh, lk_rh], rgb, options

    mapped, lk, rgb = _lookup_one_hemi(
        subject,
        vals_arr,
        str(hemi).lower(),
        view_az_el_tilt,
        lookup if isinstance(lookup, dict) else None,
        options,
    )
    return mapped, lk, rgb, options
