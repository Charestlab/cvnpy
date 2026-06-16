"""Overlay helpers (thresholds, alpha, ROI outlines).

Parity target:
- Core overlay/ROI portions of cvnlookupimages.m
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.colors import to_rgb
from scipy.ndimage import binary_dilation
from skimage.segmentation import find_boundaries

from .colormap import mat_to_rgb


def _as_rgb(color: Any) -> np.ndarray:
    arr = np.asarray(color, dtype=float)
    if arr.size == 3:
        if np.any(arr > 1):
            arr = arr / 255.0
        return np.clip(arr.reshape(3), 0.0, 1.0)
    if arr.size == 1:
        v = float(arr.item())
        return np.array([v, v, v], dtype=float)
    return np.array(to_rgb(color), dtype=float)


def _build_alpha(mappedvals: np.ndarray, options: dict[str, Any]) -> np.ndarray:
    alpha_opt = options.get("overlayalpha", None)
    threshold = options.get("threshold", None)
    absthreshold = options.get("absthreshold", None)
    overlayrange = options.get("overlayrange", None)
    inclusive = bool(options.get("inclusive", False))

    alpha: np.ndarray | None = None
    if alpha_opt is not None:
        alpha = np.asarray(alpha_opt, dtype=float)

    if threshold is not None:
        if inclusive:
            alpha = (mappedvals >= float(threshold)).astype(float)
        else:
            alpha = (mappedvals > float(threshold)).astype(float)
    if absthreshold is not None:
        if inclusive:
            alpha = (np.abs(mappedvals) >= float(absthreshold)).astype(float)
        else:
            alpha = (np.abs(mappedvals) > float(absthreshold)).astype(float)
    if overlayrange is not None:
        lo, hi = float(overlayrange[0]), float(overlayrange[1])
        if inclusive:
            alpha = ((mappedvals >= lo) & (mappedvals <= hi)).astype(float)
        else:
            alpha = ((mappedvals > lo) & (mappedvals < hi)).astype(float)

    if alpha is None:
        return np.ones_like(mappedvals, dtype=float)

    if alpha.ndim == 0:
        return np.full_like(mappedvals, float(alpha), dtype=float)
    if alpha.shape != mappedvals.shape:
        raise ValueError("overlayalpha must be scalar or image-shaped when used here")
    alpha = np.nan_to_num(alpha, nan=0.0)
    return np.clip(alpha, 0.0, 1.0)


def _resolve_background_image(
    mappedvals: np.ndarray,
    lookup: dict[str, Any],
    options: dict[str, Any],
    vals: np.ndarray,
    subject: str,
    hemi: str,
    metric_loader: Any,
    map_func: Any,
) -> np.ndarray:
    bg = options.get("background", "curv")

    if isinstance(bg, str):
        metric_kwargs: dict[str, Any] = {}
        surfdir_opt = options.get("surfdir")
        if isinstance(surfdir_opt, str) and surfdir_opt:
            metric_kwargs["surfdir"] = surfdir_opt
        metric = metric_loader(
            subject,
            hemi,
            bg,
            "",
            str(options.get("surfsuffix", "orig")),
            **metric_kwargs,
        )
        if metric is None:
            return np.zeros_like(mappedvals, dtype=float)
        bg_vals = np.asarray(metric, dtype=float)
        if bg == "curv":
            bg_vals = (bg_vals < 0).astype(float)
        elif bg == "sulc":
            if np.nanmean(np.abs(bg_vals)) > 1:
                bg_vals = bg_vals / 10.0
            bg_vals = -bg_vals
        return map_func(bg_vals, lookup, badval=np.nan)

    bg_arr = np.asarray(bg, dtype=float)
    if bg_arr.ndim == 0:
        return np.full_like(mappedvals, float(bg_arr), dtype=float)
    if bg_arr.shape == mappedvals.shape:
        return bg_arr
    if bg_arr.ndim == 1 and bg_arr.shape[0] == vals.shape[0]:
        return map_func(bg_arr, lookup, badval=np.nan)
    raise ValueError("Unsupported background format")


def _apply_roi_outlines(
    rgbimg: np.ndarray,
    lookup: dict[str, Any],
    options: dict[str, Any],
    vals: np.ndarray,
    map_func: Any,
) -> np.ndarray:
    roimask_opt = options.get("roimask", None)
    if roimask_opt is None:
        return rgbimg

    roi_list = roimask_opt if isinstance(roimask_opt, list) else [roimask_opt]

    roicolor_opt = options.get("roicolor", [0.0, 0.0, 0.0])
    if isinstance(roicolor_opt, (list, tuple, np.ndarray)):
        rc_arr = np.asarray(roicolor_opt, dtype=object)
        # single RGB triplet
        if rc_arr.ndim == 1 and rc_arr.size == 3 and all(
            isinstance(x, (int, float, np.integer, np.floating)) for x in rc_arr
        ):
            colors = [roicolor_opt]
        # list of per-ROI colors
        elif isinstance(roicolor_opt, list):
            colors = roicolor_opt
        else:
            colors = [roicolor_opt]
    else:
        colors = [roicolor_opt]
    roiwidth_opt = options.get("roiwidth", 1.0)
    widths = roiwidth_opt if isinstance(roiwidth_opt, list) else [roiwidth_opt]

    out = rgbimg.copy()
    for i, rm in enumerate(roi_list):
        rm_arr = np.asarray(rm)
        if rm_arr.ndim == 1 and rm_arr.shape[0] == vals.shape[0]:
            mapped = map_func(rm_arr, lookup, badval=0.0)
        elif rm_arr.shape == rgbimg.shape[:2]:
            mapped = rm_arr
        else:
            continue

        if np.issubdtype(mapped.dtype, np.floating):
            label_img = np.nan_to_num(mapped, nan=0.0)
        else:
            label_img = mapped

        if np.max(label_img) <= 1:
            boundary = find_boundaries(label_img > 0, mode="outer")
        else:
            boundary = find_boundaries(label_img.astype(int), mode="outer")

        width = float(widths[min(i, len(widths) - 1)])
        if width > 1:
            boundary = binary_dilation(boundary, iterations=max(1, int(round(width))))

        c = _as_rgb(colors[min(i, len(colors) - 1)])
        out[boundary, :] = c.reshape(1, 3)

    return out


def apply_overlays(
    mappedvals: Any,
    rgbimg: Any,
    lookup: dict[str, Any],
    options: dict[str, Any],
    vals: Any,
    subject: str,
    hemi: str,
    metric_loader: Any,
    map_func: Any,
) -> Any:
    """Apply background, alpha masks, thresholding, ROI outlines.

    Implements a practical subset of cvnlookupimages overlay behavior.
    """
    mapped = np.asarray(mappedvals, dtype=float)
    rgb = np.asarray(rgbimg, dtype=float)
    vals_arr = np.asarray(vals, dtype=float)

    alpha = _build_alpha(mapped, options)
    need_bg = not np.all(alpha >= 0.999)
    if need_bg:
        bg_img = _resolve_background_image(
            mapped,
            lookup,
            options,
            vals_arr,
            subject,
            hemi,
            metric_loader,
            map_func,
        )
        bg_rgb = mat_to_rgb(
            bg_img,
            clim=options.get("bg_clim", (-1.0, 2.0)),
            cmap=options.get("bg_cmap", "gray"),
            rgbnan=options.get("rgbnan", 1.0),
        )
        rgb = alpha[:, :, None] * rgb + (1.0 - alpha[:, :, None]) * bg_rgb

    rgb = _apply_roi_outlines(rgb, lookup, options, vals_arr, map_func)
    return np.clip(rgb, 0.0, 1.0)
