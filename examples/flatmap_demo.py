"""Generate a flatmap PNG using cvnpy."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["image.composite_image"] = False
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.ndimage import distance_transform_edt

from cvnpy.core.lookup_images import lookup_images
from cvnpy.io.metrics import read_surface_metric
from cvnpy.io.surfaces import read_surface
from cvnpy.lookup.map import vert_to_image
from cvnpy.render.colormap import mat_to_rgb
from cvnpy.view.viewpoints import lookup_viewpoint


def _parse_view(s: str) -> list[float]:
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError("view must be az,el,tilt")
    return parts


def _parse_pair(s: str) -> tuple[float, float]:
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 2:
        raise ValueError("expected two comma-separated values")
    return float(parts[0]), float(parts[1])


def _load_metric_or_fallback(
    subject: str,
    hemi: str,
    metric: str,
    surftype: str,
    surfsuffix: str,
    surfdir: str | None,
) -> np.ndarray:
    vals = read_surface_metric(
        subject,
        hemi,
        metric,
        "",
        surfsuffix,
        **({"surfdir": surfdir} if surfdir else {}),
    )
    if vals is not None:
        return np.asarray(vals, dtype=float)

    n = int(
        read_surface(
            subject,
            hemi,
            surftype,
            surfsuffix,
            justcount=True,
            **({"surfdir": surfdir} if surfdir else {}),
        )
    )
    rng = np.random.default_rng(0 if hemi == "lh" else 1)
    return rng.standard_normal(n)


def _load_stat_file(
    path: str,
    surfdir: str | None,
    subject: str,
    surftype: str,
    surfsuffix: str,
) -> tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(np.load(path), dtype=float).ravel()
    n_lh = int(
        read_surface(
            subject,
            "lh",
            surftype,
            surfsuffix,
            justcount=True,
            **({"surfdir": surfdir} if surfdir else {}),
        )
    )
    n_rh = int(
        read_surface(
            subject,
            "rh",
            surftype,
            surfsuffix,
            justcount=True,
            **({"surfdir": surfdir} if surfdir else {}),
        )
    )
    if vals.size == n_lh + n_rh:
        return vals[:n_lh], vals[n_lh : n_lh + n_rh]
    if vals.size == n_lh:
        return vals, np.full((n_rh,), np.nan, dtype=float)
    if vals.size == n_rh:
        return np.full((n_lh,), np.nan, dtype=float), vals
    raise ValueError(
        f"stat-file length {vals.size} does not match lh+rh ({n_lh+n_rh}), lh ({n_lh}), or rh ({n_rh})"
    )


def _load_vertex_file_split(
    path: str,
    surfdir: str | None,
    subject: str,
    surftype: str,
    surfsuffix: str,
) -> tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(np.load(path)).ravel()
    n_lh = int(
        read_surface(
            subject,
            "lh",
            surftype,
            surfsuffix,
            justcount=True,
            **({"surfdir": surfdir} if surfdir else {}),
        )
    )
    n_rh = int(
        read_surface(
            subject,
            "rh",
            surftype,
            surfsuffix,
            justcount=True,
            **({"surfdir": surfdir} if surfdir else {}),
        )
    )
    if vals.size == n_lh + n_rh:
        return vals[:n_lh], vals[n_lh : n_lh + n_rh]
    if vals.size == n_lh:
        return vals, np.zeros((n_rh,), dtype=vals.dtype)
    if vals.size == n_rh:
        return np.zeros((n_lh,), dtype=vals.dtype), vals
    raise ValueError(
        f"vertex-file length {vals.size} does not match lh+rh ({n_lh+n_rh}), lh ({n_lh}), or rh ({n_rh})"
    )


def _rotate_rgb_90(rgb: np.ndarray, degrees: int) -> np.ndarray:
    if degrees % 90 != 0:
        raise ValueError("rotation must be a multiple of 90 degrees")
    k = (degrees // 90) % 4
    return np.rot90(rgb, k=k)


def _flip_lr(arr: np.ndarray) -> np.ndarray:
    return np.fliplr(arr)


def _compose_side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    gap_px: int,
    bg_gray: float,
    margin_px: int = 24,
    rh_shift_px: int = 0,
) -> np.ndarray:
    h = max(left.shape[0], right.shape[0]) + 2 * margin_px
    w = left.shape[1] + gap_px + right.shape[1] + 2 * margin_px + max(0, int(rh_shift_px))
    channels = left.shape[2] if left.ndim == 3 else 1
    fill = [float(bg_gray)] * channels
    if channels == 4:
        fill[3] = 0.0
    out = np.full((h, w, channels), fill, dtype=float)
    y0_left = (h - left.shape[0]) // 2
    y0_right = (h - right.shape[0]) // 2
    x0 = margin_px
    out[y0_left : y0_left + left.shape[0], x0 : x0 + left.shape[1], :] = left
    x1 = x0 + left.shape[1] + gap_px + int(rh_shift_px)
    out[y0_right : y0_right + right.shape[0], x1 : x1 + right.shape[1], :] = right
    return out


def _alpha_bbox(alpha: np.ndarray, tol: float = 1e-6) -> tuple[int, int, int, int]:
    content = np.asarray(alpha, dtype=float) > tol
    ys, xs = np.nonzero(content)
    if ys.size == 0 or xs.size == 0:
        return 0, alpha.shape[0], 0, alpha.shape[1]
    return int(ys.min()), int(ys.max() + 1), int(xs.min()), int(xs.max() + 1)


def _lookup_alpha(lookup: dict) -> np.ndarray:
    imglookup = np.asarray(lookup["imglookup"])
    extrapmask = np.asarray(lookup.get("extrapmask", np.zeros_like(imglookup, dtype=bool)))
    return (~extrapmask).astype(float)


def _composite_rgba(
    curv_rgb: np.ndarray,
    curv_alpha: np.ndarray,
    stat_rgb: np.ndarray,
    stat_alpha: np.ndarray,
) -> np.ndarray:
    ca = np.clip(np.asarray(curv_alpha, dtype=float), 0.0, 1.0)
    sa = np.clip(np.asarray(stat_alpha, dtype=float), 0.0, 1.0) * ca
    curv = np.clip(np.asarray(curv_rgb, dtype=float), 0.0, 1.0)
    stat = np.clip(np.asarray(stat_rgb, dtype=float), 0.0, 1.0)
    rgb = (1.0 - sa[:, :, None]) * curv + sa[:, :, None] * stat
    return np.dstack([rgb, ca])


def _rgb_with_alpha(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return np.dstack(
        [np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0), np.clip(alpha, 0.0, 1.0)]
    )


def _crop_rgba_to_alpha(rgba: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    y0, y1, x0, x1 = _alpha_bbox(rgba[:, :, 3], tol=tol)
    return rgba[y0:y1, x0:x1, :]


def _compose_side_by_side_layers(
    left_curv: np.ndarray,
    left_curv_alpha: np.ndarray,
    left_stat: np.ndarray,
    left_alpha: np.ndarray,
    right_curv: np.ndarray,
    right_curv_alpha: np.ndarray,
    right_stat: np.ndarray,
    right_alpha: np.ndarray,
    gap_px: int,
    bg_gray: float,
    margin_px: int = 24,
    rh_shift_px: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h = max(left_curv.shape[0], right_curv.shape[0]) + 2 * margin_px
    w = (
        left_curv.shape[1]
        + gap_px
        + right_curv.shape[1]
        + 2 * margin_px
        + max(0, int(rh_shift_px))
    )
    curv = np.full((h, w, 3), float(bg_gray), dtype=float)
    curv_alpha = np.zeros((h, w), dtype=float)
    stat = np.zeros((h, w, 3), dtype=float)
    alpha = np.zeros((h, w), dtype=float)

    y0_left = (h - left_curv.shape[0]) // 2
    y0_right = (h - right_curv.shape[0]) // 2
    x0 = margin_px
    x1 = x0 + left_curv.shape[1] + gap_px + int(rh_shift_px)

    curv[y0_left : y0_left + left_curv.shape[0], x0 : x0 + left_curv.shape[1], :] = left_curv
    curv[y0_right : y0_right + right_curv.shape[0], x1 : x1 + right_curv.shape[1], :] = right_curv
    curv_alpha[y0_left : y0_left + left_curv_alpha.shape[0], x0 : x0 + left_curv_alpha.shape[1]] = left_curv_alpha
    curv_alpha[y0_right : y0_right + right_curv_alpha.shape[0], x1 : x1 + right_curv_alpha.shape[1]] = right_curv_alpha
    stat[y0_left : y0_left + left_stat.shape[0], x0 : x0 + left_stat.shape[1], :] = left_stat
    stat[y0_right : y0_right + right_stat.shape[0], x1 : x1 + right_stat.shape[1], :] = right_stat
    alpha[y0_left : y0_left + left_alpha.shape[0], x0 : x0 + left_alpha.shape[1]] = left_alpha
    alpha[y0_right : y0_right + right_alpha.shape[0], x1 : x1 + right_alpha.shape[1]] = right_alpha
    return curv, curv_alpha, stat, alpha


def _hemi_rects_px(
    left_shape: tuple[int, int, int],
    right_shape: tuple[int, int, int],
    gap_px: int,
    margin_px: int,
    rh_shift_px: int,
) -> dict[str, object]:
    h = max(left_shape[0], right_shape[0]) + 2 * margin_px
    w = left_shape[1] + gap_px + right_shape[1] + 2 * margin_px + max(0, int(rh_shift_px))
    y0_left = (h - left_shape[0]) // 2
    y0_right = (h - right_shape[0]) // 2
    x0 = margin_px
    x1 = x0 + left_shape[1] + gap_px + int(rh_shift_px)
    return {
        "canvas_h": int(h),
        "canvas_w": int(w),
        "lh": {
            "x": int(x0),
            "y": int(y0_left),
            "w": int(left_shape[1]),
            "h": int(left_shape[0]),
        },
        "rh": {
            "x": int(x1),
            "y": int(y0_right),
            "w": int(right_shape[1]),
            "h": int(right_shape[0]),
        },
    }


def _resolve_mpl_cmap(name: str) -> object:
    if name.lower() == "cmapsign4":
        colors = np.array(
            [
                [0.8, 1.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.8],
            ],
            dtype=float,
        )
        return LinearSegmentedColormap.from_list("cmapsign4", colors, N=256)
    return colormaps.get_cmap(name)


def _render_and_save(
    out: Path,
    rgba: np.ndarray,
    show_colorbar: bool,
    cmap_name: str,
    clim: tuple[float, float],
    inset_lh: np.ndarray | None = None,
    inset_rh: np.ndarray | None = None,
    inset_size_frac: float = 0.20,
    inset_margin_px: int = 14,
) -> None:
    if (not show_colorbar) and inset_lh is None and inset_rh is None:
        plt.imsave(out, np.clip(rgba, 0.0, 1.0))
        return

    h, w = rgba.shape[:2]
    dpi = 100.0
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor="none")
    ax = fig.add_axes([0.0, 0.08, 1.0, 0.92])
    ax.patch.set_alpha(0.0)
    ax.imshow(np.clip(rgba, 0.0, 1.0), interpolation="nearest")
    ax.axis("off")

    _draw_bottom_insets(
        fig=fig,
        base_shape=rgba.shape,
        main_rect=(0.0, 0.08, 1.0, 0.92),
        inset_lh=inset_lh,
        inset_rh=inset_rh,
        inset_size_frac=inset_size_frac,
        inset_margin_px=inset_margin_px,
        with_gids=False,
    )

    cax = fig.add_axes([0.35, 0.03, 0.30, 0.02])
    sm = ScalarMappable(
        norm=Normalize(vmin=float(clim[0]), vmax=float(clim[1])),
        cmap=_resolve_mpl_cmap(cmap_name),
    )
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([float(clim[0]), float(clim[1])])
    cb.set_ticklabels([f"{float(clim[0]):g}", f"{float(clim[1]):g}"])
    cb.outline.set_linewidth(0.8)

    fig.savefig(out, dpi=dpi, facecolor="none", transparent=True)
    plt.close(fig)


def _render_layered_svg(
    out: Path,
    curv_rgb: np.ndarray,
    curv_alpha: np.ndarray,
    stat_rgb: np.ndarray,
    stat_alpha: np.ndarray,
    show_colorbar: bool,
    cmap_name: str,
    clim: tuple[float, float],
    roi_color: str,
    roi_width: float,
    draw_roi_names: bool,
    roi_names: list[str] | None,
    roi_label_color: str,
    roi_label_size: float,
    roi_img: np.ndarray | None = None,
    roi_hemi: str | None = None,
    inset_lh: np.ndarray | None = None,
    inset_rh: np.ndarray | None = None,
    inset_size_frac: float = 0.20,
    inset_margin_px: int = 14,
    hemi_layers: dict[str, object] | None = None,
) -> None:
    h, w = curv_rgb.shape[:2]
    dpi = 100.0
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor="none")
    main_rect = (0.0, 0.08 if show_colorbar else 0.0, 1.0, 0.92 if show_colorbar else 1.0)
    ax = fig.add_axes(list(main_rect))
    ax.set_gid("layers")
    ax.patch.set_alpha(0.0)
    if hemi_layers is None:
        curv_rgba = np.dstack(
            [np.clip(curv_rgb, 0.0, 1.0), np.clip(curv_alpha, 0.0, 1.0)]
        )
        im_curv = ax.imshow(curv_rgba, interpolation="nearest")
        im_curv.set_gid("curv")
        stat_rgba = np.dstack(
            [np.clip(stat_rgb, 0.0, 1.0), np.clip(stat_alpha, 0.0, 1.0)]
        )
        im_stat = ax.imshow(stat_rgba, interpolation="nearest")
        im_stat.set_gid("stat")
        if roi_img is not None:
            roi_arr = np.asarray(roi_img)
            hemi_name = str(roi_hemi) if roi_hemi else "single"
            ax_roi = fig.add_axes(list(main_rect), zorder=4)
            ax_roi.set_gid(f"roi_{hemi_name}_axes")
            ax_roi.patch.set_alpha(0.0)
            roi_vals = np.unique(roi_arr[np.isfinite(roi_arr)])
            roi_vals = roi_vals[roi_vals > 0]
            if roi_vals.size > 0:
                if np.nanmax(roi_arr) <= 1:
                    roi_ids = np.array([1.0], dtype=float)
                    roi_masks = [roi_arr > 0]
                else:
                    roi_ids = roi_vals
                    roi_masks = [roi_arr == rv for rv in roi_ids]

                for ridx, rmask in enumerate(roi_masks):
                    cs = ax_roi.contour(
                        rmask.astype(float),
                        levels=[0.5],
                        colors=[roi_color],
                        linewidths=[float(roi_width)],
                        origin="upper",
                    )
                    cs.set_gid(f"roi_{hemi_name}_{int(roi_ids[ridx])}")
                    if draw_roi_names and np.any(rmask):
                        md = distance_transform_edt(rmask)
                        yy, xx = np.unravel_index(int(np.argmax(md)), md.shape)
                        rid = int(roi_ids[ridx])
                        if roi_names and rid - 1 < len(roi_names):
                            name = str(roi_names[rid - 1])
                        elif roi_names and len(roi_names) == len(roi_ids):
                            name = str(roi_names[ridx])
                        elif roi_ids.size == 1 and roi_names and len(roi_names) >= 1:
                            name = str(roi_names[0])
                        else:
                            name = str(rid)
                        txt = ax_roi.text(
                            float(xx),
                            float(yy),
                            name,
                            color=roi_label_color,
                            fontsize=float(roi_label_size),
                            ha="center",
                            va="center",
                            bbox={
                                "boxstyle": "round,pad=0.15",
                                "facecolor": "black",
                                "edgecolor": "none",
                                "alpha": 0.5,
                            },
                        )
                        txt.set_gid(f"roi_label_{hemi_name}_{rid}")
            ax_roi.set_xlim(0, roi_arr.shape[1] - 1)
            ax_roi.set_ylim(roi_arr.shape[0] - 1, 0)
            ax_roi.axis("off")
    else:
        canvas_w = float(int(hemi_layers["canvas_w"]))
        canvas_h = float(int(hemi_layers["canvas_h"]))
        for hemi_name in ("lh", "rh"):
            layer = hemi_layers.get(hemi_name)
            if not isinstance(layer, dict):
                continue
            rect = layer.get("rect")
            if not isinstance(rect, dict):
                continue
            x_px = float(rect["x"])
            y_top_px = float(rect["y"])
            ww_px = float(rect["w"])
            hh_px = float(rect["h"])
            y_bottom_px = canvas_h - (y_top_px + hh_px)
            x_fig = main_rect[0] + main_rect[2] * (x_px / canvas_w)
            y_fig = main_rect[1] + main_rect[3] * (y_bottom_px / canvas_h)
            w_fig = main_rect[2] * (ww_px / canvas_w)
            h_fig = main_rect[3] * (hh_px / canvas_h)
            ax_curv = fig.add_axes([x_fig, y_fig, w_fig, h_fig], zorder=2)
            ax_curv.set_gid(f"curv_{hemi_name}_axes")
            ax_curv.patch.set_alpha(0.0)
            curv_rgba = np.dstack(
                [
                    np.clip(np.asarray(layer["curv"], dtype=float), 0.0, 1.0),
                    np.clip(np.asarray(layer["curv_alpha"], dtype=float), 0.0, 1.0),
                ]
            )
            im_curv = ax_curv.imshow(curv_rgba, interpolation="nearest")
            im_curv.set_gid(f"curv_{hemi_name}")
            ax_curv.axis("off")

            ax_stat = fig.add_axes([x_fig, y_fig, w_fig, h_fig], zorder=3)
            ax_stat.set_gid(f"stat_{hemi_name}_axes")
            ax_stat.patch.set_alpha(0.0)
            stat_rgba = np.dstack(
                [
                    np.clip(np.asarray(layer["stat"], dtype=float), 0.0, 1.0),
                    np.clip(np.asarray(layer["alpha"], dtype=float), 0.0, 1.0),
                ]
            )
            im_stat = ax_stat.imshow(stat_rgba, interpolation="nearest")
            im_stat.set_gid(f"stat_{hemi_name}")
            ax_stat.axis("off")

            roi_img = layer.get("roi", None)
            if roi_img is not None:
                roi_arr = np.asarray(roi_img)
                ax_roi = fig.add_axes([x_fig, y_fig, w_fig, h_fig], zorder=4)
                ax_roi.set_gid(f"roi_{hemi_name}_axes")
                ax_roi.patch.set_alpha(0.0)
                roi_vals = np.unique(roi_arr[np.isfinite(roi_arr)])
                roi_vals = roi_vals[roi_vals > 0]
                if roi_vals.size == 0:
                    ax_roi.axis("off")
                else:
                    if np.nanmax(roi_arr) <= 1:
                        roi_ids = np.array([1.0], dtype=float)
                        roi_masks = [roi_arr > 0]
                    else:
                        roi_ids = roi_vals
                        roi_masks = [roi_arr == rv for rv in roi_ids]

                    for ridx, rmask in enumerate(roi_masks):
                        cs = ax_roi.contour(
                            rmask.astype(float),
                            levels=[0.5],
                            colors=[roi_color],
                            linewidths=[float(roi_width)],
                            origin="upper",
                        )
                        cs.set_gid(f"roi_{hemi_name}_{int(roi_ids[ridx])}")

                        if draw_roi_names and np.any(rmask):
                            # MATLAB parity (drawroinames.m):
                            # place label at argmax(bwdist(~m)), i.e. deepest interior point.
                            md = distance_transform_edt(rmask)
                            yy, xx = np.unravel_index(int(np.argmax(md)), md.shape)
                            rid = int(roi_ids[ridx])
                            if roi_names and rid - 1 < len(roi_names):
                                name = str(roi_names[rid - 1])
                            elif roi_names and len(roi_names) == len(roi_ids):
                                name = str(roi_names[ridx])
                            elif roi_ids.size == 1 and roi_names and len(roi_names) >= 1:
                                name = str(roi_names[0])
                            else:
                                name = str(rid)
                            txt = ax_roi.text(
                                float(xx),
                                float(yy),
                                name,
                                color=roi_label_color,
                                fontsize=float(roi_label_size),
                                ha="center",
                                va="center",
                                bbox={
                                    "boxstyle": "round,pad=0.15",
                                    "facecolor": "black",
                                    "edgecolor": "none",
                                    "alpha": 0.5,
                                },
                            )
                            txt.set_gid(f"roi_label_{hemi_name}_{rid}")

                    ax_roi.set_xlim(0, roi_arr.shape[1] - 1)
                    ax_roi.set_ylim(roi_arr.shape[0] - 1, 0)
                    ax_roi.axis("off")
    ax.axis("off")

    t1 = ax.text(0.02, 0.98, "LH", transform=ax.transAxes, va="top", ha="left", fontsize=10, color="black")
    t2 = ax.text(0.98, 0.98, "RH", transform=ax.transAxes, va="top", ha="right", fontsize=10, color="black")
    t1.set_gid("labels_lh")
    t2.set_gid("labels_rh")

    _draw_bottom_insets(
        fig=fig,
        base_shape=curv_rgb.shape,
        main_rect=main_rect,
        inset_lh=inset_lh,
        inset_rh=inset_rh,
        inset_size_frac=inset_size_frac,
        inset_margin_px=inset_margin_px,
        with_gids=True,
    )

    if show_colorbar:
        cax = fig.add_axes([0.35, 0.03, 0.30, 0.07])
        cax.set_gid("colorbar")
        sm = ScalarMappable(
            norm=Normalize(vmin=float(clim[0]), vmax=float(clim[1])),
            cmap=_resolve_mpl_cmap(cmap_name),
        )
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_ticks([float(clim[0]), float(clim[1])])
        cb.set_ticklabels([f"{float(clim[0]):g}", f"{float(clim[1]):g}"])
        cb.outline.set_linewidth(0.8)

    fig.savefig(out, dpi=dpi, facecolor="none", transparent=True, format="svg")
    plt.close(fig)


def _tag_svg_image_layers(path: Path) -> None:
    txt = path.read_text(encoding="utf-8")
    ids = ["curv", "stat"]
    start = 0
    for layer_id in ids:
        i = txt.find("<image ", start)
        if i < 0:
            break
        j = i + len("<image ")
        txt = txt[:j] + f'id="{layer_id}" ' + txt[j:]
        start = j + len(layer_id) + 6
    path.write_text(txt, encoding="utf-8")


def _draw_bottom_insets(
    fig: plt.Figure,
    base_shape: tuple[int, int, int],
    main_rect: tuple[float, float, float, float],
    inset_lh: np.ndarray | None,
    inset_rh: np.ndarray | None,
    inset_size_frac: float,
    inset_margin_px: int,
    with_gids: bool,
) -> None:
    fig_h_px = float(base_shape[0])
    fig_w_px = float(base_shape[1])
    x0_px = main_rect[0] * fig_w_px
    y0_px = main_rect[1] * fig_h_px
    w_px = main_rect[2] * fig_w_px
    target_w = max(1.0, float(inset_size_frac) * fig_w_px)
    max_h = max(1.0, 0.35 * fig_h_px)

    for img, side, gid in (
        (inset_lh, "left", "inset_lh"),
        (inset_rh, "right", "inset_rh"),
    ):
        if img is None:
            continue
        ih, iw = img.shape[:2]
        scale = target_w / max(1.0, float(iw))
        disp_w = target_w
        disp_h = max(1.0, float(ih) * scale)
        if disp_h > max_h:
            scale2 = max_h / disp_h
            disp_w *= scale2
            disp_h *= scale2

        x_px = x0_px + float(inset_margin_px)
        if side == "right":
            x_px = x0_px + w_px - float(inset_margin_px) - disp_w
        y_px = y0_px + float(inset_margin_px)
        rect = [x_px / fig_w_px, y_px / fig_h_px, disp_w / fig_w_px, disp_h / fig_h_px]
        ax_in = fig.add_axes(rect, zorder=10)
        ax_in.patch.set_alpha(0.0)
        im = ax_in.imshow(np.clip(img, 0.0, 1.0), interpolation="nearest")
        if with_gids:
            ax_in.set_gid(f"{gid}_axes")
            im.set_gid(gid)
        ax_in.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description="cvnpy flatmap demo")
    parser.add_argument("--subject", required=False, default=None)
    parser.add_argument(
        "--subject-dir",
        default=None,
        help="Path to a subject directory containing surf/ (e.g. /path/nsd_fsaverage).",
    )
    parser.add_argument(
        "--subjects-dir",
        default=None,
        help="Path to SUBJECTS_DIR root (parent that contains subject folders).",
    )
    parser.add_argument("--hemi", default="lh", choices=["lh", "rh", "both"])
    parser.add_argument("--surftype", default="full.flat.patch.3d")
    parser.add_argument("--surfsuffix", default="orig")
    parser.add_argument("--metric", default="curv")
    parser.add_argument(
        "--stat-map",
        default=None,
        help="Surface metric name for statistical overlay (e.g., tstat, zmap).",
    )
    parser.add_argument(
        "--stat-file",
        default=None,
        help="Path to .npy/.npz file with vertex values (lh+rh concatenated or single hemi length).",
    )
    parser.add_argument(
        "--cmap",
        default="cmapsign4",
        help="Colormap for mapped values/stat map.",
    )
    parser.add_argument(
        "--clim",
        default=None,
        help="Color limits 'min,max' (optional).",
    )
    parser.add_argument(
        "--symmetric-clim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use clim = [-max(abs(vals)), +max(abs(vals))] from loaded stat values.",
    )
    parser.add_argument(
        "--absthreshold",
        type=float,
        default=None,
        help="Absolute threshold; values with abs<=thr show background instead.",
    )
    parser.add_argument(
        "--background",
        default="curv",
        help="Background metric name or scalar value passed to lookup_images.",
    )
    parser.add_argument(
        "--show-colorbar",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show bottom-centered horizontal colorbar with clim labels.",
    )
    parser.add_argument(
        "--show-insets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show bottom-corner inflated insets (LH left, RH right).",
    )
    parser.add_argument(
        "--inset-viewname",
        default="lateral",
        help="View name for inflated insets (e.g., lateral, medial, occip).",
    )
    parser.add_argument(
        "--inset-imageres",
        type=int,
        default=320,
        help="Image resolution used to render each inflated inset.",
    )
    parser.add_argument(
        "--inset-size-frac",
        type=float,
        default=0.20,
        help="Inset width as fraction of full figure width.",
    )
    parser.add_argument(
        "--inset-margin-px",
        type=int,
        default=14,
        help="Margin from figure edge (in px) for inset placement.",
    )
    parser.add_argument(
        "--roi-mask-file",
        default=None,
        help="Optional .npy ROI/parcellation in vertex space (lh+rh or single-hemi length).",
    )
    parser.add_argument(
        "--roi-color",
        default="white",
        help="ROI contour color for SVG overlay objects.",
    )
    parser.add_argument(
        "--roi-width",
        type=float,
        default=1.0,
        help="ROI contour line width in SVG.",
    )
    parser.add_argument(
        "--draw-roi-names",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw ROI text labels in SVG.",
    )
    parser.add_argument(
        "--roi-names",
        default=None,
        help="Optional comma-separated ROI names for labels 1..N.",
    )
    parser.add_argument(
        "--roi-label-color",
        default="white",
        help="ROI label text color.",
    )
    parser.add_argument(
        "--roi-label-size",
        type=float,
        default=9.0,
        help="ROI label text size in points.",
    )
    parser.add_argument("--view", default="0,0,0")
    parser.add_argument("--imageres", type=int, default=900)
    parser.add_argument("--xyextent", default="1,1")
    parser.add_argument("--out", default="flatmap_demo.png")
    parser.add_argument("--out-svg", default=None, help="Optional layered SVG output path.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--flat-margin-frac",
        type=float,
        default=0.04,
        help="Fractional XY padding so map does not touch PNG borders (e.g., 0.04).",
    )
    parser.add_argument(
        "--pair-gap-px",
        type=int,
        default=24,
        help="Gap in pixels between hemispheres when --hemi both.",
    )
    parser.add_argument(
        "--pair-bg-gray",
        type=float,
        default=1.0,
        help="Background gray level for combined image when --hemi both.",
    )
    parser.add_argument(
        "--pair-margin-px",
        type=int,
        default=24,
        help="Outer white margin around combined hemisphere image.",
    )
    parser.add_argument(
        "--pair-rh-shift-px",
        type=int,
        default=0,
        help="Additional horizontal translation (px) applied to RH when --hemi both.",
    )
    parser.add_argument(
        "--lh-rotate-deg",
        type=int,
        default=0,
        help="Rotation for LH image in degrees (clockwise is negative).",
    )
    parser.add_argument(
        "--rh-rotate-deg",
        type=int,
        default=0,
        help="Rotation for RH image in degrees (counterclockwise is positive).",
    )
    parser.add_argument(
        "--rh-flip-lr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mirror RH flatmap left-right after rotation (puts calcarine-side cut inward).",
    )
    args = parser.parse_args()

    surfdir = None
    if args.subject_dir:
        subj_path = Path(args.subject_dir).resolve()
        surfdir = str(subj_path / "surf")
        subject = subj_path.name
    else:
        subject = args.subject
        if subject is None:
            raise ValueError("Provide --subject or --subject-dir")

    if args.subjects_dir:
        os.environ["SUBJECTS_DIR"] = str(Path(args.subjects_dir).resolve())

    xy = [float(x.strip()) for x in args.xyextent.split(",")]
    if len(xy) != 2:
        raise ValueError("xyextent must be w,h")

    clim = _parse_pair(args.clim) if args.clim else (-np.inf, np.inf)
    stat_metric = args.stat_map if args.stat_map else args.metric

    common_kwargs = {
        "surftype": args.surftype,
        "surfsuffix": args.surfsuffix,
        "imageres": args.imageres,
        "xyextent": (xy[0], xy[1]),
        "background": args.background,
        "surfshading": False,
        "cmap": args.cmap,
        "clim": clim,
        "rgbnan": 1.0,
        "cache_dir": args.cache_dir,
        "savelookup": not args.no_cache,
        "reset": bool(args.no_cache),
        "surfdir": surfdir,
        "flat_margin_frac": float(args.flat_margin_frac),
        "absthreshold": args.absthreshold,
        "bg_cmap": "gray",
        "bg_clim": (-1.0, 2.0),
    }
    view = _parse_view(args.view)

    all_vals_for_clim: np.ndarray | None = None
    vals_lh_data: np.ndarray | None = None
    vals_rh_data: np.ndarray | None = None
    roi_names_list = (
        [s.strip() for s in str(args.roi_names).split(",") if s.strip()]
        if args.roi_names
        else None
    )
    curv_layer: np.ndarray | None = None
    curv_alpha_layer: np.ndarray | None = None
    stat_layer: np.ndarray | None = None
    alpha_layer: np.ndarray | None = None
    rgba: np.ndarray | None = None
    roi_layer: np.ndarray | None = None
    hemi_layers: dict[str, object] | None = None
    inset_lh: np.ndarray | None = None
    inset_rh: np.ndarray | None = None
    if args.hemi == "both":
        if args.stat_file:
            vals_lh, vals_rh = _load_stat_file(
                args.stat_file, surfdir, subject, args.surftype, args.surfsuffix
            )
        else:
            vals_lh = _load_metric_or_fallback(
                subject, "lh", stat_metric, args.surftype, args.surfsuffix, surfdir
            )
            vals_rh = _load_metric_or_fallback(
                subject, "rh", stat_metric, args.surftype, args.surfsuffix, surfdir
            )
        all_vals_for_clim = np.concatenate([vals_lh.ravel(), vals_rh.ravel()], axis=0)
        vals_lh_data = vals_lh
        vals_rh_data = vals_rh
        if bool(args.symmetric_clim) and args.clim is None:
            m = float(
                np.nanmax(
                    np.abs(np.concatenate([vals_lh.ravel(), vals_rh.ravel()], axis=0))
                )
            )
            if np.isfinite(m) and m > 0:
                common_kwargs["clim"] = (-m, m)

        mapped_lh, lookup_lh, rgb_lh, _options_lh = lookup_images(
            subject, vals_lh, "lh", view, **common_kwargs
        )
        mapped_rh, lookup_rh, rgb_rh, _options_rh = lookup_images(
            subject, vals_rh, "rh", view, **common_kwargs
        )
        curv_lh = read_surface_metric(subject, "lh", "curv", "", args.surfsuffix, **({"surfdir": surfdir} if surfdir else {}))
        curv_rh = read_surface_metric(subject, "rh", "curv", "", args.surfsuffix, **({"surfdir": surfdir} if surfdir else {}))
        curv_lh_img = vert_to_image(np.asarray(curv_lh, dtype=float) < 0, lookup_lh, badval=np.nan)
        curv_rh_img = vert_to_image(np.asarray(curv_rh, dtype=float) < 0, lookup_rh, badval=np.nan)
        roi_lh_img = None
        roi_rh_img = None
        if args.roi_mask_file:
            roi_lh, roi_rh = _load_vertex_file_split(
                args.roi_mask_file, surfdir, subject, args.surftype, args.surfsuffix
            )
            roi_lh_img = vert_to_image(np.asarray(roi_lh), lookup_lh, badval=np.nan)
            roi_rh_img = vert_to_image(np.asarray(roi_rh), lookup_rh, badval=np.nan)
        curv_alpha_lh = _lookup_alpha(lookup_lh)
        curv_alpha_rh = _lookup_alpha(lookup_rh)
        curv_lh_rgb = mat_to_rgb(curv_lh_img, clim=(-1.0, 2.0), cmap="gray", rgbnan=1.0)
        curv_rh_rgb = mat_to_rgb(curv_rh_img, clim=(-1.0, 2.0), cmap="gray", rgbnan=1.0)
        stat_lh_rgb = mat_to_rgb(mapped_lh, clim=common_kwargs["clim"], cmap=args.cmap, rgbnan=1.0)
        stat_rh_rgb = mat_to_rgb(mapped_rh, clim=common_kwargs["clim"], cmap=args.cmap, rgbnan=1.0)
        alpha_lh = np.isfinite(mapped_lh).astype(float)
        alpha_rh = np.isfinite(mapped_rh).astype(float)
        if args.absthreshold is not None:
            thr = float(args.absthreshold)
            alpha_lh = alpha_lh * (np.abs(mapped_lh) > thr).astype(float)
            alpha_rh = alpha_rh * (np.abs(mapped_rh) > thr).astype(float)

        rgb_lh = _rotate_rgb_90(rgb_lh, int(args.lh_rotate_deg))
        rgb_rh = _rotate_rgb_90(rgb_rh, int(args.rh_rotate_deg))
        curv_lh_rgb = _rotate_rgb_90(curv_lh_rgb, int(args.lh_rotate_deg))
        curv_rh_rgb = _rotate_rgb_90(curv_rh_rgb, int(args.rh_rotate_deg))
        if roi_lh_img is not None:
            roi_lh_img = _rotate_rgb_90(roi_lh_img, int(args.lh_rotate_deg))
        if roi_rh_img is not None:
            roi_rh_img = _rotate_rgb_90(roi_rh_img, int(args.rh_rotate_deg))
        curv_alpha_lh = _rotate_rgb_90(curv_alpha_lh, int(args.lh_rotate_deg))
        curv_alpha_rh = _rotate_rgb_90(curv_alpha_rh, int(args.rh_rotate_deg))
        stat_lh_rgb = _rotate_rgb_90(stat_lh_rgb, int(args.lh_rotate_deg))
        stat_rh_rgb = _rotate_rgb_90(stat_rh_rgb, int(args.rh_rotate_deg))
        alpha_lh = _rotate_rgb_90(alpha_lh, int(args.lh_rotate_deg))
        alpha_rh = _rotate_rgb_90(alpha_rh, int(args.rh_rotate_deg))
        if bool(args.rh_flip_lr):
            rgb_rh = _flip_lr(rgb_rh)
            curv_rh_rgb = _flip_lr(curv_rh_rgb)
            curv_alpha_rh = _flip_lr(curv_alpha_rh)
            stat_rh_rgb = _flip_lr(stat_rh_rgb)
            alpha_rh = _flip_lr(alpha_rh)
            if roi_rh_img is not None:
                roi_rh_img = _flip_lr(roi_rh_img)
        bg_gray = float(np.clip(args.pair_bg_gray, 0.0, 1.0))
        y0, y1, x0, x1 = _alpha_bbox(curv_alpha_lh)
        rgb_lh = rgb_lh[y0:y1, x0:x1, :]
        curv_lh_rgb = curv_lh_rgb[y0:y1, x0:x1, :]
        curv_alpha_lh = curv_alpha_lh[y0:y1, x0:x1]
        stat_lh_rgb = stat_lh_rgb[y0:y1, x0:x1, :]
        alpha_lh = alpha_lh[y0:y1, x0:x1]
        if roi_lh_img is not None:
            roi_lh_img = roi_lh_img[y0:y1, x0:x1]
        y0, y1, x0, x1 = _alpha_bbox(curv_alpha_rh)
        rgb_rh = rgb_rh[y0:y1, x0:x1, :]
        curv_rh_rgb = curv_rh_rgb[y0:y1, x0:x1, :]
        curv_alpha_rh = curv_alpha_rh[y0:y1, x0:x1]
        stat_rh_rgb = stat_rh_rgb[y0:y1, x0:x1, :]
        alpha_rh = alpha_rh[y0:y1, x0:x1]
        if roi_rh_img is not None:
            roi_rh_img = roi_rh_img[y0:y1, x0:x1]
        rgba_lh = _composite_rgba(curv_lh_rgb, curv_alpha_lh, stat_lh_rgb, alpha_lh)
        rgba_rh = _composite_rgba(curv_rh_rgb, curv_alpha_rh, stat_rh_rgb, alpha_rh)
        rgba = _compose_side_by_side(
            rgba_lh,
            rgba_rh,
            gap_px=max(0, int(args.pair_gap_px)),
            bg_gray=bg_gray,
            margin_px=max(0, int(args.pair_margin_px)),
            rh_shift_px=int(args.pair_rh_shift_px),
        )
        curv_layer, curv_alpha_layer, stat_layer, alpha_layer = _compose_side_by_side_layers(
            curv_lh_rgb,
            curv_alpha_lh,
            stat_lh_rgb,
            alpha_lh,
            curv_rh_rgb,
            curv_alpha_rh,
            stat_rh_rgb,
            alpha_rh,
            gap_px=max(0, int(args.pair_gap_px)),
            bg_gray=bg_gray,
            margin_px=max(0, int(args.pair_margin_px)),
            rh_shift_px=int(args.pair_rh_shift_px),
        )
        rects = _hemi_rects_px(
            rgb_lh.shape,
            rgb_rh.shape,
            gap_px=max(0, int(args.pair_gap_px)),
            margin_px=max(0, int(args.pair_margin_px)),
            rh_shift_px=int(args.pair_rh_shift_px),
        )
        hemi_layers = {
            "canvas_h": int(rects["canvas_h"]),
            "canvas_w": int(rects["canvas_w"]),
            "lh": {
                "rect": rects["lh"],
                "curv": curv_lh_rgb,
                "curv_alpha": curv_alpha_lh,
                "stat": stat_lh_rgb,
                "alpha": alpha_lh,
                "roi": roi_lh_img,
            },
            "rh": {
                "rect": rects["rh"],
                "curv": curv_rh_rgb,
                "curv_alpha": curv_alpha_rh,
                "stat": stat_rh_rgb,
                "alpha": alpha_rh,
                "roi": roi_rh_img,
            },
        }
    else:
        if args.stat_file:
            vals_lh, vals_rh = _load_stat_file(
                args.stat_file, surfdir, subject, args.surftype, args.surfsuffix
            )
            vals = vals_lh if args.hemi == "lh" else vals_rh
        else:
            vals = _load_metric_or_fallback(
                subject, args.hemi, stat_metric, args.surftype, args.surfsuffix, surfdir
            )
        all_vals_for_clim = vals.ravel()
        vals_lh_data = vals if args.hemi == "lh" else None
        vals_rh_data = vals if args.hemi == "rh" else None
        if bool(args.symmetric_clim) and args.clim is None:
            m = float(np.nanmax(np.abs(vals)))
            if np.isfinite(m) and m > 0:
                common_kwargs["clim"] = (-m, m)
        mapped, lookup, _rgb, _options = lookup_images(
            subject,
            vals,
            args.hemi,
            view,
            **common_kwargs,
        )
        if args.roi_mask_file:
            roi_lh, roi_rh = _load_vertex_file_split(
                args.roi_mask_file, surfdir, subject, args.surftype, args.surfsuffix
            )
            roi_vals = roi_lh if args.hemi == "lh" else roi_rh
            roi_layer = vert_to_image(np.asarray(roi_vals), lookup, badval=np.nan)
        curv = read_surface_metric(subject, args.hemi, "curv", "", args.surfsuffix, **({"surfdir": surfdir} if surfdir else {}))
        curv_img = vert_to_image(np.asarray(curv, dtype=float) < 0, lookup, badval=np.nan)
        curv_layer = mat_to_rgb(curv_img, clim=(-1.0, 2.0), cmap="gray", rgbnan=1.0)
        curv_alpha_layer = _lookup_alpha(lookup)
        stat_layer = mat_to_rgb(mapped, clim=common_kwargs["clim"], cmap=args.cmap, rgbnan=1.0)
        alpha_layer = np.isfinite(mapped).astype(float)
        if args.absthreshold is not None:
            alpha_layer = alpha_layer * (np.abs(mapped) > float(args.absthreshold)).astype(float)
        rgba = _composite_rgba(curv_layer, curv_alpha_layer, stat_layer, alpha_layer)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    final_clim = tuple(float(x) for x in common_kwargs.get("clim", (-np.inf, np.inf)))
    if (not np.isfinite(final_clim[0]) or not np.isfinite(final_clim[1])) and all_vals_for_clim is not None:
        vv = np.asarray(all_vals_for_clim, dtype=float)
        final_clim = (float(np.nanmin(vv)), float(np.nanmax(vv)))

    if bool(args.show_insets):
        inset_kwargs = dict(common_kwargs)
        inset_kwargs["surftype"] = "inflated"
        inset_kwargs["imageres"] = int(args.inset_imageres)
        inset_kwargs["surfshading"] = False
        inset_kwargs["clim"] = final_clim
        inset_kwargs["xyextent"] = (1.0, 1.0)
        if vals_lh_data is not None:
            v_lh, _f_lh, _h_lh = lookup_viewpoint(
                subject, "lh", str(args.inset_viewname), "inflated"
            )
            _m_lh, _lk_lh, in_lh, _o_lh = lookup_images(
                subject, vals_lh_data, "lh", v_lh, **inset_kwargs
            )
            inset_lh = _crop_rgba_to_alpha(_rgb_with_alpha(in_lh, _lookup_alpha(_lk_lh)))
        if vals_rh_data is not None:
            v_rh, _f_rh, _h_rh = lookup_viewpoint(
                subject, "rh", str(args.inset_viewname), "inflated"
            )
            _m_rh, _lk_rh, in_rh, _o_rh = lookup_images(
                subject, vals_rh_data, "rh", v_rh, **inset_kwargs
            )
            inset_rh = _crop_rgba_to_alpha(_rgb_with_alpha(in_rh, _lookup_alpha(_lk_rh)))

    if rgba is None:
        raise RuntimeError("No rendered image was produced")

    _render_and_save(
        out,
        rgba,
        show_colorbar=bool(args.show_colorbar),
        cmap_name=str(args.cmap),
        clim=final_clim,
        inset_lh=inset_lh,
        inset_rh=inset_rh,
        inset_size_frac=float(args.inset_size_frac),
        inset_margin_px=int(args.inset_margin_px),
    )
    out_svg = Path(args.out_svg) if args.out_svg else (out if out.suffix.lower() == ".svg" else None)
    if (
        out_svg is not None
        and curv_layer is not None
        and curv_alpha_layer is not None
        and stat_layer is not None
        and alpha_layer is not None
    ):
        out_svg.parent.mkdir(parents=True, exist_ok=True)
        _render_layered_svg(
            out_svg,
            curv_rgb=curv_layer,
            curv_alpha=curv_alpha_layer,
            stat_rgb=stat_layer,
            stat_alpha=alpha_layer,
            show_colorbar=bool(args.show_colorbar),
            cmap_name=str(args.cmap),
            clim=final_clim,
            roi_color=str(args.roi_color),
            roi_width=float(args.roi_width),
            draw_roi_names=bool(args.draw_roi_names),
            roi_names=roi_names_list,
            roi_label_color=str(args.roi_label_color),
            roi_label_size=float(args.roi_label_size),
            roi_img=roi_layer,
            roi_hemi=str(args.hemi),
            inset_lh=inset_lh,
            inset_rh=inset_rh,
            inset_size_frac=float(args.inset_size_frac),
            inset_margin_px=int(args.inset_margin_px),
            hemi_layers=hemi_layers,
        )
    print(str(out.resolve()))


if __name__ == "__main__":
    main()
