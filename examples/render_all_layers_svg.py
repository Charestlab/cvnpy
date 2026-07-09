"""Batch-render layer-wise flatmaps as SVG using shared symmetric color limits.

This script:
1) discovers all layer `test_corr.npy` files under an encoding-subject directory,
2) computes one global max(abs(value)) across layers,
3) renders one production flatmap per layer via `flatmap_demo.py` with:
   - SVG export
   - inflated insets
   - symmetric colormap limits using the global max
   - no threshold (unless explicitly requested)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def _discover_stat_files(root: Path, pattern: str) -> list[Path]:
    files = sorted(p for p in root.rglob(pattern) if p.is_file())
    return files


def _finite_absmax(arr: np.ndarray) -> float:
    vals = np.asarray(arr, dtype=float).ravel()
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return 0.0
    return float(np.max(np.abs(finite)))


def _compute_global_absmax(files: list[Path]) -> tuple[float, list[tuple[Path, float]]]:
    layer_max: list[tuple[Path, float]] = []
    global_max = 0.0
    for f in files:
        arr = np.load(f)
        m = _finite_absmax(arr)
        layer_max.append((f, m))
        global_max = max(global_max, m)
    return global_max, layer_max


def _compute_global_abs_percentile(files: list[Path], percentile: float) -> float:
    if percentile <= 0 or percentile > 100:
        raise ValueError("global-clim-percentile must be in (0, 100]")
    all_abs: list[np.ndarray] = []
    for f in files:
        arr = np.asarray(np.load(f), dtype=float).ravel()
        arr = np.abs(arr[np.isfinite(arr)])
        if arr.size:
            all_abs.append(arr)
    if not all_abs:
        return 0.0
    vals = np.concatenate(all_abs, axis=0)
    return float(np.percentile(vals, percentile))


def _render_one(
    python_exe: str,
    flatmap_script: Path,
    subject_dir: Path,
    stat_file: Path,
    clim: tuple[float, float],
    out_png: Path,
    out_svg: Path,
    cache_dir: Path | None,
    surftype: str,
    view: str,
    imageres: int,
    inset_viewname: str,
    inset_imageres: int,
    inset_size_frac: float,
    inset_margin_px: int,
    lh_rotate_deg: int,
    rh_rotate_deg: int,
    pair_rh_shift_px: int,
    rh_flip_lr: bool,
    show_colorbar: bool,
    cmap: str | None,
    absthreshold: float | None,
    roi_mask_file: Path | None,
    roi_color: str,
    roi_width: float,
    draw_roi_names: bool,
    roi_names: str | None,
    roi_label_color: str,
    roi_label_size: float,
    no_cache: bool,
    overwrite: bool,
    dry_run: bool,
) -> None:
    if (not overwrite) and out_svg.exists():
        print(f"[skip] {out_svg}")
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        python_exe,
        str(flatmap_script),
        "--subject-dir",
        str(subject_dir),
        "--hemi",
        "both",
        "--surftype",
        surftype,
        "--view",
        view,
        "--imageres",
        str(int(imageres)),
        "--stat-file",
        str(stat_file),
        "--clim",
        f"{float(clim[0])},{float(clim[1])}",
        "--show-insets",
        "--inset-viewname",
        inset_viewname,
        "--inset-imageres",
        str(int(inset_imageres)),
        "--inset-size-frac",
        str(float(inset_size_frac)),
        "--inset-margin-px",
        str(int(inset_margin_px)),
        "--lh-rotate-deg",
        str(int(lh_rotate_deg)),
        "--rh-rotate-deg",
        str(int(rh_rotate_deg)),
        "--pair-rh-shift-px",
        str(int(pair_rh_shift_px)),
        "--rh-flip-lr" if rh_flip_lr else "--no-rh-flip-lr",
        "--out",
        str(out_png),
        "--out-svg",
        str(out_svg),
    ]

    if show_colorbar:
        cmd.append("--show-colorbar")

    if cmap:
        cmd.extend(["--cmap", str(cmap)])

    if absthreshold is not None:
        cmd.extend(["--absthreshold", str(float(absthreshold))])
    if roi_mask_file is not None:
        cmd.extend(["--roi-mask-file", str(roi_mask_file)])
    if roi_color:
        cmd.extend(["--roi-color", str(roi_color)])
    if roi_width is not None:
        cmd.extend(["--roi-width", str(float(roi_width))])
    cmd.append("--draw-roi-names" if draw_roi_names else "--no-draw-roi-names")
    if roi_names:
        cmd.extend(["--roi-names", str(roi_names)])
    if roi_label_color:
        cmd.extend(["--roi-label-color", str(roi_label_color)])
    if roi_label_size is not None:
        cmd.extend(["--roi-label-size", str(float(roi_label_size))])

    if cache_dir is not None:
        cmd.extend(["--cache-dir", str(cache_dir)])
    if no_cache:
        cmd.append("--no-cache")

    if dry_run:
        print("[dry-run] " + " ".join(cmd))
        return

    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch render all layer test_corr maps to SVG with shared symmetric clim."
    )
    parser.add_argument(
        "--encoding-dir",
        required=True,
        help="Directory containing one subfolder per layer/result.",
    )
    parser.add_argument(
        "--subject-dir",
        required=True,
        help="Surface subject dir containing surf/ (e.g., /path/to/nsd_fsaverage).",
    )
    parser.add_argument(
        "--pattern",
        default="test_corr.npy",
        help="Filename pattern to discover under encoding-dir (default: test_corr.npy).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for per-layer figures. Default: <encoding-dir>/flatmaps_svg.",
    )
    parser.add_argument(
        "--flatmap-script",
        default=None,
        help="Path to flatmap_demo.py. Default: examples/flatmap_demo.py next to this script.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable to run demo.")
    parser.add_argument("--surftype", default="full.flat.patch.3d")
    parser.add_argument("--view", default="0,0,0")
    parser.add_argument("--imageres", type=int, default=900)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--show-colorbar", action="store_true")
    parser.add_argument(
        "--cmap",
        default=None,
        help="Optional cmap passed to flatmap_demo.py (default uses flatmap_demo default).",
    )
    parser.add_argument(
        "--global-clim-mode",
        default="max",
        choices=["max", "percentile"],
        help="How to pick shared symmetric clim magnitude across layers.",
    )
    parser.add_argument(
        "--global-clim-percentile",
        type=float,
        default=99.5,
        help="Used when --global-clim-mode percentile.",
    )
    parser.add_argument(
        "--absthreshold",
        type=float,
        default=None,
        help="Optional threshold; default None (no threshold).",
    )
    parser.add_argument("--inset-viewname", default="lateral")
    parser.add_argument("--inset-imageres", type=int, default=320)
    parser.add_argument("--inset-size-frac", type=float, default=0.20)
    parser.add_argument("--inset-margin-px", type=int, default=14)
    parser.add_argument(
        "--lh-rotate-deg",
        type=int,
        default=0,
        help="LH flatmap rotation in degrees; negative values rotate clockwise.",
    )
    parser.add_argument(
        "--rh-rotate-deg",
        type=int,
        default=0,
        help="RH flatmap rotation in degrees; positive values rotate counterclockwise.",
    )
    parser.add_argument(
        "--pair-rh-shift-px",
        type=int,
        default=0,
        help="Extra rightward translation for RH in bilateral composition.",
    )
    parser.add_argument(
        "--rh-flip-lr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mirror RH flatmap left-right after rotation.",
    )
    parser.add_argument(
        "--roi-mask-file",
        default=None,
        help="Optional .npy ROI/parcellation in vertex space for SVG contours.",
    )
    parser.add_argument("--roi-color", default="white")
    parser.add_argument("--roi-width", type=float, default=1.0)
    parser.add_argument(
        "--draw-roi-names",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--roi-names",
        default=None,
        help="Optional comma-separated ROI names (label index 1..N).",
    )
    parser.add_argument("--roi-label-color", default="white")
    parser.add_argument("--roi-label-size", type=float, default=9.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Pass --no-cache to flatmap_demo.py for fresh lookup generation.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    encoding_dir = Path(args.encoding_dir).resolve()
    subject_dir = Path(args.subject_dir).resolve()
    if not encoding_dir.is_dir():
        raise FileNotFoundError(f"encoding-dir does not exist: {encoding_dir}")
    if not (subject_dir / "surf").is_dir():
        raise FileNotFoundError(f"subject-dir missing surf/: {subject_dir}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (encoding_dir / "flatmaps_svg")
    out_dir.mkdir(parents=True, exist_ok=True)

    this_dir = Path(__file__).resolve().parent
    flatmap_script = (
        Path(args.flatmap_script).resolve()
        if args.flatmap_script
        else (this_dir / "flatmap_demo.py").resolve()
    )
    if not flatmap_script.is_file():
        raise FileNotFoundError(f"flatmap script not found: {flatmap_script}")

    stat_files = _discover_stat_files(encoding_dir, args.pattern)
    if not stat_files:
        raise FileNotFoundError(
            f"No files matching '{args.pattern}' found under {encoding_dir}"
        )

    global_absmax, layer_stats = _compute_global_absmax(stat_files)
    if str(args.global_clim_mode) == "percentile":
        global_absmax = _compute_global_abs_percentile(
            stat_files, float(args.global_clim_percentile)
        )
    if not np.isfinite(global_absmax) or global_absmax <= 0:
        raise ValueError("Global abs max is not positive/finite; check input test_corr files")

    clim = (-global_absmax, global_absmax)

    print(f"Discovered {len(stat_files)} layer files")
    print(f"Global symmetric clim: [{clim[0]:.6g}, {clim[1]:.6g}]")

    summary_path = out_dir / "render_summary.tsv"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("layer\tstat_file\tlayer_absmax\tglobal_absmax\tout_svg\n")
        for stat_file, layer_max in layer_stats:
            layer_name = stat_file.parent.name
            out_png = out_dir / f"{layer_name}.png"
            out_svg = out_dir / f"{layer_name}.svg"

            _render_one(
                python_exe=args.python,
                flatmap_script=flatmap_script,
                subject_dir=subject_dir,
                stat_file=stat_file,
                clim=clim,
                out_png=out_png,
                out_svg=out_svg,
                cache_dir=Path(args.cache_dir).resolve() if args.cache_dir else None,
                surftype=str(args.surftype),
                view=str(args.view),
                imageres=int(args.imageres),
                inset_viewname=str(args.inset_viewname),
                inset_imageres=int(args.inset_imageres),
                inset_size_frac=float(args.inset_size_frac),
                inset_margin_px=int(args.inset_margin_px),
                lh_rotate_deg=int(args.lh_rotate_deg),
                rh_rotate_deg=int(args.rh_rotate_deg),
                pair_rh_shift_px=int(args.pair_rh_shift_px),
                rh_flip_lr=bool(args.rh_flip_lr),
                show_colorbar=bool(args.show_colorbar),
                cmap=args.cmap,
                absthreshold=args.absthreshold,
                roi_mask_file=Path(args.roi_mask_file).resolve() if args.roi_mask_file else None,
                roi_color=str(args.roi_color),
                roi_width=float(args.roi_width),
                draw_roi_names=bool(args.draw_roi_names),
                roi_names=args.roi_names,
                roi_label_color=str(args.roi_label_color),
                roi_label_size=float(args.roi_label_size),
                no_cache=bool(args.no_cache),
                overwrite=bool(args.overwrite),
                dry_run=bool(args.dry_run),
            )

            f.write(
                f"{layer_name}\t{stat_file}\t{layer_max:.10g}\t{global_absmax:.10g}\t{out_svg}\n"
            )

    print(f"Wrote summary: {summary_path}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
