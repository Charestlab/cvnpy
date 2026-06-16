# Flatmap Demo Reference

File: `/Users/iancharest/Documents/codex/cvncode/cvnpy/examples/flatmap_demo.py`

## Purpose

`flatmap_demo.py` renders cortical flatmaps with optional stat overlays, thresholding, colorbar, bilateral composition, inflated insets, and layered SVG output.

## Common Commands

### Flatmap, both hemispheres

```bash
.venv/bin/python examples/flatmap_demo.py \
  --subject-dir /Users/iancharest/data/nsd_fsaverage \
  --hemi both \
  --surftype full.flat.patch.3d \
  --view 0,0,0 \
  --out out/fullflat_bihemi.png
```

### Real stat file, symmetric color limits, threshold

```bash
.venv/bin/python examples/flatmap_demo.py \
  --subject-dir /Users/iancharest/data/nsd_fsaverage \
  --hemi both \
  --stat-file /Users/iancharest/Downloads/rcnn-mpnet/encoding/subj01/shared_pred/test_corr.npy \
  --symmetric-clim \
  --absthreshold 0.05 \
  --show-colorbar \
  --out out/subj01_corr_thresh005.png
```

### Add insets + layered SVG

```bash
.venv/bin/python examples/flatmap_demo.py \
  --subject-dir /Users/iancharest/data/nsd_fsaverage \
  --hemi both \
  --stat-file /Users/iancharest/Downloads/rcnn-mpnet/encoding/subj01/shared_pred/test_corr.npy \
  --show-colorbar \
  --show-insets \
  --out out/subj01_corr_insets.png \
  --out-svg out/subj01_corr_insets.svg
```

## Inputs

### Subject / surf resolution

- `--subject` and optional `--subjects-dir`
- or `--subject-dir` (preferred for explicit local data)

### Statistical values

- `--stat-map <metricname>`: reads surface metric by name
- `--stat-file <path.npy>`: loads values from numpy file

`--stat-file` shape expectations:
- `n_lh + n_rh`: split into LH and RH
- `n_lh`: LH only
- `n_rh`: RH only

## Core Visual Options

- `--cmap` (default `cmapsign4`)
- `--clim min,max`
- `--symmetric-clim/--no-symmetric-clim`
- `--absthreshold <float>`
- `--background <metric-or-scalar>` (default `curv`)
- `--show-colorbar/--no-show-colorbar`

Threshold behavior:
- pixels with `abs(value) <= absthreshold` render background curvature (gray)
- above threshold render stat colormap

## Bilateral Layout

When `--hemi both`:
- LH and RH are independently rendered, rotated, trimmed, then composed side-by-side
- defaults:
  - `--lh-rotate-deg -90`
  - `--rh-rotate-deg 90`

Useful controls:
- `--pair-gap-px`
- `--pair-margin-px`
- `--pair-bg-gray` (for white figure use `1.0`)

## Insets

- `--show-insets`
- `--inset-viewname` (default `lateral`)
- `--inset-imageres`
- `--inset-size-frac`
- `--inset-margin-px`

Insets are rendered from `surftype='inflated'` using the same stat values and color settings.

## Flatmap Artifact Controls

These are exposed for boundary/edge artifact tuning:
- `--flat-support-ratio`
- `--trim-extrapolation-px`
- `--flat-prune-artifact-faces`
- `--flat-prune-quantile`
- `--flat-prune-factor`
- `--flat-filter-sparse-vertices`
- `--flat-sparse-vertex-quantile`
- `--flat-sparse-vertex-factor`
- `--flat-edge-trim-steps`
- `--flat-margin-frac`
- `--flat-fill-small-holes-px`

## SVG Layering

Use `--out-svg` for layered export. The output includes IDs for:
- `curv`
- `stat`
- `labels_lh`
- `labels_rh`
- `colorbar`
- `inset_lh`, `inset_rh` (if enabled)

This enables downstream vector overlays for ROI workflows.
