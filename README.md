# cvnpy

Python port of key CVN visualization workflows from `cvncode` (MATLAB), focused on production-quality:
- flatmap rendering (including Kendrick-style fsaverage cuts)
- inflated-surface rendering
- sphere/surface lookup compatibility
- layered SVG export for downstream overlays (ROIs/labels)

Documentation: <https://cvnpy.readthedocs.io/>

## Current Status

`cvnpy` is now functional for the visualization path and includes:
- lookup generation + caching
- flat/inflated demos
- stat-map overlays with thresholding
- symmetric color limits
- optional colorbar
- optional bilateral inflated insets
- layered SVG output with stable object IDs
- MATLAB-parity nearest-neighbor vertex-to-image mapping for flatmaps

## Repository Layout

- `src/cvnpy/core/lookup_images.py`: main rendering engine (`lookup_images`)
- `src/cvnpy/io/surfaces.py`: surface + patch loading
- `src/cvnpy/io/metrics.py`: per-vertex metric loading
- `src/cvnpy/view/viewpoints.py`: named viewpoints (sphere/non-sphere/flat)
- `src/cvnpy/render/colormap.py`: MATLAB-like colormap conversion (`cmapsign4`)
- `examples/flatmap_demo.py`: full flatmap pipeline (PNG + layered SVG)
- `examples/inflated_demo.py`: inflated-surface rendering demo
- `tests/`: unit tests for scaffold milestones and colormap behavior

## Install

From PyPI:

```bash
python -m pip install cvnpy
```

To run the example scripts from this repository:

```bash
git clone https://github.com/Charestlab/cvnpy.git
cd cvnpy
python -m pip install -e ".[dev,docs]"
```

For development and tests:

```bash
python -m pytest -q
```

## Data Setup

`cvnpy` needs a FreeSurfer-style subject `surf/` directory.

### Option A: Use `SUBJECTS_DIR`

```bash
export SUBJECTS_DIR=/path/to/freesurfer/subjects
```

Then pass `--subject fsaverage` (or your subject name).

### Option B: Use explicit subject directory

```bash
--subject-dir /path/to/nsd_fsaverage
```

This expects:

```text
/path/to/nsd_fsaverage/
  surf/
    lh.full.flat.patch.3d
    rh.full.flat.patch.3d
    lh.inflated
    rh.inflated
    lh.curv
    rh.curv
    ...
```

## Quick Start

### 1) Flatmap (single hemi)

```bash
python examples/flatmap_demo.py \
  --subject-dir /path/to/nsd_fsaverage \
  --hemi lh \
  --surftype full.flat.patch.3d \
  --view 0,0,0 \
  --cache-dir /tmp/cvnpy_cache \
  --out out/nsd_lh_flat.png
```

### 2) Flatmap (both hemispheres, production defaults)

```bash
python examples/flatmap_demo.py \
  --subject-dir /path/to/nsd_fsaverage \
  --hemi both \
  --surftype full.flat.patch.3d \
  --view 0,0,0 \
  --cache-dir /tmp/cvnpy_cache \
  --out out/nsd_bihemi_flat.png
```

### 3) Stat-map overlay from `.npy`

```bash
python examples/flatmap_demo.py \
  --subject-dir /path/to/nsd_fsaverage \
  --hemi both \
  --surftype full.flat.patch.3d \
  --stat-file /path/to/stats/test_corr.npy \
  --symmetric-clim \
  --absthreshold 0.05 \
  --show-colorbar \
  --cache-dir /tmp/cvnpy_cache \
  --out out/corr_bihemi.png
```

### 4) Add inflated insets

```bash
python examples/flatmap_demo.py \
  --subject-dir /path/to/nsd_fsaverage \
  --hemi both \
  --stat-file /path/to/stats/test_corr.npy \
  --absthreshold 0.05 \
  --show-colorbar \
  --show-insets \
  --inset-viewname lateral \
  --out out/corr_bihemi_insets.png
```

### 5) Layered SVG export

```bash
python examples/flatmap_demo.py \
  --subject-dir /path/to/nsd_fsaverage \
  --hemi both \
  --stat-file /path/to/stats/test_corr.npy \
  --show-colorbar \
  --out out/corr_bihemi.png \
  --out-svg out/corr_bihemi.svg
```

## Layered SVG IDs

The SVG contains stable IDs for overlay workflows:
- `layers`
- `curv`
- `stat`
- `labels_lh`
- `labels_rh`
- `colorbar`
- `inset_lh` and `inset_rh` (when enabled)

This is designed to support Kendrick-style downstream SVG overlays.

## Default Colormap

The default data colormap is `cmapsign4`, ported from:
- `knkutils/colormap/cmapsign4.m`

You can override with `--cmap`.

## Full Documentation

- [Getting Started](docs/GETTING_STARTED.md)
- [Flatmap Demo Reference](docs/FLATMAP_DEMO.md)
- [Inflated Demo Reference](docs/INFLATED_DEMO.md)
- [Python API Reference](docs/API.md)
- [MATLAB to Python Parity Map](docs/MATLAB_PARITY.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Development and Release](docs/DEVELOPMENT.md)

## Known Limitations

- Exact pixel-perfect parity with MATLAB outputs is still in progress.
- `cvnpy` does not ship FreeSurfer/NSD/CVN surface files; users must provide compatible subject data.

## Citation / Attribution

This project ports concepts and workflows from `cvncode` and related CVN tooling by Kendrick Kay and collaborators.
