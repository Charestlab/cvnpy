# cvnpy

`cvnpy` is a Python port of key CVN visualization workflows from Kendrick Kay's MATLAB `cvncode`, focused on production-quality cortical flatmaps, inflated surfaces, statistical overlays, ROI/text overlays, and layered SVG export.

## Core Workflows

- Render full flatmaps using Kendrick-style fsaverage cuts.
- Render inflated FreeSurfer surfaces.
- Overlay per-vertex statistical maps with thresholding and symmetric color limits.
- Export layered SVGs with independent objects for curvature, stats, ROIs, labels, colorbars, and insets.
- Batch-render model-layer results for publication figures.

## Example

![cvnpy bilateral flatmap example](assets/mid_pred_flatmap.png)

## Install

```bash
python -m pip install cvnpy
```

The example scripts used in this documentation live in the GitHub repository. To run them directly:

```bash
git clone https://github.com/Charestlab/cvnpy.git
cd cvnpy
python -m pip install -e ".[dev,docs]"
```

## Data

`cvnpy` does not ship cortical surface data. Provide a FreeSurfer-style subject directory with a `surf/` folder, such as an NSD fsaverage derivative.

```bash
python examples/flatmap_demo.py \
  --subject-dir /path/to/nsd_fsaverage \
  --hemi both \
  --surftype full.flat.patch.3d \
  --out out/flatmap.png
```
