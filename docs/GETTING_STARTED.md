# Getting Started

## Requirements

- Python `>=3.10`
- FreeSurfer-compatible surface files (`surf/`)
- macOS/Linux supported (tested on macOS ARM)

## Installation

For library use:

```bash
python -m pip install cvnpy
```

To run the example scripts from the documentation:

```bash
git clone https://github.com/Charestlab/cvnpy.git
cd cvnpy
python -m pip install -e ".[dev,docs]"
```

## Data Discovery Modes

You can provide surface data in one of two ways.

### 1) `SUBJECTS_DIR` + `--subject`

```bash
export SUBJECTS_DIR=/path/to/freesurfer/subjects
python examples/inflated_demo.py --subject fsaverage --hemi lh --out out/lh_inflated.png
```

### 2) `--subject-dir`

```bash
python examples/inflated_demo.py \
  --subject-dir /path/to/nsd_fsaverage \
  --hemi lh \
  --out out/lh_inflated.png
```

`--subject-dir` should point at a folder that contains `surf/`.

## Cache Behavior

Lookup generation is cached by default.

- default cache location:
  - `SUBJECTS_DIR/<subject>/surf/imglookup`
  - or fallback temp cache if path is read-only
- override explicitly:
  - `--cache-dir /tmp/cvnpy_cache`
- disable cache writes/reads for a run:
  - `--no-cache`

## Smoke Test

```bash
python examples/flatmap_demo.py \
  --subject-dir /path/to/nsd_fsaverage \
  --hemi lh \
  --out out/smoke_flat_lh.png
```

Success criteria:
- command exits without traceback
- output PNG exists
- output is non-empty and visibly cortical (not blank or all-white)
