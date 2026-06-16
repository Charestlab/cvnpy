# Getting Started

## Requirements

- Python `>=3.10`
- FreeSurfer-compatible surface files (`surf/`)
- macOS/Linux supported (tested on macOS ARM)

## Installation

```bash
cd /Users/iancharest/Documents/codex/cvncode/cvnpy
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e '.[dev]'
```

## Data Discovery Modes

You can provide surface data in one of two ways.

### 1) `SUBJECTS_DIR` + `--subject`

```bash
export SUBJECTS_DIR=/Applications/freesurfer/8.2.0/subjects
.venv/bin/python examples/inflated_demo.py --subject fsaverage --hemi lh --out out/lh_inflated.png
```

### 2) `--subject-dir`

```bash
.venv/bin/python examples/inflated_demo.py \
  --subject-dir /Users/iancharest/data/nsd_fsaverage \
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
.venv/bin/python examples/flatmap_demo.py \
  --subject-dir /Users/iancharest/data/nsd_fsaverage \
  --hemi lh \
  --out out/smoke_flat_lh.png
```

Success criteria:
- command exits without traceback
- output PNG exists
- output is non-empty and visibly cortical (not blank or all-white)
