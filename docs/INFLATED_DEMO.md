# Inflated Demo Reference

File: `/Users/iancharest/Documents/codex/cvncode/cvnpy/examples/inflated_demo.py`

## Purpose

`inflated_demo.py` renders a single-hemisphere inflated surface image.

## Basic Usage

```bash
.venv/bin/python examples/inflated_demo.py \
  --subject-dir /Users/iancharest/data/nsd_fsaverage \
  --hemi lh \
  --surftype inflated \
  --viewname occip \
  --out out/lh_inflated_occip.png
```

## CLI Highlights

- `--subject`, `--subject-dir`, `--subjects-dir`
- `--hemi {lh,rh}`
- `--surftype` (default `inflated`)
- `--surfsuffix`
- `--metric` (default `curv`, fallback random if missing)
- `--view az,el,tilt` or `--viewname`
- `--imageres`
- `--cache-dir`
- `--no-cache`

## Example with system FreeSurfer + writable cache

```bash
.venv/bin/python examples/inflated_demo.py \
  --subject fsaverage \
  --subjects-dir /Applications/freesurfer/8.2.0/subjects \
  --cache-dir /tmp/cvnpy_cache \
  --hemi rh \
  --out out/fsaverage_rh_inflated.png
```
