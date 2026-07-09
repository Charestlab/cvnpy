# Troubleshooting

## 1) `SUBJECTS_DIR is not set`

Error:

```text
ValueError: SUBJECTS_DIR is not set and no surfdir was provided
```

Fix:
- set `SUBJECTS_DIR`, or
- pass `--subject-dir /path/to/subject`

## 2) FreeSurfer install is read-only (cache write failures)

Symptom:
- caching under `/Applications/.../surf/imglookup` fails silently or cannot write

Fix:
- pass `--cache-dir /tmp/cvnpy_cache`

## 3) `read_geometry` reshape errors on some `fsaverage` files

Example:

```text
ValueError: cannot reshape array of size 0 into shape (...)
```

Cause:
- incompatible/legacy geometry format for a given reader path

Fix:
- use NSD/CVN-compatible fsaverage surfaces (e.g., custom subject-dir)
- verify surface files are valid and complete for both hemispheres

## 4) Flatmap touches image border / appears cropped

Fix:
- increase `--flat-margin-frac` (e.g. `0.06`)
- increase `--pair-margin-px` for bilateral output

## 5) Renders look stale after updating `cvnpy`

Fix:
- rerun once with `--no-cache`
- or use a fresh `--cache-dir`

## 6) Colorbar labels or limits look wrong

Fix:
- set explicit `--clim min,max`
- or use `--symmetric-clim` for signed effects

## 7) `.npy` stat file length mismatch

Error indicates expected lengths (`n_lh`, `n_rh`, `n_lh+n_rh`).

Fix:
- ensure vector length matches one of those forms
- if concatenated, order must be `LH` then `RH`

## 8) SVG opens but layers are not easy to target

Fix:
- export with `--out-svg`
- inspect element IDs (`curv`, `stat`, `colorbar`, `labels_lh`, `labels_rh`, insets)
- use an SVG editor/script that preserves group IDs
