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
- caching under a system-managed FreeSurfer subject directory fails or cannot write

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

## Lookup caches in 0.1.6

Version 0.1.6 uses lookup cache version 11. Older files are ignored and the first
render regenerates its lookup. Cache identity includes the subject, resolved
surface directory, geometry/patch file contents, and exact view, extent,
resolution, and flat-margin options. A shared cache directory can therefore
serve different meshes without reusing another subject's lookup.

Source geometry must remain available even on cache hits so its contents can be
verified. To render without reading geometry, explicitly pass a previously
returned `lookup` dictionary; responsibility for matching that dictionary to the
data and requested view then belongs to the caller.

## Hidden vertices and medial views

Reverse inflated-surface lookups leave occluded vertices unmapped (`-1`).
`vertidx_to_imagexy` returns NaN coordinates for them, and `image_to_vert` retains
`badval`. Nearby unsampled vertices belonging to visible faces can be recovered
within a two-pixel 3D distance; this is a conservative approximation, not exact
MATLAB reverse-lookup parity.

Medial/lateral camera presets and correlation signs are unchanged in 0.1.6.
Compare the same anatomical vertex labels across flat and inflated views: the
cut and projection can make neighboring regions look quite different. The
renderer does not automatically exclude the medial wall. Apply an appropriate
cortex mask to the values when those vertices should not be interpreted.
