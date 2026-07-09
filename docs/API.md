# Python API Reference

## Public API

Primary user-facing entry point:

- `cvnpy.lookup(...)` in `/Users/iancharest/Documents/codex/cvncode/cvnpy/src/cvnpy/api.py`

This is a thin wrapper around:
- `cvnpy.core.lookup_images.lookup_images(...)`

## `lookup_images` Signature

```python
lookup_images(
    subject,
    vals,
    hemi,
    view_az_el_tilt,
    lookup=None,
    **kwargs,
)
```

Returns:

```python
(mapped, lookup, rgb, options)
```

- `mapped`: image-space scalar map (NaN where invalid/outside)
- `lookup`: lookup dictionary used/generated
- `rgb`: rendered RGB image
- `options`: resolved option dictionary

## Key kwargs

Geometry / viewpoint:
- `surftype` (`sphere`, `inflated`, `full.flat.patch.3d`, etc.)
- `surfsuffix`
- `xyextent`
- `imageres`

Color / mapping:
- `cmap` (default `cmapsign4`)
- `clim`
- `rgbnan`
- `background`, `bg_cmap`, `bg_clim`

Overlay / threshold:
- `threshold`
- `absthreshold`
- `overlayrange`
- `overlayalpha`

ROI:
- `roimask`
- `roicolor`
- `roiwidth`

Caching:
- `savelookup`
- `reset`
- `cache_dir`

Flatmap controls:
- `flat_margin_frac`

`flat_margin_frac` controls padding around the patch view.

The flatmap value path is nearest-neighbor by design and matches MATLAB `spherelookup_vert2image.m`.

## Minimal API Example

```python
import numpy as np
from cvnpy import lookup

vals = np.random.randn(10000)
mapped, lk, rgb, opts = lookup(
    subject="fsaverage",
    vals=vals,
    hemi="lh",
    view=[0, 0, 0],
    surftype="full.flat.patch.3d",
    cmap="cmapsign4",
    cache_dir="/tmp/cvnpy_cache",
)
```

## Example Figure

The example demos can export publication-style PNG and layered SVG outputs:

![Example bilateral flatmap](assets/mid_pred_flatmap.png)

## Viewpoint Helper

Use `/Users/iancharest/Documents/codex/cvncode/cvnpy/src/cvnpy/view/viewpoints.py`:

```python
from cvnpy.view.viewpoints import lookup_viewpoint

view, flip, viewhemi = lookup_viewpoint("fsaverage", "lh", "occip", "inflated")
```

## IO Helpers

Surface loading:
- `cvnpy.io.surfaces.read_surface(...)`

Metric loading:
- `cvnpy.io.metrics.read_surface_metric(...)`
