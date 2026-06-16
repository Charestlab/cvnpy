# MATLAB to Python Parity Map

This file tracks where core CVN MATLAB responsibilities currently live in `cvnpy`.

## Main Rendering Path

- MATLAB: `cvnlookupimages.m`
- Python: `/Users/iancharest/Documents/codex/cvncode/cvnpy/src/cvnpy/core/lookup_images.py`

## Viewpoint Resolution

- MATLAB: `cvnlookupviewpoint.m`
- Python: `/Users/iancharest/Documents/codex/cvncode/cvnpy/src/cvnpy/view/viewpoints.py`

## Surface IO

- MATLAB: `cvnreadsurface.m`
- Python: `/Users/iancharest/Documents/codex/cvncode/cvnpy/src/cvnpy/io/surfaces.py`

## Metric IO

- MATLAB: `cvnreadsurfacemetric.m`
- Python: `/Users/iancharest/Documents/codex/cvncode/cvnpy/src/cvnpy/io/metrics.py`

## Lookup Generators

- Sphere lookup family (`spherelookup_*`):
  - `/Users/iancharest/Documents/codex/cvncode/cvnpy/src/cvnpy/lookup/generate_sphere.py`
- Flat lookup generation:
  - `/Users/iancharest/Documents/codex/cvncode/cvnpy/src/cvnpy/lookup/generate_flat.py`
- Non-sphere surface lookup:
  - `/Users/iancharest/Documents/codex/cvncode/cvnpy/src/cvnpy/lookup/generate_surface.py`
- Vertex-to-image transfer:
  - `/Users/iancharest/Documents/codex/cvncode/cvnpy/src/cvnpy/lookup/map.py`

## Colormap Parity

- MATLAB source: `knkutils/colormap/cmapsign4.m`
- Python port: `/Users/iancharest/Documents/codex/cvncode/cvnpy/src/cvnpy/render/colormap.py`

## Demo Entrypoints

- Flatmap end-to-end demo:
  - `/Users/iancharest/Documents/codex/cvncode/cvnpy/examples/flatmap_demo.py`
- Inflated demo:
  - `/Users/iancharest/Documents/codex/cvncode/cvnpy/examples/inflated_demo.py`

## Remaining Gaps

- Pixel-perfect MATLAB parity is not guaranteed for every subject/view.
- Additional parity tests against MATLAB goldens are still needed.
- Packaging/release docs for public distribution are not finalized.
