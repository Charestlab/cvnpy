# MATLAB to Python Parity Map

This file tracks where core CVN MATLAB responsibilities currently live in `cvnpy`.

## Main Rendering Path

- MATLAB: `cvnlookupimages.m`
- Python: `src/cvnpy/core/lookup_images.py`

## Viewpoint Resolution

- MATLAB: `cvnlookupviewpoint.m`
- Python: `src/cvnpy/view/viewpoints.py`

## Surface IO

- MATLAB: `cvnreadsurface.m`
- Python: `src/cvnpy/io/surfaces.py`

## Metric IO

- MATLAB: `cvnreadsurfacemetric.m`
- Python: `src/cvnpy/io/metrics.py`

## Lookup Generators

- Sphere lookup family (`spherelookup_*`):
  - `src/cvnpy/lookup/generate_sphere.py`
- Flat lookup generation:
  - `src/cvnpy/lookup/generate_flat.py`
- Non-sphere surface lookup:
  - `src/cvnpy/lookup/generate_surface.py`
- Vertex-to-image transfer:
  - `src/cvnpy/lookup/map.py`

## Colormap Parity

- MATLAB source: `knkutils/colormap/cmapsign4.m`
- Python port: `src/cvnpy/render/colormap.py`

## Demo Entrypoints

- Flatmap end-to-end demo:
  - `examples/flatmap_demo.py`
- Inflated demo:
  - `examples/inflated_demo.py`

## Remaining Gaps

- Pixel-perfect MATLAB parity is not guaranteed for every subject/view.
- Additional parity tests against MATLAB goldens are still needed.
- Packaging/release docs for public distribution are not finalized.
