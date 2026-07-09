# Changelog

## 0.1.3 - 2026-07-09

- Remove obsolete flatmap artifact-tuning options and the associated dead code paths.
- Simplify flat patch lookup generation to the production MATLAB-parity path.
- Remove unused flatmap barycentric image mapping and keep nearest-vertex lookup as the only value-mapping path.
- Add Read the Docs example figures and improve option documentation.
- Bump flat lookup cache version because the lookup dictionary schema changed.

## 0.1.2 - 2026-07-09

- Restore MATLAB-parity nearest-vertex image mapping as the default for `vert_to_image`.
- Keep barycentric image mapping available only as an explicit opt-in via `lookup["use_barycentric"] = True`.
- Fix optional flatmap barycentric rasterization coordinates so explicit barycentric mapping uses the same pixel grid as the nearest lookup.

## 0.1.1 - 2026-07-07

- Fix packaged production flatmap defaults: LH now rotates 90 degrees clockwise and RH rotates 90 degrees counterclockwise by default.
- Add a regression test for bilateral flatmap rotation defaults.

## 0.1.0 - 2026-06-16

- Initial Python package for CVN-style flatmap and inflated-surface visualization.
- Adds flatmap/inflated demos, stat overlays, symmetric color limits, colorbars, insets, ROI/label SVG overlays, and layered SVG export.
