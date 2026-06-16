from __future__ import annotations

import numpy as np

from cvnpy.render.colormap import mat_to_rgb


def test_cmapsign4_endpoints() -> None:
    vals = np.array([0.0, 1.0], dtype=float)
    rgb = mat_to_rgb(vals, cmap="cmapsign4", clim=(0.0, 1.0), rgbnan=1.0)
    # cmapsign4 anchors: start cyan-white, end yellow-white
    assert np.allclose(rgb[0], np.array([0.8, 1.0, 1.0]), atol=1e-6)
    assert np.allclose(rgb[1], np.array([1.0, 1.0, 0.8]), atol=1e-6)

