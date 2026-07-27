from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FLATMAP_DEMO = ROOT / "examples" / "flatmap_demo.py"


def _load_flatmap_demo():
    spec = importlib.util.spec_from_file_location("flatmap_demo_for_tests", FLATMAP_DEMO)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_production_flatmap_rotation_args_are_available() -> None:
    """Rotation remains an explicit layout choice, not part of value mapping."""
    flatmap_demo = FLATMAP_DEMO.read_text()
    batch_demo = (ROOT / "examples" / "render_all_layers_svg.py").read_text()

    assert '"--lh-rotate-deg"' in flatmap_demo
    assert '"--rh-rotate-deg"' in flatmap_demo
    assert '"--lh-rotate-deg"' in batch_demo
    assert '"--rh-rotate-deg"' in batch_demo


def test_alpha_bbox_is_independent_of_stat_content() -> None:
    demo = _load_flatmap_demo()
    cortex_alpha = np.zeros((6, 7), dtype=float)
    cortex_alpha[1:5, 2:6] = 1.0

    sparse_stat_alpha = np.zeros_like(cortex_alpha)
    sparse_stat_alpha[3, 4] = 1.0
    dense_stat_alpha = cortex_alpha.copy()

    assert demo._alpha_bbox(cortex_alpha) == (1, 5, 2, 6)
    assert demo._alpha_bbox(cortex_alpha) == demo._alpha_bbox(dense_stat_alpha)
    assert demo._alpha_bbox(cortex_alpha) != demo._alpha_bbox(sparse_stat_alpha)


def test_composite_rgba_keeps_invalid_pixels_transparent() -> None:
    demo = _load_flatmap_demo()
    curv = np.full((2, 2, 3), 0.5, dtype=float)
    stat = np.ones((2, 2, 3), dtype=float)
    curv_alpha = np.array([[1.0, 0.0], [1.0, 1.0]])
    stat_alpha = np.ones((2, 2), dtype=float)

    rgba = demo._composite_rgba(curv, curv_alpha, stat, stat_alpha)

    assert rgba.shape == (2, 2, 4)
    assert rgba[0, 1, 3] == 0.0
    assert np.allclose(rgba[0, 0, :3], 1.0)
    assert np.allclose(rgba[1, 0, 3], 1.0)
