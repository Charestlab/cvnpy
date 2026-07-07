from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_production_flatmap_rotation_defaults_are_not_zero() -> None:
    """Guard against regressing the Kendrick/NSD fsaverage flatmap orientation."""
    flatmap_demo = (ROOT / "examples" / "flatmap_demo.py").read_text()
    batch_demo = (ROOT / "examples" / "render_all_layers_svg.py").read_text()

    assert '"--lh-rotate-deg",\n        type=int,\n        default=-90' in flatmap_demo
    assert '"--rh-rotate-deg",\n        type=int,\n        default=90' in flatmap_demo
    assert '"--lh-rotate-deg",\n        type=int,\n        default=-90' in batch_demo
    assert '"--rh-rotate-deg",\n        type=int,\n        default=90' in batch_demo
