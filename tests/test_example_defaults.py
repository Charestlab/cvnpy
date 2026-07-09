from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_production_flatmap_rotation_args_are_available() -> None:
    """Rotation remains an explicit layout choice, not part of value mapping."""
    flatmap_demo = (ROOT / "examples" / "flatmap_demo.py").read_text()
    batch_demo = (ROOT / "examples" / "render_all_layers_svg.py").read_text()

    assert '"--lh-rotate-deg"' in flatmap_demo
    assert '"--rh-rotate-deg"' in flatmap_demo
    assert '"--lh-rotate-deg"' in batch_demo
    assert '"--rh-rotate-deg"' in batch_demo
