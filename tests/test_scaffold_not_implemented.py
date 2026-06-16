from __future__ import annotations

import numpy as np
import pytest

from cvnpy.core.lookup_images import lookup_images


def test_lookup_images_raises_for_missing_surface_file(tmp_path, monkeypatch: object) -> None:
    (tmp_path / "subj01" / "surf").mkdir(parents=True)
    monkeypatch.setenv("SUBJECTS_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError):
        lookup_images(
            "subj01",
            np.array([1.0, 2.0, 3.0]),
            "lh",
            [0, 0, 0],
            surftype="inflated",
            imageres=8,
        )
