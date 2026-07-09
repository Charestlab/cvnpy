from __future__ import annotations

from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np

from cvnpy.core.lookup_images import lookup_images
from cvnpy.lookup.generate_flat import generate_flat_lookup
from cvnpy.lookup.generate_surface import generate_surface_lookup
from cvnpy.lookup.generate_sphere import generate_sphere_lookup
from cvnpy.lookup.map import image_to_vert, vert_to_image


def _write_geom(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    fsio.write_geometry(str(path), verts.astype(float), faces.astype(np.int32))


def _make_subject_tree(tmp_path: Path, subject: str = "subj01") -> Path:
    surfdir = tmp_path / subject / "surf"
    surfdir.mkdir(parents=True)
    return surfdir


def test_generate_sphere_lookup_basic() -> None:
    verts = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    lk = generate_sphere_lookup(verts, 10, -40, 0, (1.0, 1.0), 32)

    assert lk["imglookup"].shape == (32, 32)
    assert lk["extrapmask"].shape == (32, 32)
    assert lk["reverselookup"].shape == (verts.shape[0],)
    assert lk["imglookup"].min() >= 0
    assert lk["imglookup"].max() < verts.shape[0]


def test_generate_flat_lookup_basic() -> None:
    verts = np.array(
        [
            [-1.0, -0.5, 0.0],
            [1.0, -0.5, 0.0],
            [1.0, 0.5, 0.0],
            [-1.0, 0.5, 0.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    lk = generate_flat_lookup(verts, faces, 0, 0, 0, (1.0, 1.0), 40)
    assert lk["imglookup"].shape[0] == 40
    assert lk["pixels_per_mm"] > 0
    assert lk["imglookup"].max() < verts.shape[0]


def test_generate_flat_lookup_masks_outside_triangles() -> None:
    # Single triangle should leave substantial outside area masked.
    verts = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    lk = generate_flat_lookup(verts, faces, 0, 0, 0, (1.0, 1.0), 64)
    extrap_frac = np.mean(lk["extrapmask"])
    assert extrap_frac > 0.1


def test_map_roundtrip_with_handcrafted_lookup() -> None:
    lookup = {
        "imglookup": np.array([[0, 1], [2, 3]], dtype=np.int64),
        "extrapmask": np.array([[False, False], [False, False]]),
        "is_extrapolated": False,
        "reverselookup": np.array([0, 1, 2, 3], dtype=np.int64),
        "vertsN": 4,
    }
    vals = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)

    img = vert_to_image(vals, lookup, badval=np.nan)
    assert np.allclose(img, np.array([[10.0, 20.0], [30.0, 40.0]]))

    back = image_to_vert(img, lookup, badval=np.nan)
    assert np.allclose(back, vals)


def test_vert_to_image_defaults_to_nearest_lookup_even_when_barycentric_present() -> None:
    lookup = {
        "imglookup": np.array([[0]], dtype=np.int64),
        "extrapmask": np.array([[False]], dtype=bool),
        "is_extrapolated": False,
        "bary_face": np.array([[0]], dtype=np.int64),
        "bary_faces": np.array([[0, 1, 2]], dtype=np.int64),
        "bary_weights": np.array([[[0.2, 0.3, 0.5]]], dtype=np.float32),
    }
    vals = np.array([10.0, 20.0, 40.0], dtype=float)
    img = vert_to_image(vals, lookup, badval=np.nan)
    assert img.shape == (1, 1)
    assert np.isclose(img[0, 0], 10.0)


def test_vert_to_image_uses_barycentric_lookup_when_explicitly_enabled() -> None:
    lookup = {
        "imglookup": np.array([[0]], dtype=np.int64),
        "extrapmask": np.array([[False]], dtype=bool),
        "is_extrapolated": False,
        "use_barycentric": True,
        "bary_face": np.array([[0]], dtype=np.int64),
        "bary_faces": np.array([[0, 1, 2]], dtype=np.int64),
        "bary_weights": np.array([[[0.2, 0.3, 0.5]]], dtype=np.float32),
    }
    vals = np.array([10.0, 20.0, 40.0], dtype=float)
    img = vert_to_image(vals, lookup, badval=np.nan)
    assert img.shape == (1, 1)
    assert np.isclose(img[0, 0], 0.2 * 10.0 + 0.3 * 20.0 + 0.5 * 40.0)


def test_lookup_images_sphere_integration(tmp_path: Path, monkeypatch: object) -> None:
    surfdir = _make_subject_tree(tmp_path)
    monkeypatch.setenv("SUBJECTS_DIR", str(tmp_path))

    verts = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [5, 1, 2]], dtype=np.int64)
    _write_geom(surfdir / "lh.sphere", verts, faces)

    vals = np.linspace(0, 1, verts.shape[0])
    mapped, lk, rgb, options = lookup_images(
        "subj01",
        vals,
        "lh",
        [10, -40, 0],
        surftype="sphere",
        imageres=24,
        xyextent=(1.0, 1.0),
    )

    assert mapped.shape == (24, 24)
    assert rgb.shape == (24, 24, 3)
    assert lk["imglookup"].shape == (24, 24)
    assert options["surftype"] == "sphere"


def test_lookup_images_flat_integration(tmp_path: Path, monkeypatch: object) -> None:
    surfdir = _make_subject_tree(tmp_path)
    monkeypatch.setenv("SUBJECTS_DIR", str(tmp_path))

    verts = np.array(
        [
            [-1.0, -0.5, 0.0],
            [1.0, -0.5, 0.0],
            [1.0, 0.5, 0.0],
            [-1.0, 0.5, 0.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    _write_geom(surfdir / "lh.full.flat.patch.3d", verts, faces)

    vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    mapped, lk, rgb, options = lookup_images(
        "subj01",
        vals,
        "lh",
        [0, 0, 0],
        surftype="full.flat.patch.3d",
        imageres=30,
        xyextent=(1.0, 1.0),
    )

    assert mapped.shape[0] == 30
    assert rgb.shape[-1] == 3
    assert "pixels_per_mm" in lk
    assert options["surftype"] == "full.flat.patch.3d"


def test_generate_surface_lookup_basic() -> None:
    verts = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    lk = generate_surface_lookup(verts, faces, 0, 0, 0, imgheight=32)
    assert lk["imglookup"].shape == (32, 32)
    assert lk["reverselookup"].shape == (4,)
    assert lk["shading"].shape == (32, 32)
    assert np.nanmin(lk["shading"]) >= 0.0
    assert np.nanmax(lk["shading"]) <= 1.0


def test_lookup_images_inflated_integration(tmp_path: Path, monkeypatch: object) -> None:
    surfdir = _make_subject_tree(tmp_path)
    monkeypatch.setenv("SUBJECTS_DIR", str(tmp_path))

    verts = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    _write_geom(surfdir / "lh.inflated", verts, faces)

    vals = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    mapped, lk, rgb, options = lookup_images(
        "subj01",
        vals,
        "lh",
        [0, 0, 0],
        surftype="inflated",
        imageres=28,
        surfshading=True,
    )

    assert mapped.shape == (28, 28)
    assert rgb.shape == (28, 28, 3)
    assert "shading" in lk
    assert options["surftype"] == "inflated"


def test_lookup_images_threshold_uses_background(tmp_path: Path, monkeypatch: object) -> None:
    surfdir = _make_subject_tree(tmp_path)
    monkeypatch.setenv("SUBJECTS_DIR", str(tmp_path))

    verts = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [5, 1, 2]], dtype=np.int64)
    _write_geom(surfdir / "lh.sphere", verts, faces)

    vals = np.linspace(0, 1, verts.shape[0])
    _mapped, _lk, rgb, _options = lookup_images(
        "subj01",
        vals,
        "lh",
        [10, -40, 0],
        surftype="sphere",
        imageres=22,
        threshold=10.0,  # hide all overlay
        background=0.2,
        bg_cmap="gray",
        bg_clim=(0.0, 1.0),
        surfshading=False,
    )

    assert np.allclose(rgb[:, :, 0], 0.2, atol=1e-6)
    assert np.allclose(rgb[:, :, 1], 0.2, atol=1e-6)
    assert np.allclose(rgb[:, :, 2], 0.2, atol=1e-6)


def test_lookup_images_roi_outline_draws_color(tmp_path: Path, monkeypatch: object) -> None:
    surfdir = _make_subject_tree(tmp_path)
    monkeypatch.setenv("SUBJECTS_DIR", str(tmp_path))

    verts = np.array(
        [
            [-1.0, -0.5, 0.0],
            [1.0, -0.5, 0.0],
            [1.0, 0.5, 0.0],
            [-1.0, 0.5, 0.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    _write_geom(surfdir / "lh.full.flat.patch.3d", verts, faces)

    vals = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    roimask = np.array([1, 1, 0, 0], dtype=float)
    _mapped, _lk, rgb, _options = lookup_images(
        "subj01",
        vals,
        "lh",
        [0, 0, 0],
        surftype="full.flat.patch.3d",
        imageres=30,
        roimask=roimask,
        roicolor=[1.0, 0.0, 0.0],
        roiwidth=1.0,
        surfshading=False,
    )

    red_pixels = (rgb[:, :, 0] > 0.95) & (rgb[:, :, 1] < 0.1) & (rgb[:, :, 2] < 0.1)
    assert np.any(red_pixels)


def test_lookup_cache_roundtrip(tmp_path: Path, monkeypatch: object) -> None:
    surfdir = _make_subject_tree(tmp_path)
    monkeypatch.setenv("SUBJECTS_DIR", str(tmp_path))

    verts = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [5, 1, 2]], dtype=np.int64)
    _write_geom(surfdir / "lh.sphere", verts, faces)
    vals = np.linspace(0, 1, verts.shape[0])

    cache_dir = tmp_path / "lookup_cache"
    _mapped1, _lk1, _rgb1, _opts1 = lookup_images(
        "subj01",
        vals,
        "lh",
        [10, -40, 0],
        surftype="sphere",
        imageres=20,
        cache_dir=str(cache_dir),
        savelookup=True,
    )
    cache_files = list(cache_dir.glob("*.pkl"))
    assert len(cache_files) == 1

    # Remove source geometry: second run should still work from cache.
    (surfdir / "lh.sphere").unlink()
    mapped2, _lk2, _rgb2, _opts2 = lookup_images(
        "subj01",
        vals,
        "lh",
        [10, -40, 0],
        surftype="sphere",
        imageres=20,
        cache_dir=str(cache_dir),
        reset=False,
    )
    assert mapped2.shape == (20, 20)
