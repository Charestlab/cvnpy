"""Regression tests for rendering and lookup correctness."""

import nibabel.freesurfer.io as fs
import numpy as np
import pytest

from cvnpy.core.lookup_images import _cache_file_path, _merge_options, lookup_images
from cvnpy.lookup.generate_sphere import generate_sphere_lookup
from cvnpy.lookup.generate_surface import generate_surface_lookup
from cvnpy.lookup.map import image_to_vert
from cvnpy.render.overlay import _as_rgb


def layers():
    yz = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
    vertices = np.concatenate([np.c_[np.ones(4), yz], np.c_[-np.ones(4), yz]])
    return vertices, np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])


@pytest.mark.parametrize("az,visible", [(90, set(range(4))), (-90, set(range(4, 8)))])
def test_camera_selects_front_layer(az, visible):
    verts, faces = layers()
    lk = generate_surface_lookup(verts, faces, az, 0, 0, imgheight=31)
    assert set(lk["imglookup"][~lk["extrapmask"]]) == visible


def test_cache_isolated_by_subject(tmp_path):
    opts = _merge_options({"cache_dir": str(tmp_path)})
    assert _cache_file_path("A", "lh", [90, 0, 0], opts) != _cache_file_path(
        "B", "lh", [90, 0, 0], opts
    )


def test_cache_retains_fractional_angle(tmp_path):
    opts = _merge_options({"cache_dir": str(tmp_path)})
    assert _cache_file_path("A", "lh", [90.1, 0, 0], opts) != _cache_file_path(
        "A", "lh", [90.2, 0, 0], opts
    )


def test_cache_retains_margin(tmp_path):
    a = _merge_options({"cache_dir": str(tmp_path), "flat_margin_frac": 0})
    b = {**a, "flat_margin_frac": 0.2}
    assert _cache_file_path("A", "lh", [0, 0, 0], a) != _cache_file_path(
        "A", "lh", [0, 0, 0], b
    )


def test_hidden_vertices_stay_unmapped():
    verts, faces = layers()
    lk = generate_surface_lookup(verts, faces, 90, 0, 0, imgheight=31)
    assert np.all(lk["reverselookup"][4:] == -1)


def test_edge_on_triangle_covers_no_area():
    verts = np.array([[0, -1, -1], [0, 0, 0], [0, 1, 1]], float)
    lk = generate_surface_lookup(verts, np.array([[0, 1, 2]]), 90, 0, 0, imgheight=31)
    assert np.all(lk["extrapmask"])


def test_sphere_masks_outside_silhouette():
    verts = np.array(
        [[0, 0, 1], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], float
    )
    lk = generate_sphere_lookup(verts, 0, 0, 0, (1, 1), 31)
    assert lk["extrapmask"][0, 0]


def test_background_does_not_assign_vertex_zero():
    lk = {
        "imglookup": np.array([[0, 1]]),
        "extrapmask": np.array([[True, False]]),
        "is_extrapolated": True,
        "reverselookup": np.array([-1, 1]),
        "vertsN": 2,
    }
    out = image_to_vert(np.array([[77.0, 1.0]]), lk, badval=np.nan)
    assert np.isnan(out[0])


def test_named_roi_color():
    assert np.array_equal(_as_rgb("red"), [1, 0, 0])


def test_bilateral_vertex_alpha(tmp_path):
    verts, faces = layers()
    for hemi in ["lh", "rh"]:
        fs.write_geometry(str(tmp_path / f"{hemi}.inflated"), verts, faces)
    lookup_images(
        "test",
        np.arange(16.0),
        ["lh", "rh"],
        "medial",
        surfdir=str(tmp_path),
        surftype="inflated",
        imageres=31,
        cache_dir=str(tmp_path / "cache"),
        overlayalpha=np.ones(16),
        background=0,
    )


def test_changed_geometry_invalidates_cache(tmp_path):
    verts, faces = layers()
    path = tmp_path / "lh.inflated"
    fs.write_geometry(str(path), verts, faces)
    kwargs = {
        "surfdir": str(tmp_path),
        "surftype": "inflated",
        "imageres": 31,
        "cache_dir": str(tmp_path / "cache"),
        "clim": (0, 7),
        "surfshading": False,
    }
    before, _, _, _ = lookup_images("test", np.arange(8.0), "lh", [90, 0, 0], **kwargs)
    # Move the back plane in front without changing vertex/face counts.
    verts[4:, 0] = 2
    fs.write_geometry(str(path), verts, faces)
    after, _, _, _ = lookup_images("test", np.arange(8.0), "lh", [90, 0, 0], **kwargs)
    assert np.nanmax(before) < 4
    assert np.nanmin(after) >= 4
    assert len(list((tmp_path / "cache").glob("*.pkl"))) == 2


def test_subject_geometry_cache_isolation(tmp_path):
    verts, faces = layers()
    for subject, flip in [("A", 1), ("B", -1)]:
        surf = tmp_path / subject
        surf.mkdir()
        v = verts.copy()
        v[:, 0] *= flip
        fs.write_geometry(str(surf / "lh.inflated"), v, faces)
        mapped, _, _, _ = lookup_images(
            subject,
            np.arange(8.0),
            "lh",
            [90, 0, 0],
            surfdir=str(surf),
            surftype="inflated",
            imageres=31,
            cache_dir=str(tmp_path / "cache"),
            clim=(0, 7),
            surfshading=False,
        )
        assert np.nanmax(mapped) < 4 if subject == "A" else np.nanmin(mapped) >= 4


def test_hidden_vertex_coordinates_are_missing():
    from cvnpy.lookup.map import vertidx_to_imagexy

    verts, faces = layers()
    lk = generate_surface_lookup(verts, faces, 90, 0, 0, imgheight=31)
    assert np.all(np.isnan(vertidx_to_imagexy(np.arange(4, 8), lk)))
    assert np.all(np.isfinite(vertidx_to_imagexy(np.arange(4), lk)))


def test_sphere_retains_foreground_and_masks_background():
    from cvnpy.lookup.map import vert_to_image

    verts = np.array(
        [[0, 0, 1], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], float
    )
    lk = generate_sphere_lookup(verts, 0, 0, 0, (1, 1), (31, 41))
    mapped = vert_to_image(np.ones(6), lk, badval=np.nan)
    assert mapped.shape == (31, 41)
    assert mapped[15, 20] == 1
    assert np.isnan(mapped[0, 0])


def test_bilateral_options_match_separate_hemispheres(tmp_path):
    verts, faces = layers()
    for hemi in ["lh", "rh"]:
        fs.write_geometry(str(tmp_path / f"{hemi}.inflated"), verts, faces)
    values = np.linspace(-1, 1, 16)
    alpha = np.linspace(0.1, 0.9, 16)
    background = np.linspace(0, 1, 16)
    roi = (np.arange(16) % 3 == 0).astype(float)
    options = {
        "surfdir": str(tmp_path),
        "surftype": "inflated",
        "imageres": 31,
        "cache_dir": str(tmp_path / "cache"),
        "clim": (-1, 1),
        "surfshading": False,
        "roicolor": "red",
    }
    _, _, both, _ = lookup_images(
        "test",
        values,
        ["lh", "rh"],
        "medial",
        overlayalpha=alpha,
        background=background,
        roimask=[roi],
        **options,
    )
    separate = []
    for i, hemi in enumerate(["lh", "rh"]):
        sl = slice(i * 8, (i + 1) * 8)
        _, _, rgb, _ = lookup_images(
            "test",
            values[sl],
            hemi,
            "medial",
            overlayalpha=alpha[sl],
            background=background[sl],
            roimask=[roi[sl]],
            **options,
        )
        separate.append(rgb)
    np.testing.assert_array_equal(both, np.concatenate(separate, axis=1))


def test_degenerate_face_cannot_overwrite_valid_surface():
    verts, faces = layers()
    baseline = generate_surface_lookup(verts, faces, -90, 0, 15, imgheight=31)
    # A collinear foreground face used to paint depth zero over the back plane.
    vertices = np.concatenate([verts, [[0, -0.8, -0.8], [0, 0, 0], [0, 0.8, 0.8]]])
    triangles = np.concatenate([faces, [[8, 9, 10]]])
    result = generate_surface_lookup(vertices, triangles, -90, 0, 15, imgheight=31)
    np.testing.assert_array_equal(result["imglookup"], baseline["imglookup"])
    np.testing.assert_array_equal(result["extrapmask"], baseline["extrapmask"])


def test_missing_source_does_not_reuse_unverifiable_cache(tmp_path):
    verts, faces = layers()
    path = tmp_path / "lh.inflated"
    fs.write_geometry(str(path), verts, faces)
    options = {
        "surfdir": str(tmp_path),
        "surftype": "inflated",
        "imageres": 31,
        "cache_dir": str(tmp_path / "cache"),
    }
    lookup_images("test", np.arange(8.0), "lh", "medial", **options)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        lookup_images("test", np.arange(8.0), "lh", "medial", **options)
