from __future__ import annotations

from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np

from cvnpy.io.metrics import read_surface_metric
from cvnpy.io.surfaces import read_surface
from cvnpy.types import ValStruct
from cvnpy.view.viewpoints import lookup_viewpoint


def _make_subject_tree(tmp_path: Path, subject: str = "subj01") -> Path:
    surfdir = tmp_path / subject / "surf"
    surfdir.mkdir(parents=True)
    return surfdir


def _write_geom(path: Path, nverts: int = 4) -> None:
    verts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )[:nverts]
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    fsio.write_geometry(str(path), verts, faces)


def test_lookup_viewpoint_sphere_occip_single() -> None:
    viewpt, fliphemi, viewhemi = lookup_viewpoint("fsaverage", "lh", "occip", "sphere")
    assert viewpt == [10, -40, 0]
    assert fliphemi is False
    assert viewhemi == "lh"


def test_lookup_viewpoint_inflated_ventral_flips() -> None:
    viewpt, fliphemi, viewhemi = lookup_viewpoint(
        "fsaverage", ["lh", "rh"], "ventral", "inflated"
    )
    assert viewpt == [[0, -90, -90], [0, -90, -90]]
    assert fliphemi is True
    assert viewhemi == ["rh", "lh"]


def test_lookup_viewpoint_flat_defaults_zero() -> None:
    viewpt, fliphemi, _viewhemi = lookup_viewpoint(
        "fsaverage", ["lh", "rh"], "occip", "full.flat.patch.3d"
    )
    assert viewpt == [[0, 0, 0], [0, 0, 0]]
    assert fliphemi is False


def test_read_surface_justcount_and_full(tmp_path: Path, monkeypatch: object) -> None:
    surfdir = _make_subject_tree(tmp_path)
    _write_geom(surfdir / "lh.sphere")
    _write_geom(surfdir / "rh.sphere")

    monkeypatch.setenv("SUBJECTS_DIR", str(tmp_path))

    numlh, numrh = read_surface("subj01", ["lh", "rh"], "sphere", justcount=True)
    assert numlh == 4
    assert numrh == 4

    surf_lh = read_surface("subj01", "lh", "sphere")
    assert surf_lh.vertices.shape == (4, 3)
    assert surf_lh.faces.shape[1] == 3


def test_read_surface_with_suffix(tmp_path: Path, monkeypatch: object) -> None:
    surfdir = _make_subject_tree(tmp_path)
    _write_geom(surfdir / "lh.sphereDENSE")

    monkeypatch.setenv("SUBJECTS_DIR", str(tmp_path))

    surf = read_surface("subj01", "lh", "sphere", "DENSE")
    assert surf.vertices.shape[0] == 4


def test_read_metric_morph_and_mgz(tmp_path: Path, monkeypatch: object) -> None:
    surfdir = _make_subject_tree(tmp_path)
    monkeypatch.setenv("SUBJECTS_DIR", str(tmp_path))

    lh = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    rh = np.array([4.0, 5.0], dtype=np.float32)

    fsio.write_morph_data(str(surfdir / "lh.curv"), lh)
    img = nib.MGHImage(rh.reshape(-1, 1, 1), np.eye(4))
    nib.save(img, str(surfdir / "rh.curv.mgz"))

    out_lh = read_surface_metric("subj01", "lh", "curv")
    assert np.allclose(out_lh, lh)

    out_both = read_surface_metric("subj01", ["lh", "rh"], "curv")
    assert isinstance(out_both, ValStruct)
    assert out_both.numlh == 3
    assert out_both.numrh == 2
    assert np.allclose(out_both.data, np.concatenate([lh, rh]))


def test_read_metric_fallback_blank_surftype(tmp_path: Path, monkeypatch: object) -> None:
    surfdir = _make_subject_tree(tmp_path)
    monkeypatch.setenv("SUBJECTS_DIR", str(tmp_path))

    vals = np.array([7.0, 8.0], dtype=np.float32)
    fsio.write_morph_data(str(surfdir / "lh.sulc"), vals)

    # asks for surftype-specific file first; should fall back to blank surftype.
    out = read_surface_metric("subj01", "lh", "sulc", surftype="inflated")
    assert np.allclose(out, vals)


def test_read_surface_patch_binary(tmp_path: Path, monkeypatch: object) -> None:
    surfdir = _make_subject_tree(tmp_path)
    monkeypatch.setenv("SUBJECTS_DIR", str(tmp_path))

    # Base white mesh.
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    fsio.write_geometry(str(surfdir / "lh.white"), verts, faces)

    # Binary patch format: version=-1, npts, then (ind,x,y,z) big-endian.
    # ind uses 1-based positive encoding for normal vertices.
    patch_records = np.array(
        [
            (1, 0.0, 0.0, 0.0),
            (2, 2.0, 0.0, 0.0),
            (3, 2.0, 2.0, 0.0),
            (4, 0.0, 2.0, 0.0),
        ],
        dtype=[("ind", ">i4"), ("x", ">f4"), ("y", ">f4"), ("z", ">f4")],
    )
    patch_path = surfdir / "lh.cortex.patch.3d"
    with patch_path.open("wb") as f:
        np.asarray([-1], dtype=">i4").tofile(f)
        np.asarray([patch_records.shape[0]], dtype=">i4").tofile(f)
        patch_records.tofile(f)

    surf = read_surface(
        "subj01",
        "lh",
        "cortex.patch.3d",
        "orig",
        noscale=True,
    )
    assert surf.patchmask is not None
    assert surf.patchmask.shape[0] == 4
    assert surf.faces.shape[0] == 2
    # Patch coordinates should be inserted.
    assert np.allclose(surf.vertices[:, :2], np.array([[0, 0], [2, 0], [2, 2], [0, 2]]))
