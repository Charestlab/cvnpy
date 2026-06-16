"""Surface metric loading.

Parity target:
- /Users/iancharest/Documents/codex/cvncode/cvnreadsurfacemetric.m
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np

from ..types import ValStruct


def _resolve_surfdir(subject: str, surfdir: str | None = None) -> Path:
    if surfdir:
        p = Path(surfdir)
        if not p.is_dir():
            raise FileNotFoundError(f"surfdir does not exist: {p}")
        return p

    subjects_dir = os.environ.get("SUBJECTS_DIR")
    if not subjects_dir:
        raise ValueError("SUBJECTS_DIR is not set and no surfdir was provided")

    p = Path(subjects_dir) / subject / "surf"
    if not p.is_dir():
        raise FileNotFoundError(f"missing surf directory: {p}")
    return p


def _suffix_file(surfsuffix: str) -> str:
    return "" if surfsuffix == "orig" else surfsuffix


def _find_metric_file(
    surfdir: Path,
    hemi: str,
    metricname: str,
    surftype: str,
    suffix_file: str,
) -> Path | None:
    surftype_file = f"_{surftype}_" if surftype else ""
    patterns = [
        f"{hemi}.{metricname}{surftype_file}{suffix_file}",
        f"{hemi}.{metricname}{surftype_file}{suffix_file}.mgz",
        f"{hemi}{surftype_file}{suffix_file}.{metricname}",
    ]
    for name in patterns:
        p = surfdir / name
        if p.exists():
            return p
    return None


def _read_metric_file(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".mgz":
        data = np.asarray(nib.load(str(path)).get_fdata(), dtype=float)
        return np.ravel(data)
    data = fsio.read_morph_data(str(path))
    return np.asarray(data, dtype=float).ravel()


def read_surface_metric(subject: str, hemi: str | list[str], metricname: str, surftype: str = "", surfsuffix: str = "orig", **kwargs: Any) -> Any:
    """Read curv/sulc/other per-vertex metrics.

    Supported kwargs:
    - surfdir: override subjects/<subject>/surf
    - verbose: bool
    """
    surfdir_arg = kwargs.get("surfdir")
    verbose = bool(kwargs.get("verbose", False))
    surfdir = _resolve_surfdir(subject, surfdir_arg if isinstance(surfdir_arg, str) else None)

    hemis = [hemi] if isinstance(hemi, str) else list(hemi)
    if not hemis:
        hemis = ["lh", "rh"]

    if hemis == ["rh", "lh"]:
        hemis = ["lh", "rh"]

    suffix_file = _suffix_file(surfsuffix)
    values: list[np.ndarray] = []

    for h in hemis:
        metric_file = _find_metric_file(surfdir, h.lower(), metricname, surftype, suffix_file)
        if metric_file is None and surftype:
            metric_file = _find_metric_file(surfdir, h.lower(), metricname, "", suffix_file)
        if metric_file is None:
            if verbose:
                print(
                    f"No metric file found for hemi={h}, metric={metricname}, "
                    f"surftype={surftype}, suffix={surfsuffix}"
                )
            return None
        values.append(_read_metric_file(metric_file))

    if hemis == ["lh", "rh"]:
        return ValStruct(
            data=np.concatenate(values, axis=0),
            numlh=int(values[0].shape[0]),
            numrh=int(values[1].shape[0]),
        )

    if len(values) == 1:
        return values[0]
    return np.concatenate(values, axis=0)
