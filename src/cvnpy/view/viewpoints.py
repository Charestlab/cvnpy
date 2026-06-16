"""Predefined viewpoints for sphere/inflated/flat surfaces.

Parity target:
- /Users/iancharest/Documents/codex/cvncode/cvnlookupviewpoint.m
"""

from __future__ import annotations

from typing import Any


def _sphere_views(viewname: str) -> tuple[list[float], list[float]]:
    mapping: dict[str, tuple[list[float], list[float]]] = {
        "occip": ([10, -40, 0], [-10, -40, 0]),
        "ventral": ([10, -70, 0], [-10, -70, 0]),
        "occipa1": ([10, -40, 0], [-10, -40, 0]),
        "occipa2": ([10 - 45, -40, 0], [-10 + 45, -40, 0]),
        "occipa3": ([10 - 2 * 45, -40, 0], [-10 + 2 * 45, -40, 0]),
        "occipa4": ([10 - 3 * 45, -40, 0], [-10 + 3 * 45, -40, 0]),
        "occipa5": ([10 - 4 * 45, -40, 0], [-10 + 4 * 45, -40, 0]),
        "occipa6": ([10 - 5 * 45, -40, 0], [-10 + 5 * 45, -40, 0]),
        "occipa7": ([10 - 6 * 45, -40, 0], [-10 + 6 * 45, -40, 0]),
        "occipa8": ([10 - 7 * 45, -40, 0], [-10 + 7 * 45, -40, 0]),
        "occipb1": ([10, -40, 0], [-10, -40, 0]),
        "occipb2": ([10, -40 + 45, 0], [-10, -40 + 45, 0]),
        "occipb3": ([10, -40 + 2 * 45, 0], [-10, -40 + 2 * 45, 0]),
        "occipb4": ([10, -40 + 3 * 45, 180], [-10, -40 + 3 * 45, 180]),
        "occipb5": ([10, -40 + 4 * 45, 180], [-10, -40 + 4 * 45, 180]),
        "occipb6": ([10, -40 + 5 * 45, 180], [-10, -40 + 5 * 45, 180]),
        "occipb7": ([10, -40 + 6 * 45, 180], [-10, -40 + 6 * 45, 180]),
        "occipb8": ([10, -40 + 7 * 45, 0], [-10, -40 + 7 * 45, 0]),
        "occipc1": ([10 - 2 * 45, -40, 0], [-10 + 2 * 45, -40, 0]),
        "occipc2": ([10 - 2 * 45, -40 + 45, 0], [-10 + 2 * 45, -40 + 45, 0]),
        "occipc3": (
            [10 - 2 * 45, -40 + 2 * 45, 0],
            [-10 + 2 * 45, -40 + 2 * 45, 0],
        ),
        "occipc4": (
            [10 - 2 * 45, -40 + 3 * 45, 180],
            [-10 + 2 * 45, -40 + 3 * 45, 180],
        ),
        "occipc5": (
            [10 - 2 * 45, -40 + 4 * 45, 180],
            [-10 + 2 * 45, -40 + 4 * 45, 180],
        ),
        "occipc6": (
            [10 - 2 * 45, -40 + 5 * 45, 180],
            [-10 + 2 * 45, -40 + 5 * 45, 180],
        ),
        "occipc7": (
            [10 - 2 * 45, -40 + 6 * 45, 180],
            [-10 + 2 * 45, -40 + 6 * 45, 180],
        ),
        "occipc8": (
            [10 - 2 * 45, -40 + 7 * 45, 0],
            [-10 + 2 * 45, -40 + 7 * 45, 0],
        ),
    }
    if viewname not in mapping:
        raise ValueError(f"Unknown sphere view name: {viewname}")
    return mapping[viewname]


def _nonsphere_views(viewname: str) -> tuple[tuple[list[float], list[float]], bool]:
    mapping: dict[str, tuple[tuple[list[float], list[float]], bool]] = {
        "dorsal": (([0, 90, 90], [0, 90, 90]), False),
        "parietal": (([0, 45, 0], [0, 45, 0]), False),
        "ventral": (([0, -90, -90], [0, -90, -90]), True),
        "ventral-lateral": (([95, -66, 270], [-95, -66, 90]), True),
        "lateral-auditory": (([-120, 15, 0], [120, 15, 0]), False),
        "medial": (([90, 0, 15], [-90, 0, -15]), False),
        "lateral": (([270, 0, -15], [-270, 0, 15]), False),
        "medial-ventral": (([45, -45, 15], [-45, -45, -15]), False),
        "occip": (([10, -45, 0], [-10, -45, 0]), False),
        "superior": (([0, 90, 90], [90, 90, 0]), False),
        "frontal": (([-180, 0, 0], [-180, 0, 0]), True),
    }
    if viewname not in mapping:
        raise ValueError(f"Unknown non-sphere view name: {viewname}")
    return mapping[viewname]


def lookup_viewpoint(subject: str, hemi: str | list[str], viewname: str, surftype: str) -> tuple[Any, bool, Any]:
    """Return [az,el,tilt] style views and hemi flip hints.

    This mirrors cvnlookupviewpoint.m behavior for known view names.
    """
    _ = subject  # subject is currently unused, kept for API parity.
    viewname_n = viewname.lower()
    surftype_n = surftype.lower()

    fliphemi = False
    if surftype_n.startswith("sphere"):
        viewpt_lr = _sphere_views(viewname_n)
    elif ".flat.patch." in surftype_n:
        viewpt_lr = ([0, 0, 0], [0, 0, 0])
        if viewname_n == "ventral":
            fliphemi = True
    else:
        viewpt_lr, fliphemi = _nonsphere_views(viewname_n)

    input_is_cell = isinstance(hemi, list)
    hemis = hemi if isinstance(hemi, list) else [hemi]

    viewpt: list[list[float]] = []
    for h in hemis:
        h_n = h.lower()
        if h_n == "lh":
            viewpt.append(viewpt_lr[0])
        elif h_n == "rh":
            viewpt.append(viewpt_lr[1])
        else:
            raise ValueError(f"Unsupported hemisphere: {h}")

    if len(hemis) > 1:
        viewhemi: Any = ["lh", "rh"]
        if fliphemi:
            viewhemi = ["rh", "lh"]
    else:
        viewhemi = hemis[0]

    if not input_is_cell:
        return viewpt[0], fliphemi, viewhemi
    return viewpt, fliphemi, viewhemi
