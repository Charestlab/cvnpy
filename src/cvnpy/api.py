"""User-facing APIs."""

from __future__ import annotations

from typing import Any

from .core.lookup_images import lookup_images


def lookup(subject: str, vals: Any, hemi: str | list[str], view: Any, lookup: Any | None = None, **kwargs: Any) -> Any:
    """Pythonic wrapper analogous to MATLAB `cvnlookup.m` -> `cvnlookupimages.m`."""
    return lookup_images(subject, vals, hemi, view, lookup, **kwargs)
