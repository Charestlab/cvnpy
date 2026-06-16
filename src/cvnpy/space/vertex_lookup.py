"""Map values between surface suffix spaces.

Parity target:
- /Users/iancharest/Documents/codex/cvncode/cvnlookupvertex.m
"""

from __future__ import annotations

from typing import Any


def lookup_vertex(surfdir_or_subject: str, hemi: str | list[str], sourcesuffix: str, destsuffix: str, sourcevals: Any | None = None, badval: float = float("nan")) -> tuple[Any, Any, Any, str]:
    """Transfer values between source/destination mesh vertex spaces.

    TODO: implement parity with cvnlookupvertex.m.
    """
    raise NotImplementedError("lookup_vertex is not implemented yet")
