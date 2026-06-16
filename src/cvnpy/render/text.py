"""Text and scalebar rendering."""

from __future__ import annotations

from typing import Any


def add_text(rgbimg: Any, text: Any, textsize: float, textcolor: Any) -> Any:
    """Add hemi text labels and optional annotations.

    TODO: implement parity with add_hemi_text/addtext2img behavior.
    """
    raise NotImplementedError("add_text is not implemented yet")


def add_scalebar(rgbimg: Any, lookup: Any, options: dict[str, Any]) -> Any:
    """Add flatmap scalebar when pixels_per_mm is available.

    TODO: implement parity with cvnlookupimages.m scalebar behavior.
    """
    raise NotImplementedError("add_scalebar is not implemented yet")
