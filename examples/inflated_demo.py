"""Generate an inflated-surface PNG using cvnpy."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

from cvnpy.core.lookup_images import lookup_images
from cvnpy.io.metrics import read_surface_metric
from cvnpy.io.surfaces import read_surface
from cvnpy.view.viewpoints import lookup_viewpoint


def _parse_view(s: str) -> list[float]:
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError("view must be az,el,tilt")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="cvnpy inflated demo")
    parser.add_argument("--subject", required=False, default=None)
    parser.add_argument(
        "--subject-dir",
        default=None,
        help="Path to a subject directory containing surf/ (e.g. /path/nsd_fsaverage).",
    )
    parser.add_argument(
        "--subjects-dir",
        default=None,
        help="Path to SUBJECTS_DIR root (parent that contains subject folders).",
    )
    parser.add_argument("--hemi", default="lh", choices=["lh", "rh"])
    parser.add_argument("--surftype", default="inflated")
    parser.add_argument("--surfsuffix", default="orig")
    parser.add_argument("--metric", default="curv")
    parser.add_argument("--view", default=None, help="az,el,tilt override")
    parser.add_argument("--viewname", default="occip")
    parser.add_argument("--imageres", type=int, default=700)
    parser.add_argument("--out", default="inflated_demo.png")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    surfdir = None
    if args.subject_dir:
        subj_path = Path(args.subject_dir).resolve()
        surfdir = str(subj_path / "surf")
        subject = subj_path.name
    else:
        subject = args.subject
        if subject is None:
            raise ValueError("Provide --subject or --subject-dir")

    if args.subjects_dir:
        os.environ["SUBJECTS_DIR"] = str(Path(args.subjects_dir).resolve())

    if args.view:
        view = _parse_view(args.view)
    else:
        view, _flip, _viewhemi = lookup_viewpoint(
            subject, args.hemi, args.viewname, args.surftype
        )

    vals = read_surface_metric(
        subject,
        args.hemi,
        args.metric,
        "",
        args.surfsuffix,
        **({"surfdir": surfdir} if surfdir else {}),
    )
    if vals is None:
        n = int(
            read_surface(
                subject,
                args.hemi,
                args.surftype,
                args.surfsuffix,
                justcount=True,
                **({"surfdir": surfdir} if surfdir else {}),
            )
        )
        rng = np.random.default_rng(1)
        vals = rng.standard_normal(n)

    _mapped, _lookup, rgb, _options = lookup_images(
        subject,
        np.asarray(vals, dtype=float),
        args.hemi,
        view,
        surftype=args.surftype,
        surfsuffix=args.surfsuffix,
        imageres=args.imageres,
        surfshading=True,
        background="curv",
        cache_dir=args.cache_dir,
        savelookup=not args.no_cache,
        reset=bool(args.no_cache),
        surfdir=surfdir,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out, np.clip(rgb, 0.0, 1.0))
    print(str(out.resolve()))


if __name__ == "__main__":
    main()
