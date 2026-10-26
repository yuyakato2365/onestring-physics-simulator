#!/usr/bin/env python3
"""Actually GENERATE all-quad M2D from saved OptCuts NPZ.

This uses our independently developed coarse-patch EXPERIMENT,
NOT Zhang et al. (CAD 2021) authors' C++ reference implementation.
No Gmsh and no implicit fallback.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from onestring_physics.zhang2021_field_layout_experiment import generate_field_layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-quads", type=int, default=35)
    parser.add_argument("--max-missing-fraction", type=float, default=.02)
    parser.add_argument("--no-simplify", action="store_true")
    args = parser.parse_args()
    with np.load(args.source, allow_pickle=False) as d:
        source = SimpleNamespace(
            uv_vertices_2d=d["source_uv_vertices"].copy(),
            uv_faces=d["source_uv_faces"].copy(),
            surface_vertices_3d=d["source_xyz_vertices"].copy(),
            surface_faces=d["source_xyz_faces"].copy(),
            omega_boundary=d["omega_boundary"].copy(),
            method="optcuts_test", metrics={},
        )
    candidate = generate_field_layout(
        source, target_quads=args.target_quads,
        max_missing_fraction=args.max_missing_fraction,
        allow_outline_simplification=not args.no_simplify,
        require_seam_correspondence=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, vertices_uv=candidate.uv,
                        faces=candidate.quads)
    args.report.write_text(
        json.dumps(candidate.metrics, indent=2, default=float) + "\n",
        encoding="utf-8")
    print(json.dumps(candidate.metrics, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
