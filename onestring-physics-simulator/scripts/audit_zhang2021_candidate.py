#!/usr/bin/env python3
"""Audit (NOT generate) quadrangulations exported by a Zhang reference adapter.

Standalone: never imports Streamlit or changes any production pipeline.
Our custom NPZ input/output contract is documented in
docs/zhang2021_isolated_experiment_20261027.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from onestring_physics.zhang2021_reference_bridge import (
    Zhang2021BridgeError, audit_reference_quads)


SOURCE_KEYS = ("source_uv_vertices", "source_uv_faces",
               "source_xyz_vertices", "source_xyz_faces", "omega_boundary")


def _load_source(file_path: Path) -> SimpleNamespace:
    with np.load(file_path, allow_pickle=False) as d:
        missing = [k for k in SOURCE_KEYS if k not in d]
        if missing:
            raise Zhang2021BridgeError("SOURCE_FILE_MISSING_KEYS: " + ",".join(missing))
        return SimpleNamespace(
            uv_vertices_2d=d["source_uv_vertices"].copy(),
            uv_faces=d["source_uv_faces"].copy(),
            surface_vertices_3d=d["source_xyz_vertices"].copy(),
            surface_faces=d["source_xyz_faces"].copy(),
            omega_boundary=d["omega_boundary"].copy(),
            method="optcuts_test", metrics={},
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True,
                   help="NPZ input with original OptCuts chart provenance")
    p.add_argument("--candidate", type=Path, required=True,
                   help="NPZ output from externally verified meshing adapter")
    p.add_argument("--report", type=Path, required=True,
                   help="Write strict preflight diagnostics JSON")
    p.add_argument("--allow-incomplete-boundary-for-study", action="store_true",
                   help="WARNING: disables seam audit; never use as certification")
    p.add_argument("--max-missing-fraction", type=float, default=1e-7)
    args = p.parse_args()
    source = _load_source(args.source)
    with np.load(args.candidate, allow_pickle=False) as d:
        vertices = d["vertices_uv"].copy()
        faces = d["faces"].copy()
    audited = audit_reference_quads(
        source, vertices, faces,
        max_undercoverage_fraction=args.max_missing_fraction,
        require_seam_correspondence=not args.allow_incomplete_boundary_for_study)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(audited.diagnostics, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(audited.diagnostics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
