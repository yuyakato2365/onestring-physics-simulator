"""Cross-platform preprocessing for the OneString Bunny workflow.

Workflow:
    closed triangle Bunny
      -> Botsch-Kobbelt isotropic remeshing (GPyToolbox)
      -> export closed remeshed OBJ
      -> remove only the bottom faces in Blender
      -> feed the resulting single-boundary open disk mesh to OneString

Mitsuba 3 is detected only to report the best renderer/AD variant for the
current machine. The actual remeshing step is GPyToolbox's remesh_botsch,
matching the remeshing stage used in Mitsuba's official shape-optimization
tutorial. LargeSteps is not required for this preprocessing experiment.
"""
from __future__ import annotations

import argparse
import platform
from pathlib import Path
import sys

import numpy as np
import trimesh
from gpytoolbox import remesh_botsch


def choose_mitsuba_variant() -> tuple[str, list[str]]:
    """Return the best available Mitsuba variant for this machine.

    Preference:
      Apple Silicon: metal_ad_rgb -> llvm_ad_rgb -> scalar_rgb
      NVIDIA/other:  cuda_ad_rgb  -> llvm_ad_rgb -> scalar_rgb
    """
    try:
        import mitsuba as mi
    except Exception:
        return "mitsuba-not-installed", []

    variants = list(mi.variants())
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}
    candidates = (
        ["metal_ad_rgb", "metal_rgb", "llvm_ad_rgb", "llvm_rgb", "scalar_rgb"]
        if is_apple_silicon
        else ["cuda_ad_rgb", "cuda_rgb", "llvm_ad_rgb", "llvm_rgb", "scalar_rgb"]
    )
    selected = next((name for name in candidates if name in variants), "unavailable")
    return selected, variants


def unique_edges(faces: np.ndarray) -> np.ndarray:
    f = np.asarray(faces, dtype=np.int64)[:, :3]
    edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def mesh_quality(vertices: np.ndarray, faces: np.ndarray) -> dict[str, float]:
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)[:, :3]
    tri = v[f]
    e01 = tri[:, 1] - tri[:, 0]
    e12 = tri[:, 2] - tri[:, 1]
    e20 = tri[:, 0] - tri[:, 2]
    lengths = np.stack(
        [np.linalg.norm(e01, axis=1), np.linalg.norm(e12, axis=1), np.linalg.norm(e20, axis=1)],
        axis=1,
    )
    area2 = np.linalg.norm(np.cross(e01, tri[:, 2] - tri[:, 0]), axis=1)
    quality = 2.0 * np.sqrt(3.0) * area2 / np.maximum(np.sum(lengths * lengths, axis=1), 1e-30)

    a, b, c = lengths[:, 0], lengths[:, 1], lengths[:, 2]

    def angle(opposite: np.ndarray, side1: np.ndarray, side2: np.ndarray) -> np.ndarray:
        cosine = (side1 * side1 + side2 * side2 - opposite * opposite) / np.maximum(
            2.0 * side1 * side2, 1e-30
        )
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    minimum_angles = np.min(
        np.stack([angle(b, a, c), angle(c, a, b), angle(a, b, c)], axis=1), axis=1
    )
    edges = unique_edges(f)
    edge_lengths = np.linalg.norm(v[edges[:, 1]] - v[edges[:, 0]], axis=1)
    edge_mean = float(np.mean(edge_lengths)) if len(edge_lengths) else 0.0
    return {
        "min_angle_deg": float(np.min(minimum_angles)) if len(minimum_angles) else 0.0,
        "angle_p05_deg": float(np.percentile(minimum_angles, 5.0)) if len(minimum_angles) else 0.0,
        "quality_min": float(np.min(quality)) if len(quality) else 0.0,
        "quality_p01": float(np.percentile(quality, 1.0)) if len(quality) else 0.0,
        "quality_p05": float(np.percentile(quality, 5.0)) if len(quality) else 0.0,
        "edge_median": float(np.median(edge_lengths)) if len(edge_lengths) else 0.0,
        "edge_cv": float(np.std(edge_lengths) / max(edge_mean, 1e-30)) if len(edge_lengths) else 0.0,
    }


def print_quality(label: str, q: dict[str, float]) -> None:
    print(
        f"{label}: minAngle={q['min_angle_deg']:.3f} deg, "
        f"angleP05={q['angle_p05_deg']:.3f} deg, "
        f"q01={q['quality_p01']:.4f}, q05={q['quality_p05']:.4f}, "
        f"edgeCV={q['edge_cv']:.4f}, edgeMedian={q['edge_median']:.6g}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Closed triangular Bunny OBJ/PLY/STL")
    parser.add_argument("output", type=Path, help="Output closed remeshed OBJ/PLY/STL")
    parser.add_argument("--iterations", type=int, default=10, help="Botsch remeshing iterations")
    parser.add_argument(
        "--edge-scale",
        type=float,
        default=1.0,
        help="Target edge length / original median edge length. Start with 1.0; use 0.75 or 0.5 for a denser mesh.",
    )
    parser.add_argument(
        "--allow-open-input",
        action="store_true",
        help="Allow a non-watertight input. For the intended workflow, remesh before cutting the bottom, so this should normally stay off.",
    )
    args = parser.parse_args()

    selected_variant, variants = choose_mitsuba_variant()
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Mitsuba preferred variant: {selected_variant}")
    if variants:
        print("Mitsuba variants:", ", ".join(variants))

    loaded = trimesh.load(args.input, process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise RuntimeError("Input scene contains no geometry")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh, got {type(mesh)!r}")
    if not mesh.is_watertight and not args.allow_open_input:
        raise RuntimeError(
            "Input is not watertight. For this experiment, use the CLOSED Bunny here, "
            "remesh it first, and remove bottom faces afterwards in Blender."
        )

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int32)[:, :3]
    if len(f) == 0 or f.shape[1] != 3:
        raise RuntimeError("Input must contain triangular faces")

    before = mesh_quality(v, f)
    print(f"Before: V={len(v)}, F={len(f)}, watertight={mesh.is_watertight}")
    print_quality("Before quality", before)

    target_edge = max(float(before["edge_median"]) * float(args.edge_scale), 1e-12)
    print(f"Target edge length: {target_edge:.9g} (edge-scale={args.edge_scale:g})")

    v_new, f_new = remesh_botsch(
        v,
        f,
        i=max(1, int(args.iterations)),
        h=target_edge,
        project=True,
    )
    v_new = np.asarray(v_new, dtype=np.float64)
    f_new = np.asarray(f_new, dtype=np.int32)[:, :3]

    result = trimesh.Trimesh(vertices=v_new, faces=f_new, process=False)
    after = mesh_quality(v_new, f_new)
    print(f"After: V={len(v_new)}, F={len(f_new)}, watertight={result.is_watertight}")
    print_quality("After quality", after)

    if not result.is_watertight and mesh.is_watertight:
        raise RuntimeError("Remeshing unexpectedly opened the closed input mesh; output was not written")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.export(args.output)
    print(f"Saved: {args.output}")
    print("Next: open this CLOSED remeshed mesh in Blender, delete bottom FACES only, verify one boundary loop, then export for OneString.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
