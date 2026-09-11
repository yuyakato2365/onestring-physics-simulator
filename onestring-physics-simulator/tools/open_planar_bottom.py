"""Automatically open a mesh by removing its dominant planar support patch.

Intended use in the OneString Bunny workflow:

    closed remeshed mesh
      -> detect a broad, densely triangulated coplanar support patch
      -> delete only faces lying on that plane
      -> export the resulting open mesh

The detector does not assume a particular axis (e.g. Z) or require the bottom
faces to be labeled. It finds connected coplanar face patches, keeps patches
whose plane places almost the whole mesh on one side (a support plane), and
selects the patch with the strongest combination of surface area and number of
vertices. The selected plane is then refit from its vertices and every triangle
lying on that plane is removed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import trimesh


@dataclass
class PlanarPatch:
    faces: np.ndarray
    vertices: np.ndarray
    area: float
    normal: np.ndarray
    origin: np.ndarray
    support_fraction: float
    score: float


def load_triangle_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise RuntimeError("Input scene contains no geometry")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh, got {type(loaded)!r}")
    if loaded.faces.ndim != 2 or loaded.faces.shape[1] != 3:
        raise RuntimeError("Input must contain triangular faces")
    return loaded


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares plane fit. Returns (origin, unit normal)."""
    points = np.asarray(points, dtype=np.float64)
    origin = np.mean(points, axis=0)
    centered = points - origin
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal /= max(float(np.linalg.norm(normal)), 1e-30)
    return origin, normal


def find_largest_support_planar_patch(
    mesh: trimesh.Trimesh,
    min_faces: int = 20,
    min_support_fraction: float = 0.995,
) -> tuple[PlanarPatch, list[PlanarPatch]]:
    """Find the dominant broad planar patch without assuming a world axis.

    ``mesh.facets`` groups connected coplanar faces. A plausible bottom cap is
    additionally expected to be a support plane: nearly every mesh vertex lies
    on one side of the fitted plane.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    bbox_diag = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    support_tol = max(bbox_diag * 1e-6, 1e-9)

    patches: list[PlanarPatch] = []
    for face_ids_raw in mesh.facets:
        face_ids = np.asarray(face_ids_raw, dtype=np.int64)
        if len(face_ids) < min_faces:
            continue

        vertex_ids = np.unique(np.asarray(mesh.faces[face_ids], dtype=np.int64).ravel())
        points = vertices[vertex_ids]
        origin, normal = fit_plane(points)

        signed = (vertices - origin) @ normal
        positive_side = float(np.mean(signed >= -support_tol))
        negative_side = float(np.mean(signed <= support_tol))
        support_fraction = max(positive_side, negative_side)
        if support_fraction < min_support_fraction:
            continue

        area = float(np.sum(face_areas[face_ids]))
        # Area represents "broad" while sqrt(vertex count) rewards a patch
        # supported by many actual mesh samples without over-dominating area.
        score = area * math.sqrt(float(len(vertex_ids)))
        patches.append(
            PlanarPatch(
                faces=face_ids,
                vertices=vertex_ids,
                area=area,
                normal=normal,
                origin=origin,
                support_fraction=support_fraction,
                score=score,
            )
        )

    if not patches:
        raise RuntimeError(
            "No sufficiently large planar support patch was found. "
            "Try lowering --min-faces or --support-fraction if the intended cap is slightly noisy."
        )

    patches.sort(key=lambda patch: patch.score, reverse=True)
    return patches[0], patches


def boundary_stats(faces: np.ndarray) -> tuple[int, int, int]:
    """Return (boundary edge count, boundary components, closed boundary loops)."""
    faces = np.asarray(faces, dtype=np.int64)
    edges = np.vstack(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]
    )
    edges.sort(axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    if len(boundary_edges) == 0:
        return 0, 0, 0

    adjacency: dict[int, set[int]] = {}
    for a, b in boundary_edges:
        a_i, b_i = int(a), int(b)
        adjacency.setdefault(a_i, set()).add(b_i)
        adjacency.setdefault(b_i, set()).add(a_i)

    visited: set[int] = set()
    components = 0
    loops = 0
    for start in adjacency:
        if start in visited:
            continue
        components += 1
        stack = [start]
        component: list[int] = []
        while stack:
            vertex = stack.pop()
            if vertex in visited:
                continue
            visited.add(vertex)
            component.append(vertex)
            stack.extend(adjacency[vertex] - visited)
        if all(len(adjacency[vertex]) == 2 for vertex in component):
            loops += 1

    return int(len(boundary_edges)), components, loops


def open_planar_patch(
    mesh: trimesh.Trimesh,
    patch: PlanarPatch,
    plane_tolerance: float | None,
    max_normal_angle_deg: float,
) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray, np.ndarray, float]:
    """Delete triangles that lie on the refitted dominant plane."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    bbox_diag = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    tolerance = (
        float(plane_tolerance)
        if plane_tolerance is not None
        else max(bbox_diag * 1e-5, 1e-8)
    )

    origin, normal = fit_plane(vertices[patch.vertices])
    distances = np.abs((vertices - origin) @ normal)
    vertex_on_plane = distances <= tolerance

    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    alignment = np.abs(face_normals @ normal)
    normal_ok = alignment >= math.cos(math.radians(max_normal_angle_deg))

    delete_mask = np.all(vertex_on_plane[faces], axis=1) & normal_ok
    deleted_faces = np.flatnonzero(delete_mask)
    if len(deleted_faces) == 0:
        raise RuntimeError(
            "A planar support patch was detected, but no faces matched the final deletion tolerance."
        )

    result = trimesh.Trimesh(
        vertices=vertices.copy(),
        faces=faces[~delete_mask].copy(),
        process=False,
    )
    result.remove_unreferenced_vertices()
    return result, deleted_faces, origin, normal, tolerance


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect and remove the dominant broad planar support patch from a triangle mesh."
    )
    parser.add_argument("input", type=Path, help="Input triangle OBJ/PLY/STL")
    parser.add_argument("output", type=Path, help="Output opened OBJ/PLY/STL")
    parser.add_argument(
        "--min-faces",
        type=int,
        default=20,
        help="Minimum coplanar face count for a plane candidate (default: 20).",
    )
    parser.add_argument(
        "--support-fraction",
        type=float,
        default=0.995,
        help="Required fraction of all vertices lying on one side of a candidate plane (default: 0.995).",
    )
    parser.add_argument(
        "--plane-tolerance",
        type=float,
        default=None,
        help="Absolute distance from the detected plane used for deletion. Default: 1e-5 * mesh bounding-box diagonal.",
    )
    parser.add_argument(
        "--max-normal-angle",
        type=float,
        default=5.0,
        help="Maximum face-normal deviation from the detected plane normal in degrees (default: 5).",
    )
    args = parser.parse_args()

    if args.min_faces < 2:
        raise ValueError("--min-faces must be >= 2")
    if not 0.5 <= args.support_fraction <= 1.0:
        raise ValueError("--support-fraction must be between 0.5 and 1.0")
    if args.plane_tolerance is not None and args.plane_tolerance <= 0.0:
        raise ValueError("--plane-tolerance must be positive")
    if not 0.0 < args.max_normal_angle < 90.0:
        raise ValueError("--max-normal-angle must be between 0 and 90 degrees")

    mesh = load_triangle_mesh(args.input)
    print(
        f"Input: V={len(mesh.vertices)}, F={len(mesh.faces)}, "
        f"watertight={mesh.is_watertight}"
    )

    selected, candidates = find_largest_support_planar_patch(
        mesh,
        min_faces=args.min_faces,
        min_support_fraction=args.support_fraction,
    )

    print(f"Planar support candidates: {len(candidates)}")
    for rank, patch in enumerate(candidates[:5], start=1):
        print(
            f"  #{rank}: area={patch.area:.9g}, faces={len(patch.faces)}, "
            f"vertices={len(patch.vertices)}, support={patch.support_fraction:.6f}, "
            f"score={patch.score:.9g}"
        )

    result, deleted_faces, origin, normal, tolerance = open_planar_patch(
        mesh,
        selected,
        plane_tolerance=args.plane_tolerance,
        max_normal_angle_deg=args.max_normal_angle,
    )

    boundary_edges, boundary_components, boundary_loops = boundary_stats(result.faces)
    print(
        "Selected plane: "
        f"origin=({origin[0]:.9g}, {origin[1]:.9g}, {origin[2]:.9g}), "
        f"normal=({normal[0]:.9g}, {normal[1]:.9g}, {normal[2]:.9g})"
    )
    print(f"Plane deletion tolerance: {tolerance:.9g}")
    print(f"Deleted faces: {len(deleted_faces)}")
    print(
        f"Output: V={len(result.vertices)}, F={len(result.faces)}, "
        f"watertight={result.is_watertight}"
    )
    print(
        f"Boundary: edges={boundary_edges}, components={boundary_components}, "
        f"closed_loops={boundary_loops}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.export(args.output)
    print(f"Saved: {args.output}")

    if boundary_components == 1 and boundary_loops == 1:
        print("OK: exactly one closed boundary loop was produced.")
    else:
        print(
            "WARNING: output does not have exactly one closed boundary loop. "
            "Inspect the detected plane/cut before using it in OneString."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
