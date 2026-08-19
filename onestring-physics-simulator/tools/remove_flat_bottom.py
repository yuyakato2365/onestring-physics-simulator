#!/usr/bin/env python3
"""Remove an artificial planar cap from a closed triangle mesh by normal-space density.

Designed for the Stanford Bunny / OneString workflow:

closed remeshed Bunny
    -> find an anomalously dense cluster of nearly identical face normals
    -> split those faces into edge-connected components
    -> score components by normal coherence, planarity, and size
    -> delete the best planar component only
    -> remove newly unreferenced vertices
    -> verify one connected component with one boundary loop

IMPORTANT: geometric height is deliberately NOT used. The detector does not
assume that the cap is the lowest part of the mesh or that any world axis is
"up". It relies on the Stanford Bunny cap being an artificial region where many
spatially adjacent triangles have nearly the same normal direction and lie on
one plane.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh


def _load_single_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if len(geometries) != 1:
            raise ValueError(
                f"Expected exactly one triangle mesh in {path}, found {len(geometries)} geometries"
            )
        loaded = geometries[0]
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh object: {type(loaded)!r}")
    mesh = loaded.copy()
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        raise ValueError("Input must be a triangle mesh")
    if not np.isfinite(np.asarray(mesh.vertices)).all():
        raise ValueError("Mesh contains non-finite vertex coordinates")
    return mesh


def _face_components(faces: np.ndarray, candidate_ids: np.ndarray) -> list[np.ndarray]:
    """Return edge-connected components of a subset of faces."""
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    if len(candidate_ids) == 0:
        return []

    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_id in candidate_ids.tolist():
        a, b, c = [int(v) for v in faces[face_id]]
        for u, v in ((a, b), (b, c), (c, a)):
            edge_to_faces[tuple(sorted((u, v)))].append(face_id)

    adjacency: dict[int, set[int]] = {int(face_id): set() for face_id in candidate_ids.tolist()}
    for touching in edge_to_faces.values():
        if len(touching) < 2:
            continue
        for i in range(len(touching)):
            for j in range(i + 1, len(touching)):
                a, b = int(touching[i]), int(touching[j])
                adjacency[a].add(b)
                adjacency[b].add(a)

    components: list[np.ndarray] = []
    unseen = set(int(v) for v in candidate_ids.tolist())
    while unseen:
        start = unseen.pop()
        queue: deque[int] = deque([start])
        group = [start]
        while queue:
            current = queue.popleft()
            for neighbour in adjacency[current]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)
                    group.append(neighbour)
        components.append(np.asarray(group, dtype=np.int64))
    return components


def _component_area(vertices: np.ndarray, faces: np.ndarray, ids: np.ndarray) -> float:
    tri = vertices[faces[ids]]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def _boundary_edges(faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    keys = np.sort(edges, axis=1)
    unique, counts = np.unique(keys, axis=0, return_counts=True)
    return unique[counts == 1]


def _boundary_loop_count(faces: np.ndarray) -> tuple[int, int]:
    edges = _boundary_edges(faces)
    if len(edges) == 0:
        return 0, 0

    graph: dict[int, set[int]] = defaultdict(set)
    for a, b in edges:
        graph[int(a)].add(int(b))
        graph[int(b)].add(int(a))

    unseen = set(graph.keys())
    components = 0
    while unseen:
        components += 1
        start = unseen.pop()
        queue: deque[int] = deque([start])
        while queue:
            current = queue.popleft()
            for neighbour in graph[current]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)
    return components, len(edges)


def _face_connected_component_count(faces: np.ndarray) -> int:
    if len(faces) == 0:
        return 0
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_id, face in enumerate(faces):
        a, b, c = [int(v) for v in face]
        for u, v in ((a, b), (b, c), (c, a)):
            edge_to_faces[tuple(sorted((u, v)))].append(face_id)

    adjacency: list[set[int]] = [set() for _ in range(len(faces))]
    for touching in edge_to_faces.values():
        if len(touching) < 2:
            continue
        for i in range(len(touching)):
            for j in range(i + 1, len(touching)):
                a, b = touching[i], touching[j]
                adjacency[a].add(b)
                adjacency[b].add(a)

    unseen = set(range(len(faces)))
    count = 0
    while unseen:
        count += 1
        start = unseen.pop()
        queue: deque[int] = deque([start])
        while queue:
            current = queue.popleft()
            for neighbour in adjacency[current]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)
    return count


def _compact_vertices(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    used = np.unique(faces.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return vertices[used], remap[faces]


def _unit_face_normals(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    safe = np.maximum(lengths, 1.0e-30)
    return cross / safe[:, None], 0.5 * lengths


def _normal_density_counts(normals: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Count face normals inside a spherical angular neighbourhood.

    Uses scipy.spatial.cKDTree when available. Unit-vector Euclidean distance
    corresponding to angle theta is 2*sin(theta/2).
    """
    theta = np.deg2rad(float(angle_degrees))
    radius = float(2.0 * np.sin(0.5 * theta))
    try:
        from scipy.spatial import cKDTree
    except Exception as exc:
        raise RuntimeError(
            "Normal-density cap detection requires scipy (scipy.spatial.cKDTree)."
        ) from exc
    tree = cKDTree(np.asarray(normals, dtype=np.float64))
    try:
        counts = tree.query_ball_point(normals, r=radius, return_length=True)
        return np.asarray(counts, dtype=np.int64)
    except TypeError:
        return np.asarray([len(v) for v in tree.query_ball_point(normals, r=radius)], dtype=np.int64)


def _component_planarity(
    vertices: np.ndarray,
    faces: np.ndarray,
    ids: np.ndarray,
    reference_normal: np.ndarray,
) -> tuple[float, float, float]:
    """Return normalized RMS plane thickness, max thickness, and normal coherence."""
    component_faces = faces[ids]
    vertex_ids = np.unique(component_faces.reshape(-1))
    points = vertices[vertex_ids]
    if len(points) < 3:
        return float("inf"), float("inf"), 0.0

    center = np.mean(points, axis=0)
    centered = points - center
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    plane_normal = vh[-1]
    if np.dot(plane_normal, reference_normal) < 0.0:
        plane_normal = -plane_normal
    distances = np.abs(centered @ plane_normal)

    bounds = np.ptp(points, axis=0)
    scale = max(float(np.linalg.norm(bounds)), 1.0e-30)
    rms = float(np.sqrt(np.mean(distances * distances)) / scale)
    maximum = float(np.max(distances) / scale)

    normals, _ = _unit_face_normals(vertices, component_faces)
    coherence = float(np.mean(np.clip(normals @ reference_normal, -1.0, 1.0)))
    return rms, maximum, coherence


def remove_flat_bottom(
    mesh: trimesh.Trimesh,
    *,
    normal_cluster_angle_degrees: float = 3.0,
    component_normal_angle_degrees: float = 4.0,
    minimum_cluster_faces: int = 12,
    maximum_planarity_rms: float = 0.004,
    maximum_planarity_max: float = 0.012,
    require_watertight_input: bool = True,
) -> tuple[trimesh.Trimesh, dict[str, float | int | bool | str]]:
    """Remove the most anomalously dense, coherent, planar face component.

    No coordinate axis or geometric height participates in detection.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if require_watertight_input and not bool(mesh.is_watertight):
        raise ValueError("Input is not watertight. Expected a CLOSED mesh before cap removal.")

    normals, face_areas = _unit_face_normals(vertices, faces)
    finite = np.isfinite(normals).all(axis=1) & (face_areas > 1.0e-30)
    if not np.any(finite):
        raise ValueError("Mesh has no finite non-degenerate face normals")

    density = np.zeros(len(faces), dtype=np.int64)
    density[finite] = _normal_density_counts(normals[finite], normal_cluster_angle_degrees)
    seed_face = int(np.argmax(density))
    peak_density = int(density[seed_face])
    if peak_density < int(minimum_cluster_faces):
        raise RuntimeError(
            f"No sufficiently dense face-normal cluster found: peak={peak_density}, "
            f"required>={minimum_cluster_faces}."
        )

    seed_normal = normals[seed_face]
    cos_component = float(np.cos(np.deg2rad(float(component_normal_angle_degrees))))
    directed_similarity = normals @ seed_normal
    candidate_mask = finite & (directed_similarity >= cos_component)
    candidate_ids = np.flatnonzero(candidate_mask)

    components = _face_components(faces, candidate_ids)
    if not components:
        raise RuntimeError("Dense normal cluster was found but no connected face component could be formed")

    component_records: list[dict[str, float | int | np.ndarray]] = []
    for component in components:
        if len(component) < int(minimum_cluster_faces):
            continue
        area = _component_area(vertices, faces, component)
        rms, max_thickness, coherence = _component_planarity(
            vertices, faces, component, seed_normal
        )
        local_density = float(np.mean(density[component]))
        # Size and normal-space density are primary. Planarity/coherence suppress
        # smooth curved surface regions whose normals merely happen to be similar.
        planarity_factor = 1.0 / (1.0 + 500.0 * rms + 100.0 * max_thickness)
        score = float(len(component)) * local_density * max(coherence, 0.0) * planarity_factor
        component_records.append(
            {
                "ids": component,
                "faces": int(len(component)),
                "area": float(area),
                "rms": float(rms),
                "max": float(max_thickness),
                "coherence": float(coherence),
                "density": float(local_density),
                "score": float(score),
            }
        )

    if not component_records:
        raise RuntimeError(
            "Dense normal directions exist, but no connected component is large enough. "
            "Lower --minimum-cluster-faces or increase --component-normal-angle-degrees."
        )

    component_records.sort(key=lambda record: float(record["score"]), reverse=True)
    chosen = component_records[0]
    if float(chosen["rms"]) > float(maximum_planarity_rms) or float(chosen["max"]) > float(maximum_planarity_max):
        raise RuntimeError(
            "Best dense-normal component is not planar enough to remove safely: "
            f"rms={float(chosen['rms']):.6g}, max={float(chosen['max']):.6g}. "
            "Inspect the mesh or relax the planarity limits explicitly."
        )

    remove_ids = np.asarray(chosen["ids"], dtype=np.int64)
    keep = np.ones(len(faces), dtype=bool)
    keep[remove_ids] = False
    new_faces = faces[keep]
    new_vertices, new_faces = _compact_vertices(vertices, new_faces)

    result = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    boundary_loops, boundary_edge_count = _boundary_loop_count(new_faces)
    connected_components = _face_connected_component_count(new_faces)

    metrics: dict[str, float | int | bool | str] = {
        "input_vertices": int(len(vertices)),
        "input_faces": int(len(faces)),
        "input_watertight": bool(mesh.is_watertight),
        "detector": "dense_coherent_face_normals_plus_planarity",
        "height_used": False,
        "normal_cluster_angle_degrees": float(normal_cluster_angle_degrees),
        "component_normal_angle_degrees": float(component_normal_angle_degrees),
        "normal_density_peak": int(peak_density),
        "candidate_faces": int(len(candidate_ids)),
        "candidate_components": int(len(components)),
        "scored_components": int(len(component_records)),
        "removed_faces": int(len(remove_ids)),
        "removed_area": float(chosen["area"]),
        "removed_mean_normal_density": float(chosen["density"]),
        "removed_normal_coherence": float(chosen["coherence"]),
        "removed_planarity_rms": float(chosen["rms"]),
        "removed_planarity_max": float(chosen["max"]),
        "output_vertices": int(len(new_vertices)),
        "output_faces": int(len(new_faces)),
        "output_watertight": bool(result.is_watertight),
        "connected_components": int(connected_components),
        "boundary_loops": int(boundary_loops),
        "boundary_edges": int(boundary_edge_count),
        "disk_topology_candidate": bool(connected_components == 1 and boundary_loops == 1),
    }
    return result, metrics


def _print_metrics(metrics: dict[str, float | int | bool | str]) -> None:
    print(
        f"Input:  V={metrics['input_vertices']}, F={metrics['input_faces']}, "
        f"watertight={metrics['input_watertight']}"
    )
    print(
        "Detector: dense coherent face normals + planarity "
        f"(height_used={metrics['height_used']})"
    )
    print(
        f"Normal cluster: peak_density={metrics['normal_density_peak']}, "
        f"candidates={metrics['candidate_faces']} faces in "
        f"{metrics['candidate_components']} connected component(s)"
    )
    print(
        f"Removed: {metrics['removed_faces']} faces, area={float(metrics['removed_area']):.9g}, "
        f"meanNormalDensity={float(metrics['removed_mean_normal_density']):.3f}, "
        f"normalCoherence={float(metrics['removed_normal_coherence']):.6f}"
    )
    print(
        f"Planarity: rms={float(metrics['removed_planarity_rms']):.6g}, "
        f"max={float(metrics['removed_planarity_max']):.6g} (normalized by component extent)"
    )
    print(
        f"Output: V={metrics['output_vertices']}, F={metrics['output_faces']}, "
        f"watertight={metrics['output_watertight']}"
    )
    print(
        f"Topology: connected_components={metrics['connected_components']}, "
        f"boundary_loops={metrics['boundary_loops']}, boundary_edges={metrics['boundary_edges']}"
    )
    if bool(metrics["disk_topology_candidate"]):
        print("Result: OK candidate for OneString open-disk input (1 component, 1 boundary loop).")
    else:
        print("Result: WARNING - expected 1 connected component and 1 boundary loop; inspect before OneString.")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Delete an artificial planar cap by detecting a dense, coherent face-normal cluster. "
            "Geometric height is not used."
        )
    )
    parser.add_argument("input", type=Path, help="Closed input mesh (PLY/OBJ/etc.)")
    parser.add_argument("output", type=Path, help="Open output mesh")
    parser.add_argument(
        "--normal-cluster-angle",
        type=float,
        default=3.0,
        help="Angular radius in degrees for global face-normal density (default: 3.0)",
    )
    parser.add_argument(
        "--component-normal-angle",
        type=float,
        default=4.0,
        help="Maximum directed normal difference from densest normal for candidate faces (default: 4.0)",
    )
    parser.add_argument(
        "--minimum-cluster-faces",
        type=int,
        default=12,
        help="Minimum connected face count for a removable normal cluster (default: 12)",
    )
    parser.add_argument(
        "--maximum-planarity-rms",
        type=float,
        default=0.004,
        help="Maximum normalized RMS plane thickness of removed component (default: 0.004)",
    )
    parser.add_argument(
        "--maximum-planarity-max",
        type=float,
        default=0.012,
        help="Maximum normalized maximum plane thickness of removed component (default: 0.012)",
    )
    parser.add_argument(
        "--allow-open-input",
        action="store_true",
        help="Allow a non-watertight input mesh (not recommended for the Bunny workflow)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not (0.05 <= args.normal_cluster_angle <= 45.0):
        parser.error("--normal-cluster-angle must be between 0.05 and 45 degrees")
    if not (0.05 <= args.component_normal_angle <= 45.0):
        parser.error("--component-normal-angle must be between 0.05 and 45 degrees")
    if args.minimum_cluster_faces < 2:
        parser.error("--minimum-cluster-faces must be >= 2")
    if args.maximum_planarity_rms <= 0.0 or args.maximum_planarity_max <= 0.0:
        parser.error("planarity limits must be positive")

    mesh = _load_single_mesh(args.input)
    result, metrics = remove_flat_bottom(
        mesh,
        normal_cluster_angle_degrees=float(args.normal_cluster_angle),
        component_normal_angle_degrees=float(args.component_normal_angle),
        minimum_cluster_faces=int(args.minimum_cluster_faces),
        maximum_planarity_rms=float(args.maximum_planarity_rms),
        maximum_planarity_max=float(args.maximum_planarity_max),
        require_watertight_input=not bool(args.allow_open_input),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.export(args.output)
    _print_metrics(metrics)
    print(f"Saved: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
