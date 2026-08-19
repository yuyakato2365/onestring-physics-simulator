#!/usr/bin/env python3
"""Remove the lowest nearly planar connected cap from a closed triangle mesh.

Designed for the OneString Bunny workflow:

closed remeshed Bunny
    -> detect lowest nearly horizontal planar face set
    -> keep the largest connected candidate component
    -> delete those faces only
    -> remove newly unreferenced vertices
    -> verify that the result is one connected component with one boundary loop

The detector is intentionally conservative. A face must satisfy both:
1) all three vertices lie close to the lowest support plane, and
2) the face normal is nearly parallel to the selected up axis.

Among all such candidates, only the largest edge-connected component is removed.
This avoids deleting isolated low faces on feet or curved parts of the model.
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


def _axis_index(name: str) -> int:
    table = {"x": 0, "y": 1, "z": 2}
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise ValueError("up-axis must be one of x, y, z") from exc


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


def remove_flat_bottom(
    mesh: trimesh.Trimesh,
    *,
    up_axis: str = "z",
    height_fraction: float = 0.015,
    normal_threshold: float = 0.97,
    require_watertight_input: bool = True,
) -> tuple[trimesh.Trimesh, dict[str, float | int | bool]]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if require_watertight_input and not bool(mesh.is_watertight):
        raise ValueError("Input is not watertight. Expected a CLOSED mesh before bottom removal.")

    axis = _axis_index(up_axis)
    coordinates = vertices[:, axis]
    minimum = float(np.min(coordinates))
    maximum = float(np.max(coordinates))
    span = maximum - minimum
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError("Degenerate mesh extent along selected up axis")

    threshold_height = minimum + float(height_fraction) * span

    tri = vertices[faces]
    face_max_height = np.max(tri[:, :, axis], axis=1)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    safe_norm = np.maximum(norm, 1.0e-30)
    normals = cross / safe_norm[:, None]
    vertical_alignment = np.abs(normals[:, axis])

    candidate_mask = (
        (face_max_height <= threshold_height)
        & (vertical_alignment >= float(normal_threshold))
        & np.isfinite(vertical_alignment)
    )
    candidate_ids = np.flatnonzero(candidate_mask)
    if len(candidate_ids) == 0:
        raise RuntimeError(
            "No flat-bottom candidate faces found. Increase --height-fraction or lower --normal-threshold."
        )

    components = _face_components(faces, candidate_ids)
    if not components:
        raise RuntimeError("Could not form connected bottom candidate components")
    areas = np.asarray([_component_area(vertices, faces, component) for component in components])
    chosen_index = int(np.argmax(areas))
    remove_ids = components[chosen_index]

    keep = np.ones(len(faces), dtype=bool)
    keep[remove_ids] = False
    new_faces = faces[keep]
    new_vertices, new_faces = _compact_vertices(vertices, new_faces)

    result = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    boundary_loops, boundary_edge_count = _boundary_loop_count(new_faces)
    connected_components = _face_connected_component_count(new_faces)

    metrics: dict[str, float | int | bool] = {
        "input_vertices": int(len(vertices)),
        "input_faces": int(len(faces)),
        "input_watertight": bool(mesh.is_watertight),
        "candidate_faces": int(len(candidate_ids)),
        "candidate_components": int(len(components)),
        "removed_faces": int(len(remove_ids)),
        "removed_area": float(areas[chosen_index]),
        "output_vertices": int(len(new_vertices)),
        "output_faces": int(len(new_faces)),
        "output_watertight": bool(result.is_watertight),
        "connected_components": int(connected_components),
        "boundary_loops": int(boundary_loops),
        "boundary_edges": int(boundary_edge_count),
        "height_fraction": float(height_fraction),
        "normal_threshold": float(normal_threshold),
        "support_plane_min": float(minimum),
        "candidate_height_max": float(threshold_height),
        "disk_topology_candidate": bool(connected_components == 1 and boundary_loops == 1),
    }
    return result, metrics


def _print_metrics(metrics: dict[str, float | int | bool]) -> None:
    print(
        f"Input:  V={metrics['input_vertices']}, F={metrics['input_faces']}, "
        f"watertight={metrics['input_watertight']}"
    )
    print(
        f"Detector: candidates={metrics['candidate_faces']} faces in "
        f"{metrics['candidate_components']} connected component(s)"
    )
    print(
        f"Removed: {metrics['removed_faces']} faces, area={float(metrics['removed_area']):.9g}"
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
        description="Delete the lowest nearly horizontal connected cap from a closed triangle mesh."
    )
    parser.add_argument("input", type=Path, help="Closed input mesh (PLY/OBJ/etc.)")
    parser.add_argument("output", type=Path, help="Open output mesh")
    parser.add_argument("--up-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument(
        "--height-fraction",
        type=float,
        default=0.015,
        help="Bottom slab thickness as a fraction of total mesh height (default: 0.015)",
    )
    parser.add_argument(
        "--normal-threshold",
        type=float,
        default=0.97,
        help="Minimum |dot(face_normal, up_axis)| for a flat face (default: 0.97)",
    )
    parser.add_argument(
        "--allow-open-input",
        action="store_true",
        help="Allow a non-watertight input mesh (not recommended for the Bunny workflow)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not (0.0 < args.height_fraction < 0.5):
        parser.error("--height-fraction must be between 0 and 0.5")
    if not (0.0 <= args.normal_threshold <= 1.0):
        parser.error("--normal-threshold must be between 0 and 1")

    mesh = _load_single_mesh(args.input)
    result, metrics = remove_flat_bottom(
        mesh,
        up_axis=args.up_axis,
        height_fraction=float(args.height_fraction),
        normal_threshold=float(args.normal_threshold),
        require_watertight_input=not bool(args.allow_open_input),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.export(args.output)
    _print_metrics(metrics)
    print(f"Saved: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
