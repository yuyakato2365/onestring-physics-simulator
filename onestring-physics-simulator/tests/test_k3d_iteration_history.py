"""Regression tests for the authoritative K3D optimization history and the 3D views.

These lock in three things the UI depends on:

* the history comes from the solver that actually produces K3D (no shadow solve),
* the last history row describes exactly the K3D that is returned,
* the K3D figure carries finite 3D geometry.
"""
from __future__ import annotations

import numpy as np
import pytest

from onestring_physics import onestring_pipeline as pipeline
from onestring_physics.paper_local_global_solvers import (
    _best_fit_plane_projection,
    _closest_square_projection,
    _edges,
    _surface_project,
    optimize_paper_local_global_k3d,
)

REQUIRED_FIELDS = (
    "EAssembled",
    "w1EPlanar",
    "w2ESquare",
    "w3ESurface",
    "planarity_max",
    "planarity_rms",
    "planarity_mean",
)


def _dome_surface(n: int = 7, amplitude: float = 0.35):
    """A small doubly curved triangle mesh used as the target surface."""
    xs = np.linspace(-1.0, 1.0, n)
    grid_x, grid_y = np.meshgrid(xs, xs, indexing="ij")
    grid_z = amplitude * np.cos(1.1 * grid_x) * np.cos(1.1 * grid_y)
    vertices = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            faces.append([a, a + 1, a + n + 1])
            faces.append([a, a + n + 1, a + n])
    return vertices, np.asarray(faces, dtype=int)


class _Parameterization:
    def __init__(self, vertices, faces):
        self.surface_vertices_3d = vertices
        self.surface_faces = faces
        self.method = "test"


class _Target:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces


@pytest.fixture(scope="module")
def solved_k3d():
    surface_vertices, surface_faces = _dome_surface()
    n = 5
    xs = np.linspace(-0.8, 0.8, n)
    grid_x, grid_y = np.meshgrid(xs, xs, indexing="ij")
    grid_z = 0.3 * np.cos(1.1 * grid_x) * np.cos(1.1 * grid_y)
    m3d_vertices = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)
    quads = np.asarray(
        [[i * n + j, i * n + j + 1, (i + 1) * n + j + 1, (i + 1) * n + j]
         for i in range(n - 1) for j in range(n - 1)],
        dtype=int,
    )
    mesh = pipeline.QuadMesh(m3d_vertices, quads, (n, n), "M3D", {})
    parameterization = _Parameterization(surface_vertices, surface_faces)
    target = _Target(surface_vertices, surface_faces)
    out, _report = optimize_paper_local_global_k3d(
        target, mesh, parameterization, pipeline.PipelineParameters(), pipeline=pipeline
    )
    return dict(mesh=mesh, out=out, target=target, parameterization=parameterization)


def test_k3d_history_has_multiple_checkpoints(solved_k3d):
    history = solved_k3d["out"].metrics["k3d_iteration_history"]
    records = history["records"]

    assert history["solver"] == "optimize_paper_local_global_k3d"
    assert history["shadow_solve"] is False, "history must come from the authoritative solve"
    assert len(records) > 1, "expected iteration 0 plus at least one optimizer iteration"
    assert [r["iteration"] for r in records] == list(range(len(records)))
    for record in records:
        for field in REQUIRED_FIELDS:
            assert field in record, f"missing {field} in K3D history record"
            assert np.isfinite(record[field])
    # The optimizer must actually make the tiles more planar.
    assert records[-1]["planarity_max"] < records[0]["planarity_max"]
    assert records[-1]["EAssembled"] < records[0]["EAssembled"]


def test_k3d_history_final_matches_authoritative_output(solved_k3d):
    out = solved_k3d["out"]
    mesh = solved_k3d["mesh"]
    history = out.metrics["k3d_iteration_history"]
    record = history["records"][-1]
    weights = history["weights"]

    vertices = np.asarray(out.vertices, dtype=float)
    faces = np.asarray(out.faces, dtype=int)
    edges = _edges(faces)

    source = np.asarray(mesh.vertices, dtype=float)
    source_faces = np.asarray(mesh.faces, dtype=int)
    face_mean = np.asarray(
        [np.mean(np.linalg.norm(np.roll(source[f], -1, axis=0) - source[f], axis=1)) for f in source_faces]
    )
    owners: dict[tuple[int, int], list[int]] = {}
    for index, face in enumerate(source_faces):
        for k in range(4):
            owners.setdefault(tuple(sorted((int(face[k]), int(face[(k + 1) % 4])))), []).append(index)

    planar_sum = 0.0
    deviations = []
    square_sum = 0.0
    for face in faces:
        quad = vertices[face]
        planar_offset = quad - _best_fit_plane_projection(quad)
        planar_sum += float(np.sum(planar_offset * planar_offset))
        deviations.append(np.linalg.norm(planar_offset, axis=1))
        square_offset = quad - _closest_square_projection(quad)
        square_sum += float(np.sum(square_offset * square_offset))
    for a, b in edges:
        direction = vertices[int(b)] - vertices[int(a)]
        length = float(np.linalg.norm(direction))
        if length < 1e-12:
            continue
        incident = owners.get((int(a), int(b)), [])
        target_length = float(np.mean(face_mean[incident])) if incident else length
        offset = direction - target_length * direction / length
        square_sum += float(np.dot(offset, offset))
    surface_offset = vertices - _surface_project(vertices, solved_k3d["target"], solved_k3d["parameterization"])
    surface_sum = float(np.sum(surface_offset * surface_offset))
    deviation = np.concatenate(deviations)

    expected = {
        "w1EPlanar": weights["w1_planar"] * planar_sum,
        "w2ESquare": weights["w2_square"] * square_sum,
        "w3ESurface": weights["w3_surface"] * surface_sum,
        "planarity_max": float(np.max(deviation)),
        "planarity_rms": float(np.sqrt(np.mean(deviation * deviation))),
        "planarity_mean": float(np.mean(deviation)),
    }
    expected["EAssembled"] = expected["w1EPlanar"] + expected["w2ESquare"] + expected["w3ESurface"]
    for name, value in expected.items():
        assert record[name] == pytest.approx(value, rel=1e-9, abs=1e-15), name


def test_k3d_figure_contains_finite_3d_geometry(solved_k3d):
    from onestring_physics.visualization import figure_quad_mesh

    figure = figure_quad_mesh(solved_k3d["out"], title="K3D")
    assert len(figure.data) > 0, "K3D figure must not be empty"

    three_d = [trace for trace in figure.data if trace.type in {"mesh3d", "scatter3d"}]
    assert three_d, "K3D figure must contain 3D traces"
    for trace in three_d:
        for axis in ("x", "y", "z"):
            values = np.asarray([v for v in getattr(trace, axis) if v is not None], dtype=float)
            assert values.size > 0, f"{trace.type}.{axis} is empty"
            assert np.all(np.isfinite(values)), f"{trace.type}.{axis} has non-finite values"
        assert np.ptp(np.asarray([v for v in trace.x if v is not None], dtype=float)) > 0

    scene = figure.layout.scene
    for axis in (scene.xaxis, scene.yaxis, scene.zaxis):
        if axis.range is not None:
            assert np.all(np.isfinite(np.asarray(axis.range, dtype=float)))


def test_k3d_history_view_reports_missing_history_without_solving():
    """The view must never fall back to running its own optimizer."""
    from onestring_physics.k3d_iteration_history_view import get_k3d_history

    class _State:
        mesh_3d_optimized = pipeline.QuadMesh(
            np.zeros((4, 3)), np.asarray([[0, 1, 2, 3]]), (2, 2), "K3D", {"k3d_solver_model": "other"}
        )

    assert get_k3d_history(_State()) is None


def test_k3d_history_is_not_shadow_solve(solved_k3d, monkeypatch):
    from types import SimpleNamespace
    import onestring_physics.paper_local_global_solvers as solvers
    from onestring_physics.k3d_iteration_history_view import get_k3d_history
    def forbidden(*args, **kwargs):
        raise AssertionError("display must not run the optimizer")
    monkeypatch.setattr(solvers, "optimize_paper_local_global_k3d", forbidden)
    mesh = solved_k3d["out"]
    state = SimpleNamespace(mesh_3d_optimized=mesh)
    history = get_k3d_history(state)
    assert history is mesh.metrics["k3d_iteration_history"]
    assert history["shadow_solve"] is False
