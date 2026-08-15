import numpy as np
import pytest

from onestring_physics import PipelineParameters, build_onestring_design, create_builtin_shape
from onestring_physics.bijective_free_boundary import (
    BijectiveFreeBoundaryConfig,
    bijective_free_boundary_parameterization,
)
from onestring_physics.visualization import figure_domain


def _grid_mesh(nx: int = 8, ny: int = 7, *, curved: bool = True) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(-1.0, 1.0, nx)
    ys = np.linspace(-0.8, 0.8, ny)
    vertices = []
    for y in ys:
        for x in xs:
            z = 0.25 * np.sin(1.4 * x) * np.cos(1.2 * y) if curved else 0.0
            vertices.append([x, y, z])
    faces = []
    for row in range(ny - 1):
        for column in range(nx - 1):
            a = row * nx + column
            b = a + 1
            c = a + nx
            d = c + 1
            faces.extend([[a, b, d], [a, d, c]])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def test_flat_grid_is_finite_bijective_and_optimized() -> None:
    vertices, faces = _grid_mesh(curved=False)
    uv, boundary_loop, metrics = bijective_free_boundary_parameterization(vertices, faces)

    assert uv.shape == (len(vertices), 2)
    assert len(boundary_loop) == 2 * (8 - 1) + 2 * (7 - 1)
    assert np.isfinite(uv).all()
    assert metrics["uv_triangle_flip_count"] == 0
    assert metrics["uv_degenerate_triangle_count"] == 0
    assert metrics["internal_triangle_overlap_count"] == 0
    assert metrics["boundary_self_intersection_count"] == 0
    assert metrics["optimization_succeeded"] is True
    assert metrics["optimization_converged"] is True
    assert metrics["final_energy"] < metrics["initial_energy"]
    assert metrics["onestring_grid_loss_used"] is False
    assert metrics["lambda_directly_optimized"] is False


def test_curved_grid_stays_bijective_with_a_free_boundary() -> None:
    vertices, faces = _grid_mesh(curved=True)
    uv, _boundary_loop, metrics = bijective_free_boundary_parameterization(
        vertices,
        faces,
        BijectiveFreeBoundaryConfig(max_iterations=60),
    )

    assert np.isfinite(uv).all()
    assert metrics["omega_boundary_fixed"] is False
    assert metrics["omega_boundary_shape"] == "free"
    assert metrics["uv_triangle_flip_count"] == 0
    assert metrics["internal_triangle_overlap_count"] == 0
    assert metrics["boundary_self_intersection_count"] == 0
    assert metrics["initial_internal_triangle_overlap_count"] == 0
    assert metrics["initial_uv_triangle_flip_count"] == 0
    assert np.isfinite(metrics["lambda_min"])
    assert np.isfinite(metrics["log_lambda_max"])
    assert metrics["anisotropy_max"] >= 1.0


def test_closed_non_disk_mesh_is_rejected() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=int)

    with pytest.raises(RuntimeError, match="boundary|disk"):
        bijective_free_boundary_parameterization(vertices, faces)


def test_pipeline_mode_is_explicit_and_does_not_replace_bff() -> None:
    target = create_builtin_shape("dome", {"amplitude": 0.35, "radius": 2.0})
    state = build_onestring_design(
        target,
        PipelineParameters(
            nx=2,
            max_3d_iterations=2,
            max_2d_iterations=2,
            omega_parameterization_mode="bijective_free_boundary",
            bijective_free_boundary_max_iterations=40,
        ),
    )
    metrics = state.surface_parameterization.metrics

    assert state.surface_parameterization.method == "bijective_free_boundary"
    assert metrics["parameterization_method"] == "bijective_free_boundary"
    assert metrics["omega_boundary_shape"] == "free"
    assert metrics["omega_boundary_fixed"] is False
    assert metrics.get("bff_backend_used") != "local_discrete_bff"
    assert metrics["uv_triangle_flip_count"] == 0
    assert metrics["internal_triangle_overlap_count"] == 0
    assert metrics["boundary_self_intersection_count"] == 0
    figure = figure_domain(state)
    trace_names = {str(trace.name) for trace in figure.data}
    assert "parameterized UV mesh" in trace_names
    assert "local log(lambda)" in trace_names
