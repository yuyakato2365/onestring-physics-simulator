import numpy as np
import pytest

from onestring_physics.discrete_bff import discrete_bff_rectangle


class _FixedRectangleParams:
    boundary_target_aspect_mode = "fixed"
    boundary_target_aspect_ratio = 1.5
    boundary_target_aspect_min = 0.2
    boundary_target_aspect_max = 5.0


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
            faces.append([a, b, d])
            faces.append([a, d, c])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def test_discrete_bff_curved_disk_is_closed_flip_free_and_not_lscm() -> None:
    vertices, faces = _grid_mesh(curved=True)
    uv, boundary_loop, metrics = discrete_bff_rectangle(vertices, faces, _FixedRectangleParams())

    assert uv.shape == (len(vertices), 2)
    assert len(boundary_loop) == 2 * (8 - 1) + 2 * (7 - 1)
    assert np.isfinite(uv).all()
    assert metrics["bff_implemented"] is True
    assert metrics["bff_cherrier_formula_implemented"] is True
    assert metrics["bff_best_fit_curve_implemented"] is True
    assert metrics["bff_uses_lscm"] is False
    assert metrics["lscm_implemented_in_this_path"] is False
    assert metrics["bff_best_fit_closure_error"] < 1e-8
    assert metrics["bff_best_fit_min_corrected_length"] > 0.0
    assert metrics["boundary_target_exterior_angle_sum"] == pytest.approx(2.0 * np.pi, abs=1e-9)
    assert metrics["uv_triangle_flip_count"] == 0
    assert metrics["uv_degenerate_triangle_count"] == 0


def test_discrete_bff_planar_disk_satisfies_gauss_bonnet_and_neumann_compatibility() -> None:
    vertices, faces = _grid_mesh(curved=False)
    uv, _boundary_loop, metrics = discrete_bff_rectangle(vertices, faces, _FixedRectangleParams())

    assert np.isfinite(uv).all()
    assert metrics["bff_gauss_bonnet_error"] < 1e-8
    assert metrics["bff_neumann_rhs_sum_after_projection"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["bff_best_fit_closure_error"] < 1e-8
    assert metrics["uv_triangle_flip_count"] == 0


def test_discrete_bff_rejects_closed_non_disk_mesh() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=int)

    with pytest.raises(RuntimeError, match="boundary|disk"):
        discrete_bff_rectangle(vertices, faces, _FixedRectangleParams())
