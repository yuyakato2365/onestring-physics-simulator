import json

import numpy as np

from onestring_physics import PipelineParameters, build_paper_reference_initialization
from onestring_physics.input_shape import create_builtin_shape


def _params(tmp_path, name, **kwargs):
    values = dict(
        nx=5,
        ny=5,
        surface_mesh_subdivisions=2,
        omega_parameterization_mode="paper_reference_bff",
        bff_boundary_policy="automatic_reference",
        reference_grid_spacing=0.5,
        reference_grid_rotation_degrees=0.0,
        reference_grid_origin_u=0.25,
        reference_grid_origin_v=0.25,
        reference_stop_on_required_split=False,
        reference_diagnostics_path=str(tmp_path / f"{name}.json"),
    )
    values.update(kwargs)
    return PipelineParameters(**values)


def _similarity_rms(a, b):
    aa = np.asarray(a, dtype=float) - np.mean(a, axis=0)
    bb = np.asarray(b, dtype=float) - np.mean(b, axis=0)
    u, _s, vt = np.linalg.svd(aa.T @ bb)
    rotation = u @ vt
    aligned = aa @ rotation
    scale = float(np.sum(aligned * bb) / max(np.sum(aligned * aligned), 1e-300))
    return float(np.sqrt(np.mean(np.sum((scale * aligned - bb) ** 2, axis=1))))


def test_reference_flat_disk_matches_input_plane_up_to_similarity_and_round_trips(tmp_path):
    target = create_builtin_shape("dome", {"amplitude": 0.0, "radius": 2.0})
    state = build_paper_reference_initialization(target, _params(tmp_path, "flat"))
    rms = _similarity_rms(
        state.surface_parameterization.uv_vertices_2d,
        state.target_surface.vertices[:, :2],
    )
    assert rms < 1e-5
    assert state.diagnostics["M3D_round_trip_error_max"] < 1e-9
    assert state.diagnostics["fallbacks_used"] == []


def test_reference_gaussian_bump_saves_flip_singular_value_and_lambda_diagnostics(tmp_path):
    target = create_builtin_shape("gaussian", {"amplitude": 0.25, "radius": 2.0, "sigma": 0.9})
    state = build_paper_reference_initialization(target, _params(tmp_path, "gaussian"))
    saved = json.loads((tmp_path / "gaussian.json").read_text(encoding="utf-8"))
    assert saved["uv_triangle_flip_count"] == 0
    assert saved["boundary_self_intersection_count"] == 0
    assert len(saved["per_triangle_sigma1"]) == len(state.target_surface.faces)
    assert saved["lambda_statistics"]["max"] >= saved["lambda_statistics"]["min"] >= 1.0 - 1e-9


def test_reference_saddle_completes_with_negative_gaussian_curvature_and_reports_overlap(tmp_path):
    target = create_builtin_shape("saddle", {"amplitude": 0.2, "radius": 2.0})
    state = build_paper_reference_initialization(target, _params(tmp_path, "saddle"))
    assert state.diagnostics["uv_triangle_flip_count"] == 0
    assert state.diagnostics["internal_triangle_overlap_count"] >= 0
    assert state.diagnostics["M3D_round_trip_error_max"] < 1e-8


def test_reference_half_snowman_user_shape_produces_comparable_initialization(tmp_path):
    target = create_builtin_shape("snowman_half", {"amplitude": 0.25, "radius": 2.0})
    state = build_paper_reference_initialization(target, _params(tmp_path, "snowman_half", reference_grid_spacing=0.35))
    assert state.surface_parameterization.method == "paper_reference_bff"
    assert state.diagnostics["M2D_quad_count"] > 0
    assert state.diagnostics["scope"]["K3D_AND_LATER"] == "not executed by build_paper_reference_initialization"

