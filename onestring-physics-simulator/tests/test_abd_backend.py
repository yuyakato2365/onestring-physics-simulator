import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics.abd_backend import (
    ABDBackendConfig,
    ABDBackendError,
    ABDBackendUnavailableError,
    ShakeTrajectory,
    prepare_abd_job,
    run_abd_backend,
)


def _minimal_state():
    tile0 = np.asarray(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, -0.1], [1, 0, -0.1], [1, 1, -0.1], [0, 1, -0.1]],
        dtype=float,
    )
    tile1 = tile0 + np.asarray([1.2, 0.0, 0.0])
    assembly = SimpleNamespace(vertices=np.asarray([tile0, tile1]), metrics={"tile_thickness": 0.1})
    hinge = SimpleNamespace(
        tile_a=0, tile_b=1, surface="top", local_vertex_a=1, local_vertex_b=0,
        target_position_3d=np.asarray([1.1, 0.0, 0.0]),
    )
    gaps = [
        SimpleNamespace(id=0, surrounding_tiles=[0], centroid_2d=np.asarray([0.0, 0.0]), centroid_3d=np.asarray([0.0, 0.0, 0.0])),
        SimpleNamespace(id=1, surrounding_tiles=[1], centroid_2d=np.asarray([1.2, 0.0]), centroid_3d=np.asarray([1.2, 0.0, 0.0])),
    ]
    return SimpleNamespace(
        tiles_3d=assembly,
        tiles_2d_dual_hinge=SimpleNamespace(vertices=np.asarray([tile0, tile1])),
        hinge_graph=SimpleNamespace(hinges=[hinge]),
        gap_graph=SimpleNamespace(gaps=gaps),
        string_path=SimpleNamespace(gap_ids=[0, 1]),
    )


def test_abd_job_contains_official_scene_and_onestring_extension_manifest(tmp_path):
    config = ABDBackendConfig(
        steps=12,
        timestep=0.005,
        shake=ShakeTrajectory(amplitude=0.02, frequency_hz=3.0, direction=(0.0, 1.0, 0.0), end_time=1.0),
    )
    job = prepare_abd_job(_minimal_state(), config, tmp_path / "abd_job")
    scene = json.loads(job.scene_path.read_text(encoding="utf-8"))
    manifest = json.loads(job.manifest_path.read_text(encoding="utf-8"))
    assert scene["solver"] == "ipc_solver"
    assert scene["distance_barrier_constraint"]["trajectory_type"] == "ACCD"
    assert len(scene["rigid_body_problem"]["rigid_bodies"]) == 2
    assert manifest["desk"]["enabled"] is True
    assert manifest["desk"]["model"] == "smooth_unilateral_support_plane_penalty"
    assert len(manifest["desk"]["support_points"]) == 8
    assert manifest["tile_count"] == 2
    assert manifest["simulation_body_count"] == 2
    assert scene["rigid_body_problem"]["linear_constraints"][0]["type"] == "pin_joint"
    assert manifest["string"]["inequality"] == "L(q) <= L_command(t)"
    assert manifest["string"]["compression_force_when_slack"] == 0.0
    assert manifest["shake_trajectory"]["frequency_hz"] == 3.0
    assert job.guide_count == 2
    assert job.initial_layout == "t2d_dual_hinge_exact_positive_clearance"
    assert manifest["collision_skin"] > 0.0
    assert manifest["rest_collision_geometry"] == "exact_t2d_thick_panel_proxy"


def test_abd_prefers_dual_hinge_t2d_layout_over_top_hinge_layout(tmp_path):
    state = _minimal_state()
    top_hinge_vertices = state.tiles_2d_dual_hinge.vertices.copy()
    top_hinge_vertices[1] += np.asarray([2.0, 0.0, 0.0])
    state.tiles_2d_top_hinge = SimpleNamespace(vertices=top_hinge_vertices)

    job = prepare_abd_job(state, ABDBackendConfig(steps=2), tmp_path / "abd_top_hinge")
    scene = json.loads(job.scene_path.read_text(encoding="utf-8"))

    assert job.initial_layout == "t2d_dual_hinge_exact_positive_clearance"
    positions = np.asarray([body["position"] for body in scene["rigid_body_problem"]["rigid_bodies"]])
    assert np.linalg.norm(positions[1, :2] - positions[0, :2]) == pytest.approx(1.2)


def test_abd_job_accepts_lifted_three_component_centroid_2d(tmp_path):
    state = _minimal_state()
    state.gap_graph.gaps[0].centroid_2d = np.asarray([0.0, 0.0, 7.5])
    state.gap_graph.gaps[1].centroid_2d = np.asarray([1.2, 0.0, -3.0])

    job = prepare_abd_job(state, ABDBackendConfig(steps=2), tmp_path / "abd_job_3d_centroid")
    manifest = json.loads(job.manifest_path.read_text(encoding="utf-8"))

    assert job.guide_count == 2
    assert len(manifest["string"]["guide_points"]) == 2


def test_abd_bottom_hinge_accepts_absolute_proxy_vertex_indices(tmp_path):
    state = _minimal_state()
    state.hinge_graph.hinges[0].surface = "bottom"
    state.hinge_graph.hinges[0].local_vertex_a = 5
    state.hinge_graph.hinges[0].local_vertex_b = 4

    job = prepare_abd_job(state, ABDBackendConfig(steps=2), tmp_path / "abd_bottom_hinge")
    scene = json.loads(job.scene_path.read_text(encoding="utf-8"))
    constraint = scene["rigid_body_problem"]["linear_constraints"][0]

    assert constraint["type"] == "pin_joint"
    assert np.all(np.isfinite(np.asarray(constraint["bodyA_point"], dtype=float)))


def test_abd_bottom_hinge_keeps_legacy_face_local_indices_compatible(tmp_path):
    state = _minimal_state()
    state.hinge_graph.hinges[0].surface = "bottom"
    state.hinge_graph.hinges[0].local_vertex_a = 1
    state.hinge_graph.hinges[0].local_vertex_b = 0

    job = prepare_abd_job(state, ABDBackendConfig(steps=2), tmp_path / "abd_legacy_bottom_hinge")
    scene = json.loads(job.scene_path.read_text(encoding="utf-8"))
    constraint = scene["rigid_body_problem"]["linear_constraints"][0]

    assert constraint["type"] == "pin_joint"
    assert np.all(np.isfinite(np.asarray(constraint["bodyA_point"], dtype=float)))


def test_abd_selection_never_falls_back_to_legacy_when_executable_is_missing(tmp_path):
    config = ABDBackendConfig(executable=str(tmp_path / "missing-abd-sim.exe"))
    with pytest.raises(ABDBackendUnavailableError):
        run_abd_backend(_minimal_state(), config, tmp_path / "job", tmp_path)


def test_abd_timeout_is_reported_as_backend_error(monkeypatch, tmp_path):
    executable = tmp_path / "abd_sim.exe"
    executable.write_bytes(b"placeholder")

    def fake_run(command, **kwargs):
        if "--help" in command:
            return subprocess.CompletedProcess(
                command, 0,
                stdout="--scene-path --output-path --onestring-manifest",
                stderr="",
            )
        raise subprocess.TimeoutExpired(command, timeout=0.01, output="0%", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    config = ABDBackendConfig(executable=str(executable), steps=48, timeout_seconds=0.01)
    with pytest.raises(ABDBackendError, match="exceeded the 0.01-second time limit"):
        run_abd_backend(_minimal_state(), config, tmp_path / "timeout_job", tmp_path)
