import json
from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics.abd_backend import (
    ABDBackendConfig,
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
    assert scene["rigid_body_problem"]["linear_constraints"][0]["type"] == "pin_joint"
    assert manifest["string"]["inequality"] == "L(q) <= L_command(t)"
    assert manifest["string"]["compression_force_when_slack"] == 0.0
    assert manifest["shake_trajectory"]["frequency_hz"] == 3.0
    assert job.guide_count == 2


def test_abd_selection_never_falls_back_to_legacy_when_executable_is_missing(tmp_path):
    config = ABDBackendConfig(executable=str(tmp_path / "missing-abd-sim.exe"))
    with pytest.raises(ABDBackendUnavailableError):
        run_abd_backend(_minimal_state(), config, tmp_path / "job", tmp_path)
