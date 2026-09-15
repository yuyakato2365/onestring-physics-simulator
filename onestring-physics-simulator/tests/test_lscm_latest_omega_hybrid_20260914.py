from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

from onestring_physics.lscm_latest_omega_hybrid_20260914_patch import (
    MODE,
    install_lscm_latest_omega_hybrid_patch,
)


@dataclass
class Params:
    omega_parameterization_mode: str = MODE


class Result:
    def __init__(self):
        self.metrics = {}
        self.method = ""
        self.faces = [0, 1, 2]


def _pipeline_with_stubs(calls):
    def latest_parameterization(surface, target, grid, params):
        del surface, target, grid
        calls.append(("latest_omega", params.omega_parameterization_mode))
        return Result()

    def latest_m2d(grid, domain, params=None):
        del grid, domain, params
        calls.append(("outer_m2d", None))
        return Result()

    def latest_k3d(target, mesh, parameterization, params):
        del target, mesh, parameterization
        calls.append(("latest_k3d", params.omega_parameterization_mode))
        return Result(), SimpleNamespace()

    def latest_k2d(mesh_2d, mesh_3d, params, progress_callback=None):
        del mesh_2d, mesh_3d, params, progress_callback
        calls.append(("outer_k2d", None))
        return Result(), SimpleNamespace()

    def latest_layout(mesh, params=None):
        del mesh, params
        calls.append(("outer_layout", None))
        return Result()

    build_globals = {}

    def build_onestring_design():
        return build_globals

    pipeline = SimpleNamespace(
        _build_surface_parameterization=latest_parameterization,
        _build_m2d=latest_m2d,
        _optimize_k3d=latest_k3d,
        _optimize_k2d=latest_k2d,
        _make_flat_tile_layout=latest_layout,
        build_onestring_design=build_onestring_design,
        _ORIGINAL_BUILD_ONESTRING_DESIGN=None,
        _original=None,
    )
    return pipeline


def test_hybrid_routes_only_omega_and_k3d_to_latest(monkeypatch):
    calls = []
    pipeline = _pipeline_with_stubs(calls)

    def lscm_m2d(grid, domain, params=None):
        del grid, domain, params
        calls.append(("lscm_m2d", None))
        return Result()

    def lscm_k2d(mesh_2d, mesh_3d, params, progress_callback=None):
        del mesh_2d, mesh_3d, params, progress_callback
        calls.append(("lscm_k2d", None))
        return Result(), SimpleNamespace()

    def lscm_layout(mesh, params=None):
        del mesh, params
        calls.append(("lscm_layout", None))
        return Result()

    # Avoid touching the real Simple Split module during this unit test.
    import onestring_physics.simple_split_panel_patch as split_module

    monkeypatch.setattr(split_module, "_onestring_lscm_hybrid_rewire_installed", True, raising=False)

    install_lscm_latest_omega_hybrid_patch(
        pipeline,
        lscm_build_m2d=lscm_m2d,
        lscm_optimize_k2d=lscm_k2d,
        lscm_make_flat_tile_layout=lscm_layout,
    )

    params = Params()
    omega = pipeline._build_surface_parameterization(None, None, None, params)
    assert omega.method == MODE
    assert ("latest_omega", "optcuts_test") in calls

    pipeline._build_m2d(None, None, params)
    assert ("lscm_m2d", None) in calls
    assert ("outer_m2d", None) not in calls

    pipeline._optimize_k3d(None, None, omega, params)
    assert ("latest_k3d", "optcuts_test") in calls

    pipeline._optimize_k2d(None, None, params)
    assert ("lscm_k2d", None) in calls
    assert ("outer_k2d", None) not in calls

    pipeline._make_flat_tile_layout(None, params)
    assert ("lscm_layout", None) in calls
    assert ("outer_layout", None) not in calls


def test_non_hybrid_mode_keeps_existing_routes(monkeypatch):
    calls = []
    pipeline = _pipeline_with_stubs(calls)

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("captured LSCM override must not run for non-hybrid modes")

    import onestring_physics.simple_split_panel_patch as split_module

    monkeypatch.setattr(split_module, "_onestring_lscm_hybrid_rewire_installed", True, raising=False)

    install_lscm_latest_omega_hybrid_patch(
        pipeline,
        lscm_build_m2d=forbidden,
        lscm_optimize_k2d=forbidden,
        lscm_make_flat_tile_layout=forbidden,
    )

    params = replace(Params(), omega_parameterization_mode="lscm")
    pipeline._build_surface_parameterization(None, None, None, params)
    pipeline._build_m2d(None, None, params)
    pipeline._optimize_k3d(None, None, Result(), params)
    pipeline._optimize_k2d(None, None, params)
    pipeline._make_flat_tile_layout(None, params)

    assert ("latest_omega", "lscm") in calls
    assert ("outer_m2d", None) in calls
    assert ("latest_k3d", "lscm") in calls
    assert ("outer_k2d", None) in calls
    assert ("outer_layout", None) in calls
