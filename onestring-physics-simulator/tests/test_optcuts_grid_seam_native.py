"""Isolated native C++ OptCuts seam penalty / OneString routing regressions."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess

import numpy as np
import pytest

from onestring_physics.optcuts_grid_seam_native import (
    resolve_grid_seam_binary, candidate_roots, MARKER,
)
from onestring_physics.optcuts_backend import OptCutsUnavailableError
from onestring_physics.optcuts_pipeline_patch import _config_from_params


def test_cpp_patcher_injects_split_merge_scores_at_verified_upstream_anchors():
    # Use precise upstream anchors: the patcher MUST fail closed on change.
    from runpy import run_path
    script = Path(__file__).resolve().parents[1] / "scripts" / "setup_optcuts_grid_seam.py"
    patch = run_path(str(script))["patch_trimesh_source"]
    source = r"""
#include <fstream>
namespace OptCuts {
                    const double curEwDec = (1.0 - lambda_t) * SDDec - lambda_t * seInc;
                    if(curEwDec > maxEwDec) {
                    const double EwDec = (1.0 - lambda_t) * SDDec - lambda_t * seInc;
                    if(EwDec > EwDec_max) {
                    return lambda_t * seDec - (1.0 - lambda_t) * SDInc;
}
"""
    out = patch(source)
    assert out.count(MARKER) == 1
    assert out.count("onestringGridEdgeCost(") == 5  # definition + 4 edge-score uses
    assert out.count("onestringGridWeight() * gridCost") == 2
    assert "onestringGridWeight() * gridRemoved" in out
    assert "std::strtod" in out
    assert "std::isfinite" in out
    assert "std::cos" in out
    with pytest.raises(ValueError, match="already patched"):
        patch(out)
    with pytest.raises(ValueError, match="upstream changed"):
        patch(source.replace("const double EwDec = ", "const double X = "))


def test_grid_alignment_penalty_is_zero_for_axis_and_high_for_diagonal():
    def score(a, b):
        d=np.asarray(b)-np.asarray(a)
        z=float(d@d)
        return 4.*d[0]**2*d[1]**2/z**2
    assert score([0.,0.],[1.,0.])==pytest.approx(0.)
    assert score([0.,0.],[0.,1.])==pytest.approx(0.)
    assert score([0.,0.],[1.,1.])==pytest.approx(1.)
    assert score([0.,0.],[1.,.2])<score([0.,0.],[1.,1.])


def test_resolver_refuses_clean_official_binary_and_accepts_patched_one(monkeypatch, tmp_path):
    from onestring_physics import optcuts_grid_seam_native as mod
    project=tmp_path/"project"
    root=project/"third_party"/"OptCuts_grid_seam_onestring"
    source=root/"src"/"TriMesh.cpp"
    binary=root/"build"/"OptCuts_bin"
    source.parent.mkdir(parents=True)
    binary.parent.mkdir(parents=True)
    monkeypatch.setattr(mod,"candidate_roots",lambda _project:[root])
    source.write_text("// Clean Official OptCuts")
    binary.write_bytes(b"clean")
    with pytest.raises(OptCutsUnavailableError,match="Grid-Seam C\\+\\+"):
        resolve_grid_seam_binary(project)
    source.write_text("// "+MARKER)
    os.utime(source, (10.,10.))
    os.utime(binary, (11.,11.))
    assert resolve_grid_seam_binary(project)==binary.resolve()
    os.utime(binary, (9.,9.))
    with pytest.raises(OptCutsUnavailableError,match="missing/stale"):
        resolve_grid_seam_binary(project)


def test_config_routes_new_optcuts_native_settings():
    param=SimpleNamespace(
        omega_parameterization_mode="optcuts",
        optcuts_grid_seam_native_enabled=True,
        optcuts_grid_seam_native_weight=2.5,
        optcuts_try_alternative_initial_cut=True)
    cfg=_config_from_params(param,SimpleNamespace(nx=24,ny=24))
    assert cfg.grid_seam_native_enabled is True
    assert cfg.grid_seam_native_weight==pytest.approx(2.5)
    assert cfg.try_alternative_initial_cut is True
    param.optcuts_grid_seam_native_enabled=False
    other=_config_from_params(param,SimpleNamespace(nx=24,ny=24))
    assert not other.grid_seam_native_enabled


def test_cpp_route_never_calls_official_binary_resolver(monkeypatch, tmp_path):
    from onestring_physics import optcuts_backend as backend
    from onestring_physics import optcuts_grid_seam_native as native
    from onestring_physics.optcuts_backend import OptCutsConfig
    # Deliberately invalid input: resolution fails BEFORE any launch,
    # and the official resolver must not be consulted even if present.
    def forbidden(_):
        raise AssertionError("Called Official OptCuts in native grid mode")
    monkeypatch.setattr(backend,"resolve_optcuts_executable",forbidden)
    monkeypatch.setattr(native,"resolve_grid_seam_binary",
                        lambda project: (_ for _ in ()).throw(
                            OptCutsUnavailableError("Grid-Seam C++ missing")))
    with pytest.raises(OptCutsUnavailableError,match="missing"):
        backend._run_official_optcuts_once(
            np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),
            np.array([[0,1,2]]),
            OptCutsConfig(grid_seam_native_enabled=True))


def test_ui_and_parameter_names_are_isolated_from_10_23_baseline():
    root=Path(__file__).resolve().parents[1]
    selector=(root/"app_optcuts_core_20260916.py").read_text()
    ui=(root/"app_runtime_base.py").read_text()
    params=(root/"src/onestring_physics/onestring_pipeline.py").read_text()
    for token in ("optcuts_grid_seam_native","_onestring_grid_seam_native_selected"):
        assert token in selector
    assert "Grid seam penalty strength" in selector
    assert "optcuts_grid_seam_native_enabled=optcuts_grid_seam_native_enabled" in ui
    assert "optcuts_grid_seam_native_enabled: bool = False" in params
    assert "optcuts_grid_seam_native_weight: float = 0.0" in params
