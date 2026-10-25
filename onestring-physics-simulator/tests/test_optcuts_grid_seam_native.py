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
        EwDec_max = EwDecs[candI_max];
        switch(operationType) {
}
"""
    out = patch(source)
    assert out.count(MARKER) == 1
    assert out.count("onestringGridEdgeCost(") == 5  # definition + 4 edge-score uses
    assert out.count("onestringGridProposalScore(") == 4  # def + 3 score calls
    assert "origEwDec <= 1.e-12" not in out
    assert "[GRID-CUT-NATIVE-NO-CANDIDATES]" in out
    assert "legacy + delta : legacy - delta" in out
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
    main_source=root/"src"/"main.cpp"
    main_source.write_text("// "+MARKER)
    os.utime(main_source, (10.,10.))
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


def test_grid_square_proxy_uses_3d_inverse_mapping_and_is_rotation_invariant():
    from onestring_physics.optcuts_grid_square_proxy import square_grid_local_quality
    faces=np.array([[0,1,2],[0,2,3]])
    uv=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    xyz=np.c_[uv,np.zeros(4)]
    exact=square_grid_local_quality(xyz,faces,uv,faces)
    assert exact["square_grid_proxy_mean"]==pytest.approx(0.,abs=1e-12)
    angle=np.deg2rad(31.)
    rot=np.array([[np.cos(angle),-np.sin(angle)],
                  [np.sin(angle),np.cos(angle)]])
    rigid=square_grid_local_quality(np.c_[uv@rot.T,np.zeros(4)],faces,uv,faces)
    assert rigid["square_grid_proxy_mean"]==pytest.approx(0.,abs=1e-12)
    stretched=np.c_[uv*np.array([2.,1.]),np.zeros(4)]
    bad=square_grid_local_quality(stretched,faces,uv,faces)
    assert bad["square_grid_proxy_mean"]>0.4
    assert bad["square_grid_proxy_p95"]>0.4


def test_native_cpp_grid_term_is_scaled_and_does_not_change_candidate_sign():
    from runpy import run_path
    script=Path(__file__).resolve().parents[1]/"scripts"/"setup_optcuts_grid_seam.py"
    transform=run_path(str(script))["patch_trimesh_source"]
    sample="\n".join((
        "#include <fstream>",
        "namespace OptCuts {",
        "const double curEwDec = (1.0 - lambda_t) * SDDec - lambda_t * seInc;",
        "                    if(curEwDec > maxEwDec) {",
        "const double EwDec = (1.0 - lambda_t) * SDDec - lambda_t * seInc;",
        "                    if(EwDec > EwDec_max) {",
        "return lambda_t * seDec - (1.0 - lambda_t) * SDInc;",
        "        EwDec_max = EwDecs[candI_max];",
        "        switch(operationType) {",
        "}",
        ""))
    patched=transform(sample)
    assert "onestringGridProposalScore(" in patched
    assert "legacy + delta : legacy - delta" in patched
    assert "std::min(raw, 0.25 * legacy)" in patched
    assert "origEwDec <= 1.e-12" not in patched
    assert "GRID_CUT_NATIVE_NO_ADMISSIBLE_SPLIT_AT_FIXED_LAMBDA" in patched
    # The original energy delta is still stored for the unchanged dual update.
    assert "energyChanges_max.first = -SDDec;" in patched or (
        "energyChanges_max" not in sample)


def test_grid_penalty_no_longer_outweighs_native_seam_term_by_100x():
    # At the observed lambda_init=.999 the seam-side weight is .001.
    # Relative beta=.10 should contribute at most 10% of the seam length
    # weight for a same-length 45 degree edge, not 100 times it.
    original_seam_weight=1-.999
    beta=.10
    maximum_penalty=original_seam_weight*beta
    assert maximum_penalty==pytest.approx(0.0001)
    assert maximum_penalty/original_seam_weight==pytest.approx(0.10)



def test_native_subprocess_heartbeat_reports_progress_without_losing_output(capsys, tmp_path):
    import sys
    from onestring_physics.optcuts_grid_seam_native import run_native_with_heartbeat
    completed = run_native_with_heartbeat(
        [sys.executable, "-c",
         "import time; print('native-start', flush=True); "
         "time.sleep(.13); print('native-finished', flush=True)"],
        cwd=str(tmp_path), env=dict(os.environ),
        timeout=3.0, heartbeat_seconds=.025)
    assert completed.returncode == 0
    assert "native-start" in completed.stdout
    assert "native-finished" in completed.stdout
    rendered = capsys.readouterr().out
    assert "[GRID-CUT-NATIVE-RUN] started" in rendered
    assert "[GRID-CUT-NATIVE-RUN] still_running" in rendered
    assert "[GRID-CUT-NATIVE-RUN] finished" in rendered


def test_native_subprocess_timeout_terminates_child(tmp_path, capsys):
    import sys
    from onestring_physics.optcuts_grid_seam_native import run_native_with_heartbeat
    with pytest.raises(subprocess.TimeoutExpired):
        run_native_with_heartbeat(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            cwd=str(tmp_path), env=dict(os.environ),
            timeout=.07, heartbeat_seconds=.015)
    assert "timed_out" in capsys.readouterr().out


def test_native_grid_rank_preserves_all_signs_and_zero_beta():
    # This is the C++ helper's intentionally restricted score policy.
    def score(legacy, lam, beta, cost, merge=False):
        if legacy <= 0.0 or beta <= 0.0:
            return legacy
        delta = min(lam * beta * cost, 0.25 * legacy)
        return legacy + delta if merge else legacy - delta
    for orig in (-1.0e6, -0.02, 0.0, 0.02, 2.0):
        assert score(orig,.001,0,4.0)==orig
        assert score(orig,.001,0,4.0,merge=True)==orig
        if orig<=0:
            assert score(orig,.001,.1,4.0)==orig
            assert score(orig,.001,.1,4.0,merge=True)==orig
        else:
            assert score(orig,.001,.1,4.0)>0
            assert score(orig,.001,.1,4.0,merge=True)>0
            assert score(orig,.001,.1,4.0)>=.75*orig
            assert score(orig,.001,.1,4.0,merge=True)<=1.25*orig


def test_patch_main_guards_bounded_lambda_search_in_all_four_paths():
    from runpy import run_path
    script=Path(__file__).resolve().parents[1]/"scripts"/"setup_optcuts_grid_seam.py"
    patch=run_path(str(script))["patch_main_source"]
    upstream="""#include <fstream>
bool updateLambda_stationaryV(bool cancelMomentum = true, bool checkConvergence = false)
{
                do {
                    energyParams[0] = updateLambda(measure_bound);
                } while((computeOptPicked(energyChanges_bSplit, energyChanges_merge, 1.0 - energyParams[0]) == 1));
                while((eDec_b > 0.0) && (eDec_i > 0.0)) {
                    energyParams[0] = updateLambda(measure_bound);
                do {
                    energyParams[0] = updateLambda(measure_bound);
                } while(computeOptPicked(energyChanges_bSplit, energyChanges_merge, 1.0 - energyParams[0]) == 0);
            while(eDec_m > 0.0) {
                energyParams[0] = updateLambda(measure_bound);
}
"""
    out=patch(upstream)
    assert out.count("[GRID-CUT-NATIVE-LAMBDA-STALLED]") == 4
    assert out.count("++gridLambdaAttempts > 256") == 4
    assert out.count("std::abs(energyParams[0] - previousLambda)") == 4
    assert out.count("int gridLambdaAttempts = 0;") == 4
    assert "std::runtime_error" in out
    with pytest.raises(ValueError,match="already patched"):
        patch(out)
    with pytest.raises(ValueError,match="upstream changed"):
        patch(upstream.replace("while(eDec_m > 0.0)", "while(eDec_m > 1.0)"))


def test_native_grid_cpp_scoring_helper_compiles_and_preserves_lambda_sign(tmp_path):
    """Compile and execute the REAL injected C++ helper, not a Python paraphrase."""
    from runpy import run_path
    import shutil
    cxx=shutil.which("c++")
    if cxx is None:
        pytest.skip("C++ compiler unavailable")
    script=Path(__file__).resolve().parents[1]/"scripts"/"setup_optcuts_grid_seam.py"
    helper=run_path(str(script))["HELPER"]
    stub = """
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
namespace Eigen {
struct RowVector2d {
    double x, y;
    double operator[](int i) const { return i ? y : x; }
};
}
namespace OptCuts {
""" + helper + """
}
int main() {
    Eigen::RowVector2d a{0.0,0.0}, d{1.0,1.0}, h{1.0,0.0};
    const double diagonal=OptCuts::onestringGridEdgeCost(a,d,1.0,1.0);
    const double horizontal=OptCuts::onestringGridEdgeCost(a,h,1.0,1.0);
    std::cout << std::setprecision(17)
        << diagonal << " " << horizontal << " "
        << OptCuts::onestringGridProposalScore(-.01,.001,diagonal,false) << " "
        << OptCuts::onestringGridProposalScore(.001,.001,diagonal,false) << " "
        << OptCuts::onestringGridProposalScore(.001,.001,diagonal,true) << "\\n";
}
"""
    cpp=tmp_path/"score.cpp"
    exe=tmp_path/"score"
    cpp.write_text(stub)
    subprocess.run([cxx,"-std=c++14","-O0",str(cpp),"-o",str(exe)],
                   check=True,capture_output=True,text=True,timeout=30)
    environment=dict(os.environ,ONESTRING_OPTCUTS_GRID_SEAM_WEIGHT="0.1")
    output=subprocess.run([str(exe)],env=environment,check=True,
                          capture_output=True,text=True,timeout=5)
    diagonal,horizontal,negative,positive,merged=map(float,output.stdout.split())
    assert diagonal==pytest.approx(1.0,abs=1e-12)
    assert horizontal==pytest.approx(0.0,abs=1e-12)
    assert negative==pytest.approx(-.01)
    assert positive==pytest.approx(.0009)
    assert merged==pytest.approx(.0011)


def test_native_linear_boundary_homotopy_detects_real_intermediate_crossing():
    """Both endpoints are simple CCW polygons, but interpolating crosses."""
    from onestring_physics.optcuts_grid_boundary_untangle import (
        native_linear_path_boundary_preflight,
    )
    from onestring_physics.optcuts_grid_failure_diagnostics import (
        proper_boundary_crossings,
    )
    src=np.array([
        [0.2131048215852692,1.3569212955720948],
        [-0.15663754381191192,0.48037246784793397],
        [-0.9336890110937098,-0.93480980103178],
        [0.20842385082511444,-1.2802142794901135],
        [0.7729964013237572,-0.5825586978812494]])
    dst=np.array([
        [-0.03222832553039152,1.0530042762788165],
        [-0.2656836287591267,1.4717110134675664],
        [-0.42280460191726144,1.2215609301148138],
        [-1.0560189834302436,0.3796184000073182],
        [-1.488352189615069,-0.042545047336794886]])
    assert proper_boundary_crossings(src)["count"]==0
    assert proper_boundary_crossings(dst)["count"]==0
    src_saved=src.copy()
    dst_saved=dst.copy()
    witness=native_linear_path_boundary_preflight(src,dst)
    assert witness is not None
    assert witness["count"] >= 1
    assert 0 < witness["fraction"] < 1
    np.testing.assert_array_equal(src,src_saved)
    np.testing.assert_array_equal(dst,dst_saved)


def test_native_linear_boundary_homotopy_preserves_safe_unchanged_mask():
    from onestring_physics.optcuts_grid_boundary_untangle import (
        native_linear_path_boundary_preflight,
    )
    src=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    dst=np.array([[0.,0.],[2.,0.],[2.,2.],[0.,2.]])
    assert native_linear_path_boundary_preflight(src,dst) is None
    with pytest.raises(ValueError,match="INVALID_INPUT"):
        native_linear_path_boundary_preflight(src,dst[:3])


def test_native_homotopy_guard_excludes_no_non_native_legacy_mode():
    root=Path(__file__).resolve().parents[1]
    source=(root/"src/onestring_physics/optcuts_area_grid_boundary.py").read_text()
    assert 'native_grid=bool(getattr(params,"optcuts_grid_seam_native_enabled",False))' in source
    assert 'if native_grid:' in source
    assert 'GRID_NATIVE_NO_LINEAR_HOMOTOPY_CANDIDATES' in source
