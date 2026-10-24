"""Phase-0 artifacts must be a faithful, repeatable record of one run."""
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics.stage_audit_bundle import StageAuditWriter


@dataclass
class Params:
    tile_size:float=1.
    omega_parameterization_mode:str="optcuts_test"


def synthetic():
    xyz=np.array([
        [0.,0.,0.],[1.,0.,0.],[1.,1.,.01],[0.,1.,0.],
        [1.,0.,0.],[2.,0.,0.],[2.,1.,0.],[1.,1.,.01],
    ])
    faces=np.array([[0,1,2,3],[4,5,6,7]],int)
    surf_f=np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7]],int)
    return xyz,faces,surf_f


def test_audit_bundle_one_run_and_partial_failure_capture(tmp_path):
    xyz,faces,surf=synthetic()
    writer=StageAuditWriter(
        tmp_path,Params(),run_id="test-run-123",
        runtime_functions={"_build_m2d":test_audit_bundle_one_run_and_partial_failure_capture},
    )
    writer.record("S",xyz,surf)
    writer.record_quad("M3D",SimpleNamespace(
        vertices=xyz.copy(),faces=faces.copy(),metrics={}
    ))
    writer.fail("K3D",RuntimeError("simulated optimizer failure"))
    folder=tmp_path/"test-run-123"
    assert (folder/"stages.npz").exists()
    metrics=json.loads((folder/"metrics.json").read_text())
    assert metrics["run_id"]=="test-run-123"
    assert metrics["failure_stage"]=="K3D"
    assert metrics["status"]=="failed_before_complete"
    assert "_build_m2d" in metrics["runtime_functions"]
    assert metrics["parameters"]["tile_size"]==1.
    with np.load(folder/"stages.npz",allow_pickle=False) as bundle:
        assert "S_vertices" in bundle
        assert "M3D_vertices" in bundle
        assert "K3D_vertices" not in bundle
        np.testing.assert_allclose(bundle["S_vertices"],xyz)


def test_audit_json_preserves_inf_for_failed_planarity(tmp_path):
    xyz,faces,surf=synthetic()
    w=StageAuditWriter(tmp_path,Params(),run_id="infinite")
    w.record("S",xyz,surf,{"plane_distance":float("inf")})
    report=json.loads((w.path/"metrics.json").read_text())
    assert report["stage_metrics"]["S"]["plane_distance"]=="Infinity"


def test_stage_renderer_one_camera_no_fabricated_geometry(tmp_path):
    pytest.importorskip("matplotlib")
    xyz,faces,surf=synthetic()
    w=StageAuditWriter(tmp_path,Params(),run_id="render-smoke")
    w.record("S",xyz,surf)
    w.record_quad("M2D",SimpleNamespace(
        vertices=xyz.copy(),faces=faces.copy(),metrics={}
    ))
    w.record_quad("M3D",SimpleNamespace(
        vertices=xyz.copy(),faces=faces.copy(),metrics={}
    ))
    w.record_quad("K3D_before_T3D",SimpleNamespace(
        vertices=xyz.copy(),faces=faces.copy(),metrics={}
    ))
    w.record_quad("K3D",SimpleNamespace(
        vertices=xyz.copy(),faces=faces.copy(),metrics={
            "seam_stitch_mode":"soft",
            "seam_exact_affine_rows":[[[1,1.],[4,-1.]]],
        }
    ))
    tiles=np.concatenate([xyz[faces],xyz[faces]-np.array([0,0,.03])],axis=1)
    w.record_t3d(SimpleNamespace(
        vertices=tiles,metrics={"paper_t3d_planarity_iterations":2}
    ))
    w.finish()
    script=Path(__file__).resolve().parents[1]/"scripts"/"render_stage_audit.py"
    subprocess.run([
        sys.executable,str(script),
        "--bundle",str(w.path/"stages.npz"),
        "--out",str(w.path),"--no-videos",
    ],check=True,timeout=100)
    for stage in ("S","M3D","K3D","T3D"):
        for camera in ("front","side","back","ears"):
            assert (w.path/f"{stage}_{camera}.png").stat().st_size>1000
    assert (w.path/"K3D_seam_ribbon.png").exists()
    assert (w.path/"K3D_degenerate_quads.png").exists()
    with np.load(w.path/"stages.npz",allow_pickle=False) as bundle:
        np.testing.assert_allclose(bundle["K3D_vertices"],xyz)
    report=json.loads((w.path/"metrics.json").read_text())
    assert report["render_protocol"]["same_camera"]
    assert report["render_protocol"]["backface_culling"] is False
    assert report["run_id"]=="render-smoke"


def test_reduction_loss_threshold_is_not_silent():
    from onestring_physics.optcuts_seam_soft_constraints import remap_rows
    rows=[[[0,1.],[1,-1.]],[[2,1.],[3,-1.]]]
    with pytest.raises(ValueError,match="SEAM_SOFT_REMAP_CONSTRAINT_LOSS_EXCESS"):
        remap_rows(rows,[0,0,1,2],return_stats=True)
    keep,audit=remap_rows(rows,[0,1,2,3],return_stats=True)
    assert len(keep)==2
    assert audit["seam_rows_before_reduction"]==2
    assert audit["seam_rows_after_reduction"]==2
    assert audit["seam_rows_dropped_by_reduction"]==0


def test_large_rank_skip_is_explicit_never_a_success():
    from onestring_physics.optcuts_seam_soft_constraints import (
        seam_matrix_rank_diagnostics,
    )
    out=seam_matrix_rank_diagnostics(
        [[[0,1.],[1,-1.]],[[2,1.],[3,-1.]]],
        4,max_seam_vertices=3,
    )
    assert not out["rank_evaluated"]
    assert out["rank_skip_reason"]=="above_configured_svd_cap"



def test_comparator_rejects_different_source_even_when_camera_matches(tmp_path):
    from importlib.util import spec_from_file_location,module_from_spec
    script=Path(__file__).resolve().parents[1]/"scripts"/"compare_stage_audits.py"
    spec=spec_from_file_location("stage_audit_compare_test",script)
    mod=module_from_spec(spec)
    spec.loader.exec_module(mod)
    xyz,faces,surf=synthetic()
    first=StageAuditWriter(tmp_path,Params(),run_id="previous")
    second=StageAuditWriter(tmp_path,Params(),run_id="candidate")
    first.record("S",xyz,surf)
    altered=xyz.copy()
    altered[0,2]+=.1
    second.record("S",altered,surf)
    first.summary["render_protocol"]={
        "same_camera":True,"backface_culling":False
    }
    second.summary["render_protocol"]={
        "same_camera":True,"backface_culling":False
    }
    first.flush()
    second.flush()
    with pytest.raises(ValueError,match="STAGE_AUDIT_DIFFERENT_SOURCE_OR_CAMERA"):
        mod.validate_runs(first.path,second.path)
    # Identical source meshes and correct stage provenance allow A/B.
    second.record("S",xyz,surf)
    a,b=mod.validate_runs(first.path,second.path)
    assert a["run_id"]=="previous"
    assert b["run_id"]=="candidate"
