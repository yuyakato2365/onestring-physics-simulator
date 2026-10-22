"""Paper rectangle failure recovers with official BFF only, no Tutte/LSCM.

The BFF chart is explicitly NONRECTANGULAR and is never falsely certified as
a prescribed-rectangle Paper reproduction. Original succeeded charts stay put.
"""
from pathlib import Path
import numpy as np
import pytest
from onestring_physics.bff_rectangle_selection import (
    choose_rectangle_by_source_boundary_fit,
)
from onestring_physics.reference_bff import BFFRunResult, ReferenceBFFError


def sample():
    vertices=np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])
    faces=np.array([[0,1,2],[0,2,3]])
    return vertices,faces


def test_all_25_rectangle_charts_invalid_official_free_bff_succeeds():
    xyz,faces=sample()
    old=xyz.copy();calls=[]
    def official(_vertices,_faces,**kwargs):
        calls.append(kwargs)
        if kwargs["boundary_policy"]=="target_rectangle":
            raise ValueError("Official rectangular BFF has collapsed UV triangles")
        assert kwargs["boundary_policy"]=="automatic_reference"
        assert kwargs["free_boundary_precision_adapter"] is True
        return BFFRunResult(xyz[:,:2].copy(),{
            "bff_boundary_policy_effective":"boundary_scale_zero",
            "parameterization_backend_name":"GeometryCollective/boundary-first-flattening",
            "fallbacks_used":[],
        })
    result=choose_rectangle_by_source_boundary_fit(
        xyz,faces,runner=official,allow_official_free_boundary_recovery=True
    )
    np.testing.assert_array_equal(xyz,old)
    # Boundary-PCA candidates depend on the source shape; all available
    # rectangle candidates must fail before the final free BFF attempt.
    assert len(calls) >= 10
    assert all(item["boundary_policy"] == "target_rectangle" for item in calls[:-1])
    assert calls[-1]["boundary_policy"] == "automatic_reference"
    assert result.metrics["official_bff_free_boundary_recovery_used"] is True
    assert result.metrics["paper_rectangular_boundary_condition_satisfied"] is False
    # Invalid corner heuristics that fail before calling native BFF are also
    # recorded, so attempt records may exceed the native invocation count.
    assert result.metrics["bff_corner_selection_attempt_count"]>=len(calls)-1
    assert result.metrics["bff_corner_selection_attempt_count"]==len(result.metrics["bff_corner_selection_attempts"])
    assert result.metrics["bff_boundary_policy_effective"]=="boundary_scale_zero"
    assert result.metrics["fallbacks_used"]==[]


def test_original_valid_rectangle_is_not_replaced():
    xyz,faces=sample();count=[]
    def official(_vertices,_faces,**kwargs):
        count.append(kwargs["boundary_policy"])
        return BFFRunResult(xyz[:,:2].copy(),{
            "bff_rectangle_verified": True, "fallbacks_used":[],
        })
    result=choose_rectangle_by_source_boundary_fit(
        xyz,faces,runner=official,allow_official_free_boundary_recovery=True
    )
    assert len(count)<26
    assert "automatic_reference" not in count
    assert not result.metrics.get("official_bff_free_boundary_recovery_used",False)


def test_bad_free_boundary_also_rejected_after_failed_rectangle():
    xyz,faces=sample()
    def official(_vertices,_faces,**kwargs):
        if kwargs["boundary_policy"]=="target_rectangle":
            raise ValueError("collapsed UV")
        uv=xyz[:,:2].copy()
        uv[2]=[2.,0.] # genuinely collapsed source face
        return BFFRunResult(uv,{
            "bff_boundary_policy_effective":"boundary_scale_zero",
        })
    with pytest.raises(ReferenceBFFError,match="candidates failed"):
        choose_rectangle_by_source_boundary_fit(
            xyz,faces,runner=official,allow_official_free_boundary_recovery=True
        )


def test_nonpaper_route_cannot_turn_on_paper_only_recovery_implicitly():
    path=Path(__file__).parents[1]/"src/onestring_physics/onestring_pipeline.py"
    content=path.read_text()
    # Paper-oriented mode was restored to the validated strict rectangle route:
    # the free-boundary recovery is not enabled by any pipeline route.
    assert 'allow_official_free_boundary_recovery=False,' in content
    assert 'allow_official_free_boundary_recovery=bool(' not in content
    assert '"omega_boundary_forced_rectangle": _actual_bff_boundary_policy=="target_rectangle"' in content
    assert '"official_bff_free_boundary_recovery_used": _official_free_recovery' in content
    assert '"official_bff_free_boundary_recovery_used", False' in content
