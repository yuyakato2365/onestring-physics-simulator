"""Experimental rectangle corner choice uses original area distortion, never Split count."""
from types import SimpleNamespace
import numpy as np
import pytest
from onestring_physics.bff_rectangle_selection import (
    PHASES, choose_rectangle_by_min_area_span, whole_surface_area_span
)
from onestring_physics.reference_bff import BFFRunResult


def _mesh():
    vertices=np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])
    faces=np.array([[0,1,2],[0,2,3]])
    return vertices,faces


def test_selects_lowest_whole_source_area_jacobian_span_without_split_count():
    xyz,faces=_mesh()
    checked=[]
    dy={0.:4.0,0.0625:1.5,0.125:2.0,0.1875:3.0}
    def stub(vertices, triangles, *, rectangle_corner_phase_fraction,
             boundary_policy, rectangle_executable, timeout_seconds):
        assert boundary_policy=="target_rectangle"
        phase=rectangle_corner_phase_fraction
        checked.append(phase)
        uv=np.array([[0.,0.],[1.,0.],[1.,dy[phase]],[0.,1.]])
        return BFFRunResult(uv,{"bff_rectangle_corner_boundary_wedge_indices":[1,4,7,10]})
    result=choose_rectangle_by_min_area_span(xyz,faces,runner=stub)
    assert tuple(checked)==PHASES
    assert result.metrics["bff_corner_selection_selected_phase"]==0.0625
    assert result.metrics["bff_corner_selection_author_method_verified"] is False
    assert result.metrics["bff_corner_selection_certifies_area_limit"] is False
    assert result.metrics["bff_corner_selection_attempts"][1]["whole_source_area_jacobian_span"]==pytest.approx(1.5)


def test_degenerate_uv_candidate_fails_instead_of_silent_skip():
    xyz,faces=_mesh()
    def stub(_xyz,_faces,*,rectangle_corner_phase_fraction,**_):
        uv=np.array([[0.,0.],[1.,0.],[1.,float(rectangle_corner_phase_fraction)],[0.,1.]])
        return BFFRunResult(uv,{})
    with pytest.raises(ValueError,match="degenerate"):
        choose_rectangle_by_min_area_span(xyz,faces,runner=stub)


def test_global_uniform_uv_scale_cannot_change_area_span():
    xyz,faces=_mesh()
    uv=np.array([[0.,0.],[1.,0.],[1.,1.5],[0.,1.]])
    initial=whole_surface_area_span(xyz,uv,faces)
    assert initial==pytest.approx(1.5)
    for factor in [0.25,2.0,50.0]:
        assert whole_surface_area_span(xyz,uv*factor,faces)==pytest.approx(initial)
