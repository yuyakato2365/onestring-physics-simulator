"""Regression tests for initial-Omega input-boundary-fitted BFF rectangle."""
import numpy as np

from onestring_physics.bff_rectangle_selection import (
    _boundary_pca_projection, _boundary_similarity_error,
    _corner_candidates, choose_rectangle_by_source_boundary_fit,
)
from onestring_physics.reference_bff import BFFRunResult


def rectangle_disk():
    border=np.asarray([
        [-2.,-1.,0.], [-1.,-1.,0.], [0.,-1.,0.], [1.,-1.,0.],
        [2.,-1.,0.], [2.,0.,0.], [2.,1.,0.], [1.,1.,0.],
        [0.,1.,0.], [-1.,1.,0.], [-2.,1.,0.], [-2.,0.,0.],
    ])
    vertices=np.vstack((border,[[0.,0.,0.]]))
    faces=np.asarray([[12,i,(i+1)%12] for i in range(12)],dtype=int)
    return vertices,faces


def test_shape_corner_candidates_are_distinct_and_span_boundary():
    x,_=rectangle_disk()
    loop=np.arange(12,dtype=int)
    uv,weights,nonplanarity=_boundary_pca_projection(x,loop)
    assert abs(sum(weights)-1.)<1e-12
    assert nonplanarity<1e-10
    candidates=_corner_candidates(x[loop],uv,loop)
    assert candidates
    for c in candidates:
        assert len(set(c["source_vertex_ids"]))==4
        assert c["min_boundary_arc_fraction"]>.02


def test_boundary_similarity_does_not_penalize_rotation_or_uniform_scale():
    x,_=rectangle_disk()
    uv,w,_=_boundary_pca_projection(x,np.arange(12))
    angle=.48
    r=np.asarray([[np.cos(angle),-np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    assert _boundary_similarity_error(uv,9.4*(uv@r)+[7.,-5.],w)<1e-10
    assert _boundary_similarity_error(uv,uv@np.diag([2.,.2]),w)>.05


def test_initial_rectangle_candidates_choose_source_boundary_fit_not_split_count():
    xyz,faces=rectangle_disk()
    calls=[]
    def fake_official_rect(vertices,faces,**kwargs):
        calls.append(kwargs)
        assert kwargs["boundary_policy"]=="target_rectangle"
        points=np.asarray(vertices)[:,:2]
        if "rectangle_corner_positions" in kwargs:
            # Exact PCA-projected boundary fit is a better candidate than
            # badly anisotropic quarter-arclength baselines.
            uv=points.copy()
        else:
            uv=points*np.asarray([3.5,.25])
        return BFFRunResult(
            uv_vertices=uv,
            metrics={
                "bff_rectangle_verified":True,
                "bff_rectangle_corner_boundary_wedge_indices":[0,3,6,9],
            },
        )
    result=choose_rectangle_by_source_boundary_fit(
        xyz,faces,runner=fake_official_rect)
    assert result.metrics["bff_corner_selection_method"]=="source_boundary_pca_fit_EXPERIMENTAL"
    assert len(calls)>=5
    assert any("rectangle_corner_positions" in call for call in calls)
    assert any("rectangle_corner_phase_fraction" in call for call in calls)
    assert result.metrics["bff_corner_selection_selected_source_vertex_ids"] is not None
    assert len(result.metrics["bff_corner_selection_attempts"])==len(calls)
    assert result.metrics["bff_corner_selection_author_method_verified"] is False
