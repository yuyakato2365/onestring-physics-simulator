"""Regressions for explicit Split equality and geometric collision diagnostics."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
from onestring_physics.split_geometry_constraints import optimize_shared, propagate, pair_errors
from onestring_physics.solid_collision_audit import audit_solid_collisions
from onestring_physics.optcuts_backend import _triangle_differential_metrics
from onestring_physics.paper_eq5_k2d_solver import collision_energy_gradient


def test_shared_variables_keep_distinct_output_indices_and_transitive_seams():
    vertices=np.array([[0.,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,0],[0,0,0]])
    mesh=SimpleNamespace(vertices=vertices,faces=np.array([[0,1,2,3],[4,1,2,3],[5,1,2,3]]),metrics={'split_vertex_pairs':[[0,4],[4,5]]})
    def solve(target,reduced,par,params):
        assert len(reduced.vertices)==4
        out=copy.deepcopy(reduced)
        out.vertices += np.arange(4)[:,None]*.2
        return out,None
    out,_=optimize_shared(solve,None,mesh,None,None)
    np.testing.assert_array_equal(out.faces,mesh.faces)
    assert len(out.vertices)==6
    assert out.faces[0,0]!=out.faces[1,0]
    assert pair_errors(out.vertices,[[0,4],[4,5]])['split_pair_error_max']==0
    np.testing.assert_array_equal(mesh.vertices,vertices)
    target=SimpleNamespace(metrics={})
    propagate(out,target)
    assert target.metrics['split_vertex_pairs']==[[0,4],[4,5]]
    target.metrics['split_vertex_pairs'][0][0]=99
    assert out.metrics['split_vertex_pairs'][0][0]==0


def box(center,angle=0.):
    q=np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])
    c,s=np.cos(angle),np.sin(angle)
    q=q@np.array([[c,s],[-s,c]])
    return np.array([[x,y,z] for z in (-.1,.1) for x,y in q])+center


def test_solid_sat_distinguishes_projection_overlap_from_3d_collision():
    a=box(np.zeros(3)); b=box(np.array([.3,.2,1.]),.3)
    assert audit_solid_collisions([a,b])['solid_collision_pair_count']==0
    b[:,2]-=1
    result=audit_solid_collisions([a,b])
    assert result['solid_collision_pair_count']==1
    assert result['solid_penetration_max']==pytest.approx(.2)
    assert audit_solid_collisions([a,box(np.array([1.,0,0]))])['solid_collision_pair_count']==0


def test_rotated_sat_rejects_overlapping_aabb_without_polygon_overlap():
    a=box(np.zeros(3),np.pi/4)[:4,:2]
    b=a+[1.,1.]
    xy=np.vstack([a,b]);faces=np.arange(8).reshape(2,4)
    energy,_,count,stats=collision_energy_gradient(xy,faces,return_diagnostics=True)
    assert stats['collision_candidate_pairs']==1
    assert energy==0 and count==0


def test_global_uv_reflection_is_not_a_local_flip():
    xyz=np.array([[0.,0,0],[1,0,0],[1,1,0],[0,1,0]])
    faces=np.array([[0,1,2],[0,2,3]])
    uv=xyz[:,:2]*[1,-1]
    metrics=_triangle_differential_metrics(xyz,faces,uv,faces)
    assert metrics['uv_triangle_flip_count']==0
    assert metrics['uv_degenerate_triangle_count']==0
    mixed=faces.copy();mixed[1]=mixed[1,::-1]
    assert _triangle_differential_metrics(xyz,faces,uv,mixed)['uv_triangle_flip_count']==1




def _optcuts_quality_fixture(**updates):
    keys = ('distortion_max', 'distortion_p95', 'distortion_area_mean',
            'csf_max', 'csf_p95', 'csf_over_2_area_fraction', 'seam_length')
    result = {key: 10.0 for key in keys}
    result.update(csf_max=2.4, csf_p95=1.8, csf_over_2_area_fraction=.12,
                  flipped_triangles=0, degenerate_triangles=0,
                  injectivity_overlap_pairs=0)
    result.update(updates)
    return result


def test_optcuts_selection_prioritizes_csf_tail_without_seam_veto():
    from onestring_physics.optcuts_quality_audit import pareto_improves
    baseline = _optcuts_quality_fixture()
    candidate = _optcuts_quality_fixture(
        csf_over_2_area_fraction=.04,
        seam_length=20.0,
        distortion_max=12.0,
        distortion_p95=12.0,
        distortion_area_mean=12.0,
    )
    assert pareto_improves(candidate, baseline)


def test_optcuts_selection_rejects_worse_tail_invalid_and_catastrophic_distortion():
    from onestring_physics.optcuts_quality_audit import pareto_improves
    baseline = _optcuts_quality_fixture()
    assert not pareto_improves(
        _optcuts_quality_fixture(csf_over_2_area_fraction=.13), baseline)
    assert not pareto_improves(
        _optcuts_quality_fixture(csf_over_2_area_fraction=.04,
                                 distortion_p95=16.0), baseline)
    assert not pareto_improves(
        _optcuts_quality_fixture(csf_over_2_area_fraction=.04,
                                 injectivity_overlap_pairs=1), baseline)
    assert not pareto_improves(
        _optcuts_quality_fixture(csf_over_2_area_fraction=.04,
                                 csf_max=float('nan')), baseline)


def test_optcuts_csf_uses_global_linear_scale_not_minimum_triangle_anchor():
    from onestring_physics.optcuts_quality_audit import _relative_linear_scale
    area3 = np.array([1.0e-6, 1.0, 1.0])
    area2 = np.ones(3)
    csf, global_scale = _relative_linear_scale(area3, area2)
    assert global_scale == pytest.approx(np.sqrt(area3.sum()/area2.sum()))
    # A tiny finite triangle no longer makes every ordinary triangle enormous.
    assert csf[1] == pytest.approx(1.0/global_scale)
    assert csf[1] < 2.0
    scaled, _ = _relative_linear_scale(area3, 4.0*area2)
    np.testing.assert_allclose(scaled, csf)
