"""Guard real K3D edge-length growth without blaming pre-existing M3D edges.

All tests run with synthetic meshes, without loading an external BFF binary.
The reference pipeline still rejects ANY remaining failed K3D guard/fallback.
"""
from types import SimpleNamespace
import numpy as np

from onestring_physics.onestring_pipeline import (
    _k3d_quality_metrics,
    _k3d_quality_reject_reason,
    _k3d_damp_new_edge_outlier,
)


def _mesh(n=4):
    vertices=np.array([
        [float(i),float(j),0.] for j in range(n+1) for i in range(n+1)
    ])
    faces=np.array([
        [j*(n+1)+i,j*(n+1)+i+1,
         (j+1)*(n+1)+i+1,(j+1)*(n+1)+i]
        for j in range(n) for i in range(n)
    ], dtype=int)
    return vertices,faces


def test_normal_off_guard_remains_strict_and_reference_does_not_reject_inherited_outlier():
    grid=SimpleNamespace(tile_size=1.)
    vals=dict(
        k3d_edge_max_to_median_ratio=9.1,
        m3d_edge_max_to_median_ratio=9.0,
        k3d_edge_median_ratio_to_m3d=1.,
        k3d_quad_area_min_to_median_ratio=.9,
        k3d_quad_area_median_ratio_to_m3d=1.,
        k3d_vertex_displacement_max=0.1,
    )
    assert _k3d_quality_reject_reason(vals,grid)=="edge_length_outlier_guard"
    assert _k3d_quality_reject_reason(vals,grid,input_aware=True) is None
    vals["k3d_edge_max_to_median_ratio"]=15.0
    assert _k3d_quality_reject_reason(vals,grid,input_aware=True)=="edge_length_outlier_guard"


def test_new_extreme_edge_is_rejected_but_can_be_damped_without_going_back_to_m3d():
    base,faces=_mesh()
    candidate=base.copy()
    candidate[12,0]+=40.
    grid=SimpleNamespace(tile_size=1.)
    initial=_k3d_quality_metrics(base,candidate,faces)
    assert initial["m3d_edge_max_to_median_ratio"]==1.
    assert _k3d_quality_reject_reason(initial,grid,input_aware=True)==(
        "edge_length_outlier_guard"
    )
    rescued,metrics,alpha=_k3d_damp_new_edge_outlier(base,candidate,faces,grid)
    assert rescued is not None
    assert 0.125<=alpha<1.
    assert _k3d_quality_reject_reason(metrics,grid,input_aware=True) is None
    assert not np.array_equal(rescued,base)
    assert not np.array_equal(rescued,candidate)
    assert metrics["k3d_edge_max_to_median_ratio"]<8.
    assert metrics["k3d_vertex_displacement_max"]>0.


def test_no_safe_nontrivial_damping_keeps_a_strict_rejection():
    base,faces=_mesh()
    candidate=base.copy()
    candidate[12,0]+=250.
    rescued,metrics,alpha=_k3d_damp_new_edge_outlier(
        base,candidate,faces,SimpleNamespace(tile_size=1.)
    )
    assert rescued is None
    assert metrics is None
    assert alpha is None
