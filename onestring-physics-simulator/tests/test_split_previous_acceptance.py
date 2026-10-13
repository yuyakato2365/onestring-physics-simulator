"""The former hierarchical CSF Split rule must not fragment a fixed chart indefinitely.

Regression for a real reported failure: 64/64 cuts and 65 islands while
multiple components still exceed CSF 2. These synthetic charts test the
decision rule, not a claim about Lilium/SIGGRAPH reproduction.
"""
from types import SimpleNamespace

import numpy as np

from onestring_physics import paper_mesh_splitting as split


def grid(n=4):
    vertices=np.array([[float(i),float(j),0.] for j in range(n+1)
                       for i in range(n+1)],dtype=float)
    faces=np.array([[j*(n+1)+i,j*(n+1)+i+1,
                     (j+1)*(n+1)+i+1,(j+1)*(n+1)+i]
                    for j in range(n) for i in range(n)],dtype=int)
    return vertices,faces


def domain(vertices, params):
    return SimpleNamespace(
        parameterization=SimpleNamespace(
            metrics={},uv_vertices_2d=np.asarray(vertices)[:, :2],
        ),
        csf_split_enabled=True,csf_split_threshold=2.,
        max_csf_splits=64,reference_grid_rotation_degrees=0.,
    )


def setup(monkeypatch, lo, hi, vertices, faces):
    # Return source IDs at the mesh's central vertex as highest curvature:
    # both horizontal and vertical complete grid cuts are possible.
    center=((int(round(len(vertices)**.5))**2)-1)//2
    curvature=np.full(len(vertices),-1.,dtype=float)
    curvature[center]=10.
    samples=[np.array([center],dtype=int) for _ in faces]
    monkeypatch.setattr(split,"_face_samples",lambda *a:(lo,hi,samples))
    monkeypatch.setattr(split,"gaussian_curvature",lambda *a:curvature)


def test_restore_stops_when_no_cut_can_improve_even_with_budget_64(monkeypatch):
    vertices,faces=grid()
    lo=np.ones(len(faces))
    hi=np.full(len(faces),4.2)
    setup(monkeypatch,lo,hi,vertices,faces)
    params=SimpleNamespace(paper_fidelity_mode=True,collision_split_forced_lines=())
    v,f,records,metrics=split._split_mesh_single(vertices,faces,domain(vertices,params),params)
    assert not records
    assert metrics["number_of_splits"]==0
    assert metrics["csf_split_step_count"]==0
    assert metrics["csf_split_unresolved_component_count"]==1
    assert not metrics["csf_split_budget_exhausted"]
    assert metrics["csf_split_rejected_components"][0]["reason"]=="within_quad_distortion_floor"
    np.testing.assert_array_equal(f,faces)
    np.testing.assert_array_equal(v,vertices)


def test_restore_only_accepts_real_child_csf_improvement(monkeypatch):
    vertices,faces=grid()
    # Every quad individually has CSF 1.1. Left/right halves have distinct
    # absolute area scales: split at x=2 reduces the component max/min span.
    lo=np.array([1. if k%4<2 else 4. for k in range(len(faces))])
    hi=lo*1.1
    setup(monkeypatch,lo,hi,vertices,faces)
    params=SimpleNamespace(paper_fidelity_mode=True,collision_split_forced_lines=())
    _,_,records,metrics=split._split_mesh_single(vertices,faces,domain(vertices,params),params)
    assert records
    assert len(records)<64
    assert all(r["worst_sigma_reduction"]>1e-10 for r in records)
    assert metrics["max_csf_after_split"]<metrics["max_csf_before_split"]
    assert not metrics["csf_split_budget_exhausted"]
