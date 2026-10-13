"""Paper Split: CSF > 2 -> highest-Gaussian-curvature complete cut -> repeat.

The earlier user instruction was NOT to veto cuts based on immediate CSF
improvement. Failed geometrical cuts/budget must be reported as unresolved.
These are synthetic checks, not a claim about Lilium reproduction.
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


def test_paper_accepts_complete_cut_even_without_immediate_csf_gain(monkeypatch):
    vertices,faces=grid()
    lo=np.ones(len(faces))
    hi=np.full(len(faces),4.2)
    setup(monkeypatch,lo,hi,vertices,faces)
    params=SimpleNamespace(paper_fidelity_mode=True,collision_split_forced_lines=())
    v,f,records,metrics=split._split_mesh_single(vertices,faces,domain(vertices,params),params)
    # A valid complete cut is accepted even when every tile has CSF=4.2.
    assert records, "Paper mode must not veto a legal curvature cut for no CSF gain"
    assert all(r["gaussian_curvature"] == 10. for r in records)
    assert len(records) < 64  # geometric impossibility stops futile recuts
    assert metrics["max_csf_after_split"]>2.
    assert metrics["csf_split_unresolved_component_count"]>0
    assert metrics["csf_split_stopping_reason"] == "no_admissible_cut_remaining"
    assert not metrics["csf_split_bound_certified"]
    assert len(f)==len(faces)
    assert len(v)>len(vertices)


def test_paper_stops_as_soon_as_all_component_csf_values_are_at_most_two(monkeypatch):
    vertices,faces=grid()
    lo=np.ones(len(faces))
    hi=np.full(len(faces),1.9)
    setup(monkeypatch,lo,hi,vertices,faces)
    params=SimpleNamespace(paper_fidelity_mode=True,collision_split_forced_lines=())
    _,_,records,metrics=split._split_mesh_single(vertices,faces,domain(vertices,params),params)
    assert records==[]
    assert metrics["csf_split_stopping_reason"]=="threshold_met"
    assert metrics["csf_split_unresolved_component_count"]==0


def test_paper_keeps_cutting_until_component_spans_fall_below_two(monkeypatch):
    vertices,faces=grid()
    # Every quad individually has area span 1.1, while the two halves
    # use strongly different absolute scales. The first x=2 cut
    # establishes two components, both with CSF 1.1.
    lo=np.array([1. if k%4<2 else 4. for k in range(len(faces))])
    hi=lo*1.1
    setup(monkeypatch,lo,hi,vertices,faces)
    params=SimpleNamespace(paper_fidelity_mode=True,collision_split_forced_lines=())
    _,_,records,metrics=split._split_mesh_single(vertices,faces,domain(vertices,params),params)
    assert records
    assert len(records) < 64
    assert metrics["csf_split_stopping_reason"]=="threshold_met"
    assert metrics["max_csf_after_split"]<=2.0
    assert not metrics["csf_split_budget_exhausted"]
