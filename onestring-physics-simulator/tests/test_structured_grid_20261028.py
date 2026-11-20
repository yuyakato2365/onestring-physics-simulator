"""2026-10-28 structured-grid mode: minimal smoke tests (pure Python)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from test_zhang2021_quad_20261027 import cylinder_chart


def _params(**kw):
    from onestring_physics.onestring_pipeline import PipelineParameters
    p = PipelineParameters()
    p.optcuts_structured_grid_20261028_enabled = True
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def test_mode_uses_09_18_route_and_zhang_hooks():
    from onestring_physics import optcuts_zhang2021_quad_20261027 as z
    p = _params()
    assert z.mode_enabled(p) and z.structured_grid_enabled(p)
    assert z.downstream_route(p) == z.ROUTE_PAPER_LOCAL_GLOBAL
    assert not z.mode_enabled(SimpleNamespace())
    assert p.zhang2021_initial_csf_split_enabled is False


def test_cylinder_structured_grid_is_regular_and_seam_closable():
    from onestring_physics.structured_grid_20261028 import plan_structured_grid
    xyz, f3, uv, f2 = cylinder_chart(n_theta=24, n_z=6, h=1.0)
    plan = plan_structured_grid(xyz, f3, uv, f2, _params(structured_grid_auto_panels=60))
    m = plan.metrics
    assert m["structured_grid_interior_valence_all_4"]
    assert m["zhang2021_m2d_quads_valid"]
    assert m["zhang2021_seam_preflight_mode"] == "hard"
    assert m["structured_grid_unresolved"] == []
    # a flat open cylinder needs no seam cone: the logical domain is a rectangle
    assert m["structured_grid_convex_corners"] == 4 and m["structured_grid_reflex_corners"] == 0
    assert m["structured_grid_seam_cones"] == []


# Regressions: old zipper cut only the path to the first tip of a Y tree.
def _slit_domain(dirs, lengths, pairs):
    from onestring_physics.structured_grid_20261028 import DIRS
    points = [np.zeros(2, int)]
    for d, n in zip(dirs, lengths):
        points.append(points[-1] + n * DIRS[d])
    assert np.array_equal(points[-1], points[0])
    return dict(polygon=np.asarray(points[:-1]), dirs=dirs, lengths=lengths, cut_pairs=pairs)


def test_zipper_cuts_every_y_branch_and_preserves_tip_and_junction_valence(monkeypatch):
    from onestring_physics import structured_grid_20261028 as sg
    monkeypatch.setattr(sg, "ZIPPER", True)
    dom = _slit_domain([0,1,2,0,0,2,3,0,1,2,3], [3,3,2,2,2,2,3,3,6,6,6],
                       [(1,6),(2,3),(4,5)])
    v, q, _, walk = sg.lattice_grid(dom)
    audit = sg._validate_boundary_walk(q, walk)
    assert audit["boundary_edges"] == audit["walk_edges"] == 38
    assert len(q) == 36
    for point, copies in (([3,3],3), ([1,3],1), ([5,3],1)):
        ids = np.flatnonzero(np.all(v == point, axis=1))
        assert len(ids) == copies
        assert sum(np.count_nonzero(q == i) for i in ids) == 4


def test_zipper_straight_slit_and_cyclic_rotation(monkeypatch):
    from onestring_physics import structured_grid_20261028 as sg
    monkeypatch.setattr(sg, "ZIPPER", True)
    ds, ns, pairs = [0,1,3,0,1,2,3], [3,3,3,3,6,6,6], [(1,2)]
    for offset in range(len(ds)):
        d = ds[offset:] + ds[:offset]; n = ns[offset:] + ns[:offset]
        ps = [((a-offset)%len(ds), (b-offset)%len(ds)) for a,b in pairs]
        v, q, _, walk = sg.lattice_grid(_slit_domain(d, n, ps))
        assert sg._validate_boundary_walk(q, walk)["boundary_edges"] == 30
        assert len(q) == 36


def test_boundary_audit_rejects_internal_edge_masquerading_as_boundary():
    import pytest
    from onestring_physics import structured_grid_20261028 as sg
    q = np.array([[0,1,4,3], [1,2,5,4]])
    # 1->4 is internal; old Y output passed this sort of walk downstream.
    walk = [(v,0,0) for v in [0,1,4,5,2,1,4,3]]
    with pytest.raises(sg.StructuredGridError, match="WALK_MISMATCH"):
        sg._validate_boundary_walk(q, walk)


def test_lift_guard_rejects_nonconvex_and_collapsed_panels():
    from onestring_physics.zhang2021_panel_optimizer import _lift_is_valid
    q = np.array([[0,1,2,3]])
    good = np.array([[0,0,0], [2,0,0], [2,2,0], [0,2,0]], float)
    assert _lift_is_valid(good, q)
    bad = good.copy(); bad[2] = [1,.3,0]
    assert not _lift_is_valid(bad, q)
    bad[2] = bad[1]
    assert not _lift_is_valid(bad, q)


def test_paired_boundary_slide_moves_both_copies_without_crossing():
    from onestring_physics.zhang2021_panel_optimizer import (
        ChartLift, optimize_panel_quality, _lift_is_valid)
    from test_zhang2021_quad_20261027 import grid_quads
    xyz, f3, uv, f2 = cylinder_chart(n_theta=24, n_z=12, h=2.0)
    p, q = grid_quads(8, 4, 0, 2*np.pi, 0, 2.0)
    # Narrow first row: boundary positions should move, but seam copies must agree.
    for old, new in ((.5,.16), (1.,.8), (1.5,1.4)):
        p[np.isclose(p[:,1],old),1] = new
    fixed = np.isclose(p[:,0],0) | np.isclose(p[:,0],2*np.pi) | np.isclose(p[:,1],0) | np.isclose(p[:,1],2)
    out, x, report = optimize_panel_quality(ChartLift(uv,f2,xyz,f3), p,q,fixed,
                                          max_iter=100, boundary_slide={"max_shift":.35})
    assert report.get("panel_optimizer_boundary_slide_groups",0) > 0
    assert report["panel_optimizer_boundary_slide_max_shift"] > 1e-6
    assert report["panel_optimizer_boundary_slide_pair_error"] < 1e-8
    assert _lift_is_valid(x,q)
    left, right = np.arange(0,45,9), np.arange(8,45,9)
    assert np.allclose(x[left],x[right],atol=1e-8)
    assert np.all(np.diff(x[left,2]) > 0)


def test_strict_final_gate_rejects_invalid_lift_even_in_report_only_mode(monkeypatch):
    import pytest
    from onestring_physics import optcuts_zhang2021_quad_20261027 as mode
    from onestring_physics import zhang2021_panel_optimizer as po
    from test_zhang2021_quad_20261027 import grid_quads
    xyz, f3, uv, f2 = cylinder_chart()
    p, q = grid_quads(6,3,0,2*np.pi,0,1)
    fake = SimpleNamespace(quad_vertices=p,quads=q,metrics={})
    real = po.optimize_panel_quality
    def corrupt(*a,**kw):
        out, x, report = real(*a,**kw)
        x[q[0,1]] = x[q[0,0]]
        return out, x, report
    monkeypatch.setattr(po,"optimize_panel_quality",corrupt)
    with pytest.raises(mode.Zhang2021QuadModeError,match="SG_M3D_GEOMETRY_REJECTED"):
        mode.plan_zhang2021_m2d(xyz,f3,uv,f2,SimpleNamespace(),zhang_result=fake,
                                report_only=True,require_valid_geometry=True)


def test_zipper_handles_multiple_junctions_in_one_cut_tree(monkeypatch):
    from onestring_physics import structured_grid_20261028 as sg
    monkeypatch.setattr(sg,"ZIPPER",True)
    dom = _slit_domain([0,1,2,0,1,2,0,0,2,3,3,0,1,2,3],
                       [4,2,2,2,2,2,2,2,2,2,2,4,8,8,8],
                       [(1,10),(2,3),(4,9),(5,6),(7,8)])
    v,q,_,walk = sg.lattice_grid(dom)
    assert len(q) == 64
    assert sg._validate_boundary_walk(q,walk)["boundary_edges"] == 52
    for point in ([4,2],[4,4]):
        ids = np.flatnonzero(np.all(v == point,axis=1))
        assert len(ids) == 3
        assert sum(np.count_nonzero(q == i) for i in ids) == 4


def test_strict_gate_accepts_consistently_clockwise_chart():
    from onestring_physics import optcuts_zhang2021_quad_20261027 as mode
    from test_zhang2021_quad_20261027 import grid_quads
    xyz,f3,uv,f2 = cylinder_chart()
    p,q = grid_quads(6,3,0,2*np.pi,0,1)
    uv[:,0] *= -1; p[:,0] *= -1
    fake = SimpleNamespace(quad_vertices=p,quads=q,metrics={})
    plan = mode.plan_zhang2021_m2d(xyz,f3,uv,f2,SimpleNamespace(),zhang_result=fake,
                                 report_only=True,require_valid_geometry=True)
    assert plan.metrics["zhang2021_m2d_quads_valid"]


def test_zipper_rejects_a_cut_path_crossing_itself(monkeypatch):
    import pytest
    from onestring_physics import structured_grid_20261028 as sg
    monkeypatch.setattr(sg,"ZIPPER",True)
    dom = _slit_domain([0,1,2,3,0,2,1,0,3,0,1,2,3],
                       [4,6,2,4,4,4,4,2,6,4,8,8,8],
                       [(1,8),(2,7),(3,6),(4,5)])
    with pytest.raises(sg.StructuredGridError,match="SG_BOUNDARY"):
        sg.lattice_grid(dom)


def test_zipper_four_way_junction_gets_four_sectors(monkeypatch):
    from onestring_physics import structured_grid_20261028 as sg
    monkeypatch.setattr(sg,"ZIPPER",True)
    dom = _slit_domain([0,1,2,0,1,3,0,2,3,0,1,2,3],
                       [4,4,2,2,2,2,2,2,4,4,8,8,8],
                       [(1,8),(2,3),(4,5),(6,7)])
    v,q,_,walk = sg.lattice_grid(dom)
    assert sg._validate_boundary_walk(q,walk)["boundary_edges"] == 52
    ids = np.flatnonzero(np.all(v == [4,4],axis=1))
    assert len(ids) == 4
    assert all(np.count_nonzero(q == i) == 1 for i in ids)
