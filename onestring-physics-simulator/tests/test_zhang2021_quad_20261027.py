"""Tests for the 2026-10-27 OptCuts -> Zhang et al. (CAD 2021) quad M2D mode.

Pure-Python parts (metrics, OneString chart adapter, seam matching, setup
normalisation, UI registration) always run.  Tests that need the authors'
reference programs are skipped unless they are built
(``python3 scripts/setup_zhang2021_quad_official.py``).
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics.zhang2021_quad_metrics import (
    quad_layout_patches,
    scaled_jacobians,
    two_sided_boundary_hausdorff,
)
from onestring_physics.zhang2021_onestring_boundary import (
    build_source_boundary,
    initial_correspondence,
    quad_boundary_loop,
    uv_to_xyz_vertex_map,
)
from onestring_physics.zhang2021_onestring_reparam import (
    seam_conforming_reparameterization,
    signed_uv_areas,
)
from onestring_physics.zhang2021_onestring_seam_matching import (
    _edge_quads,
    insert_chord,
    seam_constraints,
    trace_chord,
)

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------- fixtures
def grid_quads(nx: int, ny: int, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    xs, ys = np.meshgrid(np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1))
    uv = np.c_[xs.ravel(), ys.ravel()]
    quads = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            quads.append([a, a + 1, a + nx + 2, a + nx + 1])
    return uv, np.asarray(quads)


def triangulate(quads):
    return np.asarray([[q[0], q[1], q[2]] for q in quads] + [[q[0], q[2], q[3]] for q in quads])


def cylinder_chart(n_theta=24, n_z=8, h=1.0):
    """Open cylinder cut along theta=0: UV rectangle whose left/right sides are
    the two copies of the cut, top/bottom sides are open boundary."""
    th = np.linspace(0.0, 2 * np.pi, n_theta + 1)
    zs = np.linspace(0.0, h, n_z + 1)
    uv = np.asarray([[t, z] for z in zs for t in th])
    xyz_id = np.asarray([(j * n_theta + (i % n_theta)) for j in range(n_z + 1) for i in range(n_theta + 1)])
    xyz = np.asarray([[np.cos(t), np.sin(t), z] for z in zs for t in th[:-1]])
    quads = []
    for j in range(n_z):
        for i in range(n_theta):
            a = j * (n_theta + 1) + i
            quads.append([a, a + 1, a + n_theta + 2, a + n_theta + 1])
    f2 = triangulate(quads)
    f3 = xyz_id[f2]
    return xyz, f3, uv, f2


# -------------------------------------------------------------- metrics
def test_scaled_jacobian_unit_square_and_skew():
    v = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [1.5, 1.0]], float)
    assert np.allclose(scaled_jacobians(v, np.array([[0, 1, 2, 3]])), 1.0)
    skew = scaled_jacobians(v, np.array([[0, 1, 4, 3]]))
    assert 0.0 < skew[0] < 1.0


def test_layout_patches_regular_grid_is_one_patch():
    _, q = grid_quads(6, 4)
    n, ids, _ = quad_layout_patches(q)
    assert n == 1 and len(ids) == len(q)


def test_two_sided_hausdorff_of_offset_squares():
    a = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    b = a + np.array([0.1, 0.0])
    d, dab, dba = two_sided_boundary_hausdorff(a, [[0, 1, 2, 3]], b, [[0, 1, 2, 3]])
    assert d == pytest.approx(0.1, abs=1e-9)
    assert dab == pytest.approx(0.1, abs=1e-9) and dba == pytest.approx(0.1, abs=1e-9)


# ---------------------------------------------------------- chord tools
def test_chord_insertion_keeps_a_conforming_all_quad_mesh():
    uv, q = grid_quads(4, 3)
    eq = _edge_quads(q)
    loop = quad_boundary_loop(q)
    # first boundary edge on the bottom side
    a, b = loop[0], loop[1]
    (qi, i), = eq[(min(a, b), max(a, b))]
    ch = trace_chord(q, eq, qi, i)
    assert ch is not None and len(ch.quads) in (3, 4)
    uv2, q2, vs, ve = insert_chord(uv, q, ch, 0.5, 0.5)
    assert len(q2) == len(q) + len(ch.quads)
    # every quad keeps the orientation of the input
    sa = signed_uv_areas(uv2, q2[:, [0, 1, 2]]) + signed_uv_areas(uv2, q2[:, [0, 2, 3]])
    assert np.all(sa > 0)
    # each interior edge shared by exactly two quads in opposite directions
    owners = _edge_quads(q2)
    assert all(len(o) <= 2 for o in owners.values())
    assert len(quad_boundary_loop(q2)) == len(loop) + 2
    assert vs in quad_boundary_loop(q2) and ve in quad_boundary_loop(q2)


# ------------------------------------------------- OneString chart adapter
def test_reparameterisation_matches_quad_boundary_without_flips():
    # Source chart: a slightly wavy unit square lifted onto a bumpy surface.
    uv_grid, q_src = grid_quads(16, 16)
    uv = uv_grid.copy()
    on_b = (np.isclose(uv[:, 0], 0) | np.isclose(uv[:, 0], 1) | np.isclose(uv[:, 1], 0) | np.isclose(uv[:, 1], 1))
    uv[on_b] += 0.01 * np.sin(7 * uv[on_b][:, ::-1])
    f2 = triangulate(q_src)
    xyz = np.c_[uv_grid, 0.2 * np.sin(3 * uv_grid[:, 0]) * np.cos(2 * uv_grid[:, 1])]
    f3 = f2.copy()
    # The "paper" quad mesh: a 5x5 grid of the exact unit square.
    quv, quads = grid_quads(5, 5)
    rp = seam_conforming_reparameterization(xyz, f3, uv, f2, quv, quads, seam_matching=True)
    sa = signed_uv_areas(rp.uv, rp.f2)
    assert np.all(sa > 0), "chart must stay injective"
    assert np.sum(sa) == pytest.approx(1.0, rel=1e-9), "chart region == quad region"
    # 3D geometry of S is unchanged: every chart vertex lies on the source surface
    assert np.allclose(rp.xyz[:, 2], 0.2 * np.sin(3 * rp.xyz[:, 0]) * np.cos(2 * rp.xyz[:, 1]), atol=2e-2)
    assert np.isfinite(rp.metrics["zhang2021_onestring_sd_mean_after"])
    assert rp.metrics["zhang2021_onestring_sd_max_after"] < 1e3
    assert np.array_equal(rp.quads, quads), "no cut -> seam matching must not modify Q"


def test_cylinder_cut_gives_a_hard_conforming_seam():
    xyz, f3, uv, f2 = cylinder_chart()
    u2x = uv_to_xyz_vertex_map(f3, f2, len(uv))
    src = build_source_boundary(uv, f2, u2x, xyz)
    kinds = set(src.node_kind.values())
    assert kinds == {"transition"}
    assert [c.kind for c in seam_constraints(src)] == ["pair"]
    quv, quads = grid_quads(6, 3, 0.0, 2 * np.pi, 0.0, 1.0)
    rp = seam_conforming_reparameterization(xyz, f3, uv, f2, quv, quads)
    assert rp.metrics["zhang2021_onestring_hard_seam_branches"] == 1
    assert rp.metrics["zhang2021_onestring_soft_seam_branches"] == 0
    from onestring_physics.optcuts_seam_exact_closure import discover_exact_seam_topology
    chart = SimpleNamespace(surface_vertices_3d=rp.xyz, surface_faces=rp.f3, uv_vertices_2d=rp.uv,
                            uv_faces=rp.f2, metrics={})
    topo = discover_exact_seam_topology(chart, rp.quad_uv, rp.quads)
    assert topo["status"] == "conforming"
    assert topo["unclassified_boundary_edges"] == 0
    assert len(topo["paired_edges"]) == 3


def test_monotone_correspondence_is_order_preserving():
    xyz, f3, uv, f2 = cylinder_chart()
    u2x = uv_to_xyz_vertex_map(f3, f2, len(uv))
    src = build_source_boundary(uv, f2, u2x, xyz)
    quv, quads = grid_quads(6, 3, 0.0, 2 * np.pi, 0.0, 1.0)
    init = initial_correspondence(src, quv, quads)
    s = np.asarray(init.s)
    k = int(np.argmin(s))
    ss = np.r_[s[k:], s[:k]]
    assert np.all(np.diff(ss) > 0)
    assert init.snapped_nodes == 4 and not init.unsnapped_nodes


# --------------------------------------------------------- mode wiring
def test_plan_reports_requested_versus_actual_and_ccw_faces(monkeypatch):
    from onestring_physics import optcuts_zhang2021_quad_20261027 as mode
    xyz, f3, uv, f2 = cylinder_chart()
    quv, quads = grid_quads(6, 3, 0.0, 2 * np.pi, 0.0, 1.0)

    def fake_runner(xyz_, f3_, uv_, f2_, zp, bin_dir=None):
        return SimpleNamespace(quad_vertices=quv, quads=quads, metrics={
            "zhang2021_flipped_quads": 0, "zhang2021_quad_boundary_loops": 1, "zhang2021_n_patches": 1,
            "zhang2021_j_min": 1.0, "zhang2021_d_h_percent_bbox": 0.0, "zhang2021_target_quads": zp.target_quads})

    monkeypatch.setattr(mode, "run_reference_pipeline", fake_runner)
    plan = mode.plan_zhang2021_m2d(xyz, f3, uv, f2, SimpleNamespace(zhang2021_target_quads=35))
    m = plan.metrics
    assert m["zhang2021_target_reached"] is False
    assert m["zhang2021_paper_output_quads"] == 18 and m["zhang2021_final_m2d_quads"] == 18
    assert "requested 35" in m["zhang2021_requested_vs_actual_quads"]
    sa = signed_uv_areas(plan.quad_uv, plan.quads[:, [0, 1, 2]])
    assert np.all(sa > 0), "M2D faces follow OneString's counter-clockwise QuadGrid convention"
    assert np.isfinite(plan.m3d_preview).all()
    assert np.allclose(np.linalg.norm(plan.m3d_preview[:, :2], axis=1), 1.0, atol=1e-6)


def test_mode_is_inactive_unless_flag_set():
    from onestring_physics import optcuts_zhang2021_quad_20261027 as mode
    assert not mode.mode_enabled(SimpleNamespace())
    assert mode.mode_enabled(SimpleNamespace(optcuts_zhang2021_quad_20261027_enabled=True))


def test_pipeline_parameters_default_off():
    from onestring_physics.onestring_pipeline import PipelineParameters
    p = PipelineParameters()
    assert p.optcuts_zhang2021_quad_20261027_enabled is False
    assert p.optcuts_planar_quad_20261026_enabled is False
    assert p.zhang2021_downstream_route == "optcuts_test2"
    assert p.zhang2021_initial_csf_split_enabled is False


def test_initial_omega_csf_split_policy_default_off_and_opt_in():
    from onestring_physics import optcuts_zhang2021_quad_20261027 as mode
    from onestring_physics.onestring_pipeline import PipelineParameters

    domain = SimpleNamespace()
    params = PipelineParameters(optcuts_zhang2021_quad_20261027_enabled=True)
    assert mode._configure_initial_csf_split(domain, params) is False
    assert domain.paper_split_policy is False
    assert domain.csf_split_enabled is False

    params.zhang2021_initial_csf_split_enabled = True
    params.enable_csf_splits = False
    assert mode._configure_initial_csf_split(domain, params) is False
    assert domain.paper_split_policy is False

    params.enable_csf_splits = True
    assert mode._configure_initial_csf_split(domain, params) is True
    assert domain.paper_split_policy is True
    assert domain.csf_split_enabled is True


def test_disabled_initial_split_preserves_m2d_through_simple_split_wrapper(monkeypatch):
    from onestring_physics import optcuts_zhang2021_quad_20261027 as mode
    from onestring_physics import simple_split_panel_patch as split_module

    uv, faces = grid_quads(2, 1)
    original_xyz = np.column_stack((uv, np.zeros(len(uv))))
    grid = mode.UnstructuredQuadGrid(uv, faces, 0.5, 0.04)

    def quadmesh(xyz, quads, actual_grid, stage, metrics, split_lines):
        return SimpleNamespace(vertices=np.asarray(xyz).copy(), faces=np.asarray(quads).copy(),
                               grid=actual_grid, stage=stage, metrics=dict(metrics),
                               split_lines=list(split_lines))

    initial = quadmesh(original_xyz, faces, grid, "M2D", {}, [])
    fake = SimpleNamespace(
        _original=SimpleNamespace(QuadMesh=quadmesh),
        _build_m2d=lambda *args: initial,
        _lift_m2d_to_m3d=lambda *args: (None, None),
        _optimize_k2d=lambda *args, **kwargs: (None, None),
    )
    debug = SimpleNamespace()
    monkeypatch.setattr(
        split_module, "apply_paper_split",
        lambda *args: pytest.fail("legacy initial Omega CSF Split must not run"),
    )
    split_module.install_simple_split_panel_patch(fake, debug)

    domain = SimpleNamespace(split_lines=[], localized_split_segments=[])
    params = SimpleNamespace(zhang2021_initial_csf_split_enabled=False,
                             enable_csf_splits=True)
    assert mode._configure_initial_csf_split(domain, params) is False
    result = fake._build_m2d(grid, domain, params)
    assert np.array_equal(result.vertices, original_xyz)
    assert np.array_equal(result.faces, faces)
    assert result.metrics["csf_split_applied"] is False
    assert result.metrics["number_of_splits"] == 0


# ------------------------------------------------------- downstream route
def three_quad_triangle():
    """Three quads around one interior vertex of valence 3 (odd): the face
    adjacency is a triangle, so no 2-colouring exists."""
    c = np.array([0.0, 0.0])
    corners = [np.array([np.cos(a), np.sin(a)]) for a in (np.pi / 2, np.pi / 2 + 2 * np.pi / 3,
                                                           np.pi / 2 + 4 * np.pi / 3)]
    mids = [(corners[i] + corners[(i + 1) % 3]) / 2 for i in range(3)]
    uv = np.vstack([c] + corners + mids)       # 0 centre, 1..3 corners, 4..6 mids
    quads = np.array([[0, 6, 1, 4], [0, 4, 2, 5], [0, 5, 3, 6]])
    return uv, quads


def test_odd_valence_interior_vertices():
    from onestring_physics import optcuts_zhang2021_quad_20261027 as mode
    _, quads = grid_quads(4, 3)
    assert mode.odd_valence_interior_vertices(quads) == []
    _, tri = three_quad_triangle()
    assert mode.odd_valence_interior_vertices(tri) == [0]


def test_downstream_route_validation():
    from onestring_physics import optcuts_zhang2021_quad_20261027 as mode
    assert mode.downstream_route(SimpleNamespace()) == mode.ROUTE_OPTCUTS_TEST2
    assert mode.downstream_route(SimpleNamespace(
        zhang2021_downstream_route="paper_local_global_0918")) == mode.ROUTE_PAPER_LOCAL_GLOBAL
    with pytest.raises(mode.Zhang2021QuadModeError, match="ZHANG2021_DOWNSTREAM_ROUTE_INVALID"):
        mode.downstream_route(SimpleNamespace(zhang2021_downstream_route="grid"))


def test_route_hooks_dispatch_only_for_the_selected_route(monkeypatch):
    from onestring_physics import optcuts_zhang2021_quad_20261027 as mode
    calls = []
    pipe = SimpleNamespace(
        _optimize_k3d=lambda *a: calls.append("inner_k3d") or ("k3d", None),
        _optimize_k2d=lambda *a, progress_callback=None: calls.append("inner_k2d") or ("k2d", None))
    monkeypatch.setattr(mode, "_paper_local_global_k3d", lambda *a: calls.append("lg_k3d") or ("lg", None))
    monkeypatch.setattr(mode, "_paper_local_global_k2d", lambda *a: calls.append("lg_k2d") or ("lg", None))
    mode._install_route_hooks(pipe)
    off = SimpleNamespace(zhang2021_downstream_route="paper_local_global_0918")      # mode flag off
    test2 = SimpleNamespace(optcuts_zhang2021_quad_20261027_enabled=True)
    lg = SimpleNamespace(optcuts_zhang2021_quad_20261027_enabled=True,
                         zhang2021_downstream_route="paper_local_global_0918")
    for params in (off, test2):
        pipe._optimize_k3d(None, None, None, params)
        pipe._optimize_k2d(None, None, params)
    assert calls == ["inner_k3d", "inner_k2d", "inner_k3d", "inner_k2d"]
    calls.clear()
    pipe._optimize_k3d(None, None, None, lg)
    pipe._optimize_k2d(None, None, lg)
    assert calls == ["lg_k3d", "lg_k2d"]


def test_paper_route_k2d_fails_explicitly_on_odd_valence_vertices():
    from onestring_physics import optcuts_zhang2021_quad_20261027 as mode
    uv, quads = three_quad_triangle()
    m2d = SimpleNamespace(vertices=np.c_[uv, np.zeros(len(uv))], faces=quads,
                          grid=SimpleNamespace(nx=0), metrics={})
    m3d = SimpleNamespace(vertices=np.c_[uv, np.zeros(len(uv))], faces=quads, metrics={})
    with pytest.raises(mode.Zhang2021QuadModeError, match="ZHANG2021_PAPER_LG_LINKAGE_NON_BIPARTITE"):
        mode._paper_local_global_k2d(SimpleNamespace(), m2d, m3d, SimpleNamespace())


def test_finalize_split_rebuilds_grid_and_rejects_uncertified_bounds():
    from onestring_physics import optcuts_zhang2021_quad_20261027 as mode
    uv, quads = grid_quads(2, 1)
    grid = mode.UnstructuredQuadGrid(uv, quads, 0.5, 0.04)
    # Simulated Split: the middle edge duplicated and the right quad shifted.
    split_uv = np.vstack([uv, uv[[1, 4]] + [0.1, 0.0]])
    split_uv[[2, 5]] += [0.1, 0.0]
    split_faces = np.array([[0, 1, 4, 3], [6, 2, 5, 7]])
    metrics = {"csf_split_bound_certified": True, "zhang2021_effective_uv_tile_size": 0.5,
               "zhang2021_effective_uv_gap_size": 0.04}
    mesh = SimpleNamespace(vertices=np.c_[split_uv, np.zeros(len(split_uv))], faces=split_faces,
                           grid=grid, metrics=metrics)
    domain = SimpleNamespace(csf_before=3.0)
    params = SimpleNamespace(enable_csf_splits=True, csf_split_threshold=2.0)
    out = mode._finalize_split(mesh, domain, params)
    assert out.metrics["zhang2021_post_split_grid_reconstructed"] is True
    assert [t.vertex_ids for t in out.grid.tiles] == [tuple(f) for f in split_faces.tolist()]
    assert mode._finalize_split(out, domain, params) is out, "idempotent: audited once"
    bad = SimpleNamespace(vertices=mesh.vertices, faces=split_faces, grid=grid,
                          metrics={"csf_split_bound_certified": False})
    with pytest.raises(mode.Zhang2021QuadModeError, match="ZHANG2021_CSF_SPLIT_UNCERTIFIED"):
        mode._finalize_split(bad, domain, params)


def test_m2d_hook_defers_the_split_to_the_simple_split_owner():
    src = (ROOT / "src/onestring_physics/optcuts_zhang2021_quad_20261027.py").read_text(encoding="utf-8")
    assert '_paper_split_deferred' in src and 'simple_split_wrapper' in src
    # Exactly one apply_paper_split call site, guarded by the deferral flag.
    assert src.count("apply_paper_split(") == 1


def test_ui_registers_the_dated_mode_once_with_honest_label():
    tree = ast.parse((ROOT / "app_runtime_base.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "selectbox" and n.args and isinstance(n.args[0], ast.Constant)
             and n.args[0].value == "Omega parameterization mode"]
    assert len(calls) == 1
    choices = ast.literal_eval(calls[0].args[1])
    assert choices.count("optcuts_zhang2021_quad_20261027") == 1
    assert choices.count("optcuts_planar_quad_20261026") == 1, "the Gmsh prototype stays available"
    src = ast.unparse(calls[0])
    assert "Zhang et al. 2021" in src and "authors' reference code" in src


# ----------------------------------------------------------- setup script
def _load_setup():
    spec = importlib.util.spec_from_file_location("z21setup", ROOT / "scripts" / "setup_zhang2021_quad_official.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_setup_normalisation_rules():
    z = _load_setup()
    assert z.normalise("a\r\nb".encode("utf-16-le").join([b"\xff\xfe", b""])) == b"a\nb"
    assert z.normalise(b"\xef\xbb\xbfx\r\ny") == b"x\ny"
    assert z.normalise(b"x\r\ny") == b"x\ny"
    assert ".cpp" in z.NORMALISED_EXTENSIONS and ".vcxproj" not in z.NORMALISED_EXTENSIONS
    assert len(z.ARCHIVE_SHA256) == 64
    assert z.PATCH.is_file()


# ------------------------------------------- reference programs (optional)
def _binaries_or_skip():
    from onestring_physics.zhang2021_quad_reference import Zhang2021UnavailableError, resolve_reference_binaries
    try:
        return resolve_reference_binaries()
    except Zhang2021UnavailableError:
        pytest.skip("Zhang 2021 reference programs not built (scripts/setup_zhang2021_quad_official.py)")


def test_reference_pipeline_on_a_rectangle_gives_one_patch():
    _binaries_or_skip()
    from onestring_physics.zhang2021_quad_reference import Zhang2021Parameters, run_reference_pipeline
    uv, q = grid_quads(30, 15, 0.0, 2.0, 0.0, 1.0)
    tris = triangulate(q)
    res = run_reference_pipeline(np.c_[uv, np.zeros(len(uv))], tris, uv, tris,
                                 Zhang2021Parameters(target_quads=200, hausdorff_ratio=0.01))
    m = res.metrics
    assert m["zhang2021_n_patches"] == 1
    assert m["zhang2021_flipped_quads"] == 0
    assert m["zhang2021_j_min"] == pytest.approx(1.0, abs=1e-6)
    assert m["zhang2021_hausdorff_bound_satisfied"] is True


# ------------------------------------------- Omega pre-conditioning (design Step 2)
def wavy_square_chart(amp=0.03, n=16):
    uv_grid, q_src = grid_quads(n, n)
    uv = uv_grid.copy()
    on_b = (np.isclose(uv[:, 0], 0) | np.isclose(uv[:, 0], 1) | np.isclose(uv[:, 1], 0) | np.isclose(uv[:, 1], 1))
    uv[on_b] += amp * np.sin(9 * uv[on_b][:, ::-1])
    f2 = triangulate(q_src)
    xyz = np.c_[uv_grid, 0.2 * np.sin(3 * uv_grid[:, 0]) * np.cos(2 * uv_grid[:, 1])]
    return xyz, f2.copy(), uv, f2


def test_douglas_peucker_keeps_only_corners_beyond_tolerance():
    from onestring_physics.zhang2021_onestring_precondition import _douglas_peucker
    P = np.array([[0, 0], [1, 0.01], [2, -0.01], [3, 0], [3.5, 1.0], [4, 2]], float)
    assert _douglas_peucker(P, 0.05) == [0, 3, 5]
    assert _douglas_peucker(P, 2.0) == [0, 5]


def test_preconditioning_straightens_boundary_bijectively_within_tolerance():
    from onestring_physics.zhang2021_onestring_precondition import precondition_omega
    from onestring_physics.zhang2021_onestring_boundary import oriented_boundary_loops
    xyz, f3, uv, f2 = wavy_square_chart()
    ident = precondition_omega(xyz, f3, uv, f2, 0.0)
    assert ident.accepted and np.array_equal(ident.uv, uv)
    pre = precondition_omega(xyz, f3, uv, f2, 0.06, sd_mean_ratio_max=2.0, sd_p95_ratio_max=3.0)
    assert pre.accepted, pre.reason
    assert np.all(signed_uv_areas(pre.uv, pre.f2) > 0), "h must stay bijective"
    m = pre.metrics
    assert m["zhang2021_precondition_boundary_hausdorff_percent_bbox"] <= 6.0 + 1e-6
    assert m["zhang2021_precondition_corners"] < m["zhang2021_precondition_boundary_vertices"]
    # S is unchanged: split points lie on the source surface, original vertices keep their 3D position
    assert np.allclose(pre.xyz[:len(xyz)], xyz)
    loop = oriented_boundary_loops(pre.f2)[0]
    assert len(loop) == len(oriented_boundary_loops(f2)[0]), "the boundary vertex set is unchanged"


def test_preconditioning_rejects_explicitly_when_distortion_limit_is_exceeded():
    from onestring_physics.zhang2021_onestring_precondition import precondition_omega
    xyz, f3, uv, f2 = wavy_square_chart()
    pre = precondition_omega(xyz, f3, uv, f2, 0.06, sd_mean_ratio_max=0.5, sd_p95_ratio_max=0.5)
    assert not pre.accepted
    assert pre.reason.startswith("ZHANG2021_PRECONDITION_DISTORTION_REJECTED")
    assert np.array_equal(pre.uv, uv), "a rejected candidate returns Omega_0 untouched"


def test_preconditioning_keeps_seam_nodes_fixed():
    from onestring_physics.zhang2021_onestring_precondition import precondition_omega
    xyz, f3, uv, f2 = cylinder_chart()
    u2x = uv_to_xyz_vertex_map(f3, f2, len(uv))
    src = build_source_boundary(uv, f2, u2x, xyz)
    nodes = [src.loop[j] for j in src.nodes]
    pre = precondition_omega(xyz, f3, uv, f2, 0.05, sd_mean_ratio_max=5.0, sd_p95_ratio_max=5.0)
    assert pre.accepted, pre.reason
    assert np.allclose(pre.uv[nodes], uv[nodes])


def test_coordinated_planner_reports_every_candidate(monkeypatch):
    from onestring_physics import optcuts_zhang2021_quad_20261027 as mode
    xyz, f3, uv, f2 = wavy_square_chart()
    calls = []

    def fake_runner(xyz_, f3_, uv_, f2_, zp, bin_dir=None):
        calls.append(zp.hausdorff_ratio)
        n = 6 if zp.hausdorff_ratio > 0.03 else 10
        quv, quads = grid_quads(n, n, uv_[:, 0].min(), uv_[:, 0].max(), uv_[:, 1].min(), uv_[:, 1].max())
        return SimpleNamespace(quad_vertices=quv, quads=quads, metrics={
            "zhang2021_flipped_quads": 0, "zhang2021_quad_boundary_loops": 1, "zhang2021_n_patches": 1,
            "zhang2021_j_min": 1.0, "zhang2021_d_h_percent_bbox": 1.0, "zhang2021_target_quads": zp.target_quads})

    monkeypatch.setattr(mode, "run_reference_pipeline", fake_runner)
    params = SimpleNamespace(zhang2021_target_quads=35, zhang2021_precondition_tolerances="0,0.06",
                             zhang2021_candidate_hausdorff_ratios="0.02,0.05",
                             zhang2021_precondition_sd_mean_ratio_max=2.0,
                             zhang2021_precondition_sd_p95_ratio_max=3.0)
    plan = mode.plan_zhang2021_m2d_coordinated(xyz, f3, uv, f2, params)
    rows = plan.metrics["zhang2021_coordinated_candidates"]
    assert len(rows) == 4 and len(calls) == 4
    assert plan.metrics["zhang2021_coordinated_chosen_hausdorff_ratio"] == 0.05, "36 quads beat 100 for N=35"
    assert all("score" in r for r in rows if r["status"] == "ok")
    assert np.array_equal(plan.source.uv, uv), "metrics are reported against the original OptCuts chart"
