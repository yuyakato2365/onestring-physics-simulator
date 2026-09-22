"""Small deterministic tests; no OptCuts process, Bunny solve, or physics run."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
from onestring_physics import onestring_pipeline as p
from onestring_physics import extrusion_aware as exp
from onestring_physics.extrusion_curvature import principal_curvature, align_principal_grid
from onestring_physics.extrusion_quality import evaluate_extrusion
from onestring_physics.paper_extrusion import extrude_paper_face_planarity
from onestring_physics.paper_local_global_solvers import optimize_paper_local_global_k3d, optimize_paper_local_global_k2d
from onestring_physics.quad_grid import create_quad_grid
from onestring_physics.heightfields import HeightField


def fixture_surface(kind='saddle', n=5):
    xy = np.linspace(-.7, .7, n)
    x, y = np.meshgrid(xy, xy)
    if kind == 'plane':
        z = x*0
    elif kind == 'cylinder':
        z = 2-np.sqrt(4-x*x)
    elif kind == 'sphere':
        z = 3-np.sqrt(9-x*x-y*y)
    elif kind == 'neck':
        z = .3*np.exp(-(x/.22)**2)*(1-y*y)
    else:
        z = .4*(x*x-y*y)
    vertices = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    faces = np.asarray([[j*n+i, j*n+i+1, (j+1)*n+i+1, (j+1)*n+i]
                        for j in range(n-1) for i in range(n-1)])
    triangles = np.vstack((faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]))
    parameterization = p.SurfaceParameterization('test', vertices.copy(), triangles,
        vertices[:, :2].copy(), triangles.copy(), np.array([[-.7,-.7],[.7,-.7],[.7,.7],[-.7,.7],[-.7,-.7]]))
    grid = create_quad_grid(n-1, n-1, 1.4/(n-1), .04)
    mesh = p.QuadMesh(vertices.copy(), faces, grid, 'M3D', {})
    return mesh, parameterization


@pytest.fixture(autouse=True)
def small_budget(monkeypatch):
    monkeypatch.setenv('ONESTRING_PAPER_K3D_ITERATIONS', '2')
    monkeypatch.setenv('ONESTRING_EQ5_ITERATIONS', '2')
    monkeypatch.setenv('ONESTRING_K3D_PLANARITY_POLISH', '0')
    monkeypatch.setenv('ONESTRING_K3D_PLANARITY_MODE', 'soft')


@pytest.mark.parametrize('kind', ['plane', 'cylinder', 'sphere', 'saddle', 'neck'])
def test_shared_quality_and_curvature_are_finite(kind):
    mesh, surface = fixture_surface(kind)
    q = evaluate_extrusion(mesh.vertices, mesh.faces, .08)
    assert np.isfinite(q['residuals']).all()
    k, d, confidence, _ = principal_curvature(surface.surface_vertices_3d, surface.surface_faces)
    assert np.isfinite(k).all() and np.isfinite(d).all()
    assert np.all((confidence >= 0) & (confidence <= 1))
    if kind == 'plane':
        assert q['difficulty'].max() < 1e-12
        assert confidence.max() == 0
    else:
        assert q['difficulty'].max() > 0


def test_principal_grid_rotation_confidence_and_umibilic_fallback():
    mesh, p0 = fixture_surface('plane')
    before = p0.uv_vertices_2d.copy()
    align_principal_grid(p0)
    np.testing.assert_array_equal(before, p0.uv_vertices_2d)
    _, sphere = fixture_surface('sphere', 9)
    _, cylinder = fixture_surface('cylinder', 9)
    cs = principal_curvature(sphere.surface_vertices_3d, sphere.surface_faces)[2]
    cc = principal_curvature(cylinder.surface_vertices_3d, cylinder.surface_faces)[2]
    assert np.mean(cs) < np.mean(cc)*.4
    theta = .4
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    cylinder.uv_vertices_2d = cylinder.uv_vertices_2d @ rotation
    cylinder.omega_boundary = cylinder.omega_boundary @ rotation
    align_principal_grid(cylinder)
    assert abs(cylinder.metrics['principal_grid_rotation_degrees']) > .1


@pytest.mark.parametrize('mask', ['0000','1000','0100','0001','1100','0101','1001','1101'])
def test_supported_feature_paths(mask):
    mesh, surface = fixture_surface('neck', 4)
    params = p.PipelineParameters(model_version=exp.VERSION, thickness=.08, extrusion_max_nfev=2,
        **dict(zip(exp.FEATURES, (c == '1' for c in mask))))
    exp.validate(params)
    exp.prepare_parameterization(surface, params)
    domain = SimpleNamespace(uv_vertices=surface.uv_vertices_2d, overlay_nx=3, overlay_ny=3)
    csf = np.ones(len(surface.uv_vertices_2d))*1.9
    scored = exp.split_cost_values(surface, domain, mesh.grid, params, csf, p)
    if mask[3] == '1':
        assert np.max(scored-csf) > 0
    flat = copy.deepcopy(mesh); flat.vertices[:, 2] = 0
    # Rotation modifies only UV; synthetic mesh uses original surface coordinates.
    flat.vertices[:, :2] = surface.uv_vertices_2d
    solver = lambda t, m, s, ps: optimize_paper_local_global_k3d(t, m, s, ps, pipeline=p)
    out, _ = exp.optimize_assembled(solver, None, flat, mesh, surface, params, p)
    if mask[1] == '1':
        assert out.metrics['extrusion_objective_applied']
        assert out.metrics['extrusion_objective_after'] <= out.metrics['extrusion_objective_before']+1e-10
    out.metrics['paper_t3d_planarity_iterations'] = 2
    tiles, _ = extrude_paper_face_planarity(out, .08, 'T3D', p)
    assert len(out.faces) == len(tiles.vertices) == len(mesh.faces)
    np.testing.assert_array_equal(out.vertices[out.faces], tiles.vertices[:, :4])
    np.testing.assert_array_equal(out.faces, mesh.faces)
    k2d, _ = optimize_paper_local_global_k2d(flat, out, params, pipeline=p)
    layout = p._make_flat_tile_layout(k2d, params)
    t2d, _ = p._make_t2d_from_transforms(k2d, layout, out, tiles, 'T2D top hinge')
    assert t2d.vertices.shape == tiles.vertices.shape
    assert np.isfinite(t2d.vertices).all()
    np.testing.assert_array_equal(k2d.linkage_topology.source_faces, mesh.faces)


@pytest.mark.parametrize('mask', [f'{i:04b}' for i in range(16) if i & 2])
def test_adaptive_masks_explicitly_unsupported(mask):
    params = p.PipelineParameters(model_version=exp.VERSION, **dict(zip(exp.FEATURES, (c == '1' for c in mask))))
    with pytest.raises(NotImplementedError, match='regular lattice'):
        exp.validate(params)


def test_split_geometry_equality_and_flat_topology_retained():
    mesh, surface = fixture_surface('saddle', 3)
    flat = copy.deepcopy(mesh); flat.vertices[:, 2] = 0
    # Actual CSF split duplicates along x=0 using the repository splitter.
    v, f, count = p._split_m2d_along_existing_grid_line(flat.vertices, flat.faces, ('col', 0.))
    assert count > 0
    flat.vertices, flat.faces = v, f
    mesh.vertices = np.array([p.inverse_map_uv_to_surface(point[:2], surface)[0] for point in v])
    mesh.faces = f.copy()
    params = p.PipelineParameters(model_version=exp.VERSION, use_extrusion_aware_k3d=True, extrusion_max_nfev=2)
    solver = lambda t,m,s,ps: optimize_paper_local_global_k3d(t,m,s,ps,pipeline=p)
    out, _ = exp.optimize_assembled(solver, None, flat, mesh, surface, params, p)
    mapping = np.asarray(out.metrics['assembled_geometry_map'])
    assert out.metrics['assembled_equivalence_group_count'] > 0
    assert len(out.vertices) == len(v) and len(np.unique(mapping)) < len(v)
    out.metrics['paper_t3d_planarity_iterations'] = 2
    tiles, _ = extrude_paper_face_planarity(out, params.thickness, 'T3D', p)
    for group in np.unique(mapping):
        ids = np.flatnonzero(mapping == group)
        np.testing.assert_allclose(out.vertices[ids], np.broadcast_to(out.vertices[ids[0]], (len(ids),3)), atol=0)
        copies = [tiles.vertices[fi, li+4] for fi, face in enumerate(f) for li, vid in enumerate(face) if vid in ids]
        np.testing.assert_allclose(copies, np.broadcast_to(copies[0], (len(copies),3)), atol=0)
    np.testing.assert_array_equal(flat.faces, f)
    np.testing.assert_array_equal(flat.vertices, v)


@pytest.mark.parametrize("with_split", [False, True])
def test_baseline_full_driver_bitwise(monkeypatch, tmp_path, with_split):
    # Exercise the authoritative driver, real parameterization/M2D/lift and all
    # downstream geometry on a small surface, with current paper solver routes.
    owner = p._original
    monkeypatch.setattr(owner, '_optimize_k3d', lambda t,m,s,ps: optimize_paper_local_global_k3d(t,m,s,ps,pipeline=p))
    monkeypatch.setattr(owner, '_optimize_k2d', lambda a,b,ps,progress_callback=None: optimize_paper_local_global_k2d(a,b,ps,pipeline=p))
    monkeypatch.setattr(owner, '_extrude_tiles', lambda m,h,s: extrude_paper_face_planarity(m,h,s,p))
    if with_split:
        monkeypatch.setattr(p, '_parameterization_stretch_csf', lambda surface: np.full(len(surface.uv_vertices_2d), 2.5))
    kwargs = dict(nx=2, ny=2, localize_csf_splits=False, tile_size=.5, omega_overlay_margin=0, enable_csf_splits=with_split, csf_split_threshold=1.001,
        omega_boundary_mode='shape_preserving_experimental', omega_parameterization_mode='pca_debug',
        allow_experimental_pipeline=True, surface_mesh_subdivisions=2, hinge_layout_iterations=2,
        extrusion_results_dir=str(tmp_path))
    target = HeightField('saddle', {'amplitude': .1})
    old = p.build_onestring_design(target, p.PipelineParameters(model_version='2026-09-20-paper-t3d', **kwargs))
    new = p.build_onestring_design(target, p.PipelineParameters(model_version=exp.VERSION, **kwargs))
    for key in ['mesh_2d_initial','mesh_3d_initial','mesh_3d_optimized','mesh_2d_optimized',
                'tiles_3d','tiles_2d_top_hinge','tiles_2d_dual_hinge']:
        a,b = getattr(old,key),getattr(new,key)
        np.testing.assert_array_equal(a.vertices,b.vertices)
        if hasattr(a,'faces'):
            np.testing.assert_array_equal(a.faces,b.faces)
    assert len(old.hinge_graph.hinges) == len(new.hinge_graph.hinges)
    assert old.mesh_2d_initial.split_lines == new.mesh_2d_initial.split_lines
    if with_split:
        assert new.mesh_2d_initial.split_lines
    record = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert record['feature_mask'] == '0000'
    assert set(record['runtime']) == {'parameterization','grid','K3D','T3D','K2D','T2D'}
    assert len(record['panel_extrusion_difficulty']) == record['panel_count']


def test_sparse_extrusion_gradient_matches_directional_difference():
    mesh, _ = fixture_surface('neck', 3)
    energy, grad = exp.make_extrusion_objective(mesh.vertices, mesh.faces, .08, .7)
    flat = mesh.vertices.ravel()
    direction = np.random.default_rng(4).normal(size=flat.shape)
    direction /= np.linalg.norm(direction)
    numerical = (energy(flat+1e-6*direction)-energy(flat-1e-6*direction))/2e-6
    assert np.dot(grad(flat),direction) == pytest.approx(numerical, rel=2e-3, abs=1e-5)


def test_split_candidate_additive_cost_is_used():
    mesh, surface = fixture_surface('neck', 5)
    csf = np.ones(len(mesh.vertices))*1.1
    difficulty = 4*np.exp(-(surface.uv_vertices_2d[:,0]/.3)**2)
    lines = exp.select_split_candidates(surface, csf, csf+difficulty, 1.9, 2, p)
    assert lines
    scores = surface.metrics['extrusion_split_candidate_scores']
    assert all(r['combined_cost'] == r['existing_cost']+r['extrusion_cost'] for r in scores)
    assert any(r['extrusion_cost'] > 0 for r in scores)


def test_unrelated_coincident_sheets_are_not_joined():
    mesh, surface = fixture_surface('plane', 3)
    n = len(mesh.vertices)
    surface.surface_vertices_3d = np.vstack([surface.surface_vertices_3d]*2)
    surface.surface_faces = np.vstack([surface.surface_faces, surface.surface_faces+n])
    surface.uv_vertices_2d = np.vstack([surface.uv_vertices_2d, surface.uv_vertices_2d+[3,0]])
    surface.uv_faces = np.vstack([surface.uv_faces, surface.uv_faces+n])
    mesh.vertices = np.vstack([mesh.vertices]*2)
    mesh.faces = np.vstack([mesh.faces, mesh.faces+n])
    flat = copy.deepcopy(mesh)
    flat.vertices[:,:2] = surface.uv_vertices_2d
    mapping = exp.source_equivalence(flat, mesh, surface, p)
    assert len(np.unique(mapping)) == 2*n
