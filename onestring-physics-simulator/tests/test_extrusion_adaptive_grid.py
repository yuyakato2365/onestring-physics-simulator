"""Conforming adaptive grid and 16-mask checks, using tiny synthetic surfaces."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
from test_extrusion_aware import fixture_surface, small_budget
from onestring_physics import onestring_pipeline as p
from onestring_physics import extrusion_aware as exp
from onestring_physics.extrusion_adaptive_grid import refine_overlay, attach_panel_correspondence
from onestring_physics.extrusion_curvature import principal_curvature
from onestring_physics.paper_local_global_solvers import optimize_paper_local_global_k3d, optimize_paper_local_global_k2d
from onestring_physics.paper_extrusion import extrude_paper_face_planarity
from onestring_physics.paper_eq5_k2d_solver import build_linkage_topology
from onestring_physics.heightfields import HeightField


def params(**kwargs):
    return p.PipelineParameters(model_version=exp.VERSION, use_curvature_adaptive_grid=True,
                               adaptive_normal_turn_degrees=4., adaptive_max_subdivisions=2, **kwargs)


def overlay(mesh, surface):
    # A row-major rectangular overlay, independent of later UV rotation.
    return SimpleNamespace(uv_vertices=surface.uv_vertices_2d.copy(),
                           overlay_nx=mesh.grid.nx, overlay_ny=mesh.grid.ny)


@pytest.mark.parametrize('kind', ['plane', 'cylinder', 'sphere', 'saddle', 'neck'])
def test_curvature_refinement_is_conforming(kind):
    mesh, surface = fixture_surface(kind, 7)
    domain = overlay(mesh, surface)
    before = domain.uv_vertices.copy()
    result = refine_overlay(surface, domain, mesh.grid, params())
    if kind == 'plane':
        np.testing.assert_array_equal(before, domain.uv_vertices)
        assert result['overlay_cells'] == result['base_overlay_cells']
    else:
        assert result['overlay_cells'] > result['base_overlay_cells']
    grid = p.create_quad_grid(result['nx'], result['ny'])
    faces = np.asarray([t.vertex_ids for t in grid.tiles])
    m = p.QuadMesh(np.column_stack((domain.uv_vertices, np.zeros(len(domain.uv_vertices)))), faces, grid, 'M2D')
    attach_panel_correspondence(m, domain)
    topology, _ = build_linkage_topology(m)
    assert len(topology.hinges) == result['nx']*(result['ny']-1)+result['ny']*(result['nx']-1)
    assert len(np.unique(m.metrics['adaptive_overlay_panel_ids'])) == len(faces)
    assert max(m.metrics['adaptive_base_cell_per_panel']) < result['base_overlay_cells']
    np.testing.assert_array_equal(m.grid.vertex_positions, m.vertices)
    quads = m.vertices[faces]
    assert np.all(np.cross(quads[:,1]-quads[:,0],quads[:,3]-quads[:,0])[:,2] > 0)


def test_simple_split_gaps_preserve_adaptive_panel_ids():
    from onestring_physics.simple_split_panel_patch import _cut_once, _open_split_gaps
    mesh, surface = fixture_surface('neck', 5)
    domain = overlay(mesh, surface)
    data = refine_overlay(surface, domain, mesh.grid, params())
    grid = p.create_quad_grid(data['nx'], data['ny'])
    faces = np.asarray([t.vertex_ids for t in grid.tiles])
    vertices = np.column_stack((domain.uv_vertices, np.zeros(len(domain.uv_vertices))))
    canonical, cut_faces, record = _cut_once(vertices, faces, ('col', 0.))
    assert record is not None
    opened, *_ = _open_split_gaps(canonical, cut_faces, [record], grid)
    assert not np.array_equal(opened, canonical)
    mesh = p.QuadMesh(opened, cut_faces, grid, 'M2D')
    mesh._split_panel_source_vertices = canonical
    attach_panel_correspondence(mesh, domain)
    np.testing.assert_array_equal(mesh.metrics['adaptive_overlay_panel_ids'], np.arange(len(faces)))
    np.testing.assert_array_equal(mesh.vertices, opened)
    topology, _ = build_linkage_topology(mesh)
    np.testing.assert_array_equal(topology.source_faces, cut_faces)


def test_sphere_uses_curvature_not_direction_confidence():
    mesh, surface = fixture_surface('sphere', 7)
    _, _, direction_confidence, _, fit_confidence = principal_curvature(
        surface.surface_vertices_3d, surface.surface_faces, return_fit_confidence=True)
    assert np.mean(direction_confidence) < .15
    assert np.mean(fit_confidence) > .8
    data = refine_overlay(surface, overlay(mesh, surface), mesh.grid, params())
    assert data['overlay_cells'] > data['base_overlay_cells']


def test_refinement_is_nonuniform_and_scale_invariant():
    mesh, surface = fixture_surface('plane', 17)
    v = surface.surface_vertices_3d
    v[:,2] = .15*np.exp(-(v[:,0]**2+v[:,1]**2)/.025)
    ps = params(); ps.adaptive_normal_turn_degrees = 2.
    data = refine_overlay(surface, overlay(mesh, surface), mesh.grid, ps)
    assert 1 in data['u_subdivisions'] and 2 in data['u_subdivisions']
    assert 1 in data['v_subdivisions'] and 2 in data['v_subdivisions']
    scaled = copy.deepcopy(surface); scaled.surface_vertices_3d *= 100
    other = refine_overlay(scaled, overlay(mesh, scaled), mesh.grid, ps)
    assert data['u_subdivisions'] == other['u_subdivisions']
    assert data['v_subdivisions'] == other['v_subdivisions']


def test_budget_failure_leaves_input_untouched():
    mesh, surface = fixture_surface('neck', 7)
    domain = overlay(mesh, surface)
    before = copy.deepcopy(domain)
    with pytest.raises(ValueError, match='budget'):
        refine_overlay(surface, domain, mesh.grid, params(adaptive_max_cells=1))
    np.testing.assert_array_equal(domain.uv_vertices, before.uv_vertices)
    assert (domain.overlay_nx, domain.overlay_ny) == (before.overlay_nx, before.overlay_ny)
    assert not hasattr(domain, 'adaptive_grid')


@pytest.mark.parametrize('value', [0., float('nan'), float('inf'), -1.])
def test_invalid_turn_setting_rejected(value):
    ps = params(); ps.adaptive_normal_turn_degrees = value
    with pytest.raises(ValueError, match='normal_turn'):
        exp.validate(ps)


@pytest.mark.parametrize('mask', [f'{i:04b}' for i in range(16)])
def test_all_masks_full_stage_driver(mask, monkeypatch, tmp_path):
    _, surface = fixture_surface('neck', 7)
    owner = p._original
    # Substitute only the external parameterizer and explicit solver routes.
    # The real driver performs grid/refinement/CSF split/lift/correspondence,
    # K3D/T3D/K2D/top+dual T2D/hinge/gap construction and experiment recording.
    monkeypatch.setattr(owner, '_build_surface_parameterization', lambda *args: copy.deepcopy(surface))
    monkeypatch.setattr(p, '_parameterization_stretch_csf', lambda s: np.full(len(s.uv_vertices_2d), 2.5))
    monkeypatch.setattr(owner, '_optimize_k3d', lambda t,m,s,ps: optimize_paper_local_global_k3d(t,m,s,ps,pipeline=p))
    monkeypatch.setattr(owner, '_optimize_k2d', lambda a,b,ps,progress_callback=None: optimize_paper_local_global_k2d(a,b,ps,pipeline=p))
    def extrude(mesh, h, stage):
        mesh.metrics['paper_t3d_planarity_iterations'] = 2
        return extrude_paper_face_planarity(mesh,h,stage,p)
    monkeypatch.setattr(owner, '_extrude_tiles', extrude)
    ps = p.PipelineParameters(model_version=exp.VERSION, nx=3, ny=3, tile_size=.4, thickness=.06,
        omega_overlay_margin=0, m2d_crop_policy='strict_vertices', omega_parameterization_mode='pca_debug',
        omega_boundary_mode='shape_preserving_experimental', allow_experimental_pipeline=True,
        surface_mesh_subdivisions=2, hinge_layout_iterations=2, extrusion_max_nfev=2,
        csf_split_threshold=1.9, localize_csf_splits=False,
        adaptive_normal_turn_degrees=8., adaptive_max_subdivisions=2,
        extrusion_results_dir=str(tmp_path), **dict(zip(exp.FEATURES, (c=='1' for c in mask))))
    ps.preserve_detected_symmetry = False
    state = p.build_onestring_design(HeightField('sampled', points=surface.surface_vertices_3d, faces=surface.surface_faces), ps)
    n = len(state.mesh_2d_initial.faces)
    for name in ('mesh_3d_initial','mesh_3d_optimized','mesh_2d_optimized'):
        assert len(getattr(state,name).faces) == n
    for name in ('tiles_3d','tiles_2d_top_hinge','tiles_2d_dual_hinge'):
        assert getattr(state,name).vertices.shape == (n,8,3)
        assert np.isfinite(getattr(state,name).vertices).all()
    np.testing.assert_array_equal(state.mesh_3d_optimized.vertices[state.mesh_3d_optimized.faces], state.tiles_3d.vertices[:,:4])
    np.testing.assert_array_equal(state.mesh_2d_optimized.linkage_topology.source_faces, state.mesh_3d_optimized.faces)
    assert all(0 <= h.tile_a < n and 0 <= h.tile_b < n for h in state.hinge_graph.hinges)
    record = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert record['feature_mask'] == mask
    if mask[2] == '1':
        assert record['adaptive_grid']['overlay_cells'] > record['adaptive_grid']['base_overlay_cells']
        assert len(record['adaptive_base_cell_per_panel']) == n
    else:
        assert record['adaptive_grid'] is None
    if mask[3] == '1':
        assert state.mesh_2d_initial.metrics['csf_split_applied']
        assert record['assembled_equivalence_group_count'] > 0
        assert record['assembled_split_boundary_separation_max'] == 0
