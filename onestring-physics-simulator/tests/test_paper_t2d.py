"""Flat-linkage invariants on an exactly realizable rotating-square fixture."""
import copy
import numpy as np
import pytest
from onestring_physics import onestring_pipeline as p
from onestring_physics.paper_eq5_k2d_solver import build_linkage_topology
from onestring_physics.paper_t2d import build_hinge_graph, build_gap_graph, validate_flat_linkage


def fixture(n=3):
    grid=p.create_quad_grid(n,n,1.,.08)
    m=p.QuadMesh(grid.vertex_positions.copy(),np.array([t.vertex_ids for t in grid.tiles]),grid,'M2D')
    top,xy=build_linkage_topology(m)
    source=m.vertices[m.faces].copy()
    # Alternating rigid rotations and exact corner constraints, no optimizer.
    rotated=source.copy()
    for i,tile in enumerate(source):
        angle=np.radians(20 if (i//n+i%n)%2 else -20)
        c,s=np.cos(angle),np.sin(angle)
        rotated[i,:,:2]=tile[:,:2]@np.array([[c,s],[-s,c]])
    translations={0:np.zeros(3)}
    while len(translations)<len(source):
        for a,ca,b,cb in top.hinges:
            if a in translations and b not in translations:
                translations[b]=rotated[a,ca]+translations[a]-rotated[b,cb]
            elif b in translations and a not in translations:
                translations[a]=rotated[b,cb]+translations[b]-rotated[a,ca]
    rotated+=np.array([translations[i] for i in range(len(source))])[:,None,:]
    xy3=np.zeros((len(xy),3))
    for face,tile in zip(top.faces,rotated): xy3[face]=tile
    k2=p.QuadMesh(xy3,top.faces.copy(),grid,'K2D',{'fabrication_feasible':True,'theta_min_deg':5.,'collisions':0},linkage_topology=top)
    layout=p._make_flat_tile_layout(k2)
    solid=np.concatenate([source,source+np.array([0,0,-.08])],axis=1)
    t3=p.TileAssembly(solid,np.array([[0,1,2,3]]),np.array([[4,7,6,5]]),np.array([[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]),'T3D')
    t2,_=p._make_t2d_from_transforms(k2,layout,m,t3,'T2D top hinge')
    return k2,t2,t3


@pytest.fixture
def linkage(): return fixture()


def test_t2d_preserves_tile_identity(linkage):
    k,t,_=linkage
    assert t.metrics['tile_ids']==list(range(len(k.faces)))
    np.testing.assert_array_equal(t.vertices[:,:4],k.vertices[k.faces])


def test_t2d_preserves_hinge_topology(linkage):
    k,t,t3=linkage
    graph=p._build_hinge_graph(k.grid,k.linkage_topology.source_faces,t,t3,False)
    assert [(h.tile_a,h.local_vertex_a,h.tile_b,h.local_vertex_b) for h in graph.hinges]==list(map(tuple,k.linkage_topology.hinges))
    assert t.metrics['hinge_connection_max_error']<1e-12


def test_t2d_has_no_invalid_overlap(linkage):
    _,t,_=linkage
    assert t.metrics['flat_collision_count']==0


def test_t2d_edge_lengths_match_k2d(linkage):
    k,t,_=linkage
    expected=np.linalg.norm(np.roll(k.vertices[k.faces],-1,axis=1)-k.vertices[k.faces],axis=2)
    actual=np.linalg.norm(np.roll(t.vertices[:,:4],-1,axis=1)-t.vertices[:,:4],axis=2)
    np.testing.assert_allclose(actual,expected,atol=1e-12)


def test_t2d_fabrication_constraints(linkage):
    _,t,_=linkage
    assert t.metrics['fabrication_gap_violations']==0
    assert t.metrics['fabrication_gap_min_deg']==pytest.approx(40.)
    assert t.metrics['fabrication_feasible']
    assert t.metrics['tile_shape_max_error_to_T3D']<1e-12


def test_t2d_string_graph_is_connected(linkage):
    _,t,t3=linkage
    graph=p._build_gap_graph(t.linkage_topology.source_faces,t,t3)
    assert graph.metrics['string_graph_connected']
    interior=[g for g in graph.gaps if not g.boundary]
    assert len(interior)==4
    assert all(len(g.surrounding_tiles)==4 for g in interior)
    route=p._build_string_path(graph,[],.2)
    assert route.gap_ids
    assert set(route.gap_ids)<=set(range(len(graph.gaps)))


def test_t2d_rejects_broken_hinge_even_without_collision(linkage):
    _,t,t3=linkage
    t.vertices[0]+=np.array([-100.,0.,0.])
    check=validate_flat_linkage(t,t3,p)
    assert not check['fabrication_feasible']
    assert check['hinge_connection_max_error']>1.


def test_t2d_noncongruent_frustum_is_not_claimed_exact(linkage):
    k,_,t3=linkage
    t3.vertices[:,4:,:2]*=.9
    m=copy.copy(k);m.faces=k.linkage_topology.source_faces
    t,_=p._make_t2d_from_transforms(k,p._make_flat_tile_layout(k),m,t3,'T2D')
    assert t.metrics['top_to_bottom_rigid_fit_max_error']>1e-3
    assert not t.metrics['fabrication_feasible']
    assert not t.metrics['t2d_t3d_congruent_tile_geometry']


def test_t2d_dual_solve_preserves_k2d_edge_lengths(linkage):
    k,t,t3=linkage
    from onestring_physics.paper_hinge_solver import optimize_hinge_poses
    out,graph,_=optimize_hinge_poses(k.grid,k.linkage_topology.source_faces,t,t3,p.PipelineParameters(hinge_layout_iterations=2),p)
    assert out.metrics['t2d_edge_length_max_error_to_k2d']<1e-12
    assert out.metrics['fabrication_feasible']
    assert len(graph.hinges)==len(k.linkage_topology.hinges)


def test_sat_projection_separates_contained_polygons():
    from onestring_physics.paper_local_global_solvers import _sat_projection
    a=np.array([[0,0],[4,0],[4,4],[0,4]],float)
    b=np.array([[1,1],[2,1],[2,2],[1,2]],float)
    shift=_sat_projection(a,b)
    assert np.linalg.norm(shift)==pytest.approx(1.)
    assert _sat_projection(a-shift,b+shift) is None


def test_t2d_string_graph_does_not_bridge_disconnected_tiles(linkage):
    _,t,t3=linkage
    t.linkage_topology.hinges=np.empty((0,4),dtype=int)
    graph=build_gap_graph(t,t3,p)
    assert graph.metrics['gap_graph_components']==9
    assert not graph.metrics['string_graph_connected']


def test_t2d_explicit_linkage_bypasses_independent_translation_cleanup(linkage,monkeypatch):
    from types import SimpleNamespace
    import onestring_physics.optcuts_run_flag_patch as patch
    _,t,t3=linkage
    graph=build_hinge_graph(t,t3,p,dual=True)
    module=SimpleNamespace(_optimize_dual_hinges=lambda:(t,graph,None))
    monkeypatch.setattr(patch,'_enabled',lambda:True)
    def forbidden(*args): raise AssertionError('must not disconnect linkage after Eq.6')
    monkeypatch.setattr(patch,'_project_dual_hinge_hard_nonpenetration',forbidden)
    patch._wire_dual_hard_wrapper(module)
    assert module._optimize_dual_hinges()[0] is t
