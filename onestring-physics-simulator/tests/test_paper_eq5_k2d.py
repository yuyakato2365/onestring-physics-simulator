"""Numerical and topological regressions for the Section 4.3 reconstruction."""
import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from onestring_physics import onestring_pipeline as pipeline
from onestring_physics.paper_eq5_k2d_solver import (
    build_linkage_topology, project_angle_vectors, collision_energy_gradient,
    FlatObjective, optimize_paper_eq5,
)


def mesh(nx=3,ny=3):
    grid=pipeline.create_quad_grid(nx,ny,1.,.08)
    return pipeline.QuadMesh(grid.vertex_positions.copy(),np.array([t.vertex_ids for t in grid.tiles]),grid,'M2D')


def test_grid_joint_counts_and_actual_void_stencils():
    m=mesh();top,xy=build_linkage_topology(m)
    assert len(top.hinges)==12
    assert len(xy)==24  # 36 corners minus twelve *pairwise* welds
    assert np.max(np.bincount(top.faces.ravel()))==2
    assert np.all(top.gaps[:,1]!=top.gaps[:,2])
    assert np.allclose(xy[top.gaps[:,1]],xy[top.gaps[:,2]])  # initially closed, different DOF
    for a,ca,b,cb in top.hinges:
        assert top.faces[a,ca]==top.faces[b,cb]
    # Corner-sharing diagonal tiles are not directly connected.
    assert (0,4) not in {(a,b) for a,_,b,_ in top.hinges}


def test_split_ids_are_not_rewelded():
    m=mesh(2,1)
    m.vertices=np.vstack((m.vertices,m.vertices[m.faces[1]]))
    m.faces[1]=np.arange(6,10)
    top,_=build_linkage_topology(m)
    assert len(top.hinges)==0
    assert len(top.source_vertex_ids)==8


@pytest.mark.parametrize('degrees,lengths',[(0,(1,3)),(2,(3,.4)),(120,(1,2)),(175,(.7,3)),(40,(1,2))])
def test_angle_projection_is_nearest_variable_length_solution(degrees,lengths):
    a=np.array([[lengths[0],0.]])
    b=lengths[1]*np.array([[np.cos(np.radians(degrees)),np.sin(np.radians(degrees))]])
    p,q,_=project_angle_vectors(a,b,np.radians(5))
    actual=float(np.sum((p-a)**2+(q-b)**2))
    beta=np.radians(np.clip(degrees,5,90))
    def cost(phi,sign):
        u=np.array([np.cos(phi),np.sin(phi)])
        v=np.array([np.cos(phi+sign*beta),np.sin(phi+sign*beta)])
        x=max(0.,a[0]@u)*u;y=max(0.,b[0]@v)*v
        return float(np.sum((x-a[0])**2+(y-b[0])**2))
    # Independent dense angular search followed by scalar bounded minimization.
    grid=np.linspace(-np.pi,np.pi,2001)
    reference=min(minimize_scalar(lambda t:cost(t,sign),bounds=(grid[i]-.004,grid[i]+.004),method='bounded',options={'xatol':1e-13}).fun
                  for sign in (1,-1) for i in [int(np.argmin([cost(t,sign) for t in grid]))])
    assert actual==pytest.approx(reference,abs=1e-10)
    if degrees==0:
        assert not np.isclose(np.linalg.norm(p),lengths[0])  # old rotations kept lengths


def numerical_gradient(fun,x):
    h=1e-6;g=np.zeros_like(x)
    for i in range(x.size):
        plus=x.copy();minus=x.copy();plus.flat[i]+=h;minus.flat[i]-=h
        g.flat[i]=(fun(plus)-fun(minus))/(2*h)
    return g


def test_collision_gradient_includes_moving_sat_axis():
    xy=np.array([[0.,0.],[1.2,.1],[1.1,1.],[.1,.8], [.71,.42],[1.61,.23],[1.9,1.31],[.9,1.5]])
    faces=np.arange(8).reshape(2,4)
    energy,g,n=collision_energy_gradient(xy,faces)
    assert n==1 and energy>0
    numeric=numerical_gradient(lambda x:collision_energy_gradient(x,faces)[0],xy)
    np.testing.assert_allclose(g,numeric,atol=1e-7)
    # Containment must require exit from the outer box, not just the inner width.
    xy=np.array([[0,0],[4,0],[4,4],[0,4],[1,1],[2,1],[2,2],[1,2]],float)
    assert collision_energy_gradient(xy,faces)[0]==pytest.approx(4.)


def test_combined_objective_gradient():
    m=mesh(2,1);top,xy=build_linkage_topology(m)
    xy+=np.random.default_rng(41).normal(0,.015,xy.shape)
    objective=FlatObjective(top,m.vertices,xy,(1.,.8,.4),np.radians(15))
    _,g,_=objective.evaluate(xy)
    numeric=numerical_gradient(lambda x:objective.evaluate(x)[0],xy)
    np.testing.assert_allclose(g,numeric,atol=1e-7)


def test_solver_monotone_and_pairwise_hinges_exact(monkeypatch):
    monkeypatch.setenv('ONESTRING_EQ5_ITERATIONS','160')
    monkeypatch.setenv('ONESTRING_EQ5_W_FAB','1')
    m=mesh();k=mesh();k.vertices*=.94
    out,report=optimize_paper_eq5(m,k,pipeline.PipelineParameters(),pipeline=pipeline)
    rows=out.eq5_history['records'];energy=np.array([r['EFlat'] for r in rows])
    assert energy[-1]<energy[0]*.01
    assert np.all(np.diff(energy)<=1e-12)
    assert rows[-1]['EFab']<rows[0]['EFab']*.01
    assert rows[-1]['collisions']==0
    assert rows[-1]['fab_violations']<rows[0]['fab_violations']
    assert out.eq5_history['snapshots'][0]['fab_violations']>0
    assert report.before_error==energy[0]
    assert np.max(np.bincount(out.faces.ravel()))==2
    assert np.all(np.isfinite(out.vertices))


def test_flat_layout_preserves_solved_positions_and_source_tile_ids(monkeypatch):
    monkeypatch.setenv('ONESTRING_EQ5_ITERATIONS','10')
    m=mesh(2,2)
    out,_=optimize_paper_eq5(m,m,pipeline.PipelineParameters(),pipeline=pipeline)
    layout=pipeline._make_flat_tile_layout(out,pipeline.PipelineParameters())
    np.testing.assert_array_equal(layout.tile_top_vertices_2d,out.vertices[out.faces,:2])
    assert len(layout.hinge_pairs)==4
    assert all(0<=a<len(m.faces) and 0<=b<len(m.faces) for a,b in layout.hinge_pairs)


def test_t2d_preserves_corner_correspondence_with_reversed_target_winding():
    m=mesh(2,2);top,xy=build_linkage_topology(m)
    k=mesh(2,2);k.vertices[:,0]*=-1  # inverse UV map can reverse orientation
    out=pipeline.QuadMesh(np.column_stack((xy,np.zeros(len(xy)))),top.faces,m.grid,'K2D',{'collisions':0},linkage_topology=top)
    layout=pipeline._make_flat_tile_layout(out)
    t3,_=pipeline._extrude_tiles(k,.05,'T3D')
    t2,_=pipeline._make_t2d_from_transforms(out,layout,k,t3,'T2D top hinge')
    np.testing.assert_allclose(t2.vertices[:,:4],layout.tile_top_vertices_3d,atol=1e-12)
    assert t2.metrics['tile_shape_max_error_to_T3D']<1e-12
    assert np.allclose(np.linalg.det(t2.transform_matrices[:,:3,:3]),1.)
