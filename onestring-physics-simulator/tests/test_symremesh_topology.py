"""The mirror-plane snap must not turn a valid polar fan into a non-manifold mesh."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import trimesh

_spec = importlib.util.spec_from_file_location('symremesh_topology_test', Path(__file__).resolve().parents[1] / 'tools/optcuts_soft_fit_20261104/remesh_topology.py')
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def polar_cap():
    V = [[0., 0., 1.]]
    for t in np.linspace(0, np.deg2rad(75), 21)[1:]:
        for p in np.linspace(0, 2*np.pi, 48, endpoint=False):
            V.append([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)])
    F = [[0, 1+j, 1+(j+1) % 48] for j in range(48)]
    for r in range(19):
        for j in range(48):
            a=1+48*r+j; b=1+48*r+(j+1)%48; c=a+48; d=b+48
            F.extend([[a,c,b],[b,c,d]])
    return np.asarray(V, dtype=np.float32).astype(float), np.asarray(F)


def test_dense_pole_falls_back_to_direct_cut():
    V,F=polar_cap();mod.require_manifold(V,F,'input')
    mesh=trimesh.Trimesh(V,F,process=False);h=float(mesh.edges_unique_length.mean())
    H,T,info=mod.prepare_half(V,F,np.array([1.,0.,0.]),0.,h)
    assert not info['broad_snap']
    assert info['rejected_attempts']
    mod.require_manifold(H,T,'half')
    assert np.min(H[:,0]) >= -1e-12
    edges, counts=np.unique(np.sort(np.concatenate([T[:,[0,1]],T[:,[1,2]],T[:,[2,0]]]),axis=1),axis=0,return_counts=True)
    assert len(H)-len(edges)+len(T)==1
    assert (counts==1).any()  # cut and original opening are retained
    import gpytoolbox as gpy
    Hr,Tr=gpy.remesh_botsch(H,T,i=1,h=h,project=True,feature_edges=edges[counts==1])
    mod.require_manifold(Hr,Tr,'after Botsch')


def test_existing_safe_snap_is_retained():
    V=np.array([[x,y,0.] for y in [-1.,0.,1.] for x in [-1.,0.,1.]])
    F=np.array([[a,a+1,a+4] for a in [0,1,3,4]]+[[a,a+4,a+3] for a in [0,1,3,4]])
    H,T,info=mod.prepare_half(V,F,[1.,0.,0.],0.,1.)
    assert info['broad_snap'] and not info['rejected_attempts']
    mod.require_manifold(H,T,'safe half')


def test_rejects_edge_and_vertex_nonmanifold_inputs():
    V=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,-1.,0.],[0.,0.,1.]])
    with pytest.raises(ValueError,match='non-manifold edges'):
        mod.require_manifold(V,[[0,1,2],[1,0,3],[0,1,4]],'test')
    with pytest.raises(ValueError,match='non-manifold vertex'):
        mod.require_manifold(V,[[0,1,2],[0,3,4]],'test')
