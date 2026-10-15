"""Regressions for physical distortion caused by retiling recursive cuts."""
import numpy as np
import pytest
from types import SimpleNamespace
import postcut_rebff_iterative as split
from postcut_rebff_full_pipeline import _assemble_patches
from onestring_physics.onestring_pipeline import PipelineParameters
from onestring_physics.split_geometry_constraints import optimize_shared

def chart():
    uv=np.array([[i/8,j/8] for j in range(9) for i in range(9)],float)
    faces=[]
    for j in range(8):
        for i in range(8):
            a=j*9+i;faces.extend([[a,a+1,a+10],[a,a+10,a+9]])
    xyz=np.column_stack([uv,.2*np.sin(uv[:,0]*2)*np.sin(uv[:,1]*2)])
    return dict(side='initial',xyz=xyz,new_uv=uv,faces=np.array(faces),
                bff_metadata={'bff_rectangle_verified':True})

def diagnostic_bff(x,f,rectangle_corner_positions=None):
    # Deliberately anisotropic: this must NEVER become the fixed chart.
    return x[:,:2]*[3.,.4], {'bff_rectangle_verified':True}
diagnostic_bff.seam_corners_required=True
diagnostic_bff.geometry_policy='fixed_initial_bff'
diagnostic_bff.fixed_grid_spacing=.125

def test_cut_restricts_original_inverse_instead_of_warping_it():
    patches=split._split_and_rebff(chart(),'row',.75,diagnostic_bff)
    for p in patches:
        np.testing.assert_allclose(p['new_uv'],p['xyz'][:,:2],atol=1e-15)
        assert p['postcut_joint_atlas_metadata']['postcut_joint_atlas_conformality_preserved']
        assert not np.allclose(p['new_uv'],p['independent_bff_uv'])
    children=split._split_and_rebff(patches[0],'col',.625,diagnostic_bff)
    for p in children:
        np.testing.assert_allclose(p['new_uv'],p['xyz'][:,:2],atol=1e-15)

def test_recursive_candidate_and_final_mesh_use_same_source_lattice():
    a,b=split._split_and_rebff(chart(),'row',.75,diagnostic_bff)
    candidates=split.candidate_lines(a['xyz'],a['new_uv'],a['faces'],8,fixed_spacing=.125)
    assert candidates
    for c in candidates:
        assert c['coordinate']/.125==pytest.approx(round(c['coordinate']/.125))
    children=split._split_and_rebff(a,'col',.625,diagnostic_bff)
    out=_assemble_patches(children+[b],PipelineParameters(nx=8,ny=8),8,31)
    # Density matching must NOT change the pitch to chase the requested count.
    q=out['quad_vertices'][out['quad_faces']]
    edges=np.linalg.norm(np.roll(q,-1,axis=1)-q,axis=2)
    np.testing.assert_allclose(edges,.125,atol=1e-14)
    assert len(q)==64
    assert out['audit']['postcut_seam_uv_preserves_csf_ratio']
    assert out['audit']['split_vertex_pairs']
    assert out['audit']['postcut_omega_boundary_tile_loss']==0

def test_assembled_equality_keeps_exported_split_vertex_ids():
    p=chart();a,b=split._split_and_rebff(p,'row',.5,diagnostic_bff)
    out=_assemble_patches([a,b],PipelineParameters(nx=8,ny=8),8,64)
    from onestring_physics.onestring_pipeline import QuadMesh
    mesh=QuadMesh(out['quad_vertices'],out['quad_faces'],None,'M3D',out['audit'],[])
    def solve(target,reduced,param,settings):
        # A nonuniform deterministic move exposes independently solved seam IDs.
        reduced.vertices=reduced.vertices+np.arange(len(reduced.vertices))[:,None]*.001
        return reduced,None
    result,_=optimize_shared(solve,None,mesh,None,None)
    np.testing.assert_array_equal(result.faces,mesh.faces)
    assert len(result.vertices)==len(mesh.vertices)
    for i,j in out['audit']['split_vertex_pairs']:
        assert i!=j
        np.testing.assert_array_equal(result.vertices[i],result.vertices[j])

def test_experimental_stage_hooks_restore_on_failure(monkeypatch):
    import postcut_rebff_full_pipeline as full
    a,b=split._split_and_rebff(chart(),'row',.5,diagnostic_bff)
    original=full.pipe._original
    names=['_build_m2d','_optimize_k3d','_extrude_tiles','_optimize_k2d']
    saved={name:getattr(original,name) for name in names}
    def fail(*args,**kwargs):
        assert all(getattr(original,n) is not saved[n] for n in names)
        raise RuntimeError('intentional downstream failure')
    monkeypatch.setattr(full.pipe,'_ORIGINAL_BUILD_ONESTRING_DESIGN',fail)
    monkeypatch.setattr(full,'load_target_shape',lambda path:None)
    experiment={'patches':[a,b],'report':dict(linear_limit_two_satisfied=True,
                initial_m2d={'quads':64},max_postcut_patch_linear_span=1.1,
                geometry_policy='fixed_initial_bff')}
    from pathlib import Path
    with pytest.raises(RuntimeError,match='intentional downstream failure'):
        full.run_existing_pipeline_through_dual_hinge(Path('unused.obj'),experiment,grid=8)
    assert all(getattr(original,n) is saved[n] for n in names)
