"""Read-only geometric comparison (distances are in source mesh units)."""
import sys,json,pickle
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'src'))
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from onestring_physics.paper_eq5_k2d_solver import collision_energy_gradient
from onestring_physics.solid_collision_audit import audit_solid_collisions
from onestring_physics import onestring_pipeline

def quad_metrics(q):
    e=np.roll(q,-1,axis=1)-q;length=np.linalg.norm(e,axis=2)
    _,_,vt=np.linalg.svd(q-q.mean(1)[:,None,:],full_matrices=False)
    plane=np.abs(np.einsum('fvi,fi->fv',q-q.mean(1)[:,None,:],vt[:,-1]))
    cosine=np.sum((-np.roll(e,1,axis=1))*e,axis=2)/np.maximum(length*np.roll(length,1,axis=1),1e-30)
    angles=np.rad2deg(np.arccos(np.clip(cosine,-1,1)))
    aspect=length.max(1)/np.maximum(length.min(1),1e-30)
    area=.5*(np.linalg.norm(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),axis=1)+np.linalg.norm(np.cross(q[:,2]-q[:,0],q[:,3]-q[:,0]),axis=1))
    return dict(tiles=len(q),aspect_median=float(np.median(aspect)),aspect_p95=float(np.quantile(aspect,.95)),aspect_max=float(aspect.max()),angle_min_deg=float(angles.min()),angle_max_deg=float(angles.max()),planarity_max=float(plane.max()),planarity_rms=float(np.sqrt(np.mean(plane**2))),area_min=float(area.min()),area_median=float(np.median(area)),edge_median=float(np.median(length)))

def exact_distance(points,triangles):
    d=[]
    for p in points:
        c=trimesh.triangles.closest_point(triangles,np.broadcast_to(p,(len(triangles),3)))
        d.append(float(np.min(np.linalg.norm(c-p,axis=1))))
    return dict(rms=float(np.sqrt(np.mean(np.array(d)**2))),max=float(max(d)))

def measure(s):
    source=s.target_surface
    tris=source.vertices[source.faces]
    out={}
    out['source_bbox_diagonal']=float(np.linalg.norm(np.ptp(source.vertices,axis=0)))
    for label,mesh in [('M2D',s.mesh_2d_initial),('M3D',s.mesh_3d_initial),('K3D',s.mesh_3d_optimized)]:
        q=mesh.vertices[mesh.faces];out[label]=quad_metrics(q)
        if label!='M2D':out[label]['source_distance_vertices']=exact_distance(mesh.vertices,tris)
    k=s.mesh_3d_optimized
    if hasattr(k,'k3d_before_extrusion_vertices'):
        out['K3D_before_T3D']=quad_metrics(k.k3d_before_extrusion_vertices[k.faces])
    # Bidirectional surface approximation, including panel interiors. These
    # are 512 deterministically selected triangle-centroid samples, not a
    # continuous Hausdorff bound.
    ids=np.linspace(0,len(tris)-1,min(512,len(tris)),dtype=int)
    ktri=k.vertices[np.vstack([k.faces[:,[0,1,2]],k.faces[:,[0,2,3]]])]
    out['K3D']['source_centroid_sample_distance']=exact_distance(tris[ids].mean(1),ktri)
    out['K3D']['panel_center_distance']=exact_distance(k.vertices[k.faces].mean(1),tris)
    for label,t in [('T3D',s.tiles_3d),('T2D',s.tiles_2d_dual_hinge)]:
        out[label]=quad_metrics(t.vertices[:,:4,:])
    t=s.tiles_2d_dual_hinge.vertices
    hulls=[ConvexHull(p[:,:2]).vertices.tolist() for p in t];width=max(map(len,hulls))
    faces=np.array([[8*i+j for j in h+[h[-1]]*(width-len(h))] for i,h in enumerate(hulls)])
    E,g,count,diag=collision_energy_gradient(t[:,:,:2].reshape(-1,2),faces,return_diagnostics=True)
    out['T2D'].update(footprint_colliding_pairs=count,footprint_penetration_max=diag['penetration_max'],footprint_penetration_rms=diag['penetration_rms'],footprint_energy=E)
    out['T2D']['solid_collisions']=audit_solid_collisions(t)['solid_collision_pair_count']
    d0=np.linalg.norm(s.tiles_3d.vertices[:,:,None,:]-s.tiles_3d.vertices[:,None,:,:],axis=-1)
    d1=np.linalg.norm(t[:,:,None,:]-t[:,None,:,:],axis=-1)
    out['T2D']['rigidity_max']=float(abs(d0-d1).max())
    for stage in ['M3D','K3D']:
        m=s.mesh_3d_initial if stage=='M3D' else s.mesh_3d_optimized
        from scipy.spatial import cKDTree
        pairs=np.array(sorted(cKDTree(s.mesh_2d_initial.vertices).query_pairs(1e-9)),int)
        out[stage]['seam_vertex_gap_max']=float(np.linalg.norm(m.vertices[pairs[:,0]]-m.vertices[pairs[:,1]],axis=1).max()) if len(pairs) else 0.
    out['k3d_solver']=s.mesh_3d_optimized.metrics.get('k3d_solver_model')
    out['split_csf']=s.mesh_2d_initial.metrics.get('postcut_rebff_measured_max_csf')
    out['coverage']=s.mesh_2d_initial.metrics.get('postcut_omega_rectangles_area_fraction_covered')
    out['section_4_3_linkage']=s.tiles_2d_dual_hinge.metrics.get('section_4_3_linkage',False)
    return out

if __name__=='__main__':
    p=Path(sys.argv[1]);s=pickle.loads((p/'state.pkl').read_bytes())
    result=measure(s);(p/'geometry.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
