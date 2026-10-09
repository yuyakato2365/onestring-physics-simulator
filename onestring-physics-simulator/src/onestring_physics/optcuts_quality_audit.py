"""Quality measurements and conservative selection of official OptCuts runs."""
from collections import defaultdict
import numpy as np
from .optcuts_uv_overlap_guard import positive_area_uv_overlaps


def quality(result):
    from .optcuts_backend import _triangle_differential_metrics
    xyz, sf, uv, uf = result.surface_vertices_3d, result.surface_faces, result.uv_vertices_2d, result.uv_faces
    differential = _triangle_differential_metrics(xyz, sf, uv, uf)
    sd = np.asarray(differential['per_triangle_symmetric_dirichlet'])
    tri = xyz[sf]; q = uv[uf]
    area = .5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    a,b=q[:,1]-q[:,0],q[:,2]-q[:,0]
    flat = .5*np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])
    raw = np.divide(area,flat,out=np.full(len(area),np.inf),where=flat>0)
    csf = raw / max(float(raw.min()),np.finfo(float).tiny)
    edges = defaultdict(list)
    for f,g in zip(sf,uf):
        for i in range(3):
            j=(i+1)%3
            edges[tuple(sorted((int(f[i]),int(f[j]))))].append(tuple(sorted((int(g[i]),int(g[j])))))
    seams = [edge for edge,copies in edges.items() if len(copies)==2 and copies[0]!=copies[1]]
    overlap,_=positive_area_uv_overlaps(uv,uf)
    return dict(distortion_max=float(sd.max()),distortion_p95=float(np.percentile(sd,95)),
                distortion_area_mean=float(np.dot(sd,area)/area.sum()),
                csf_max=float(csf.max()),csf_p95=float(np.percentile(csf,95)),
                csf_over_2_area_fraction=float(area[csf>2].sum()/area.sum()),
                seam_length=float(sum(np.linalg.norm(xyz[a]-xyz[b]) for a,b in seams)),
                seam_edge_count=len(seams),flipped_triangles=differential['uv_triangle_flip_count'],
                degenerate_triangles=differential['uv_degenerate_triangle_count'],
                injectivity_overlap_pairs=int(overlap),
                csf_definition='global normalized inverse area Jacobian; not a conformality certificate')


def pareto_improves(candidate, baseline):
    keys=('distortion_max','distortion_p95','distortion_area_mean','csf_max','csf_p95',
          'csf_over_2_area_fraction','seam_length')
    valid=all(candidate[k]==0 for k in ('flipped_triangles','degenerate_triangles','injectivity_overlap_pairs'))
    no_worse=all(np.isfinite(candidate[k]) and candidate[k]<=baseline[k]+1e-8*max(1.,baseline[k]) for k in keys)
    improves=any(candidate[k]<baseline[k]-1e-8*max(1.,baseline[k]) for k in keys)
    return bool(valid and no_worse and improves)
