"""Quality measurements and OneString-aware selection of official OptCuts runs."""
from collections import defaultdict
import numpy as np
from .optcuts_uv_overlap_guard import positive_area_uv_overlaps


def _relative_linear_scale(area_3d, area_2d):
    """Return local inverse linear scale relative to the chart-wide area scale.

    For a locally conformal map, A_3D/A_2D = lambda^2.  The old audit divided
    the area ratio by its minimum triangle, so one finite sliver could rescale
    the entire chart.  Use the square-root area ratio (linear scale) and
    normalize it by the global area ratio instead.
    """
    area_3d = np.asarray(area_3d, dtype=float)
    area_2d = np.asarray(area_2d, dtype=float)
    local = np.sqrt(np.divide(
        area_3d,
        area_2d,
        out=np.full(len(area_3d), np.inf, dtype=float),
        where=area_2d > 0,
    ))
    total_3d = float(np.sum(area_3d))
    total_2d = float(np.sum(area_2d))
    if total_3d <= 0 or total_2d <= 0 or not np.isfinite(total_3d + total_2d):
        return np.full(len(area_3d), np.inf, dtype=float), float("nan")
    global_scale = float(np.sqrt(total_3d / total_2d))
    if not np.isfinite(global_scale) or global_scale <= 0:
        return np.full(len(area_3d), np.inf, dtype=float), global_scale
    return local / global_scale, global_scale


def quality(result):
    from .optcuts_backend import _triangle_differential_metrics
    xyz, sf, uv, uf = result.surface_vertices_3d, result.surface_faces, result.uv_vertices_2d, result.uv_faces
    differential = _triangle_differential_metrics(xyz, sf, uv, uf)
    sd = np.asarray(differential['per_triangle_symmetric_dirichlet'])
    tri = xyz[sf]; q = uv[uf]
    area = .5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    a,b=q[:,1]-q[:,0],q[:,2]-q[:,0]
    flat = .5*np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])
    csf, global_scale = _relative_linear_scale(area, flat)
    edges = defaultdict(list)
    for f,g in zip(sf,uf):
        for i in range(3):
            j=(i+1)%3
            edges[tuple(sorted((int(f[i]),int(f[j]))))].append(tuple(sorted((int(g[i]),int(g[j])))))
    seams = [edge for edge,copies in edges.items() if len(copies)==2 and copies[0]!=copies[1]]
    overlap,_=positive_area_uv_overlaps(uv,uf)
    total_area = float(area.sum())
    area_mean = float(np.dot(sd,area)/total_area) if total_area > 0 else float('inf')
    high_area = float(area[csf>2].sum()/total_area) if total_area > 0 else float('inf')
    out=dict(distortion_max=float(sd.max()),distortion_p95=float(np.percentile(sd,95)),
             distortion_area_mean=area_mean,
             csf_max=float(csf.max()),csf_p95=float(np.percentile(csf,95)),
             csf_over_2_area_fraction=high_area,
             csf_global_linear_scale=float(global_scale),
             seam_length=float(sum(np.linalg.norm(xyz[a]-xyz[b]) for a,b in seams)),
             seam_edge_count=len(seams),flipped_triangles=differential['uv_triangle_flip_count'],
             degenerate_triangles=differential['uv_degenerate_triangle_count'],
             injectivity_overlap_pairs=int(overlap),
             csf_definition='sqrt(A3/A2) normalized by global sqrt(sum(A3)/sum(A2)); inverse local linear scale relative to chart-wide scale')
    out['csf_tail_penalty']=csf_tail_penalty_from_quality(out)
    return out



def csf_tail_penalty_from_result(result, onset=1.8, threshold=2.0, sharpness=20.0):
    """Area-weighted smooth CSF-tail energy on every source triangle.

    Unlike the compact candidate score below, this is a genuine dense tail
    objective suitable for local UV refinement:
        mean_A softplus(k * (CSF-onset))^2
    plus a stronger continuation above the hard OneString threshold.
    """
    xyz=np.asarray(result.surface_vertices_3d,float)
    sf=np.asarray(result.surface_faces,int)
    uv=np.asarray(result.uv_vertices_2d,float)
    uf=np.asarray(result.uv_faces,int)
    tri=xyz[sf]; q=uv[uf]
    area=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    a,b=q[:,1]-q[:,0],q[:,2]-q[:,0]
    flat=.5*np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])
    csf,_=_relative_linear_scale(area,flat)
    if not np.all(np.isfinite(csf)) or float(area.sum())<=0:
        return float('inf')
    w=area/float(area.sum())
    soft=lambda x: np.logaddexp(0.0,sharpness*x)/sharpness
    return float(np.dot(w,soft(csf-onset)**2 + 3.0*soft(csf-threshold)**2))

def csf_tail_penalty_from_quality(metrics, onset=1.8, threshold=2.0, sharpness=20.0):
    """Smooth diagnostic preference for the high-CSF tail."""
    p95=float(metrics['csf_p95']); maximum=float(metrics['csf_max'])
    area=float(metrics['csf_over_2_area_fraction'])
    def softplus(x):
        z=sharpness*x
        return float(np.logaddexp(0.0,z)/sharpness)
    return float(4.0*area + softplus(p95-onset)**2 + 0.25*softplus(maximum-threshold)**2)


def selection_score(metrics, split_count=None):
    """Lexicographic OneString score: real Split count first, then CSF tail."""
    invalid=sum(int(metrics[k]) for k in ('flipped_triangles','degenerate_triangles','injectivity_overlap_pairs'))
    return (invalid,
            int(split_count) if split_count is not None else 10**9,
            float(metrics.get('csf_tail_penalty', csf_tail_penalty_from_quality(metrics))),
            float(metrics['csf_over_2_area_fraction']),
            float(metrics['csf_p95']),
            float(metrics['csf_max']),
            float(metrics['distortion_area_mean']))

def pareto_improves(candidate, baseline):
    """Select an alternative OptCuts run without requiring every metric to improve.

    Invalid UVs remain a hard rejection.  Among valid results, OneString cares
    first about eliminating the area whose relative linear scale exceeds 2.
    Seam length is diagnostic, not a veto.  Distortion is allowed to trade
    moderately for a better CSF tail, but catastrophic (>1.5x) worsening is
    rejected.
    """
    finite_keys=('distortion_max','distortion_p95','distortion_area_mean',
                 'csf_max','csf_p95','csf_over_2_area_fraction','seam_length')
    valid=all(candidate[k]==0 for k in ('flipped_triangles','degenerate_triangles','injectivity_overlap_pairs'))
    valid=valid and all(np.isfinite(candidate[k]) for k in finite_keys)
    if not valid:
        return False

    # Keep a broad safety rail for parameterization quality, rather than the
    # previous all-metrics-must-not-worsen Pareto gate.
    for key in ('distortion_max','distortion_p95','distortion_area_mean'):
        base=float(baseline[key]); cand=float(candidate[key])
        if np.isfinite(base) and cand > 1.5 * max(base, np.finfo(float).tiny):
            return False

    eps=1e-8
    b_tail=float(baseline['csf_over_2_area_fraction'])
    c_tail=float(candidate['csf_over_2_area_fraction'])
    if c_tail < b_tail - eps:
        return True
    if c_tail > b_tail + eps:
        return False

    # If neither candidate has more >2 area, prefer the lower high-end scale.
    b_p95=float(baseline['csf_p95']); c_p95=float(candidate['csf_p95'])
    if c_p95 < b_p95 - eps * max(1.0, abs(b_p95)):
        return True
    if c_p95 > b_p95 + eps * max(1.0, abs(b_p95)):
        return False

    b_max=float(baseline['csf_max']); c_max=float(candidate['csf_max'])
    if c_max < b_max - eps * max(1.0, abs(b_max)):
        return True

    # With an equivalent CSF tail, accept a genuine distortion improvement.
    return any(
        float(candidate[k]) < float(baseline[k]) - eps * max(1.0, abs(float(baseline[k])))
        for k in ('distortion_max','distortion_p95','distortion_area_mean')
    )
