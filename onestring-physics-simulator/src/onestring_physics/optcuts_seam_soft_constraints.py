"""Safe, non-projective OptCuts Seam fitting for incompatible Grid partitions.

C @ x is the physical source-edge matched seam discrepancy.  Using
||C x||² as an objective is well-defined even when ker(C) contains only
translation (a collapsed seam).  NEVER project onto ker(C) in soft mode.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np
from scipy import sparse


def row_matrix(n_vertices, rows):
    data=[]; ii=[]; jj=[]
    for k,row in enumerate(rows):
        for vertex,weight in row:
            vertex=int(vertex)
            if not 0<=vertex<int(n_vertices):
                raise ValueError(
                    f"SEAM_SOFT_ROW_VERTEX_OUT_OF_BOUNDS vertex={vertex} "
                    f"n={n_vertices}"
                )
            ii.append(k);jj.append(vertex);data.append(float(weight))
    return sparse.coo_matrix(
        (data,(ii,jj)),shape=(len(rows),int(n_vertices))
    ).tocsr()


def remap_rows(rows,mapping):
    """Translate ORIGINAL Grid vertex IDs into a reduced geometry view."""
    mapping=np.asarray(mapping,int)
    out=[]
    for row in rows:
        coeff=defaultdict(float)
        for i,w in row:
            coeff[int(mapping[int(i)])]+=float(w)
        vals=[[i,float(w)] for i,w in sorted(coeff.items())
              if abs(w)>1e-12]
        if vals:out.append(vals)
    return out


def residual_stats(vertices,rows):
    verts=np.asarray(vertices,float)
    if not rows:
        return {"rms":0.,"max":0.,"row_count":0}
    C=row_matrix(len(verts),rows)
    discrepancies=C@verts
    lengths=np.linalg.norm(discrepancies,axis=1)
    return {
        "rms":float(np.sqrt(np.mean(lengths*lengths))),
        "max":float(np.max(lengths)),
        "row_count":int(len(rows)),
    }


def soft_seam_energy_gradient(vertices,rows,weight,scale=1.):
    """0.5 * weight / (m * scale²) * || C X ||² and EXACT gradient."""
    vertices=np.asarray(vertices,float)
    grad=np.zeros_like(vertices)
    if not rows or weight<=0:
        return 0.,grad
    C=row_matrix(len(vertices),rows)
    d=C@vertices
    denom=max(float(scale)**2,1e-20)*len(rows)
    energy=.5*float(weight)*float(np.sum(d*d))/denom
    grad[:]=(float(weight)/denom)*(C.T@d)
    return energy,grad


def seam_matrix_rank_diagnostics(rows,n_vertices,max_seam_vertices=200):
    """Small/medium deterministic dense SVD audit; large inputs safe-skip.

    Rank on *participating* vertices gives the physical one-coordinate
    feasible dimension of the constrained seam. A value of one when
    multiple seam bends are required exposes forced total collapse.
    """
    ids=sorted({int(i) for row in rows for i,w in row if abs(w)>1e-12})
    data={
        "seam_vertices":int(len(ids)),
        "constraint_rows":int(len(rows)),
        "seam_rank":None,
        "seam_nullity":None,
        "rank_evaluated":False,
    }
    if not ids or len(ids)>int(max_seam_vertices):
        return data
    C=row_matrix(int(n_vertices),rows)[:,ids].toarray()
    singular=np.linalg.svd(C,compute_uv=False)
    cutoff=max(C.shape)*np.finfo(float).eps*(
        float(singular.max()) if len(singular) else 0.
    )*10.
    rank=int(np.sum(singular>cutoff))
    data.update({
        "seam_rank":rank,
        "seam_nullity":int(len(ids)-rank),
        "rank_evaluated":True,
        "singular_value_threshold":float(cutoff),
    })
    return data
