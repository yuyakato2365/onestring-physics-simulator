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


def remap_rows(rows,mapping,*,return_stats=False,maximum_dropped_fraction=.05):
    """Remap seam rows and explicitly audit any vanished constraints.

    A row that becomes identically zero after vertex merging is a LOST
    source-Seam constraint, not a harmless optimization. Fail if more
    than five percent vanish. Legacy callers can still receive rows only.
    """
    mapping=np.asarray(mapping,int)
    out=[]
    for row in rows:
        coeff=defaultdict(float)
        for i,w in row:
            src=int(i)
            if src<0 or src>=len(mapping):
                raise ValueError(
                    "SEAM_SOFT_REMAP_SOURCE_VERTEX_OUT_OF_RANGE: "
                    f"{src}/{len(mapping)}"
                )
            target=int(mapping[src])
            if target<0:
                raise ValueError("SEAM_SOFT_REMAP_NEGATIVE_VERTEX_ID")
            coeff[target]+=float(w)
        vals=[[i,float(w)] for i,w in sorted(coeff.items())
              if abs(w)>1e-12]
        if vals:out.append(vals)
    lost=len(rows)-len(out)
    fraction=float(lost/max(len(rows),1))
    stats={
        "seam_rows_before_reduction":int(len(rows)),
        "seam_rows_after_reduction":int(len(out)),
        "seam_rows_dropped_by_reduction":int(lost),
        "seam_rows_dropped_fraction":float(fraction),
    }
    if fraction>float(maximum_dropped_fraction)+1e-12:
        raise ValueError(
            "SEAM_SOFT_REMAP_CONSTRAINT_LOSS_EXCESS: "
            f"dropped={lost}/{len(rows)} ({fraction:.4%}); "
            "vertex reduction erased source Seam constraints"
        )
    return (out,stats) if return_stats else out


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


def seam_matrix_rank_diagnostics(rows,n_vertices,max_seam_vertices=4000):
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
        "rank_cap_vertices":int(max_seam_vertices),
        "rank_skip_reason":None,
    }
    if not ids:
        data["rank_skip_reason"]="no_seam_vertices"
        return data
    if len(ids)>int(max_seam_vertices):
        data["rank_skip_reason"]="above_configured_svd_cap"
        data["rank_cap_vertices"]=int(max_seam_vertices)
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
