"""Exact geometric Seam constraints for non-conforming OptCuts Grid edges.

Each linear equation is Σ weight_i * v_i = 0 in 3D. Unlike vertex welding,
this can join a vertex to the INTERIOR of the other side's boundary edge
without splitting a four-corner K3D/M2D panel. It also handles Grid edges
whose initial fraction is a cut and final fraction is a genuine source-open
boundary.

Affine constraints are generated ONLY by pairing the two topological UV
copies of the *same physical source-mesh edge*, never by nearest-point
search. Every original source-cut edge must be completely represented by
both M2D boundaries; missing, self-intersecting or unclassified coverage
is an error, not silently accepted.

In the AL and Paper T3D stages, use the orthogonal projector onto ker(C)
for both positions and analytic gradients. Its linear feasible set is
independent of SmoothMax weights, so a soft square penalty cannot open
these paired boundary polylines.
"""
from collections import defaultdict

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


class SourceSeamAffineError(RuntimeError):
    pass


def _seam_overlap(source_uv_edge, grid_uv_edge, tol):
    """Return physical parameter interval and Grid edge parameter values."""
    src=np.asarray(source_uv_edge,float)
    target=np.asarray(grid_uv_edge,float)
    d=src[1]-src[0]
    e=target[1]-target[0]
    dd=float(d@d)
    ee=float(e@e)
    if dd<=1e-24 or ee<=1e-24:
        raise SourceSeamAffineError("SEAM_AFFINE_DEGENERATE_SOURCE_OR_GRID_EDGE")
    def perp(p):
        z=p-src[0]
        return abs(float(d[0]*z[1]-d[1]*z[0]))/dd**.5
    if max(perp(target[0]),perp(target[1]))>tol:
        return None
    proj=(target-src[0])@d/dd
    lower=max(0.,float(np.min(proj)))
    upper=min(1.,float(np.max(proj)))
    if upper-lower<=tol/dd**.5:
        return None
    # For a physical t in [lower,upper], the Grid edge interpolant is
    # v_a*(1-s)+v_b*s where s depends linearly on t.
    def weight(t):
        q=src[0]+t*d
        return float(np.clip((q-target[0])@e/ee,0.,1.))
    return (lower,upper,weight(lower),weight(upper))


def _find_grid_piece(pieces,t,tol):
    selected=[piece for piece in pieces
              if piece[0]-tol<=t<=piece[1]+tol]
    if not selected:
        raise SourceSeamAffineError(
            "SEAM_AFFINE_PHYSICAL_CUT_NOT_COVERED: "
            f"source_edge_t={t:.9g}"
        )
    # At a Grid corner, both incident pieces may qualify. The shared
    # endpoint will have precisely the same panel vertex reference.
    selected.sort(key=lambda p:(p[1]-p[0],p[2],p[3]))
    return selected[0]


def _full_coverage(pieces,tol):
    ordered=sorted((a,b) for a,b,*_ in pieces)
    cursor=0.
    for a,b in ordered:
        if a>cursor+tol:
            return False
        cursor=max(cursor,b)
    return cursor>=1.-tol


def _canonical_row(row):
    coefficients=defaultdict(float)
    for idx,weight in row:
        coefficients[int(idx)]+=float(weight)
    result=[(i,w) for i,w in sorted(coefficients.items())
            if abs(w)>1e-12]
    if not result:
        return None
    size=max(abs(w) for _,w in result)
    sign=1. if result[0][1]>=0 else -1.
    return [(i,sign*w/size) for i,w in result]


def source_cut_affine_rows(uv_vertices,quad_faces,source_cut_segments,
                           *,absolute_tolerance=None):
    """Pair non-identically sampled M2D boundaries by physical 3D edge ID.

    The source_cut_segments records are (UV 2x2, XYZ 2x3, physical_edge,
    side), where *both* sides use the same original physical-edge
    orientation. No original seam or target Grid vertices are modified.

    Returns (rows,diagnostics) for C * K3D_xyz = 0.
    """
    from .optcuts_seam_exact_closure import _uv_boundary_edges
    uv=np.asarray(uv_vertices,float)
    edges=_uv_boundary_edges(quad_faces)
    uvscale=max(float(np.ptp(uv,axis=0).max()),1.)
    tol=float(absolute_tolerance or max(uvscale*1e-8,1e-10))
    source=defaultdict(dict)
    for uv_edge,_xyz,key,side in source_cut_segments:
        if side in source[key]:
            raise SourceSeamAffineError(
                "SEAM_AFFINE_DUPLICATE_PHYSICAL_SOURCE_EDGE_SIDE: "
                f"physical={key} side={side}"
            )
        source[key][side]=np.asarray(uv_edge,float)
    rows=[]
    seen=set()
    mixed_source_endpoint_count=0
    nonmatching_count=0
    cut_segments_by_side=0
    for physical,side_edges in source.items():
        if set(side_edges)!={0,1}:
            raise SourceSeamAffineError(
                "SEAM_AFFINE_UNPAIRED_OR_AMBIGUOUS_SOURCE_CUT: "
                f"physical={physical} sides={list(side_edges)}"
            )
        per_side={}
        for side,src in sorted(side_edges.items()):
            pieces=[]
            for a,b in edges:
                entry=_seam_overlap(src,uv[[a,b]],tol)
                if entry is not None:
                    lo,hi,slo,shi=entry
                    pieces.append((lo,hi,int(a),int(b),slo,shi))
            if not _full_coverage(pieces,1e-6):
                raise SourceSeamAffineError(
                    "SEAM_AFFINE_SOURCE_CUT_NOT_FULLY_IN_M2D_GRID: "
                    f"physical={physical} side={side} "
                    f"covered_intervals={[(round(p[0],6),round(p[1],6)) for p in pieces[:12]]}"
                )
            per_side[side]=pieces
            cut_segments_by_side+=len(pieces)
        left=per_side[0]
        right=per_side[1]
        # Enforce coincidence at every breakpoint in the UNION of both
        # piecewise-linear boundary partitions. Then the entire 3D seam
        # segments coincide, not just a few recognized vertices.
        breaks=sorted({0.,1.,*(
            float(x) for part in (left,right) for piece in part
            for x in piece[:2]
        )})
        dedup=[]
        for t in breaks:
            if not dedup or t-dedup[-1]>1e-9:
                dedup.append(t)
        if len(left)!=len(right) or any(
            abs(x-y)>1e-8 for x,y in zip(
                [p[0] for p in sorted(left)],
                [p[0] for p in sorted(right)]
            )
        ):
            nonmatching_count+=1
        for t in dedup:
            row=[]
            for side,sign in ((0,1.),(1,-1.)):
                piece=_find_grid_piece(per_side[side],t,1e-6)
                lo,hi,a,b,slo,shi=piece
                s=float(np.clip(slo+(shi-slo)*(t-lo)/max(hi-lo,1e-14),0.,1.))
                row.extend(((a,sign*(1.-s)),(b,sign*s)))
            canonical=_canonical_row(row)
            if canonical is None:
                continue
            key=tuple((i,round(w,12)) for i,w in canonical)
            if key not in seen:
                seen.add(key)
                rows.append(canonical)
        # Source edge endpoints that occur mid-Grid-edge receive a true
        # affine constraint, never a fabricated M2D tile vertex.
        for side in (0,1):
            for t in (0.,1.):
                part=_find_grid_piece(per_side[side],t,1e-6)
                if min(abs(part[4]),abs(part[5]),
                       abs(1-part[4]),abs(1-part[5]))>1e-8:
                    mixed_source_endpoint_count+=1
    return rows,{
        "source_cut_physical_edge_count":len(source),
        "source_cut_grid_piece_count":cut_segments_by_side,
        "source_cut_different_grid_partition_count":nonmatching_count,
        "source_cut_constraint_row_count":len(rows),
        "source_cut_interior_grid_endpoint_count":mixed_source_endpoint_count,
        "source_cut_exact_polyline_correspondence":True,
    }


class AffineSeamProjector:
    """Orthogonal equality projector preserving piecewise-linear 3D seams.

    The K3D minimizer differentiates f(P x); its exact Jacobian is P
    because P is a constant orthogonal projection. The T3D local/global
    step applies P after each update. Redundant equations are allowed,
    but infeasible/ill-conditioned solves explicitly fail, not silently
    relax a user's hard seam condition.
    """
    def __init__(self,n_vertices,rows):
        self.n=int(n_vertices)
        canon=[_canonical_row(r) for r in rows]
        canon=[r for r in canon if r is not None]
        rr=[];cc=[];dd=[]
        for i,row in enumerate(canon):
            for j,w in row:
                if j<0 or j>=self.n:
                    raise SourceSeamAffineError(
                        "SEAM_AFFINE_VERTEX_OUT_OF_BOUNDS"
                    )
                rr.append(i);cc.append(j);dd.append(w)
        self.c=sparse.coo_matrix(
            (dd,(rr,cc)),shape=(len(canon),self.n)
        ).tocsr()
        self.count=len(canon)
        self._factor=None
        if self.count:
            gram=(self.c@self.c.T).tocsc()
            diag=np.asarray(gram.diagonal(),float)
            scale=max(float(np.max(diag)),1e-12)
            # Tiny ridge handles redundant endpoint equations. All
            # accepted coordinates are independently audited afterwards.
            try:
                self._factor=splu(
                    gram+sparse.eye(self.count,format="csc")*(scale*1e-13)
                )
            except RuntimeError as exc:
                raise SourceSeamAffineError(
                    "SEAM_AFFINE_CONSTRAINT_GRAM_SINGULAR"
                ) from exc

    def residual(self,coords):
        if not self.count:return 0.
        xyz=np.asarray(coords,float).reshape(self.n,3)
        return float(np.max(np.abs(self.c@xyz)))

    def project(self,coords,*,strict=False):
        xyz=np.asarray(coords,float).reshape(self.n,3)
        if not self.count:return xyz.copy()
        out=xyz.copy()
        for _ in range(3):
            rhs=self.c@out
            multiplier=self._factor.solve(rhs)
            out-=self.c.T@multiplier
        error=self.residual(out)
        scale=max(1.,float(np.max(np.abs(xyz))))
        if strict and error>scale*1e-9:
            raise SourceSeamAffineError(
                "SEAM_AFFINE_HARD_CONSTRAINT_PROJECTION_FAILED: "
                f"residual={error:.9g} tolerance={scale*1e-9:.9g}"
            )
        if not np.isfinite(out).all():
            raise SourceSeamAffineError(
                "SEAM_AFFINE_NONFINITE_PROJECTION"
            )
        return out
