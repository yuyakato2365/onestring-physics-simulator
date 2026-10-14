"""Exact (non-cropping) quad coverage of a union of rectangular BFF charts.

A conformal rectangle is NOT necessarily an integer number of uniform square
cells: demanding squares of one pitch and no boundary loss is generally
impossible. Boundary cells are allowed to be smaller rectangles, and the
global x/y grid includes all rectangle side coordinates so paired physical
Split seams have exactly identical edge segmentation. Nothing is cropped.

This is an experimental NON-UNIFORM rectangular quad mesher, not a proof that
the original OneString square-tile fabrication constraints are satisfied.
"""
from __future__ import annotations

import numpy as np


def rectangle_bounds(patch, *, tolerance=1e-6):
    uv=np.asarray(patch["new_uv"],float)
    faces=np.asarray(patch["faces"],int)
    from onestring_physics.reference_bff import validate_reference_mesh
    boundary=validate_reference_mesh(
        np.asarray(patch["xyz"],float),faces
    )["boundary_loops"][0]
    poly=uv[np.asarray(boundary,int)]
    lo=np.min(poly,axis=0)
    hi=np.max(poly,axis=0)
    span=hi-lo
    if not np.all(np.isfinite(poly)) or np.any(span<=1e-10):
        raise ValueError("BFF rectangle has degenerate width/height")
    tol=float(tolerance)*float(np.max(span))
    side_distance=np.min(np.column_stack((
        np.abs(poly[:,0]-lo[0]),np.abs(poly[:,0]-hi[0]),
        np.abs(poly[:,1]-lo[1]),np.abs(poly[:,1]-hi[1])
    )),axis=1)
    if float(np.max(side_distance))>tol:
        raise RuntimeError(
            "The reBFF chart boundary is not an axis-aligned rectangle; "
            "cannot guarantee exact four-sided grid coverage"
        )
    # Require actual four corner vertices; a bbox of some arbitrary shape
    # is NOT by itself proof of a rectangle.
    for x in (lo[0],hi[0]):
        for y in (lo[1],hi[1]):
            if np.min(np.linalg.norm(poly-[x,y],axis=1))>tol:
                raise RuntimeError("BFF rectangle is missing a real corner vertex")
    return lo,hi


def _coordinate_grid(bounds, pitch, axis, *, sliver_fraction=0.12):
    """Global identical coordinates shared by every rectangle/physical seam."""
    bottom=min(b[0][axis] for b in bounds)
    top=max(b[1][axis] for b in bounds)
    if pitch<=0 or not np.isfinite(pitch):
        raise ValueError("Grid pitch must be positive finite")
    # Align global square lattice to origin=0. Rectangular boundaries are
    # mandatory additional lines. Avoid near-zero border strips by deleting
    # nearby regular lines, never mandatory chart-boundary lines.
    left=int(np.floor(bottom/pitch))-1
    right=int(np.ceil(top/pitch))+1
    if right-left>20000:
        raise RuntimeError("Too many grid lines; pathological rectangle aspect")
    mandatory=sorted(float(v) for b in bounds for v in (b[0][axis],b[1][axis]))
    rtol=max(1e-12*pitch,1e-12*(top-bottom))
    unique=[]
    for m in mandatory:
        if not unique or abs(m-unique[-1])>rtol:
            unique.append(m)
    canonical=np.arange(left,right+1,dtype=float)*pitch
    regular=[]
    for v in canonical:
        if v<=bottom+rtol or v>=top-rtol:
            continue
        if min(abs(v-m) for m in unique)<sliver_fraction*pitch:
            continue
        regular.append(float(v))
    merged=sorted([*regular,*unique])
    out=[]
    for v in merged:
        if not out or abs(v-out[-1])>rtol:
            out.append(v)
    return np.asarray(out,dtype=float)


def _within(values,low,high,pitch):
    tol=max(1e-10*pitch,1e-11*(high-low))
    points=values[(values>=low-tol)&(values<=high+tol)]
    points=points.copy()
    # Guarantee exact boundary coordinates, rather than a snapped version.
    if len(points)<2 or abs(points[0]-low)>tol or abs(points[-1]-high)>tol:
        raise RuntimeError("Global grid lost an exact rectangular boundary")
    points[0]=low
    points[-1]=high
    if np.any(np.diff(points)<=tol):
        raise RuntimeError("Collapsed thin rectangular grid cell")
    return points


def grid_at_pitch(patches,pitch,*, max_tiles=15000):
    bounds=[rectangle_bounds(p) for p in patches]
    # Two rectangles must not have overlapping interiors: matching seam edges
    # are okay, but overlapping charts invalidate the 1:1 tiling.
    eps=1e-9*max(1.,max(float(np.max(hi-lo)) for lo,hi in bounds))
    for i,(ai,bi) in enumerate(bounds):
        for j,(aj,bj) in enumerate(bounds):
            if j<=i:
                continue
            penetration=np.maximum(
                0.,np.minimum(bi,bj)-np.maximum(ai,aj)
            )
            # Two guillotine cut rectangles can differ by a few ULPs at
            # their shared edge (the source clipper interpolates float64).
            # An area threshold eps**2 rejects an entire legitimate
            # shared edge when the normal penetration is ~1e-15 but the
            # tangential overlap is O(1). Reject only overlaps thicker
            # than the geometry tolerance on BOTH axes.
            if np.all(penetration>eps):
                raise RuntimeError(
                    f"ReBFF rectangles {i} and {j} overlap in the common UV chart "
                    f"by {penetration.tolist()}; "
                    "no valid seam-preserving grid exists"
                )
    xs=_coordinate_grid(bounds,pitch,0)
    ys=_coordinate_grid(bounds,pitch,1)
    results=[]
    quad_total=0
    for i,((lo,hi),patch) in enumerate(zip(bounds,patches)):
        xx=_within(xs,lo[0],hi[0],pitch)
        yy=_within(ys,lo[1],hi[1],pitch)
        nx,ny=len(xx)-1,len(yy)-1
        quad_total+=nx*ny
        if quad_total>max_tiles:
            raise RuntimeError("Rectangular grid would exceed max_tiles")
        points=np.asarray([[x,y,0.] for y in yy for x in xx],float)
        quad=np.asarray([
            [row*(nx+1)+col,row*(nx+1)+col+1,
             (row+1)*(nx+1)+col+1,(row+1)*(nx+1)+col]
            for row in range(ny) for col in range(nx)
        ],int)
        actual=float(np.sum(np.abs(np.cross(
            points[quad[:,1],:2]-points[quad[:,0],:2],
            points[quad[:,3],:2]-points[quad[:,0],:2]
        ))))
        expected=float(np.prod(hi-lo))
        rel=abs(actual-expected)/max(expected,1e-30)
        if rel>1e-10:
            raise RuntimeError(
                f"Chart {i} lost Omega coverage: area mismatch {rel:.3g}"
            )
        results.append({
            "vertices":points,"faces":quad,"x":xx,"y":yy,
            "rectangle_bounds":(lo,hi),
            "quad_count":len(quad),
            "area_fraction_covered":float(actual/expected),
            "min_cell_width":float(np.min(np.diff(xx))),
            "min_cell_height":float(np.min(np.diff(yy))),
        })
    return results


def choose_exact_grid(patches, target_tiles, *, grid=14, max_tiles=15000):
    if not patches:
        raise ValueError("No BFF rectangular patches")
    boxes=[rectangle_bounds(p) for p in patches]
    area=sum(float(np.prod(hi-lo)) for lo,hi in boxes)
    guess=np.sqrt(area/max(int(target_tiles),1))
    if not np.isfinite(guess) or guess<=0:
        raise ValueError("Degenerate total BFF rectangle area")
    best=None
    grid_failures=[]
    for factor in np.geomspace(.6,1.7,19):
        pitch=guess*factor
        try:
            meshes=grid_at_pitch(patches,pitch,max_tiles=max_tiles)
        except RuntimeError as exc:
            grid_failures.append(f"pitch={pitch:.6g}: {exc}")
            continue
        count=sum(len(x["faces"]) for x in meshes)
        mismatch=abs(count-int(target_tiles))
        # Keep the smallest number of very narrow boundary rectangles on
        # a tie; never discard any cell to hit the target count.
        narrow=min(min(m["min_cell_width"],m["min_cell_height"])/pitch
                   for m in meshes)
        key=(mismatch,-narrow,abs(np.log(factor)))
        if best is None or key<best[0]:
            best=(key,float(pitch),meshes)
    if best is None:
        raise RuntimeError(
            "No valid non-cropping rectangle grid at bounded density. "
            "First failures: "+ "; ".join(grid_failures[:4])
        )
    _,pitch,meshes=best
    return pitch,meshes


def matching_seam_edges(meshes,*, tolerance=1e-8):
    """Check both pieces have IDENTICAL UV edge segmentation along every seam."""
    edges=[]
    for p,m in enumerate(meshes):
        verts=np.asarray(m["vertices"],float)[:,:2]
        seg={}
        for f in m["faces"]:
            for a,b in zip(f,np.roll(f,-1)):
                pair=tuple(sorted((int(a),int(b))))
                seg[pair]=seg.get(pair,0)+1
        for (a,b),cnt in seg.items():
            if cnt!=1:
                continue
            va,vb=verts[[a,b]]
            key=tuple(sorted((
                tuple(np.rint(va/tolerance).astype(np.int64)),
                tuple(np.rint(vb/tolerance).astype(np.int64))
            )))
            edges.append((key,p))
    groups={}
    for key,p in edges:
        groups.setdefault(key,set()).add(p)
    pairs={}
    for owners in groups.values():
        if len(owners)!=2:
            continue
        i,j=sorted(owners)
        pairs[(i,j)]=pairs.get((i,j),0)+1
    return pairs
