"""SAT energy extracted unchanged from the local quad implementation."""
import numpy as np
def collision_energy_gradient(xy, faces, tolerance=1e-10, exclude_face_pairs=None, return_diagnostics=False):
    """Squared convex SAT translation depth; includes derivative of each axis.

    `exclude_face_pairs` is an explicit caller override. A physical hinge alone
    is not a reason to exclude a pair: it allows contact, not penetration.
    This surrogate is not claimed to equal Konakovic et al.'s local half-plane
    projection. All overlapping AABBs are tested, with no candidate cap. Contact
    at a shared corner/edge has zero energy. Convexity is checked by the caller.
    """
    def finish(energy, gradient, depths, candidate_count):
        penetrations = np.asarray(depths)[np.asarray(depths) > tolerance]
        count = len(penetrations)
        result = (energy, gradient, count)
        if return_diagnostics:
            return (*result, dict(collision_candidate_pairs=int(candidate_count), collision_pair_count=count,
                                 penetration_depths=penetrations.tolist(),
                                 penetration_max=float(penetrations.max()) if count else 0.,
                                 penetration_rms=float(np.sqrt(np.mean(penetrations**2))) if count else 0.,
                                 penetration_model='convex XY footprint separating translation, not 3D solid depth'))
        return result
    polygons = xy[faces]
    lo, hi = polygons.min(axis=1), polygons.max(axis=1)
    # Sweep broad phase: memory scales with nearby pairs, not all F^2 pairs.
    order = np.argsort(lo[:,0], kind='stable')
    pair_a, pair_b = [], []
    for pos, i in enumerate(order):
        end = np.searchsorted(lo[order,0], hi[i,0], side='left')
        candidates = order[pos+1:end]
        keep = (hi[candidates,1] > lo[i,1]) & (lo[candidates,1] < hi[i,1])
        candidates = candidates[keep]
        pair_a.extend([i]*len(candidates)); pair_b.extend(candidates)
    ii, jj = np.asarray(pair_a,int), np.asarray(pair_b,int)
    if exclude_face_pairs and len(ii):
        excluded={tuple(sorted((int(a),int(b)))) for a,b in exclude_face_pairs}
        keep=np.fromiter((tuple(sorted((int(a),int(b)))) not in excluded for a,b in zip(ii,jj)),dtype=bool,count=len(ii))
        ii,jj=ii[keep],jj[keep]
    grad = np.zeros_like(xy)
    if len(ii) == 0:
        return finish(0., grad, [], 0)
    a, b = polygons[ii], polygons[jj]
    edges = np.concatenate((np.roll(a,-1,axis=1)-a,np.roll(b,-1,axis=1)-b),axis=1)
    lengths = np.linalg.norm(edges,axis=2)
    normals = np.stack((-edges[...,1],edges[...,0]),axis=-1)/np.maximum(lengths[...,None],1e-30)
    ap = np.einsum('pvd,pad->pav',a,normals)
    bp = np.einsum('pvd,pad->pav',b,normals)
    d1, d2 = ap.max(axis=2)-bp.min(axis=2), bp.max(axis=2)-ap.min(axis=2)
    depths = np.minimum(d1,d2)
    depths[lengths <= 1e-30] = np.inf  # repeated hull padding is not an axis
    axis = np.argmin(depths,axis=1)
    depth = depths[np.arange(len(ii)),axis]
    live = depth > 0.
    rows = np.flatnonzero(live)
    if len(rows) == 0:
        return finish(0., grad, [], len(ii))
    ax, dp = axis[rows], depth[rows]
    n = normals[rows,ax]
    forward = d1[rows,ax] <= d2[rows,ax]
    ah = np.where(forward,ap[rows,ax].argmax(axis=1),ap[rows,ax].argmin(axis=1))
    bh = np.where(forward,bp[rows,ax].argmin(axis=1),bp[rows,ax].argmax(axis=1))
    sign = np.where(forward,1.,-1.)
    aid, bid = faces[ii[rows],ah], faces[jj[rows],bh]
    delta = sign[:,None]*(xy[aid]-xy[bid])
    weight = 2*dp
    np.add.at(grad,aid,(weight*sign)[:,None]*n)
    np.add.at(grad,bid,-(weight*sign)[:,None]*n)
    # n=J e/|e|. Chain rule for the selected support axis.
    dn = (delta-n*np.sum(delta*n,axis=1)[:,None])/lengths[rows,ax,None]
    de = np.stack((dn[:,1],-dn[:,0]),axis=1)*weight[:,None]
    m = faces.shape[1]
    owner = np.where(ax<m,ii[rows],jj[rows]); local = ax%m
    np.add.at(grad,faces[owner,local],-de)
    np.add.at(grad,faces[owner,(local+1)%m],de)
    return finish(float(dp@dp), grad, dp, len(ii))


def solid_contacts(solid, origins, tets=None, return_pairs=False, power=2):
    """Exact convex-prism SAT at a shared top Z. Depth is horizontal separation.

    Each candidate axis is a side normal or the cross product of two edges.
    Pure Z cannot separate solids resting on the same plane. Each axis is
    normalised by its XY length so the penalty is a distance on the bed.
    """
    n=len(solid); grad=np.zeros((n,3))
    def finish(result, ia=(), ib=(), depths=()):
        if return_pairs:
            return (*result, np.column_stack((ia,ib)).astype(int), np.asarray(depths))
        return result
    owner=np.arange(n)
    if tets is not None:
        solid=solid[owner[:,None,None],tets].reshape(-1,4,3)
        owner=np.repeat(owner,tets.shape[1])
    lo=solid.min(1); hi=solid.max(1); order=lo[:,0].argsort()
    pair_a=[];pair_b=[]
    stops=np.searchsorted(lo[order,0],hi[order,0])
    for ix,(i,stop) in enumerate(zip(order,stops)):
        other=order[ix+1:stop]
        other=other[(owner[other]!=owner[i]) & (lo[other,1]<hi[i,1]) & (hi[other,1]>lo[i,1]) & (lo[other,2]<hi[i,2]) & (hi[other,2]>lo[i,2])]
        pair_a.extend([i]*len(other));pair_b.extend(other)
    if not pair_a:return finish((0.,grad,0,0.))
    ia=np.asarray(pair_a);ib=np.asarray(pair_b)
    if tets is None:
        edges=np.roll(solid[:,:3],-1,axis=1)-solid[:,:3]
        legs=solid[:,3:]-solid[:,:3]
        sides=np.cross(edges,legs)
        lines=np.concatenate((edges,legs),axis=1)
    else:
        p,q=np.triu_indices(4,k=1)
        lines=solid[:,p]-solid[:,q]
        faces=solid[:,[[0,1,2],[0,1,3],[0,2,3],[1,2,3]]]
        sides=np.cross(faces[:,:,1]-faces[:,:,0],faces[:,:,2]-faces[:,:,0])
    # Unit edge vectors keep near-parallel axis rejection independent of model scale.
    lines/=np.linalg.norm(lines,axis=2)[:,:,None]
    sides/=np.linalg.norm(sides,axis=2)[:,:,None]
    def spin(v):
        return np.stack([-v[...,1],v[...,0],np.zeros(v.shape[:-1])],axis=-1)
    zspin=spin(lines)
    crosses=np.cross(lines[ia,:,None],lines[ib,None,:]).reshape(-1,36,3)
    axes=np.concatenate((sides[ia],sides[ib],crosses),axis=1)
    norms=np.linalg.norm(axes[:,:,:2],axis=2)
    vertical=(norms<1e-9)&(np.abs(axes[:,:,2])>1e-9)
    axes/=np.maximum(norms,1e-12)[:,:,None]
    left=np.einsum('pvd,pad->pav',solid[ia],axes)
    right=np.einsum('pvd,pad->pav',solid[ib],axes)
    d1=left.max(2)-right.min(2);d2=right.max(2)-left.min(2)
    depths=np.minimum(d1,d2);depths[norms<1e-9]=np.inf
    z_gap=np.minimum(hi[ia,2]-lo[ib,2],hi[ib,2]-lo[ia,2])
    depths[vertical & (z_gap[:,None]<=1e-10)] = 0.
    axis=depths.argmin(1); row=np.arange(len(axis))
    pen=depths[row,axis];keep=pen>1e-10
    if not keep.any():return finish((0.,grad,0,0.))
    row=np.flatnonzero(keep)
    if tets is not None:
        # All three tetrahedra move rigidly together. Select the deepest
        # intersecting tetrahedron pair per panel pair for its gradient.
        a0,b0=owner[ia[row]],owner[ib[row]]
        key=np.minimum(a0,b0)*n+np.maximum(a0,b0)
        order=np.lexsort((-pen[row],key))
        first=np.r_[True,np.diff(key[order])!=0]
        row=row[order[first]]
    axis=axis[row];pen=pen[row];a=ia[row];b=ib[row]
    normal=axes[row,axis]
    forward=d1[row,axis]<=d2[row,axis]
    pa=solid[a,np.where(forward,left[row,axis].argmax(1),left[row,axis].argmin(1))]
    pb=solid[b,np.where(forward,right[row,axis].argmin(1),right[row,axis].argmax(1))]
    def dot(x,y):return np.einsum('ij,ij->i',x,y)
    # Differentiate both support points and the active (rotating) axis.
    wa=dot(spin(pa[:,:2]-origins[owner[a]]),normal)
    wb=-dot(spin(pb[:,:2]-origins[owner[b]]),normal)
    da=np.zeros_like(normal);db=np.zeros_like(normal)
    ns=sides.shape[1]
    ma=axis<ns;mb=(axis>=ns)&(axis<2*ns)
    da[ma]=spin(normal[ma]);db[mb]=spin(normal[mb])
    cross=axis>=2*ns
    if cross.any():
        i,j=np.divmod(axis[cross]-2*ns,6)
        ea,eb=lines[a[cross],i],lines[b[cross],j]
        for dest,raw in [(da,np.cross(zspin[a[cross],i],eb)),(db,np.cross(ea,zspin[b[cross],j]))]:
            nn=normal[cross]
            dest[cross]=(raw-nn*dot(nn[:,:2],raw[:,:2])[:,None])/norms[row[cross],axis[cross],None]
    wa+=dot(pa-pb,da);wb+=dot(pa-pb,db)
    w=power*pen**(power-1)*np.where(forward,1.,-1.)
    np.add.at(grad,owner[a],w[:,None]*np.c_[wa,normal[:,:2]])
    np.add.at(grad,owner[b],w[:,None]*np.c_[wb,-normal[:,:2]])
    return finish((float(np.sum(pen**power)),grad,len(pen),float(pen.max())), owner[a],owner[b],pen)

def hinge_clearance_bounds(solids, tets, joints):
    """A shared-depth disk about each hinge pivot proves interference at every rotation."""
    from shapely.geometry import MultiPoint, Point
    from shapely.ops import unary_union
    low, high = np.triu_indices(4, k=1)
    bounds = np.zeros(len(joints))
    def radius(panel, corner, height):
        sections = []
        for tet in solids[panel,tets[panel]]:
            start, end = tet[low], tet[high]
            cut = (start[:,2]>height) != (end[:,2]>height)
            start, end = start[cut], end[cut]
            if len(start)<3:
                continue
            t = (height-start[:,2])/(end[:,2]-start[:,2])
            xy = start[:,:2]+t[:,None]*(end[:,:2]-start[:,:2])
            sections.append(MultiPoint(xy).convex_hull)
        section = unary_union(sections)
        pivot = Point(solids[panel,corner,:2])
        return section.boundary.distance(pivot) if section.contains(pivot) else 0.
    for i,(a,ka,b,kb) in enumerate(joints):
        roof = min(solids[a,:3,2].min(),solids[b,:3,2].min())
        floor = max(solids[a,3:,2].max(),solids[b,3:,2].max())
        if floor >= roof:
            continue
        z = (roof+floor)/2
        ra, rb = radius(a,ka,z),radius(b,kb,z)
        if min(ra,rb)>1e-10*(roof-floor):
            bounds[i] = ra+rb
    return bounds
