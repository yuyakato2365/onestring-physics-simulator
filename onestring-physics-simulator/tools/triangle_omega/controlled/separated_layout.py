"""Find a nonoverlapping rigid layout and measure the flexure spans it needs."""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree


def separated_layout(rest, joints, cells, up, opening, steps=60, compact_from=None, progress=None, compact_mean_weight=8.):
    edges=np.linalg.norm(rest[:,0]-rest[:,1],axis=1)
    pitch=edges.mean()
    P=rest[:,:,:2]/pitch
    P=P-P[:,:3].mean(1)[:,None]
    centers=cells.mean(1)
    guide=(cells-centers[:,None])
    centers=centers-centers.mean(0)
    a,ka,b,kb=joints.T
    n=len(rest)
    def rotation(angles):
        co,si=np.cos(angles),np.sin(angles)
        return np.stack([co,si,-si,co],axis=1).reshape(-1,2,2)
    desired=guide@rotation(np.where(up,-opening,opening))
    theta=np.arctan2((P[:,:3,0]*desired[:,:,1]-P[:,:3,1]*desired[:,:,0]).sum(1),(P[:,:3]*desired).sum((1,2)))
    cen=centers*2.4
    tol=.001
    from scipy.spatial import ConvexHull
    edges=[]
    for p in P:
        hull=ConvexHull(p).vertices
        e=p[np.roll(hull,-1)]-p[hull]
        vec=e[:,[1,0]]*[-1,1]
        edges.append(vec/np.linalg.norm(vec,axis=1)[:,None])
    axes=np.array([np.pad(e,((0,6-len(e)),(0,0)),mode='edge') for e in edges])
    radius=np.linalg.norm(P,axis=2).max()
    def state(c,t):
        rot=rotation(t)
        pts=P@rot+c[:,None]
        normals=axes@rot
        ij=cKDTree(c).query_pairs(2*radius+1.,output_type='ndarray')
        i,j=ij.T
        ns=np.concatenate((normals[i],normals[j]),axis=1)
        u=np.einsum('nkd,nvd->nkv',ns,pts[i])
        v=np.einsum('nkd,nvd->nkv',ns,pts[j])
        gaps=np.concatenate((v.min(2)-u.max(2),u.min(2)-v.max(2)),axis=1)
        best=gaps.argmax(1); row=np.arange(len(i))
        normal=ns[row,best%12]*np.where(best<12,1,-1)[:,None]
        return pts,ij,normal,gaps[row,best]
    mean_weight=.3
    span_limit=None
    if compact_from is not None:
        target=compact_from[:,:3,:2]/pitch
        cen=target.mean(1)
        vec=target-cen[:,None]
        mat=P[:,:3]
        theta=np.arctan2((mat[:,:,0]*vec[:,:,1]-mat[:,:,1]*vec[:,:,0]).sum((1,)),(mat*vec).sum((1,2)))
        span_limit=np.linalg.norm(compact_from[a,ka,:2]-compact_from[b,kb,:2],axis=1).max()/pitch+1e-8
        mean_weight=compact_mean_weight
    while True:
        pts,ij,normal,gap=state(cen,theta)
        if gap.min(initial=1)>(0. if compact_from is not None else tol):break
        if compact_from is not None:return compact_from
        cen*=1.1
    unit=rotation(np.arange(12)*np.pi/6)[:,0]
    def score(pts):
        dist=np.linalg.norm(pts[a,ka]-pts[b,kb],axis=1)
        return dist.max()+mean_weight*dist.mean()
    trust=.25
    for it in range(steps):
        if progress is not None and it%10==0:
            progress(it+1,steps)
        pts,ij,normal,gap=state(cen,theta); current=score(pts)
        active=gap<4*trust+tol;ij,normal,gap=ij[active],normal[active],gap[active]
        i,j=ij.T;m=len(i);nh=len(a)
        rhs=[];rows=[];cols=[];values=[]
        def block(bound,terms):
            start=sum(map(len,rhs));idx=np.arange(bound.size).reshape(bound.shape)+start
            rhs.append(bound.ravel())
            for c,v in terms:
                rows.append(idx.ravel());cols.append(np.broadcast_to(c,idx.shape).ravel());values.append(np.broadcast_to(v,idx.shape).ravel())
        rp=pts-cen[:,None];perp=rp[:,:,[1,0]]*[-1,1]
        mid=.5*((pts[i]*normal[:,None]).sum(2).max(1)+(pts[j]*normal[:,None]).sum(2).min(1))
        for ids,sign in [(i,1),(j,-1)]:
            nn=normal*sign;bound=sign*mid[:,None]-(pts[ids]*nn[:,None]).sum(2)-tol/2
            block(bound,[(3*ids[:,None],nn[:,0,None]),(3*ids[:,None]+1,nn[:,1,None]),(3*ids[:,None]+2,(perp[ids]*nn[:,None]).sum(2)),(3*n+np.arange(m)[:,None],-sign)])
        h=3*n+m+np.arange(nh)[:,None];last=3*n+m+nh
        block(-(pts[a,ka]-pts[b,kb])@unit.T,[(3*a[:,None],unit[:,0]),(3*a[:,None]+1,unit[:,1]),(3*a[:,None]+2,perp[a,ka]@unit.T),(3*b[:,None],-unit[:,0]),(3*b[:,None]+1,-unit[:,1]),(3*b[:,None]+2,-perp[b,kb]@unit.T),(h,-1)])
        block(np.zeros((nh,1)),[(h,1),(last,-1)])
        A=coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(sum(map(len,rhs)),last+1)).tocsr()
        result=linprog(np.r_[np.zeros(3*n+m),np.full(nh,mean_weight/nh),1],A_ub=A,b_ub=np.concatenate(rhs),bounds=[(-trust,trust)]*(3*n)+[(None,None)]*m+[(0,None)]*nh+[(0,span_limit)],method='highs')
        if not result.success:break
        dq=result.x[:3*n].reshape(n,3)
        for alpha in (1.,.5,.25,.125,.0625):
            C=cen+alpha*dq[:,:2];T=theta+alpha*dq[:,2];trial,_,_,g=state(C,T)
            if (g.min(initial=1)>0 and score(trial)<current-1e-6 and
                (span_limit is None or np.linalg.norm(trial[a,ka]-trial[b,kb],axis=1).max()<=span_limit)):
                cen,theta=C,T;break
        else:
            trust*=.5
            if trust<.004:break
    solid=rest.copy();solid[:,:,:2]=(P@rotation(theta)+cen[:,None])*pitch
    return solid


def validate_separated(rest, solid, tets, joints):
    """Independently check rigid tiles, all projected outlines, and all thick panels."""
    from shapely.geometry import MultiPoint, LineString
    from shapely.strtree import STRtree
    from contact import solid_contacts, hinge_clearance_bounds

    upper = np.triu_indices(6,1)
    def lengths(points):
        return np.linalg.norm(points[:,upper[0]]-points[:,upper[1]],axis=2)
    if not np.allclose(lengths(rest),lengths(solid),atol=1e-10,rtol=1e-10):
        raise ValueError('T2Dでパネル形状が変化しています。')
    outlines=[MultiPoint(p[:,:2]).convex_hull for p in solid]
    tree=STRtree(outlines)
    i,j=tree.query(outlines,predicate='intersects')
    projected=int(np.count_nonzero(i<j))
    _,_,contacts,depth=solid_contacts(solid,solid[:,:3,:2].mean(1),tets)
    if contacts or projected:
        raise ValueError('干渉回避後のT2Dにパネルの交差が残っています。')
    a,k,b,l=joints.T
    spans=np.linalg.norm(solid[a,k,:2]-solid[b,l,:2],axis=1)
    segments=[LineString([left,right]) for left,right in zip(solid[a,k,:2],solid[b,l,:2])]
    u,v=tree.query(segments,predicate='intersects')
    foreign=(v!=a[u])&(v!=b[u])
    u,v=STRtree(segments).query(segments,predicate='crosses')
    tolerance=float(np.mean(np.linalg.norm(rest[:,0]-rest[:,1],axis=1))*.001)
    bounds=hinge_clearance_bounds(rest,tets,joints) if tets is not None else np.zeros(len(joints))
    return dict(layout_seed='separated',hinges=len(joints),max_gap=float(spans.max()),median_gap=float(np.median(spans)),
                overlap_pairs=int(contacts),footprint_overlap_pairs=projected,max_penetration=float(depth),
                tolerance=tolerance,converged=bool(spans.max()<=tolerance),
                nonincident_panel_link_hits=int(foreign.sum()),link_centerline_crossings=int((u<v).sum()),
                unavoidable_hinge_pairs=joints[bounds>tolerance].tolist(),separation_lower_bound=float(bounds.max()))
