"""Open rim drawstring. Optimise straight bores against both endpoint poses."""
import json
import pickle
import numpy as np
import manifold3d as md
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-10)


def bellmouth(a, b, radius):
    axis = unit(b-a)
    length = np.linalg.norm(b-a)
    shoulder = min(1.2, length*.4)
    # Rounded flares and prolonged mouths open across oblique side walls.
    z = np.r_[-3*radius, np.linspace(0, shoulder, 6), length-np.linspace(0, shoulder, 6)[::-1], length+3*radius]
    r = (radius + .55*np.maximum(0,1-np.maximum(0,np.minimum(z,length-z))/shoulder)**2)/np.cos(np.pi/40)
    profile = np.vstack([[0,z[0]], np.c_[r,z], [0,z[-1]]])
    solid = md.CrossSection([profile]).revolve(40)
    x=unit(np.cross(axis,np.eye(3)[np.argmin(abs(axis))]))
    return solid.transform(np.column_stack([x,np.cross(axis,x),axis,a]))


def perimeter_channels(out, folded, flat, radius):
    with (out/'remesh3d.pkl').open('rb') as stream:
        faces=pickle.load(stream)['T']
    edges, counts = np.unique(np.sort(faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0,return_counts=True)
    rim=set(edges[counts==1].ravel())
    neighbors={}
    crossings={}
    for i,tri in enumerate(faces):
        for j in range(3):
            a,b=tri[j],tri[(j+1)%3]
            if (a in rim)==(b in rim): continue
            edge=(int(a),int(b)) if a in rim else (int(b),int(a))
            neighbors.setdefault(edge,[]).append(i)
            crossings.setdefault(i,[]).append(edge)
    mid=(folded[:,:3]+folded[:,7:10])/2
    rim_faces=[i for i in crossings if sum(v in rim for v in faces[i])==2]
    if not rim_faces: raise ValueError('外周ストリングに使える開いた外周がありません。')
    def rim_span(i):
        a,b=mid[i,[v in rim for v in faces[i]]]
        return np.linalg.norm(a-b)
    start=max(rim_faces,key=rim_span)
    ring=[]; cross=[]; at=start; entry=crossings[start][0]
    while at not in ring:
        ring.append(at); cross.append(entry)
        other,=[x for x in crossings[at] if x!=entry]
        at,=[i for i in neighbors[other] if i!=at]
        entry=other
    if at!=start or entry!=cross[0] or len(ring)!=len(crossings):
        raise ValueError('単一の外周経路を確定できません。')
    # Split the widest rim tile into two leads; both ends exit the physical rim.
    owners=np.array(ring+[start])
    n=len(ring); ids=np.arange(n)
    first,last=cross[1][0],cross[0][0]
    rimedge=(first,last)
    ea=[rimedge]+cross[1:]+[cross[0]]; eb=cross[1:]+[cross[0],rimedge]
    ia=np.r_[n,ids[1:],0]; ib=np.r_[ids[1:],0,n+1]
    def geometry(layer):
        center=(layer[:,:3]+layer[:,7:10])/2
        def ends(es):
            return np.array([[center[i,list(faces[i]).index(v)] for v in edge] for i,edge in zip(owners,es)])
        return ends(ea),ends(eb)
    def points(geom,t):
        return [e[:,0]+t[ix,None]*(e[:,1]-e[:,0]) for e,ix in zip(geom,(ia,ib))]
    G,H=geometry(folded),geometry(flat)
    def residual(t):
        a,b=points(G,t); u=unit(b-a)
        x,y=points(H,t); v=unit(y-x); gap=x[1:]-y[:-1]
        w=unit(gap); importance=np.minimum(np.linalg.norm(gap,axis=1)/1.5,1)[:,None]
        return np.r_[(u[1:]-u[:-1]).ravel(),((v[:-1]-w)*importance).ravel(),
                     ((v[1:]-w)*importance).ravel(),.32*(t-.38)]
    sp=lil_matrix((9*n+n+2,n+2),dtype=int)
    for i in range(n):
        for k in (0,1,2):sp[k*3*n+3*i:k*3*n+3*i+3,[ia[i],ib[i],ia[i+1],ib[i+1]]]=1
    sp[9*n:,:]=np.eye(n+2)
    t=least_squares(residual,np.r_[np.full(n,.38),.22,.78],jac_sparsity=sp.tocsr(),
                    bounds=(np.r_[np.full(n,.22),.12,.67],np.r_[np.full(n,.65),.33,.88]),max_nfev=100).x
    a,b=points(H,t); af,bf=points(G,t)
    channels=[md.Manifold() for _ in faces]
    for i,x,y in zip(owners,a,b):
        channels[i]=channels[i]+bellmouth(x,y,radius)
    angle=lambda x,y:np.degrees(np.arccos(np.clip((unit(x)*unit(y)).sum(1),-1,1)))
    gap=a[1:]-b[:-1];mask=np.linalg.norm(gap,axis=1)>.15
    flat_turn=np.r_[angle((b-a)[:-1][mask],gap[mask]),angle(gap[mask],(b-a)[1:][mask])]
    report=dict(panel_sequence=owners.tolist(),flat_entry_mm=a.tolist(),flat_exit_mm=b.tolist(),
                folded_entry_mm=af.tolist(),folded_exit_mm=bf.tolist(),diameter_mm=2*radius,
                cut_panel=int(start),unique_panels=n,internal_bends=0,
                folded_port_mismatch_mm=float(np.linalg.norm(af[1:]-bf[:-1],axis=1).max()),
                flat_mouth_turn_deg_p50_max=np.percentile(flat_turn,[50,100]).tolist(),
                folded_turn_deg_p50_max=np.percentile(angle((bf-af)[:-1],(bf-af)[1:]),[50,100]).tolist())
    (out/'perimeter_route.json').write_text(json.dumps(report))
    return channels
