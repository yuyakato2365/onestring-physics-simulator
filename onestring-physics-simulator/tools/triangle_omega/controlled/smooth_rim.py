"""Route one tension line along a rounded perimeter in the open print pose."""
import json
import pickle
import numpy as np
import manifold3d as md
from shapely.geometry import MultiPoint, LineString, Polygon
from shapely.geometry.polygon import orient
from manufacturing import union


def horizontal_section(tile, triangles, z):
    points=[]
    for tri in tile[triangles]:
        for a,b in zip(tri,np.roll(tri,-1,axis=0)):
            if min(a[2],b[2])<z<max(a[2],b[2]):
                points.append(a[:2]+(b[:2]-a[:2])*(z-a[2])/(b[2]-a[2]))
    return MultiPoint(points).convex_hull


def rim_curves(faces, edges, counts, polys, z):
    """Trace the band of triangles incident to the physical opening, in order."""
    from scipy.interpolate import CubicSpline
    from scipy.ndimage import gaussian_filter1d
    boundary=np.unique(edges[counts==1])
    rim=np.isin(faces,boundary)
    owners={}
    for i,tri in enumerate(faces):
        for k in range(3):
            j=(k+1)%3
            if rim[i,k]!=rim[i,j]:
                owners.setdefault(tuple(sorted((tri[k],tri[j]))),[]).append(i)
    adj={}
    for pair in owners.values():
        if len(pair)!=2:raise ValueError("外周帯が閉じていません。")
        i,j=pair;adj.setdefault(i,[]).append(j);adj.setdefault(j,[]).append(i)
    if not adj or any(len(a)!=2 for a in adj.values()):raise ValueError("単一の外周帯が必要です。")
    ring=[next(iter(adj))]
    while True:
        previous=ring[-2] if len(ring)>1 else None
        i=next(j for j in adj[ring[-1]] if j!=previous)
        if i==ring[0]:break
        ring.append(i)
    if len(ring)!=len(adj):raise ValueError("開口が複数あります。")
    xy=np.array([polys[i].centroid.coords[0] for i in ring])
    n=len(xy);steps=np.arange(0,n,1/12)
    for sigma in (.35,.55,.75,.95,1.2,1.5,1.9,2.4):
        points=gaussian_filter1d(xy,sigma,axis=0,mode='wrap')
        spline=CubicSpline(np.arange(n+1),np.vstack([points,points[0]]),bc_type='periodic')
        v=spline(steps,1);a=spline(steps,2)
        radius=(np.linalg.norm(v,axis=1)**3/np.maximum(abs(v[:,0]*a[:,1]-v[:,1]*a[:,0]),1e-9)).min()
        # Start between guides so a guide is not split by the periodic seam.
        path=spline(steps+.5);curve=LineString(np.vstack([path,path[0]])).simplify(.025)
        if curve.is_simple:yield curve,float(radius),float(sigma)


def smooth_channels(out, folded, flat, radius, triangles):
    with (out/'remesh3d.pkl').open('rb') as stream:
        faces=pickle.load(stream)['T']
    edges,counts=np.unique(np.sort(faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0,return_counts=True)
    rim=np.isin(faces,edges[counts==1]).any(1)
    z=float(np.percentile(flat[:,7:10,2],10)*.47)
    polys=[horizontal_section(q,tri,z) for q,tri in zip(flat,triangles)]
    hull=MultiPoint(np.vstack([np.array(g.exterior.coords) for g in polys if isinstance(g,Polygon)])).convex_hull
    best=None
    for curve,bend,smoothing in rim_curves(faces,edges,counts,polys,z):
        entries=[]
        for i,g in enumerate(polys):
            intersection=curve.intersection(g)
            parts=getattr(intersection,'geoms',[intersection])
            for seg in parts:
                if seg.length>1e-6:
                    s=curve.project(seg.interpolate(.5,normalized=True))
                    entries.append((s,seg.length,i))
        if not entries:continue
        entries.sort(); s,length,idx=np.array(entries).T
        gaps=(np.roll(s,-1)-s)%curve.length-.5*(length+np.roll(length,-1))
        usable=np.unique(idx[(length>=2*radius)&rim[idx.astype(int)]])
        score=(-len(usable),sum(length[~rim[idx.astype(int)]]),-bend,max(gaps))
        if best is None or score<best[0]:best=(score,curve,entries,gaps,bend,smoothing)
    if best is None:raise ValueError("外周ガイドを配置できません。")
    score,curve,entries,gaps,bend,smoothing=best
    # The largest free span is the opening for both pulling ends.
    cut=int(np.argmax(gaps));entries=entries[cut+1:]+entries[:cut+1]
    path=lambda s:np.r_[curve.interpolate(s%curve.length).coords[0],z]
    data=[];owners=[];channels=[md.Manifold() for _ in flat]
    # Cut the complete swept envelope, including near-grazing neighbouring panels.
    pts=np.asarray(curve.coords)[:-1];dilated=radius/np.cos(np.pi/32)+.015
    stamps=[md.Manifold.sphere(dilated,32).translate([*p,z]) for p in pts]
    tools=[md.Manifold.batch_hull([stamps[i-1],v]) for i,v in enumerate(stamps)]
    from shapely.strtree import STRtree
    footprints=[MultiPoint(q[:,:2]).convex_hull for q in flat];tree=STRtree(footprints)
    buckets=[[] for _ in flat]
    for i,tool in enumerate(tools):
        for j in tree.query(LineString([pts[i-1],pts[i]]).buffer(dilated+.1)):
            buckets[j].append(tool)
    for i,parts in enumerate(buckets):
        if parts: channels[i]=union(parts)
    for s,length,i in entries:
        owners.append(int(i));data.append([path(s-length/2),path(s+length/2)])
    data=np.array(data)
    from fixed_skin import basis
    R=basis(folded[:,:3]).transpose(0,2,1)@basis(flat[:,:3])
    folded_path=(data-flat[owners,:3].mean(1)[:,None])@R[owners].transpose(0,2,1)+folded[owners,:3].mean(1)[:,None]
    # In the closed pose the free spans change direction. Give their swept
    # cord clearance in any intervening tile and map the cut back rigidly.
    boxes_min=folded.min(1);boxes_max=folded.max(1)
    from triangle_linkage import manifold
    tile_parts=[manifold(tile,tri) for tile,tri in zip(flat,triangles)]
    relieved=set()
    for start,end in zip(folded_path[:-1,1],folded_path[1:,0]):
        if np.linalg.norm(end-start)<1e-7: continue
        candidates=np.where(((boxes_max>=np.minimum(start,end)-dilated) &
                             (boxes_min<=np.maximum(start,end)+dilated)).all(1))[0]
        for i in candidates:
            # Endpoint spheres remove the sharp exit lip for this pose.
            x,y=(np.array([start,end])-folded[i,:3].mean(0))@R[i]+flat[i,:3].mean(0)
            relief=md.Manifold.batch_hull([md.Manifold.sphere(dilated,32).translate(point) for point in (x,y)])
            if ((tile_parts[i]-channels[i]) ^ relief).volume()>1e-7:
                channels[i]=channels[i]+relief
                relieved.add(int(i))
    report=dict(folded_relief_panels=sorted(relieved),panel_sequence=owners,flat_entry_mm=data[:,0].tolist(),flat_exit_mm=data[:,1].tolist(),
                folded_entry_mm=folded_path[:,0].tolist(),folded_exit_mm=folded_path[:,1].tolist(),
                route_xy_mm=pts.tolist(),height_mm=z,bend_radius_mm=bend,smoothing=smoothing,physical_rim_total=int(rim.sum()),usable_guides=int(-score[0]),
                largest_free_span_mm=float(gaps.max()),physical_rim_guides=int(rim[owners].sum()),
                guide_passages=len(owners),diameter_mm=radius*2,
                folded_port_spacing_mm=np.percentile(np.linalg.norm(folded_path[1:,0]-folded_path[:-1,1],axis=1),[0,50,100]).tolist())
    (out/'perimeter_route.json').write_text(json.dumps(report,indent=2))
    return channels
