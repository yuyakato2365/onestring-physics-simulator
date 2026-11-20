"""Fit ordered panel corners to one rim or both banks of a cut."""
import numpy as np


def resample(points, parameters, positions):
    return np.stack([np.interp(positions, parameters, axis) for axis in points.T], axis=-1)


def arc_fractions(points):
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0., np.cumsum(lengths)]
    return cumulative / cumulative[-1]


def place_corners(banks, pitch, spacing=None):
    """Shortest path through ordered arc samples, with one chord per panel edge.

    Each bank contains its original UV polyline, normalized 3D arclength
    and initial lattice corners. Paired banks traverse the same 3D direction.
    """
    count = len(banks[0][2])-1
    uniform = np.linspace(0., 1., 8*count+1)
    knots = [uniform]
    for points, arc, _ in banks:
        if len(arc) <= 32*(count+1):
            knots.append(arc)
            continue
        # On dense input, retain the largest departure from each coarse chord.
        # Candidate count then follows panel resolution instead of scan density.
        coarse = resample(points, arc, uniform)
        errors = np.linalg.norm(points-resample(coarse, uniform, arc), axis=1)
        bins = np.minimum((arc*(len(uniform)-1)).astype(int), len(uniform)-2)
        ranked = np.lexsort((-errors, bins))
        first = np.r_[True, np.diff(bins[ranked]) != 0]
        knots.append(arc[ranked[first]])
    samples = np.unique(np.round(np.concatenate(knots), 13))
    curves = [resample(points, arc, samples) for points,arc,_ in banks]
    metric = None if spacing is None else np.interp(samples, *spacing)
    m = len(samples)
    stops = np.searchsorted(samples, samples + 3./count + 1e-12, side='right')
    width = int(np.max(stops - np.arange(m) - 1))
    predecessors = np.maximum(0, np.arange(m)[:,None] - np.arange(width,0,-1))
    chord_cost = np.full((m,width), np.inf)
    for start in range(m-1):
        # The uniform samples guarantee a feasible path with unit mean spacing.
        # Restrict expensive chord tests to the local part of the curve.
        ends = np.arange(start+1, stops[start])
        if not len(ends):
            continue
        values = np.zeros(len(ends))
        for curve in curves:
            delta = (curve[ends] - curve[start])/pitch
            length2 = (delta*delta).sum(axis=1)
            # The distance to a segment is convex along each source edge.
            # Its maximum on this arc occurs at one of the sampled vertices.
            t = np.clip((delta @ delta.T)/np.maximum(length2, 1e-20), 0., 1.)
            distance = delta[:,None,:] - t[:,:,None] * delta[None,:,:]
            distances2 = (distance*distance).sum(axis=2)
            error = np.triu(distances2).max(axis=0)
            values += 32.*error + (np.sqrt(length2)-1.)**2
        if metric is not None:
            # Both banks use this one 3D arc coordinate, in local target edge lengths.
            values += 12.*len(banks)*(metric[ends]-metric[start]-1.)**2
        chord_cost[ends, width+start-ends] = values
    choice = np.empty((count,m),int)
    previous = np.full(m,np.inf)
    previous[0] = 0.
    for j in range(count):
        costs = previous[predecessors]+chord_cost
        best = np.argmin(costs, axis=1)
        choice[j] = predecessors[np.arange(m), best]
        previous = costs[np.arange(m), best]
        for curve,(_,_,corners) in zip(curves,banks):
            previous += ((curve-corners[j+1])**2).sum(axis=1)/pitch**2
    selected = [m-1]
    for row in choice[::-1]:
        selected.append(row[selected[-1]])
    return [curve[selected[::-1]] for curve in curves]


def angle_report(triangles):
    left = np.roll(triangles, 1, axis=1) - triangles
    right = np.roll(triangles, -1, axis=1) - triangles
    lengths = np.linalg.norm(left, axis=2)
    denom = lengths * np.roll(lengths, -1, axis=1)
    angles = np.degrees(np.arccos(np.clip((left*right).sum(axis=2)/np.maximum(denom, 1e-30), -1., 1.)))
    deviation = np.abs(angles-60.).max(axis=1)
    return dict(min_angle=float(angles.min()), max_angle=float(angles.max()),
                max_edge_ratio=float((lengths.max(1)/np.maximum(lengths.min(1), 1e-30)).max()),
                max_deviation=float(deviation.max()), p95_deviation=float(np.percentile(deviation,95)),
                outside_20deg=int((deviation>20.+1e-7).sum()), panels=len(triangles))


def shape_limited_grid(points, faces, corners, targets):
    """Move the triangular lattice toward the contour while reserving corner angles."""
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import spsolve
    from shapely.geometry import LinearRing
    used = np.unique(faces)
    free = np.setdiff1d(used, corners)
    displacement = np.zeros_like(points)
    displacement[corners] = targets-points[corners]
    if len(free):
        edges = np.unique(np.sort(faces[:, [[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0)
        i,j = np.vstack((edges, edges[:,::-1])).T
        adjacency = coo_matrix((np.ones(len(i)),(i,j)),shape=(len(points),len(points))).tocsr()
        lap = diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
        displacement[free] = spsolve(lap[free][:,free], -(lap[free][:,corners] @ displacement[corners]))
    boundary = faces[np.isin(faces,corners).any(1)]
    exact = points+displacement
    def valid(scale):
        x = points+scale*displacement
        d, e = x[faces[:,1]]-x[faces[:,0]], x[faces[:,2]]-x[faces[:,0]]
        return (np.all(d[:,0]*e[:,1]-d[:,1]*e[:,0]>0) and LinearRing(x[corners]).is_simple
                and angle_report(x[boundary])['max_deviation'] <= 20.)
    lo, hi = 0.,1.
    if valid(hi):
        return exact, exact
    for _ in range(16):
        mid = .5*(lo+hi)
        if valid(mid):
            lo=mid
        else:
            hi=mid
    return points+lo*displacement, exact


def allocate_boundary_edges(targets, groups, total):
    """Integer least squares with equal counts on each pair and a fixed perimeter."""
    if total < len(targets):
        return None
    cost = np.full(total+1, np.inf)
    cost[0] = 0.
    back = []
    for group in groups:
        copies = len(group)
        target = np.mean(targets[list(group)])
        best = np.full_like(cost, np.inf)
        counts = np.zeros(total+1, int)
        for n in range(1, total//copies+1):
            used = copies*n
            candidate = cost[:total+1-used] + copies*(n-target)**2
            better = candidate < best[used:]
            best[used:][better] = candidate[better]
            counts[used:][better] = n
        back.append(counts)
        cost = best
    if not np.isfinite(cost[total]):
        return None
    result = np.empty(len(targets), int)
    for group, counts in zip(reversed(groups), reversed(back)):
        n = counts[total]
        result[list(group)] = n
        total -= n*len(group)
    return result


def independent_stars(vertex_count, faces, free):
    incidence = [[] for _ in range(vertex_count)]
    for fi, face in enumerate(faces):
        for j, vertex in enumerate(face):
            incidence[vertex].append((fi,j))
    colors = np.full(vertex_count,-1,int)
    for vertex in free:
        adjacent = faces[[fi for fi,_ in incidence[vertex]]]
        unavailable = set(colors[adjacent].ravel())
        color = 0
        while color in unavailable:
            color += 1
        colors[vertex] = color
    # All interior lattice stars have six triangles; group by degree as well
    # so this also works on imported grids with nonuniform valences.
    groups = {}
    for vertex in free:
        groups.setdefault((colors[vertex],len(incidence[vertex])),[]).append(vertex)
    stars = []
    for vertices in groups.values():
        ids = np.array(vertices)
        fi, corner = np.array([incidence[i] for i in ids]).transpose(2,0,1)
        others = np.take_along_axis(faces[fi],(corner[:,:,None]+[1,2])%3,axis=2)
        stars.append((ids,fi,others))

    return stars


def distribute_lift_distortion(grid, faces, boundary, boundary_xyz, uv, uv_faces, xyz, pitch, domain):
    """Relax independent interior stars in the chart, evaluating actual surface chords."""
    from matplotlib.tri import Triangulation
    from shapely.geometry import Polygon

    x = grid.copy()
    free = np.setdiff1d(np.arange(len(x)), boundary)
    if not len(free):
        return x, dict(moved=0)
    try:
        finder = Triangulation(*uv.T, triangles=uv_faces).get_trifinder()
    except RuntimeError:
        return x, dict(moved=0, reason='chart point location unavailable')
    chart = uv[uv_faces]
    surface = xyz[uv_faces]
    derivatives = (surface[:,1:]-surface[:,:1]).transpose(0,2,1) @ np.linalg.inv(
        (chart[:,1:]-chart[:,:1]).transpose(0,2,1))

    def lift(points, ids):
        return surface[ids,0] + np.einsum('nij,nj->ni',derivatives[ids],points-chart[ids,0])

    ids = finder(*x[free].T)
    if np.any(ids < 0):
        return x, dict(moved=0, reason='interior vertices outside chart')
    v = np.zeros((len(x),3))
    v[free] = lift(x[free],ids)
    v[boundary] = boundary_xyz
    normals = np.cross(v[faces[:,1]]-v[faces[:,0]],v[faces[:,2]]-v[faces[:,0]])
    stars = independent_stars(len(x), faces, free)

    def errors(p, q, r):
        a,b,c=q-p,r-q,p-r
        def cosine(l,r):
            denom = np.linalg.norm(l,axis=-1)*np.linalg.norm(r,axis=-1)
            return -np.sum(l*r,axis=-1)/np.maximum(denom,1e-25)
        angles=np.arccos(np.clip(np.stack((cosine(c,a),cosine(a,b),cosine(b,c)),axis=-1),-1.,1.))
        return np.max(np.abs(angles-np.pi/3),axis=-1)

    import shapely
    close = shapely.distance(shapely.points(x),domain.boundary) < 2.5*pitch
    dirs=np.array([(np.cos(t),np.sin(t)) for t in np.arange(6)*np.pi/3])
    for step in (.14,.09,.05,.025,.0125):
        for ids,fi,others in stars:
            q,r=x[others[:,:,0]],x[others[:,:,1]]
            Q,R=v[others[:,:,0]],v[others[:,:,1]]
            best_x,best_v=x[ids].copy(),v[ids].copy()
            start2=errors(best_x[:,None],q,r)
            start3=errors(best_v[:,None],Q,R)
            def merit(d2,d3):
                return np.mean(d3**4+2*d2**4,axis=1)+np.max(d3,axis=1)**4
            best=merit(start2,start3)
            for direction in dirs:
                probe=x[ids]+step*pitch*direction
                face_id=finder(*probe.T)
                mapped=lift(probe,np.maximum(face_id,0))
                a,b=q-probe[:,None],r-probe[:,None]
                ccw=a[:,:,0]*b[:,:,1]-a[:,:,1]*b[:,:,0]
                n=np.cross(Q-mapped[:,None],R-mapped[:,None])
                d2=errors(probe[:,None],q,r)
                d3=errors(mapped[:,None],Q,R)
                allowed=(face_id>=0) & np.all((ccw>0) & (np.sum(n*normals[fi],axis=2)>0)
                    & (d2<=np.maximum(start2,np.deg2rad(20))+1e-9),axis=1)
                cost=merit(d2,d3)
                chosen=allowed & (cost<best) & (d3.max(1)<=start3.max(1)+1e-9)
                best_x[chosen],best_v[chosen],best[chosen]=probe[chosen],mapped[chosen],cost[chosen]
            for row in np.flatnonzero(close[ids] & np.any(best_x!=x[ids],axis=1)):
                tris = np.stack((np.broadcast_to(best_x[row],q[row].shape),q[row],r[row]),axis=1)
                polys = shapely.polygons(tris)
                if np.any(shapely.area(shapely.intersection(polys,domain)) < .5*shapely.area(polys)):
                    best_x[row],best_v[row]=x[ids[row]],v[ids[row]]
            x[ids],v[ids]=best_x,best_v
    changed = np.any(x!=grid,axis=1)
    for tri in x[faces[changed[faces].any(1)]]:
        poly=Polygon(tri)
        if poly.intersection(domain).area < .5*poly.area:
            return grid, dict(moved=0, reason='overlap below half')
    return x, dict(moved=int(changed.sum()),max_displacement_over_pitch=float(np.linalg.norm(x-grid,axis=1).max()/pitch))
