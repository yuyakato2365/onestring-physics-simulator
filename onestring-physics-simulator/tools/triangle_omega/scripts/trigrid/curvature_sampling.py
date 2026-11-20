"""Redistribute a triangular lattice around insufficiently sampled surface bends."""
import numpy as np
from scipy.spatial import cKDTree
import shapely
from shapely.geometry import LinearRing


def fan_angles(vertices, faces):
    tri = vertices[faces]
    normals = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:,None], 1e-30)
    owners = [[] for _ in vertices]
    for i, face in enumerate(faces):
        for v in face:
            owners[v].append(i)
    result = np.zeros(len(vertices))
    for v, fan in enumerate(owners):
        if len(fan)>1:
            n = normals[fan]
            result[v] = np.degrees(np.arccos(np.clip((n @ n.T).min(),-1,1)))
    return result


def focused_chart(mesh, uv, boundary, faces, chart_xyz, pitch, minimum_depth, lateral):
    angles = fan_angles(mesh['X'],mesh['T'])
    limit = np.degrees(2*np.arctan2(lateral, minimum_depth))
    order = np.argsort(-angles)
    points = mesh['X'][order[angles[order] > limit]]
    if not len(points):
        return uv, {'centers':0}
    tree = cKDTree(chart_xyz)
    centers = []
    radius = 2.*pitch
    for point in points:
        _, j = tree.query(point)
        # All chart copies of a seam vertex receive the same sampling treatment.
        for idx in tree.query_ball_point(chart_xyz[j],1e-12):
            center = uv[idx]
            if all(np.linalg.norm(c-center)>2.*radius for c in centers):
                centers.append(center)
    shift = np.zeros_like(uv)
    for center in centers:
        d = uv-center
        shift += d*np.exp(-np.sum(d*d,axis=1)[:,None]/(2*radius**2))
    # Preserve seam-bank correspondence and the previously accepted outline.
    distance = shapely.distance(shapely.points(uv), LinearRing(uv[boundary]))
    t = np.clip(distance/radius, 0., 1.)
    shift *= (t*t*(3.-2.*t))[:,None]
    shift[boundary] = 0.
    for gain in (1.,.75,.5,.25,.125,.0625):
        trial = uv+gain*shift
        d = trial[faces[:,1:]]-trial[faces[:,0],None]
        if np.all(d[:,0,0]*d[:,1,1]-d[:,0,1]*d[:,1,0]>0) and LinearRing(trial[boundary]).is_simple:
            return trial,dict(centers=len(centers),gain=gain,angle_limit_deg=limit,radius_in_pitches=2.)
    raise ValueError('曲率に合わせて格子の密度を変えるとΩ2が反転します。')
