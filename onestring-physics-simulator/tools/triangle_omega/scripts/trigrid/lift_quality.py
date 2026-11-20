"""Geometric acceptance tests for welded panel surfaces, in panel-length units."""
import numpy as np
import igl
from igl.copyleft.cgal import remesh_self_intersections
from scipy.optimize import linprog


def surface_quality(vertices, faces, reference):
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    area = np.linalg.norm(normals, axis=1)
    if not np.all(np.isfinite(area)) or np.any(area <= 1e-14):
        return dict(valid=False, reason='degenerate face')
    normals /= area[:, None]
    fans = [[] for _ in vertices]
    for i, face in enumerate(faces):
        for vertex in face:
            fans[vertex].append(i)
    blocked = []
    for vertex, incident in enumerate(fans):
        if not incident:
            continue
        n = normals[incident]
        if np.min(n @ n.mean(0)) > 1e-8:
            continue
        lp = linprog([0, 0, 0, -1], A_ub=np.column_stack((n, np.ones(len(n)))),
                     b_ub=np.zeros(len(n)), bounds=[(-1, 1)]*3+[(0, 1)], method='highs')
        if not lp.success or lp.x[-1] < 1e-8:
            blocked.append(vertex)
    intersections = len(remesh_self_intersections(vertices, faces, detect_only=True)[2])
    pitch = np.median(np.linalg.norm(triangles-np.roll(triangles, 1, axis=1), axis=2))
    distance = np.sqrt(igl.point_mesh_squared_distance(reference, vertices, faces)[0])/pitch
    return dict(valid=not blocked and not intersections, blocked_vertices=blocked,
                intersecting_pairs=intersections, max_distance_over_pitch=float(distance.max()),
                p95_distance_over_pitch=float(np.percentile(distance, 95)),
                farthest_reference_vertex=int(np.argmax(distance)))
