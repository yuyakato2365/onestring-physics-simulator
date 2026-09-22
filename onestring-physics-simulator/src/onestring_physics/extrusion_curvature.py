"""Noise-weighted principal directions and a topology-preserving grid rotation.

A single UV rotation is the safe regular-grid subset; no singularity-crossing
quad remesher or per-panel rotation is claimed here.
"""
from __future__ import annotations
import numpy as np


def principal_curvature(vertices, triangles):
    v, f = np.asarray(vertices, float), np.asarray(triangles, int)
    normals = np.zeros_like(v)
    fn = np.cross(v[f[:, 1]]-v[f[:, 0]], v[f[:, 2]]-v[f[:, 0]])
    for k in range(3):
        np.add.at(normals, f[:, k], fn)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    neighbors = [set() for _ in v]
    for tri in f:
        for a in tri:
            neighbors[a].update(int(b) for b in tri if a != b)
    curvature = np.zeros((len(v), 2))
    directions = np.zeros((len(v), 2, 3))
    confidence = np.zeros(len(v))
    for i, ring in enumerate(neighbors):
        ids = sorted(ring | set().union(*(neighbors[j] for j in ring)))
        ids = [j for j in ids if j != i]
        if len(ids) < 5 or np.linalg.norm(normals[i]) < .5:
            continue
        n = normals[i]
        axis = np.eye(3)[np.argmin(np.abs(n))]
        e1 = np.cross(n, axis); e1 /= np.linalg.norm(e1)
        basis = np.stack((e1, np.cross(n, e1)), axis=1)
        xy = (v[ids]-v[i]) @ basis
        dn = (normals[ids]-n) @ basis
        w = 1/np.maximum(np.linalg.norm(xy, axis=1), 1e-10)
        shape, _, rank, sv = np.linalg.lstsq(xy*w[:, None], dn*w[:, None], rcond=None)
        if rank < 2:
            continue
        shape = .5*(shape+shape.T)
        values, vectors = np.linalg.eigh(shape)
        curvature[i] = values
        directions[i] = (basis @ vectors).T
        error = np.linalg.norm((xy @ shape-dn)*w[:, None])
        signal = np.linalg.norm(dn*w[:, None])
        anisotropy = abs(values[1]-values[0])/max(np.sum(abs(values)), 1e-10)
        # Absolute signal gate makes planes/near-flat numerical noise fall back.
        signal_gate = min(1., signal/0.02)
        confidence[i] = anisotropy * max(0., 1-error/max(signal, 1e-12)) * min(1., sv[-1]/sv[0]*3) * signal_gate
    return curvature, directions, confidence, neighbors


def align_principal_grid(parameterization):
    p = parameterization
    v, f = np.asarray(p.surface_vertices_3d), np.asarray(p.surface_faces, int)
    uv, uf = np.asarray(p.uv_vertices_2d), np.asarray(p.uv_faces, int)
    if f.shape != uf.shape or f.shape[1] != 3:
        raise ValueError("Principal grid needs paired surface/UV triangles")
    curvature, directions, confidence, neighbors = principal_curvature(v, f)
    cross = np.zeros(len(uv), complex)
    weights = np.zeros(len(uv))
    for sf, tf in zip(f, uf):
        xyz = v[sf]; xy = uv[tf]
        # Physical tangent vector -> UV vector, preserving chart orientation.
        jac = np.linalg.pinv(np.stack((xyz[1]-xyz[0], xyz[2]-xyz[0]), axis=1))
        uv_edges = np.stack((xy[1]-xy[0], xy[2]-xy[0]), axis=1)
        for si, ui in zip(sf, tf):
            d = uv_edges @ jac @ directions[si, 1]
            if np.linalg.norm(d) <= 1e-12:
                continue
            angle = np.arctan2(d[1], d[0])
            cross[ui] += confidence[si]*np.exp(4j*angle)
            weights[ui] += confidence[si]
    # exp(4i theta) identifies both sign and the 90-degree axis exchange.
    uv_neighbors = [set() for _ in uv]
    for tri in uf:
        for a in tri:
            uv_neighbors[a].update(int(b) for b in tri if a != b)
    field = cross/np.maximum(weights, 1e-12)
    reliability = np.minimum(weights/6., 1.)
    for _ in range(3):
        old = field.copy()
        for i, ring in enumerate(uv_neighbors):
            ids = [i, *sorted(ring)]
            w = reliability[ids]
            field[i] = np.sum(old[ids]*w)/max(np.sum(w), 1e-12)
    mean = np.sum(field*reliability)/max(np.sum(reliability), 1e-12)
    strength = min(1., float(np.mean(reliability))*2)*abs(mean)
    angle = float(np.angle((1-strength)+strength*np.exp(1j*np.angle(mean)))/4)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    center = uv.mean(axis=0)
    p.uv_vertices_2d = (uv-center) @ rotation+center
    p.omega_boundary = (np.asarray(p.omega_boundary)-center) @ rotation+center
    p.triangle_acceleration = None
    for key in list(vars(p)):
        if 'cache' in key:
            delattr(p, key)
    p.metrics.update(principal_grid_rotation_degrees=float(np.degrees(angle)),
                     principal_grid_confidence_mean=float(np.mean(confidence)),
                     principal_grid_scope="confidence-weighted global UV rotation; regular topology preserved",
                     principal_curvatures=curvature.tolist(), principal_direction_confidence=confidence.tolist())
