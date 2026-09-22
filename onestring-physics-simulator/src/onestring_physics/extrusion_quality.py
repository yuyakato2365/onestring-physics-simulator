"""Shared experimental quality of the actual paper normal-offset frustum.

All optimization residuals are dimensionless, one equally weighted row per
panel. This is a research objective, not a fabrication feasibility certificate.
"""
from __future__ import annotations
import numpy as np
from .paper_extrusion import _vertex_normals


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def evaluate_extrusion(vertices, faces, thickness, *, tiles=None):
    vertices, faces = np.asarray(vertices, float), np.asarray(faces, int)
    h = float(thickness)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("Extrusion quality requires positive finite thickness")
    if not len(faces):
        raise ValueError("Extrusion quality requires at least one quad")
    if tiles is None:
        normals = _vertex_normals(vertices, faces)
        top = vertices[faces]
        bottom = (vertices - h * normals)[faces]
    else:
        tiles = np.asarray(tiles, float)
        top, bottom = tiles[:, :4], tiles[:, 4:8]
    if not np.isfinite(top).all() or not np.isfinite(bottom).all():
        raise ValueError("Nonfinite extrusion geometry")
    edge = np.roll(top, -1, axis=1) - top
    edge_bottom = np.roll(bottom, -1, axis=1) - bottom
    scale = np.maximum(np.mean(np.linalg.norm(edge, axis=2), axis=1), 1e-12)
    nt = _unit(np.cross(top[:, 1]-top[:, 0], top[:, 2]-top[:, 0]) +
               np.cross(top[:, 2]-top[:, 0], top[:, 3]-top[:, 0]))
    nb = _unit(np.cross(bottom[:, 1]-bottom[:, 0], bottom[:, 2]-bottom[:, 0]) +
               np.cross(bottom[:, 2]-bottom[:, 0], bottom[:, 3]-bottom[:, 0]))
    side = np.stack((top, np.roll(top, -1, axis=1),
                     np.roll(bottom, -1, axis=1), bottom), axis=2)
    centered = side - side.mean(axis=2, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    side_normal = vt[:, :, -1].copy()
    reference = np.cross(side[:, :, 1]-side[:, :, 0], side[:, :, 3]-side[:, :, 0])
    sign = np.where(np.sum(side_normal*reference, axis=2) < 0, -1., 1.)
    side_normal *= sign[:, :, None]
    distance = np.einsum('pevc,pec->pev', centered, side_normal)
    depth_vector = top-bottom
    depth = np.einsum('pvc,pc->pv', depth_vector, nt)
    shear = depth_vector-depth[:, :, None]*nt[:, None, :]
    # Rigid top -> bottom fit measures departure from a constant thick plate.
    a, b = top-top.mean(axis=1, keepdims=True), bottom-bottom.mean(axis=1, keepdims=True)
    u, _, vt = np.linalg.svd(np.einsum('pvi,pvj->pij', a, b))
    corr = np.tile(np.eye(3), (len(top), 1, 1))
    corr[:, 2, 2] = np.linalg.det(u @ vt)
    pred = a @ (u @ corr @ vt)
    rigid = np.sqrt(np.mean(np.sum((pred-b)**2, axis=2), axis=1))
    # The ideal translated plate has unchanged edge vectors. This smooth
    # residual includes taper/shear without differentiating a rigid-fit SVD.
    natural = (edge_bottom-edge)/scale[:, None, None]
    residuals = np.concatenate((
        distance.reshape(len(top), -1)/scale[:, None],
        nt-nb,
        (depth-depth.mean(axis=1, keepdims=True))/h,
        shear.reshape(len(top), -1)/h,
        natural.reshape(len(top), -1),
        np.maximum(0.05-depth/h, 0),
    ), axis=1)
    difficulty = np.sqrt(np.sum(residuals**2, axis=1))
    angle = np.degrees(np.arccos(np.clip(np.sum(nt*nb, axis=1), -1, 1)))
    return dict(residuals=residuals, difficulty=difficulty,
                side_face_planarity=np.max(np.abs(distance), axis=(1, 2)),
                top_bottom_angle_deg=angle, thickness_variation=np.std(depth, axis=1),
                shear=np.sqrt(np.mean(np.sum(shear**2, axis=2), axis=1)), rigid_rms=rigid)


def quality_summary(quality):
    return {
        "extrusion_difficulty_mean": float(np.mean(quality["difficulty"])),
        "extrusion_difficulty_max": float(np.max(quality["difficulty"])),
        "side_face_planarity_error": float(np.max(quality["side_face_planarity"])),
        "top_bottom_angle_mean": float(np.mean(quality["top_bottom_angle_deg"])),
        "top_bottom_angle_max": float(np.max(quality["top_bottom_angle_deg"])),
        "thickness_variation": float(np.mean(quality["thickness_variation"])),
        "rigid_rms": float(np.mean(quality["rigid_rms"])),
    }
