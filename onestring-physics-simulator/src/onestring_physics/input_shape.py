from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np

from .heightfields import HeightField, make_height_field

CLOSED_SHAPE_WARNING = (
    "v0.1 works best for open height-field-like surfaces. Closed shapes such as "
    "bunny are not fully supported yet."
)


def create_builtin_shape(kind: str, parameters: dict | None = None) -> HeightField:
    return make_height_field(kind, parameters)


def load_target_shape(path: str | Path) -> HeightField:
    try:
        import trimesh
    except Exception as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError("trimesh is required for mesh loading") from exc

    mesh = trimesh.load(path, force="mesh")
    if getattr(mesh, "is_watertight", False):
        warnings.warn(CLOSED_SHAPE_WARNING, stacklevel=2)
    points = np.asarray(mesh.vertices, dtype=float)
    points = normalize_shape(points)
    return HeightField("sampled", points=points)


def normalize_shape(mesh_or_points) -> np.ndarray:
    if hasattr(mesh_or_points, "vertices"):
        points = np.asarray(mesh_or_points.vertices, dtype=float)
    else:
        points = np.asarray(mesh_or_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("shape points must be an (N, 3) array")

    centered = points - np.mean(points, axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(centered[:, :2], axis=1))
    if scale <= 1e-12:
        scale = np.max(np.ptp(centered, axis=0))
    return centered / max(scale, 1e-12)


def sample_target_surface(
    target: HeightField,
    nx: int,
    ny: int | None = None,
    tile_size: float = 1.0,
) -> np.ndarray:
    ny = nx if ny is None else ny
    return target.sample_grid(nx, ny, tile_size)
