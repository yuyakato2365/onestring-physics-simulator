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



def create_paper_bulb_neck_shape(
    n_x: int = 96,
    n_theta: int = 72,
) -> HeightField:
    """Create the smooth bulb-with-neck closed target used for paper-style tests.

    This is a procedural approximation of the reference target: a narrow open-looking
    neck smoothly blends into a large rounded bulb.  The returned representation is a
    sampled triangular mesh so it follows the same closed-mesh pipeline as Bunny.
    """
    n_x = max(12, int(n_x))
    n_theta = max(16, int(n_theta))
    xs = np.linspace(-1.45, 1.35, n_x)
    body_center = 0.35
    body_radius = 1.05
    sphere_r = np.sqrt(np.maximum(0.0, body_radius * body_radius - (xs - body_center) ** 2))
    neck_r = 0.34 + 0.11 * (xs + 1.45)
    blend = 1.0 / (1.0 + np.exp(-5.5 * (xs + 0.72)))
    radii = (1.0 - blend) * neck_r + blend * sphere_r
    radii += 0.10 * np.exp(-((xs + 1.40) / 0.20) ** 2)
    radii = np.maximum(radii, 0.035)

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    vertices: list[list[float]] = []
    for x, radius in zip(xs, radii):
        droop = -0.16 * np.exp(-((x + 0.95) / 0.50) ** 2)
        for angle in theta:
            vertices.append([
                float(x),
                float(radius * np.cos(angle)),
                float(radius * np.sin(angle) + droop),
            ])

    faces: list[list[int]] = []
    for ix in range(n_x - 1):
        for it in range(n_theta):
            jt = (it + 1) % n_theta
            a = ix * n_theta + it
            b = ix * n_theta + jt
            c = (ix + 1) * n_theta + jt
            d = (ix + 1) * n_theta + it
            faces.extend(([a, b, c], [a, c, d]))

    # Cap both poles so this behaves as a closed target mesh, like the paper target.
    left_center = len(vertices)
    right_center = left_center + 1
    left_droop = -0.16 * np.exp(-((xs[0] + 0.95) / 0.50) ** 2)
    right_droop = -0.16 * np.exp(-((xs[-1] + 0.95) / 0.50) ** 2)
    vertices.extend([
        [float(xs[0]), 0.0, float(left_droop)],
        [float(xs[-1]), 0.0, float(right_droop)],
    ])
    for it in range(n_theta):
        jt = (it + 1) % n_theta
        faces.append([left_center, jt, it])
        a = (n_x - 1) * n_theta + it
        b = (n_x - 1) * n_theta + jt
        faces.append([right_center, a, b])

    points = normalize_shape(np.asarray(vertices, dtype=float))
    return HeightField("sampled", points=points, faces=np.asarray(faces, dtype=int))


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
    faces = np.asarray(mesh.faces, dtype=int)
    return HeightField("sampled", points=points, faces=faces)


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
