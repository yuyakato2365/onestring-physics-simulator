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
    n_theta: int = 37,
) -> HeightField:
    """Create an open half-surface approximation of the paper bulb/neck target.

    The reference figure shows only one half of the rotational surface.  This
    mesh therefore spans theta=0..pi, leaves the symmetry-plane boundary open,
    keeps the neck axis straight, and leaves the mouth open.
    """
    n_x = max(12, int(n_x))
    n_theta = max(5, int(n_theta))

    # Stop the right end just before the spherical pole.  Keeping a finite
    # boundary there avoids the artificial pinched/capped pole used by the old
    # closed-mesh approximation.
    xs = np.linspace(-1.45, 1.34, n_x)
    body_center = 0.35
    body_radius = 1.05
    sphere_r = np.sqrt(np.maximum(0.0, body_radius * body_radius - (xs - body_center) ** 2))

    # Straight, circular neck.  The radius increases gently toward the bulb;
    # unlike the previous version its centerline is not displaced ("drooped").
    neck_r = 0.36 + 0.07 * (xs + 1.45)
    blend = 1.0 / (1.0 + np.exp(-6.0 * (xs + 0.68)))
    radii = (1.0 - blend) * neck_r + blend * sphere_r
    radii = np.maximum(radii, 0.035)

    # Half of the rotational surface only.  theta=0 and theta=pi are the two
    # open boundary curves on the symmetry plane z=0.
    theta = np.linspace(0.0, np.pi, n_theta, endpoint=True)
    vertices: list[list[float]] = []
    for x, radius in zip(xs, radii):
        for angle in theta:
            vertices.append([
                float(x),
                float(radius * np.cos(angle)),
                float(radius * np.sin(angle)),
            ])

    faces: list[list[int]] = []
    for ix in range(n_x - 1):
        for it in range(n_theta - 1):
            a = ix * n_theta + it
            b = a + 1
            d = (ix + 1) * n_theta + it
            cc = d + 1
            faces.extend(([a, b, cc], [a, cc, d]))

    # Intentionally no caps: the mouth, symmetry-plane cut, and far boundary
    # remain open, matching the open-surface input expected by the pipeline.
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
