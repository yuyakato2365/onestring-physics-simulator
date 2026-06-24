from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HeightField:
    """Analytic or sampled height-field target used by the v0.1 optimizer."""

    kind: str
    parameters: dict[str, float] = field(default_factory=dict)
    points: np.ndarray | None = None

    def height(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        amp = float(self.parameters.get("amplitude", 0.6))
        radius = float(self.parameters.get("radius", 1.8))
        wavelength = float(self.parameters.get("wavelength", 2.5))
        sigma = float(self.parameters.get("sigma", 1.0))

        if self.kind == "flat":
            return np.zeros_like(x, dtype=float)
        if self.kind == "dome":
            r2 = x * x + y * y
            z = amp * np.maximum(0.0, 1.0 - r2 / max(radius * radius, 1e-8))
            return z
        if self.kind == "saddle":
            return amp * (x * x - y * y) / max(radius * radius, 1e-8)
        if self.kind == "wave":
            return amp * np.sin(2.0 * np.pi * x / wavelength) * np.cos(
                2.0 * np.pi * y / wavelength
            )
        if self.kind in {"gaussian", "gaussian bump", "gaussian_bump"}:
            return amp * np.exp(-(x * x + y * y) / max(2.0 * sigma * sigma, 1e-8))
        if self.kind == "sampled" and self.points is not None:
            return self._nearest_height(x, y)
        raise ValueError(f"unknown height field kind: {self.kind}")

    def sample_grid(self, nx: int, ny: int, tile_size: float) -> np.ndarray:
        xs = (np.arange(nx + 1) - nx / 2.0) * tile_size
        ys = (np.arange(ny + 1) - ny / 2.0) * tile_size
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        zz = self.height(xx, yy)
        return np.stack([xx, yy, zz], axis=-1)

    def _nearest_height(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pts = np.asarray(self.points, dtype=float)
        query = np.stack([x.ravel(), y.ravel()], axis=1)
        diff = query[:, None, :] - pts[None, :, :2]
        idx = np.argmin(np.sum(diff * diff, axis=2), axis=1)
        z = pts[idx, 2]
        return z.reshape(x.shape)


def make_height_field(kind: str, parameters: dict[str, Any] | None = None) -> HeightField:
    normalized = kind.strip().lower().replace("-", "_")
    aliases = {
        "gaussian_bump": "gaussian",
        "bump": "gaussian",
    }
    return HeightField(aliases.get(normalized, normalized), dict(parameters or {}))
