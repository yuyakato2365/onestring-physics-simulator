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
    faces: np.ndarray | None = None

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
            # The original built-in wave was too aggressive for the paper-style
            # K2D/T2D pipeline.  Keep the same UI amplitude, but damp this
            # target by default and use a longer wavelength unless overridden.
            wave_scale = float(self.parameters.get("wave_amplitude_scale", 0.35))
            return amp * wave_scale * np.sin(2.0 * np.pi * x / wavelength) * np.cos(
                2.0 * np.pi * y / wavelength
            )
        if self.kind in {"gaussian", "gaussian bump", "gaussian_bump"}:
            return amp * np.exp(-(x * x + y * y) / max(2.0 * sigma * sigma, 1e-8))
        if self.kind in {"half_gourd", "gourd_half", "hyotan_half", "hyoutan_half"}:
            outline = self._half_gourd_outline(x, y)
            yn = y / max(radius, 1e-8)
            crown = np.sqrt(np.maximum(0.0, outline))
            # Slight asymmetry makes the upper lobe visibly smaller, like a cut
            # hyotan/gourd shell rather than a symmetric dumbbell.
            asym = 0.92 + 0.08 * np.tanh(-0.8 * yn)
            return amp * crown * asym
        if self.kind == "snowman_half":
            outline = self._snowman_half_outline(x, y)
            return amp * np.sqrt(np.maximum(0.0, outline))
        if self.kind == "snowman_full":
            left = self._snowman_lobe_outline(x, y, center_y=-0.46, scale=1.0)
            right = self._snowman_lobe_outline(x, y, center_y=0.46, scale=0.78)
            neck = self._snowman_neck_outline(x, y)
            return amp * np.sqrt(np.maximum(0.0, np.maximum.reduce([left, right, 0.55 * neck])))
        if self.kind == "sampled" and self.points is not None:
            return self._nearest_height(x, y)
        raise ValueError(f"unknown height field kind: {self.kind}")

    def support_mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the XY footprint of analytic targets.

        Most built-in targets occupy the full rectangular sample domain.
        half_gourd is intentionally non-rectangular so S->Omega and M2D crop
        can exercise the paper-style boundary handling path.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.kind in {"half_gourd", "gourd_half", "hyotan_half", "hyoutan_half"}:
            return self._half_gourd_outline(x, y) > 0.0
        if self.kind == "snowman_half":
            return self._snowman_half_outline(x, y) > 0.0
        if self.kind == "snowman_full":
            left = self._snowman_lobe_outline(x, y, center_y=-0.46, scale=1.0)
            right = self._snowman_lobe_outline(x, y, center_y=0.46, scale=0.78)
            neck = self._snowman_neck_outline(x, y)
            return np.maximum.reduce([left, right, neck]) > 0.0
        return np.ones_like(x, dtype=bool)

    def _half_gourd_outline(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        radius = float(self.parameters.get("radius", 1.8))
        yn = y / max(radius, 1e-8)
        # Width profile: two lobes with a narrow waist.  This is a single
        # height-field patch representing the cut half of a gourd/hyotan.
        lower = 0.78 * np.exp(-((yn + 0.42) / 0.38) ** 2)
        upper = 0.55 * np.exp(-((yn - 0.46) / 0.32) ** 2)
        waist = 0.42 * np.exp(-(yn / 0.18) ** 2)
        width_profile = np.clip(0.18 + lower + upper - waist, 0.16, 1.05)
        half_width = max(radius, 1e-8) * 0.58 * width_profile
        y_extent = 1.08
        # Superellipse in y keeps both ends rounded while leaving a visible waist.
        return 1.0 - (x / np.maximum(half_width, 1e-8)) ** 2 - (np.abs(yn) / y_extent) ** 4

    def _snowman_lobe_outline(self, x: np.ndarray, y: np.ndarray, *, center_y: float, scale: float) -> np.ndarray:
        radius = float(self.parameters.get("radius", 1.8))
        yn = y / max(radius, 1e-8)
        xn = x / max(radius, 1e-8)
        rx = 0.48 * float(scale)
        ry = 0.42 * float(scale)
        return 1.0 - (xn / max(rx, 1e-8)) ** 2 - ((yn - float(center_y)) / max(ry, 1e-8)) ** 2

    def _snowman_neck_outline(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        radius = float(self.parameters.get("radius", 1.8))
        yn = y / max(radius, 1e-8)
        xn = x / max(radius, 1e-8)
        return 1.0 - (xn / 0.24) ** 2 - (yn / 0.34) ** 4

    def _snowman_half_outline(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        lobe = self._snowman_lobe_outline(x, y, center_y=-0.15, scale=1.0)
        skirt = 0.65 * self._snowman_neck_outline(x, y)
        return np.maximum(lobe, skirt)

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
        "half-gourd": "half_gourd",
        "half gourd": "half_gourd",
        "gourd": "half_gourd",
        "gourd_half": "half_gourd",
        "hyotan": "half_gourd",
        "hyoutan": "half_gourd",
        "hyotan_half": "half_gourd",
        "hyoutan_half": "half_gourd",
        "snowman": "snowman_full",
        "two_dome_with_neck": "snowman_full",
        "two-dome-with-neck": "snowman_full",
        "snowman-full": "snowman_full",
        "snowman half": "snowman_half",
        "snowman-half": "snowman_half",
    }
    return HeightField(aliases.get(normalized, normalized), dict(parameters or {}))
