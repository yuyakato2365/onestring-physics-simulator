"""Panel size / shape uniformity metrics and acceptance gates (OneString, not Zhang 2021).

Used for M2D (UV), M3D (lifted, generally non-planar quads) and K3D (planar
panels).  Every quantity is scale free: lengths enter only as ratios or as
logarithms relative to the mesh's own geometric-mean edge length, so the
same thresholds apply to a 10 cm bunny and to a 2 m dome.

Why not the scaled Jacobian alone: a 1 x 6 rectangle has scaled Jacobian 1
(all corners are right angles) but is useless as a panel.  Size (edge length
spread, panel area spread), aspect (longest / shortest edge of each panel)
and angles are therefore measured separately, together with the worst panel
of every criterion and a smooth maximum (log-sum-exp) of the per-panel
deviation, so that one extreme panel cannot hide behind a good mean.

The thresholds are user settings with documented defaults, not "official"
values; every rejection names the criterion, the measured value, the limit
and the worst panel.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np


@dataclass
class PanelQualityThresholds:
    """Scale-free acceptance limits for one stage (M2D, M3D or K3D)."""

    min_angle_deg: float = 30.0          # every panel corner >= this
    max_angle_deg: float = 150.0         # every panel corner <= this
    max_edge_ratio: float = 3.0          # per panel: longest / shortest edge
    max_size_ratio: float = 4.0          # largest / smallest panel (sqrt of area)
    max_size_spread_p95_p5: float = 2.5  # robust size spread (95th / 5th percentile)
    max_e_size: float = 0.15             # mean log^2(edge / geometric-mean edge)
    max_e_angle: float = 0.15            # mean cos^2(corner angle)
    max_area_cv: float = 0.75            # std / mean of panel areas
    max_nonplanarity: float | None = None  # max corner-to-plane distance / mean edge (K3D)

    @classmethod
    def from_params(cls, params: Any, prefix: str, **defaults) -> "PanelQualityThresholds":
        out = cls(**defaults)
        for f in fields(cls):
            key = f"{prefix}_{f.name}"
            if params is not None and hasattr(params, key):
                value = getattr(params, key)
                if value is None:
                    setattr(out, f.name, None)
                else:
                    setattr(out, f.name, float(value))
        out.validate()
        return out

    @classmethod
    def from_text(cls, text: Any, **defaults) -> "PanelQualityThresholds":
        """'min_angle_deg=30, max_edge_ratio=3' (empty -> defaults); unknown keys are errors."""
        out = cls(**defaults)
        if text is None or not str(text).strip():
            out.validate()
            return out
        names = {f.name for f in fields(cls)}
        for part in str(text).replace(";", ",").split(","):
            if not part.strip():
                continue
            if "=" not in part:
                raise ValueError(f"ZHANG2021_QUALITY_THRESHOLD_INVALID: {part!r} (expected name=value)")
            k, v = (x.strip() for x in part.split("=", 1))
            if k not in names:
                raise ValueError(f"ZHANG2021_QUALITY_THRESHOLD_INVALID: unknown {k!r}; known {sorted(names)}")
            setattr(out, k, None if v.lower() in ("none", "off", "") else float(v))
        out.validate()
        return out

    def validate(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if v is None:
                continue
            if not np.isfinite(v) or v < 0:
                raise ValueError(f"ZHANG2021_QUALITY_THRESHOLD_INVALID: {f.name}={v!r}")
        if not (0.0 <= self.min_angle_deg < 90.0 < self.max_angle_deg <= 180.0):
            raise ValueError("ZHANG2021_QUALITY_THRESHOLD_INVALID: angle limits")
        for name in ("max_edge_ratio", "max_size_ratio", "max_size_spread_p95_p5"):
            if getattr(self, name) < 1.0:
                raise ValueError(f"ZHANG2021_QUALITY_THRESHOLD_INVALID: {name} < 1")

    def as_dict(self) -> dict:
        return asdict(self)


# Defaults per stage.  M2D lives in the (distorted) chart: near OptCuts cut
# tips the UV quads must be thin wedges so that the lifted panels are good,
# so M2D is gated for validity/non-degeneracy only; M3D and K3D carry the
# size/shape uniformity gates.
DEFAULT_M2D_THRESHOLDS = dict(min_angle_deg=3.0, max_angle_deg=177.0, max_edge_ratio=25.0,
                              max_size_ratio=1e6, max_size_spread_p95_p5=1e6, max_e_size=1e6,
                              max_e_angle=1e6, max_area_cv=1e6)
DEFAULT_M3D_THRESHOLDS: dict = {}
DEFAULT_K3D_THRESHOLDS = dict(max_nonplanarity=1e-3)


@dataclass
class PanelQualityReport:
    stage: str
    metrics: dict = field(default_factory=dict)
    per_panel: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)
    thresholds: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.failures

    def prefixed(self, prefix: str) -> dict:
        out = {f"{prefix}_{k}": v for k, v in self.metrics.items()}
        out[f"{prefix}_quality_passed"] = bool(self.passed)
        out[f"{prefix}_quality_failures"] = list(self.failures)
        out[f"{prefix}_quality_thresholds"] = dict(self.thresholds)
        return out


def _as3(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, float)
    if p.ndim != 2:
        raise ValueError("points must be (n, 2) or (n, 3)")
    if p.shape[1] == 2:
        return np.column_stack((p, np.zeros(len(p))))
    return p[:, :3]


def unique_quad_edges(faces: np.ndarray) -> np.ndarray:
    f = np.asarray(faces, int)
    e = np.concatenate([f[:, [k, (k + 1) % 4]] for k in range(4)])
    return np.unique(np.sort(e, axis=1), axis=0)


def per_panel_geometry(points: np.ndarray, faces: np.ndarray, *, planar_2d_orientation: float | None = None) -> dict:
    """Edge lengths, corner angles, areas and corner orientation per panel."""
    P = _as3(points)[np.asarray(faces, int)]                  # (m, 4, 3)
    nxt = np.roll(P, -1, axis=1) - P                           # edge k: corner k -> k+1
    prv = np.roll(P, 1, axis=1) - P                            # edge to corner k-1
    lengths = np.linalg.norm(nxt, axis=2)                      # (m, 4)
    d1 = P[:, 2] - P[:, 0]
    d2 = P[:, 3] - P[:, 1]
    vec_area = 0.5 * np.cross(d1, d2)                          # vector area of the quad
    area = np.linalg.norm(vec_area, axis=1)
    normal = vec_area / np.maximum(area[:, None], 1e-300)
    denom = np.maximum(np.linalg.norm(prv, axis=2) * lengths, 1e-300)
    cosines = np.clip(np.einsum("mkd,mkd->mk", prv, nxt) / denom, -1.0, 1.0)
    angles = np.degrees(np.arccos(cosines))
    # corner orientation w.r.t. the panel's own normal: >0 convex corner
    corner_cross = np.cross(nxt, prv)                          # (m, 4, 3)
    if planar_2d_orientation is not None:
        sign = np.sign(planar_2d_orientation) or 1.0
        orient = sign * corner_cross[:, :, 2] / np.maximum(denom, 1e-300)
    else:
        orient = np.einsum("mkd,md->mk", corner_cross, normal) / np.maximum(denom, 1e-300)
    centered = P - P.mean(axis=1, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        plane_n = vt[:, -1, :]
        nonplanar = np.max(np.abs(np.einsum("mkd,md->mk", centered, plane_n)), axis=1)
    except np.linalg.LinAlgError:
        nonplanar = np.full(len(P), np.nan)
    return {"lengths": lengths, "angles": angles, "cosines": cosines, "area": area,
            "corner_orientation": orient, "nonplanar": nonplanar,
            "diag": np.c_[np.linalg.norm(d1, axis=1), np.linalg.norm(d2, axis=1)]}


def smooth_max(values: np.ndarray, beta: float = 12.0) -> float:
    v = np.asarray(values, float)
    if not len(v):
        return 0.0
    z = beta * v
    zmax = float(np.max(z))
    return float((zmax + np.log(np.mean(np.exp(z - zmax)))) / beta)


def panel_quality(points: np.ndarray, faces: np.ndarray, *, stage: str,
                  thresholds: PanelQualityThresholds | None = None,
                  planar_2d_orientation: float | None = None) -> PanelQualityReport:
    """Measure size/shape uniformity and apply the stage's acceptance gate."""
    faces = np.asarray(faces, int)
    thr = thresholds or PanelQualityThresholds()
    thr.validate()
    if faces.ndim != 2 or faces.shape[1] != 4 or not len(faces):
        return PanelQualityReport(stage, {"panels": int(len(faces))}, {}, ["no quad panels"], thr.as_dict())
    g = per_panel_geometry(points, faces, planar_2d_orientation=planar_2d_orientation)
    pts = _as3(points)
    edges = unique_quad_edges(faces)
    el = np.linalg.norm(pts[edges[:, 1]] - pts[edges[:, 0]], axis=1)
    positive = el > 0
    l_ref = float(np.exp(np.mean(np.log(el[positive])))) if np.any(positive) else 0.0
    with np.errstate(divide="ignore"):
        log_rel = np.where(positive, np.log(np.maximum(el, 1e-300) / max(l_ref, 1e-300)), -np.inf)
    e_size = float(np.mean(log_rel ** 2)) if np.all(np.isfinite(log_rel)) else float("inf")
    lengths = g["lengths"]
    lmin = np.min(lengths, axis=1)
    lmax = np.max(lengths, axis=1)
    edge_ratio = np.where(lmin > 0, lmax / np.maximum(lmin, 1e-300), np.inf)
    area = g["area"]
    size = np.sqrt(area)
    angles = g["angles"]
    cos2 = g["cosines"] ** 2
    e_angle = float(np.mean(cos2))
    area_mean = float(np.mean(area))
    area_cv = float(np.std(area) / area_mean) if area_mean > 0 else float("inf")
    tiny = max(l_ref, 1e-300) ** 2 * 1e-8
    degenerate = (area <= tiny) | (lmin <= 1e-6 * max(l_ref, 1e-300)) | (np.min(angles, axis=1) < 0.5)
    flipped = np.any(g["corner_orientation"] <= 0.0, axis=1) & ~degenerate
    # per-panel deviation used by the smooth maximum: angle + aspect + size
    with np.errstate(divide="ignore", invalid="ignore"):
        size_dev = np.log(np.maximum(size, 1e-300) / max(float(np.exp(np.mean(np.log(np.maximum(size, 1e-300))))), 1e-300))
        dev = np.sqrt(np.mean(cos2, axis=1) + np.log(np.maximum(edge_ratio, 1.0)) ** 2 + size_dev ** 2)
    dev = np.where(np.isfinite(dev), dev, 1e6)
    s_pos = size[size > 0]
    p5, p95 = (np.percentile(s_pos, [5, 95]) if len(s_pos) else (0.0, 0.0))
    nonplanar_rel = g["nonplanar"] / max(l_ref, 1e-300)
    m = {
        "panels": int(len(faces)),
        "reference_edge_length": l_ref,
        "edge_min_relative": float(np.min(el) / max(l_ref, 1e-300)),
        "edge_max_relative": float(np.max(el) / max(l_ref, 1e-300)),
        "e_size": e_size,
        "e_angle": e_angle,
        "e_aspect": float(np.mean(np.log(np.maximum(edge_ratio, 1.0)) ** 2)),
        "area_cv": area_cv,
        "angle_min_deg": float(np.min(angles)),
        "angle_max_deg": float(np.max(angles)),
        "angle_p1_deg": float(np.percentile(np.min(angles, axis=1), 1)),
        "angle_p5_deg": float(np.percentile(np.min(angles, axis=1), 5)),
        "edge_ratio_max": float(np.max(edge_ratio)),
        "edge_ratio_p95": float(np.percentile(edge_ratio, 95)),
        "size_ratio_max_min": float(np.max(size) / np.min(size)) if np.min(size) > 0 else float("inf"),
        "size_spread_p95_p5": float(p95 / p5) if p5 > 0 else float("inf"),
        "nonplanarity_max_relative": float(np.nanmax(nonplanar_rel)) if len(nonplanar_rel) else 0.0,
        "degenerate_panels": int(np.sum(degenerate)),
        "flipped_or_nonconvex_panels": int(np.sum(flipped)),
        "worst_panel_smooth_max": smooth_max(dev),
        "worst_min_angle_panel": int(np.argmin(np.min(angles, axis=1))),
        "worst_max_angle_panel": int(np.argmax(np.max(angles, axis=1))),
        "worst_edge_ratio_panel": int(np.argmax(edge_ratio)),
        "smallest_panel": int(np.argmin(size)),
        "largest_panel": int(np.argmax(size)),
    }
    fails: list[str] = []

    def chk(ok: bool, text: str):
        if not ok:
            fails.append(text)

    chk(m["degenerate_panels"] == 0, f"{m['degenerate_panels']} degenerate panel(s)")
    chk(m["flipped_or_nonconvex_panels"] == 0,
        f"{m['flipped_or_nonconvex_panels']} flipped/non-convex panel(s)")
    chk(m["angle_min_deg"] >= thr.min_angle_deg,
        f"min angle {m['angle_min_deg']:.2f} deg < {thr.min_angle_deg:g} (panel {m['worst_min_angle_panel']})")
    chk(m["angle_max_deg"] <= thr.max_angle_deg,
        f"max angle {m['angle_max_deg']:.2f} deg > {thr.max_angle_deg:g} (panel {m['worst_max_angle_panel']})")
    chk(m["edge_ratio_max"] <= thr.max_edge_ratio,
        f"edge ratio {m['edge_ratio_max']:.3g} > {thr.max_edge_ratio:g} (panel {m['worst_edge_ratio_panel']})")
    chk(m["size_ratio_max_min"] <= thr.max_size_ratio,
        f"panel size ratio {m['size_ratio_max_min']:.3g} > {thr.max_size_ratio:g} "
        f"(panels {m['largest_panel']}/{m['smallest_panel']})")
    chk(m["size_spread_p95_p5"] <= thr.max_size_spread_p95_p5,
        f"panel size spread p95/p5 {m['size_spread_p95_p5']:.3g} > {thr.max_size_spread_p95_p5:g}")
    chk(m["e_size"] <= thr.max_e_size, f"E_size {m['e_size']:.4g} > {thr.max_e_size:g}")
    chk(m["e_angle"] <= thr.max_e_angle, f"E_angle {m['e_angle']:.4g} > {thr.max_e_angle:g}")
    chk(m["area_cv"] <= thr.max_area_cv, f"area CV {m['area_cv']:.4g} > {thr.max_area_cv:g}")
    if thr.max_nonplanarity is not None:
        chk(m["nonplanarity_max_relative"] <= thr.max_nonplanarity,
            f"non-planarity {m['nonplanarity_max_relative']:.3g} > {thr.max_nonplanarity:g}")
    per_panel = {"min_angle_deg": np.min(angles, axis=1), "max_angle_deg": np.max(angles, axis=1),
                 "edge_ratio": edge_ratio, "size": size, "deviation": dev,
                 "degenerate": degenerate, "flipped": flipped}
    return PanelQualityReport(stage, m, per_panel, fails, thr.as_dict())


def bad_panel_mask(report: PanelQualityReport) -> np.ndarray:
    """Panels violating a per-panel limit (for visualisation)."""
    pp = report.per_panel
    thr = report.thresholds
    if not pp:
        return np.zeros(0, bool)
    return (pp["degenerate"] | pp["flipped"]
            | (pp["min_angle_deg"] < thr["min_angle_deg"])
            | (pp["max_angle_deg"] > thr["max_angle_deg"])
            | (pp["edge_ratio"] > thr["max_edge_ratio"]))


__all__ = ["PanelQualityThresholds", "PanelQualityReport", "panel_quality", "per_panel_geometry",
           "smooth_max", "bad_panel_mask", "unique_quad_edges",
           "DEFAULT_M2D_THRESHOLDS", "DEFAULT_M3D_THRESHOLDS", "DEFAULT_K3D_THRESHOLDS"]
