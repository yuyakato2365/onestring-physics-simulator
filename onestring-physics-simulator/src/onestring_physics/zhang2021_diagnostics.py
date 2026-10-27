"""Diagnostics files for the 2026-10-27 Zhang2021 mode: candidate table, bad panels, seam mismatches.

Written to ``<repo>/output/zhang2021_diagnostics/<timestamp>/`` (or
``params.zhang2021_diagnostics_dir``) whenever candidates are rejected or a
verification fails, so a failure can be inspected instead of only read.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def _out_dir(params: Any) -> Path:
    base = str(getattr(params, "zhang2021_diagnostics_dir", "") or "").strip()
    root = Path(base) if base else Path(__file__).resolve().parents[2] / "output" / "zhang2021_diagnostics"
    d = root / time.strftime("%Y%m%d-%H%M%S")
    k = 0
    while d.exists():
        k += 1
        d = root / (time.strftime("%Y%m%d-%H%M%S") + f"-{k}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    return str(x)


def plot_panels(path: Path, uv: np.ndarray | None, xyz: np.ndarray, faces: np.ndarray, report: Any,
                title: str, seam_points: np.ndarray | None = None, seam_gap_edges: list | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .zhang2021_panel_quality import bad_panel_mask

    faces = np.asarray(faces, int)
    bad = bad_panel_mask(report) if report is not None else np.zeros(len(faces), bool)
    fig = plt.figure(figsize=(16, 8))
    if uv is not None:
        ax = fig.add_subplot(1, 2, 1)
        for i, f in enumerate(faces):
            poly = np.asarray(uv)[np.r_[f, f[0]]]
            ax.fill(poly[:, 0], poly[:, 1], color="#e8553d" if bad[i] else "#dde3ea", ec="k", lw=0.4)
        if seam_points is not None and len(seam_points):
            sp = np.asarray(seam_points)
            ax.plot(sp[:, 0], sp[:, 1], "o", ms=3, color="#1f6fd1")
        ax.set_aspect("equal")
        ax.set_title("M2D (UV) - red: panels violating a per-panel limit; blue: seam T-vertices")
    ax = fig.add_subplot(1, 2, 2 if uv is not None else 1, projection="3d")
    X = np.asarray(xyz, float)
    for i, f in enumerate(faces):
        poly = X[np.r_[f, f[0]]]
        ax.plot(poly[:, 0], poly[:, 1], poly[:, 2], color="#e8553d" if bad[i] else "0.35",
                lw=1.6 if bad[i] else 0.4)
    for a, b in (seam_gap_edges or []):
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="#1f6fd1", lw=2.0)
    ax.set_title(title)
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def write_candidate_diagnostics(rows: list[dict], plan: Any, params: Any) -> str:
    """Candidate table (JSON + text) and, if given, the best infeasible plan's bad panels."""
    try:
        d = _out_dir(params)
        (d / "candidates.json").write_text(json.dumps(_jsonable(rows), indent=1), encoding="utf-8")
        from .zhang2021_candidate_selection import summarize_row
        (d / "candidates.txt").write_text("\n".join(summarize_row(r) for r in rows), encoding="utf-8")
        if plan is not None:
            (d / "best_rejected_candidate_quality.json").write_text(json.dumps(_jsonable({
                "m2d": plan.m2d_report.metrics, "m2d_failures": plan.m2d_report.failures,
                "m3d": plan.m3d_report.metrics, "m3d_failures": plan.m3d_report.failures}), indent=1),
                encoding="utf-8")
            t = plan.metrics.get("zhang2021_onestring_seam_t_vertices", 0)
            try:
                plot_panels(d / "best_rejected_candidate_m3d.png", plan.quad_uv, plan.m3d_preview, plan.quads,
                        plan.m3d_report,
                        f"M3D of the best-scoring rejected candidate ({len(plan.quads)} quads, {t} seam T-vertices)")
            except ImportError as exc:
                (d / "figure_skipped.txt").write_text(f"figure not drawn: {exc} (pip install matplotlib)")
        return str(d)
    except Exception as exc:  # noqa: BLE001 - diagnostics must not mask the real failure
        return f"(diagnostics not written: {type(exc).__name__}: {exc})"


def write_verification_diagnostics(log: list[dict], state: Any, params: Any) -> str:
    try:
        d = _out_dir(params)
        (d / "verification.json").write_text(json.dumps(_jsonable(log), indent=1), encoding="utf-8")
        k3d = getattr(state, "mesh_3d_optimized", None) if state is not None else None
        if k3d is not None:
            from .zhang2021_panel_quality import panel_quality
            from .optcuts_zhang2021_quad_20261027 import stage_thresholds
            rep = panel_quality(np.asarray(k3d.vertices, float), np.asarray(k3d.faces, int), stage="K3D",
                                thresholds=stage_thresholds(params, "k3d"))
            m2d = getattr(state, "mesh_2d_initial", None)
            uv = np.asarray(m2d.vertices, float)[:, :2] if m2d is not None and len(m2d.vertices) == len(k3d.vertices) else None
            try:
                plot_panels(d / "k3d_bad_panels.png", uv, k3d.vertices, k3d.faces, rep,
                            "K3D - red: panels violating a per-panel limit")
            except ImportError as exc:
                (d / "figure_skipped.txt").write_text(f"figure not drawn: {exc} (pip install matplotlib)")
        return str(d)
    except Exception as exc:  # noqa: BLE001
        return f"(diagnostics not written: {type(exc).__name__}: {exc})"


def write_structured_grid_diagnostics(state: Any, summary: dict, params: Any) -> str:
    """2026-10-28 structured grid: numbers (JSON) and pictures (M2D/M3D, K3D) of every run."""
    try:
        base = str(getattr(params, "zhang2021_diagnostics_dir", "") or "").strip()
        root = Path(base) if base else Path(__file__).resolve().parents[2] / "output" / "structured_grid_diagnostics"
        d = root / time.strftime("%Y%m%d-%H%M%S")
        k = 0
        while d.exists():
            k += 1
            d = root / (time.strftime("%Y%m%d-%H%M%S") + f"-{k}")
        d.mkdir(parents=True, exist_ok=True)
        from .zhang2021_panel_quality import panel_quality
        from .optcuts_zhang2021_quad_20261027 import stage_thresholds
        pm = dict(getattr(getattr(state, "surface_parameterization", None), "metrics", {}) or {})
        plan = getattr(getattr(state, "surface_parameterization", None), "_zhang2021_plan", None)
        keep = {k: v for k, v in pm.items() if k.startswith(("structured_grid", "zhang2021_m2d_", "zhang2021_m3d_",
                                                             "zhang2021_seam", "zhang2021_onestring_sd",
                                                             "panel_optimizer"))}
        out = {"summary": summary, "m2d_m3d": keep}
        k3d = getattr(state, "mesh_3d_optimized", None)
        if k3d is not None:
            rep = panel_quality(np.asarray(k3d.vertices, float), np.asarray(k3d.faces, int), stage="K3D",
                                thresholds=stage_thresholds(params, "k3d"))
            out["k3d"] = {"metrics": rep.metrics, "limits_not_met": rep.failures}
        (d / "structured_grid_report.json").write_text(json.dumps(_jsonable(out), indent=1), encoding="utf-8")
        try:
            if plan is not None:
                plot_panels(d / "m2d_m3d.png", plan.quad_uv, plan.m3d_preview, plan.quads, plan.m3d_report,
                            f"M3D structured grid ({len(plan.quads)} quads) - red: limits not met (display only)")
            if k3d is not None:
                m2d = getattr(state, "mesh_2d_initial", None)
                uv = (np.asarray(m2d.vertices, float)[:, :2]
                      if m2d is not None and len(m2d.vertices) == len(k3d.vertices) else None)
                plot_panels(d / "k3d.png", uv, k3d.vertices, k3d.faces, rep,
                            "K3D - red: limits not met (display only)")
        except ImportError as exc:
            (d / "figure_skipped.txt").write_text(f"figure not drawn: {exc} (pip install matplotlib)")
        return str(d)
    except Exception as exc:  # noqa: BLE001
        return f"(diagnostics not written: {type(exc).__name__}: {exc})"


__all__ = ["write_structured_grid_diagnostics", "write_candidate_diagnostics", "write_verification_diagnostics", "plot_panels"]
