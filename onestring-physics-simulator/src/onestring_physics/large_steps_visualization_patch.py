"""Streamlit display helpers for the automatic Large Steps conditioning stage."""
from __future__ import annotations

from typing import Any

import numpy as np


def _mesh_edge_lines(vertices: np.ndarray, faces: np.ndarray):
    """Return Plotly line-coordinate lists for the unique mesh edges."""
    verts = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)[:, :3]
    if len(verts) == 0 or len(tris) == 0:
        return [], [], []
    edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    for a, b in edges:
        p = verts[[int(a), int(b)]]
        x.extend([float(p[0, 0]), float(p[1, 0]), None])
        y.extend([float(p[0, 1]), float(p[1, 1]), None])
        z.extend([float(p[0, 2]), float(p[1, 2]), None])
    return x, y, z


def figure_large_steps_comparison(
    original_vertices: np.ndarray,
    conditioned_vertices: np.ndarray,
    faces: np.ndarray,
    metrics: dict[str, Any] | None = None,
):
    """Overlay the raw S wireframe and conditioned S mesh in one 3D figure."""
    import plotly.graph_objects as go
    from .visualization import _style_scene

    original = np.asarray(original_vertices, dtype=float)
    conditioned = np.asarray(conditioned_vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)[:, :3]
    report = dict(metrics or {})

    fig = go.Figure()
    if len(conditioned) and len(tris):
        fig.add_trace(
            go.Mesh3d(
                x=conditioned[:, 0],
                y=conditioned[:, 1],
                z=conditioned[:, 2],
                i=tris[:, 0],
                j=tris[:, 1],
                k=tris[:, 2],
                color="#2dd4bf",
                opacity=0.46,
                flatshading=True,
                lighting=dict(ambient=0.92, diffuse=0.22, specular=0.02, roughness=1.0),
                name="Conditioned S",
                showlegend=True,
            )
        )

    cx, cy, cz = _mesh_edge_lines(conditioned, tris)
    if cx:
        fig.add_trace(
            go.Scatter3d(
                x=cx,
                y=cy,
                z=cz,
                mode="lines",
                line=dict(color="#0f766e", width=2),
                opacity=0.72,
                name="Conditioned S edges",
                hoverinfo="skip",
            )
        )

    ox, oy, oz = _mesh_edge_lines(original, tris)
    if ox:
        fig.add_trace(
            go.Scatter3d(
                x=ox,
                y=oy,
                z=oz,
                mode="lines",
                line=dict(color="#111827", width=4),
                opacity=0.62,
                name="Original S wireframe",
                hoverinfo="skip",
            )
        )

    before_angle = float(report.get("large_steps_before_minimum_angle_degrees", 0.0))
    after_angle = float(report.get("large_steps_after_minimum_angle_degrees", 0.0))
    before_q = float(report.get("large_steps_before_triangle_quality_p05", 0.0))
    after_q = float(report.get("large_steps_after_triangle_quality_p05", 0.0))
    max_displacement = float(report.get("large_steps_vertex_displacement_max", 0.0))
    if not np.isfinite(max_displacement) or max_displacement == 0.0:
        if len(original) == len(conditioned) and len(original):
            max_displacement = float(np.max(np.linalg.norm(conditioned - original, axis=1)))
        else:
            max_displacement = 0.0
    fig.update_layout(
        title=(
            "Conditioned S vs Original S — Large Steps completed"
            f" | min angle {before_angle:.2f}° → {after_angle:.2f}°"
            f" | q05 {before_q:.3f} → {after_q:.3f}"
            f" | max Δx={max_displacement:.3g}"
        ),
        legend=dict(orientation="h", x=0.0, y=1.02, yanchor="bottom"),
    )
    _style_scene(fig)
    return fig


def render_large_steps_live_comparison(
    original_vertices: np.ndarray,
    conditioned_vertices: np.ndarray,
    faces: np.ndarray,
    metrics: dict[str, Any] | None = None,
) -> None:
    """Render the S/Conditioned-S comparison immediately after conditioning."""
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            return
    except Exception:
        return

    st.subheader("Large Steps result: Original S ↔ Conditioned S")
    st.caption(
        "This preview is emitted immediately after Large Steps finishes and before S → Ω optimization starts. "
        "Dark wireframe = original S; translucent surface = Conditioned S."
    )
    fig = figure_large_steps_comparison(
        original_vertices,
        conditioned_vertices,
        faces,
        metrics,
    )
    try:
        st.plotly_chart(fig, use_container_width=True)
    except TypeError:
        st.plotly_chart(fig)

    report = dict(metrics or {})
    st.caption(
        "Large Steps diagnostics: "
        f"min angle {float(report.get('large_steps_before_minimum_angle_degrees', 0.0)):.3f}° → "
        f"{float(report.get('large_steps_after_minimum_angle_degrees', 0.0)):.3f}°, "
        f"quality p05 {float(report.get('large_steps_before_triangle_quality_p05', 0.0)):.4f} → "
        f"{float(report.get('large_steps_after_triangle_quality_p05', 0.0)):.4f}, "
        f"surface deviation max={float(report.get('large_steps_surface_deviation_max', 0.0)):.3g}."
    )


def install_large_steps_visualization_patch() -> None:
    try:
        import streamlit as st
        from . import visualization
    except Exception:
        return
    if getattr(visualization, "_LARGE_STEPS_VISUALIZATION_PATCH_INSTALLED", False):
        return

    original_figure_surface_mesh = visualization.figure_surface_mesh

    def figure_surface_mesh_with_conditioning(surface: Any, *args: Any, **kwargs: Any):
        show_conditioned = bool(st.session_state.get("_onestring_show_conditioned_surface", False))
        if not show_conditioned:
            return original_figure_surface_mesh(surface, *args, **kwargs)

        conditioned = getattr(surface, "_large_steps_conditioned_vertices", None)
        metrics = getattr(surface, "_large_steps_conditioning_metrics", {}) or {}
        if conditioned is None:
            fig = original_figure_surface_mesh(surface, *args, **kwargs)
            try:
                fig.update_layout(title="Conditioned S unavailable for this run")
            except Exception:
                pass
            return fig

        fig = figure_large_steps_comparison(
            np.asarray(surface.vertices, dtype=float),
            np.asarray(conditioned, dtype=float),
            np.asarray(surface.faces, dtype=int),
            metrics,
        )
        try:
            st.caption(
                "Conditioned S comparison: dark wireframe is the original S. "
                "The translucent conditioned mesh uses identical connectivity and a fixed 3D boundary."
            )
        except Exception:
            pass
        return fig

    visualization.figure_surface_mesh = figure_surface_mesh_with_conditioning
    visualization._LARGE_STEPS_VISUALIZATION_PATCH_INSTALLED = True


__all__ = [
    "figure_large_steps_comparison",
    "render_large_steps_live_comparison",
    "install_large_steps_visualization_patch",
]
