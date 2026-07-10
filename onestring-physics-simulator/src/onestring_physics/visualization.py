from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .design_optimizer import DesignResult
from .onestring_pipeline import (
    FlatTileLayout,
    GapGraph,
    HingeGraph,
    OneStringDesignState,
    QuadMesh,
    StringPath,
    SurfaceMesh,
    TileAssembly,
    _surface_peak_uvs,
)


def figure_target(design: DesignResult) -> go.Figure:
    grid = design.target.sample_grid(design.grid.nx, design.grid.ny, design.grid.tile_size)
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=grid[..., 0],
            y=grid[..., 1],
            z=grid[..., 2],
            colorscale="Viridis",
            opacity=0.72,
            showscale=False,
            name="target",
        )
    )
    _style_scene(fig)
    return fig


def figure_tiles(
    tiles: np.ndarray,
    design: DesignResult | None = None,
    rope: np.ndarray | None = None,
    pull_handle: np.ndarray | None = None,
    title: str = "tiles",
) -> go.Figure:
    fig = go.Figure()
    add_tiles(fig, tiles)
    if design is not None:
        add_hinges(fig, tiles, design)
    if rope is not None:
        add_rope(fig, rope)
    if pull_handle is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[pull_handle[0]],
                y=[pull_handle[1]],
                z=[pull_handle[2]],
                mode="markers",
                marker=dict(size=7, color="#b11226"),
                name="pull handle",
            )
        )
    fig.update_layout(title=title)
    _style_scene(fig)
    return fig


def figure_comparison(design: DesignResult, final_tiles: np.ndarray) -> go.Figure:
    fig = go.Figure()
    add_tiles(fig, design.assembled_tiles, color="#3b82f6", opacity=0.38, name="optimized assembled")
    add_tiles(fig, final_tiles, color="#f97316", opacity=0.72, name="physical final")
    _style_scene(fig)
    return fig


def figure_loss(loss_history: list[float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=loss_history, mode="lines+markers", name="mean squared residual"))
    fig.update_layout(xaxis_title="recorded step", yaxis_title="loss", height=280)
    return fig


def figure_pipeline_overview(state: OneStringDesignState) -> go.Figure:
    fig = go.Figure()
    nodes = {
        "S": (0, 1),
        "M3D": (1, 1),
        "K3D": (2, 1),
        "T3D": (3, 1),
        "Omega": (0, 0),
        "M2D": (1, 0),
        "K2D": (2, 0),
        "T2D top hinge": (3, 0),
        "T2D dual hinge": (4, 0),
        "Lift Points": (5, 0.35),
        "String Path": (6, 0.35),
        "Actuation": (7, 0.35),
        "Final": (8, 0.35),
    }
    edges = [
        ("S", "M3D"),
        ("Omega", "M2D"),
        ("M2D", "M3D"),
        ("M3D", "K3D"),
        ("K3D", "T3D"),
        ("M2D", "K2D"),
        ("K2D", "T2D top hinge"),
        ("T2D top hinge", "T2D dual hinge"),
        ("T2D dual hinge", "Lift Points"),
        ("Lift Points", "String Path"),
        ("String Path", "Actuation"),
        ("Actuation", "Final"),
        ("T3D", "Actuation"),
    ]
    for a, b in edges:
        xa, ya = nodes[a]
        xb, yb = nodes[b]
        fig.add_trace(
            go.Scatter(
                x=[xa, xb],
                y=[ya, yb],
                mode="lines",
                line=dict(color="#6b7280", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    labels = list(nodes)
    x = [nodes[label][0] for label in labels]
    y = [nodes[label][1] for label in labels]
    hover = [_stage_hover_text(state, label) for label in labels]
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            marker=dict(size=28, color="#0f766e", line=dict(color="#0f172a", width=1)),
            text=labels,
            textposition="bottom center",
            hovertext=hover,
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Fig. 5-style OneString pipeline",
        height=360,
        margin=dict(l=20, r=20, t=44, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, range=(-0.35, 1.35)),
        plot_bgcolor="white",
    )
    return fig


def figure_surface_mesh(surface: SurfaceMesh, title: str = "S target surface") -> go.Figure:
    fig = go.Figure()
    _add_quad_mesh_surface(fig, surface.vertices, surface.faces, color="#7c3aed", opacity=0.55, name="S")
    fig.update_layout(title=title)
    _style_scene(fig)
    return fig


def _split_lines_for_display(state: OneStringDesignState) -> list[tuple[str, float]]:
    lines = list(getattr(state.mesh_2d_initial, "split_lines", []) or [])
    if lines:
        return lines
    return list(getattr(state.conformal_domain, "split_lines", []) or [])


def _split_samples_on_parameterization(state: OneStringDesignState) -> tuple[np.ndarray, np.ndarray]:
    parameterization = state.surface_parameterization
    uv_vertices = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    surface_vertices = np.asarray(parameterization.surface_vertices_3d, dtype=float)
    uv_faces = np.asarray(parameterization.uv_faces, dtype=int)
    surface_faces = np.asarray(parameterization.surface_faces, dtype=int)
    lines = _split_lines_for_display(state)
    if not lines or len(uv_vertices) == 0 or len(surface_vertices) == 0:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 3), dtype=float)

    samples_uv: list[np.ndarray] = []
    samples_s: list[np.ndarray] = []
    for axis, value in lines:
        coord = 1 if axis == "row" else 0
        for uv_face, surface_face in zip(uv_faces, surface_faces):
            tri_uv = uv_vertices[np.asarray(uv_face, dtype=int)]
            tri_s = surface_vertices[np.asarray(surface_face, dtype=int)]
            for a, b in ((0, 1), (1, 2), (2, 0)):
                va = float(tri_uv[a, coord] - value)
                vb = float(tri_uv[b, coord] - value)
                if abs(va) <= 1e-12 and abs(vb) <= 1e-12:
                    for t in (0.0, 1.0):
                        samples_uv.append((1.0 - t) * tri_uv[a] + t * tri_uv[b])
                        samples_s.append((1.0 - t) * tri_s[a] + t * tri_s[b])
                    continue
                if va * vb > 0.0:
                    continue
                denom = va - vb
                if abs(denom) <= 1e-12:
                    continue
                t = float(np.clip(va / denom, 0.0, 1.0))
                samples_uv.append((1.0 - t) * tri_uv[a] + t * tri_uv[b])
                samples_s.append((1.0 - t) * tri_s[a] + t * tri_s[b])

    if not samples_uv:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 3), dtype=float)
    uv = np.asarray(samples_uv, dtype=float)
    xyz = np.asarray(samples_s, dtype=float)
    rounded = np.round(uv, 8)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    return uv[unique_idx], xyz[unique_idx]


def _high_csf_vertices(state: OneStringDesignState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parameterization = state.surface_parameterization
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    xyz = np.asarray(parameterization.surface_vertices_3d, dtype=float)
    csf = np.asarray(getattr(state.conformal_domain, "csf_values", np.zeros(0)), dtype=float)
    threshold = float(state.mesh_2d_initial.metrics.get("csf_split_threshold", 2.0))
    if len(uv) == 0 or len(xyz) == 0 or len(csf) != len(uv):
        return np.zeros((0, 2), dtype=float), np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
    mask = csf > threshold
    return uv[mask], xyz[mask], csf[mask]


def _residual_high_csf_vertices(state: OneStringDesignState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parameterization = state.surface_parameterization
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    xyz = np.asarray(parameterization.surface_vertices_3d, dtype=float)
    csf = np.asarray(getattr(state.conformal_domain, "csf_values", np.zeros(0)), dtype=float)
    indices = state.mesh_2d_initial.metrics.get("csf_split_residual_high_vertex_indices_after_all", [])
    try:
        ids = np.asarray([int(i) for i in indices], dtype=int)
    except Exception:
        ids = np.zeros(0, dtype=int)
    ids = ids[(ids >= 0) & (ids < len(uv)) & (ids < len(xyz)) & (ids < len(csf))]
    if len(ids) == 0:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
    return uv[ids], xyz[ids], csf[ids]


def _surface_peak_markers(state: OneStringDesignState) -> tuple[np.ndarray, np.ndarray]:
    parameterization = state.surface_parameterization
    peak_uv = _surface_peak_uvs(parameterization)
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    xyz = np.asarray(parameterization.surface_vertices_3d, dtype=float)
    if len(peak_uv) == 0 or len(uv) == 0 or len(xyz) != len(uv):
        return np.zeros((0, 2), dtype=float), np.zeros((0, 3), dtype=float)
    peak_xyz = []
    for point in peak_uv:
        idx = int(np.argmin(np.linalg.norm(uv - point, axis=1)))
        peak_xyz.append(xyz[idx])
    return peak_uv, np.asarray(peak_xyz, dtype=float)


def figure_split_mapping(state: OneStringDesignState) -> go.Figure:
    """Show where CSF-based splitting is detected on S and how it maps to Omega."""
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=("S: high-stretch region and mapped cut", "Omega: mapped cut line"),
        horizontal_spacing=0.06,
    )

    surface = state.target_surface
    _add_quad_mesh_surface(
        fig,
        surface.vertices,
        surface.faces,
        color="#94a3b8",
        opacity=0.32,
        name="S",
        row=1,
        col=1,
    )

    high_uv, high_s, high_csf = _high_csf_vertices(state)
    residual_uv, residual_s, residual_csf = _residual_high_csf_vertices(state)
    peak_uv, peak_s = _surface_peak_markers(state)
    if len(high_s):
        fig.add_trace(
            go.Scatter3d(
                x=high_s[:, 0],
                y=high_s[:, 1],
                z=high_s[:, 2],
                mode="markers",
                marker=dict(size=5, color=high_csf, colorscale="Inferno", colorbar=dict(title="CSF")),
                name="S vertices with CSF > threshold",
            ),
            row=1,
            col=1,
        )

    cut_uv, cut_s = _split_samples_on_parameterization(state)
    if len(cut_s):
        fig.add_trace(
            go.Scatter3d(
                x=cut_s[:, 0],
                y=cut_s[:, 1],
                z=cut_s[:, 2],
                mode="markers",
                marker=dict(size=5, color="#ef4444"),
                name="S mapped split samples",
            ),
            row=1,
            col=1,
        )
    if len(residual_s):
        fig.add_trace(
            go.Scatter3d(
                x=residual_s[:, 0],
                y=residual_s[:, 1],
                z=residual_s[:, 2],
                mode="markers",
                marker=dict(size=4, color=residual_csf, colorscale="Viridis", symbol="circle-open"),
                name="S residual CSF after split steps",
            ),
            row=1,
            col=1,
        )
    if len(peak_s):
        fig.add_trace(
            go.Scatter3d(
                x=peak_s[:, 0],
                y=peak_s[:, 1],
                z=peak_s[:, 2],
                mode="markers",
                marker=dict(size=8, color="#facc15", symbol="diamond"),
                name="S detected peaks",
            ),
            row=1,
            col=1,
        )

    boundary = state.conformal_domain.boundary
    fig.add_trace(
        go.Scatter(
            x=boundary[:, 0],
            y=boundary[:, 1],
            mode="lines",
            line=dict(color="#0f172a", width=3),
            name="Omega boundary",
        ),
        row=1,
        col=2,
    )

    mesh = state.mesh_2d_initial
    grid_x: list[float | None] = []
    grid_y: list[float | None] = []
    for face in mesh.faces:
        pts = mesh.vertices[list(face) + [face[0]], :2]
        grid_x.extend([*pts[:, 0].tolist(), None])
        grid_y.extend([*pts[:, 1].tolist(), None])
    fig.add_trace(
        go.Scatter(
            x=grid_x,
            y=grid_y,
            mode="lines",
            line=dict(color="#14b8a6", width=1),
            name="M2D grid in Omega",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    if len(high_uv):
        fig.add_trace(
            go.Scatter(
                x=high_uv[:, 0],
                y=high_uv[:, 1],
                mode="markers",
                marker=dict(size=7, color=high_csf, colorscale="Inferno", showscale=False),
                name="Omega vertices with CSF > threshold",
            ),
            row=1,
            col=2,
        )
    if len(cut_uv):
        fig.add_trace(
            go.Scatter(
                x=cut_uv[:, 0],
                y=cut_uv[:, 1],
                mode="markers",
                marker=dict(size=6, color="#ef4444"),
                name="Omega split samples",
            ),
            row=1,
            col=2,
        )
    if len(residual_uv):
        fig.add_trace(
            go.Scatter(
                x=residual_uv[:, 0],
                y=residual_uv[:, 1],
                mode="markers",
                marker=dict(size=8, color=residual_csf, colorscale="Viridis", symbol="circle-open"),
                name="Omega residual CSF after split steps",
            ),
            row=1,
            col=2,
        )
    if len(peak_uv):
        fig.add_trace(
            go.Scatter(
                x=peak_uv[:, 0],
                y=peak_uv[:, 1],
                mode="markers",
                marker=dict(size=10, color="#facc15", symbol="diamond"),
                name="Omega detected peaks",
            ),
            row=1,
            col=2,
        )

    omega_min = np.nanmin(boundary, axis=0) if len(boundary) else np.array([0.0, 0.0])
    omega_max = np.nanmax(boundary, axis=0) if len(boundary) else np.array([1.0, 1.0])
    for axis, value in _split_lines_for_display(state):
        if axis == "row":
            fig.add_trace(
                go.Scatter(
                    x=[float(omega_min[0]), float(omega_max[0])],
                    y=[float(value), float(value)],
                    mode="lines",
                    line=dict(color="#ef4444", dash="dash", width=2),
                    name="Omega split line",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[float(value), float(value)],
                    y=[float(omega_min[1]), float(omega_max[1])],
                    mode="lines",
                    line=dict(color="#ef4444", dash="dash", width=2),
                    name="Omega split line",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

    fig.update_layout(
        title="CSF split correspondence: S -> Omega -> M2D",
        height=620,
        margin=dict(l=20, r=20, t=70, b=20),
        scene=dict(aspectmode="data"),
        xaxis_title="u",
        yaxis_title="v",
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    return fig


def figure_quad_mesh(mesh: QuadMesh, title: str | None = None, show_csf: bool = False) -> go.Figure:
    fig = go.Figure()
    color = "#3b82f6" if mesh.vertices.shape[1] == 3 and np.ptp(mesh.vertices[:, 2]) > 1e-8 else "#14b8a6"
    _add_quad_mesh_surface(fig, mesh.vertices, mesh.faces, color=color, opacity=0.72, name=mesh.stage)
    if show_csf:
        fig.add_trace(
            go.Scatter3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2] + 0.01,
                mode="markers",
                marker=dict(size=5, color=np.asarray(list(mesh.metrics.values())[: len(mesh.vertices)] or [0]), colorscale="Viridis"),
                name="CSF sample",
                showlegend=False,
            )
        )
    fig.update_layout(title=title or mesh.stage)
    _style_scene(fig)
    return fig


def figure_m3d_overlay(state: OneStringDesignState) -> go.Figure:
    fig = go.Figure()
    _add_quad_mesh_surface(fig, state.target_surface.vertices, state.target_surface.faces, color="#94a3b8", opacity=0.28, name="target surface S")
    _add_quad_mesh_surface(fig, state.mesh_3d_initial.vertices, state.mesh_3d_initial.faces, color="#ef4444", opacity=0.82, name="M3D c^-1 grid")
    metrics = state.mesh_3d_initial.metrics
    failure_ids = np.asarray(metrics.get("m3d_uv_lookup_failure_vertex_ids", []), dtype=int)
    failure_ids = failure_ids[(failure_ids >= 0) & (failure_ids < len(state.mesh_3d_initial.vertices))]
    if len(failure_ids):
        points = state.mesh_3d_initial.vertices[failure_ids]
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2] + 0.03,
                mode="markers",
                marker=dict(size=7, color="#dc2626", symbol="x"),
                name="UV lookup failures",
            )
        )
    outside_ids = np.asarray(metrics.get("m3d_outside_omega_vertex_ids", []), dtype=int)
    outside_ids = outside_ids[(outside_ids >= 0) & (outside_ids < len(state.mesh_3d_initial.vertices))]
    if len(outside_ids):
        points = state.mesh_3d_initial.vertices[outside_ids]
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2] + 0.045,
                mode="markers",
                marker=dict(size=6, color="#f59e0b", symbol="diamond"),
                name="outside Omega / clamped lookup",
            )
        )
    hit_counts = np.asarray(metrics.get("m3d_surface_triangle_hit_counts", []), dtype=float)
    target_faces = np.asarray(state.target_surface.faces, dtype=int)[:, :3]
    if len(hit_counts) == len(target_faces) and np.any(hit_counts > 0):
        hit_ids = np.flatnonzero(hit_counts > 0)
        centers = np.mean(np.asarray(state.target_surface.vertices, dtype=float)[target_faces[hit_ids]], axis=1)
        fig.add_trace(
            go.Scatter3d(
                x=centers[:, 0],
                y=centers[:, 1],
                z=centers[:, 2] + 0.02,
                mode="markers",
                marker=dict(
                    size=4,
                    color=hit_counts[hit_ids],
                    colorscale="Plasma",
                    colorbar=dict(title="inverse-map hits"),
                    opacity=0.82,
                ),
                text=[f"surface triangle {triangle_id}<br>hits={int(hit_counts[triangle_id])}" for triangle_id in hit_ids],
                hoverinfo="text",
                name="surface triangle usage",
            )
        )
    fig.update_layout(title="M3D inverse parameterization overlay on target surface")
    _style_scene(fig)
    return fig


def figure_flat_tile_layout(
    layout: FlatTileLayout,
    title: str = "K2D flat tile layout",
    hinge_graph: HingeGraph | None = None,
) -> go.Figure:
    fig = go.Figure()
    tiles = layout.tile_top_vertices_3d
    x: list[float] = []
    y: list[float] = []
    z: list[float] = []
    i_idx: list[int] = []
    j_idx: list[int] = []
    k_idx: list[int] = []
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    for tile in tiles:
        base = len(x)
        x.extend(tile[:, 0].tolist())
        y.extend(tile[:, 1].tolist())
        z.extend(tile[:, 2].tolist())
        i_idx.extend([base, base])
        j_idx.extend([base + 1, base + 2])
        k_idx.extend([base + 2, base + 3])
        closed = np.vstack([tile, tile[0]])
        edge_x.extend([*closed[:, 0].tolist(), None])
        edge_y.extend([*closed[:, 1].tolist(), None])
        edge_z.extend([*closed[:, 2].tolist(), None])
    fig.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i_idx,
            j=j_idx,
            k=k_idx,
            color="#2dd4bf",
            opacity=0.78,
            flatshading=True,
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
            name="K2D independent tile faces",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(color="#111827", width=3),
            name="tile gaps / edges",
        )
    )
    if hinge_graph is not None:
        add_hinge_markers(
            fig,
            TileAssembly(
                vertices=np.concatenate([tiles, tiles], axis=1),
                top_faces=np.asarray([[0, 1, 2, 3] for _ in range(len(tiles))], dtype=int),
                bottom_faces=np.asarray([[4, 7, 6, 5] for _ in range(len(tiles))], dtype=int),
                side_faces=np.asarray([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]], dtype=int),
                stage="K2D layout hinge overlay",
            ),
            hinge_graph,
        )
    if layout.gap_polygons:
        centers = np.asarray([np.mean(poly, axis=0) for poly in layout.gap_polygons], dtype=float)
        fig.add_trace(
            go.Scatter3d(
                x=centers[:, 0],
                y=centers[:, 1],
                z=np.full(len(centers), 0.035),
                mode="markers",
                marker=dict(size=4, color="#f97316"),
                name="gap centers",
            )
        )
    fig.update_layout(title=title)
    _style_scene(fig)
    return fig


def figure_domain(state: OneStringDesignState) -> go.Figure:
    fig = go.Figure()
    parameterization = state.surface_parameterization
    metrics = getattr(parameterization, "metrics", {}) or {}
    target_corners = np.asarray(metrics.get("boundary_target_corners", np.zeros((0, 2))), dtype=float)
    if target_corners.shape == (4, 2):
        target_boundary = np.vstack([target_corners, target_corners[0]])
        fig.add_trace(
            go.Scatter(
                x=target_boundary[:, 0],
                y=target_boundary[:, 1],
                mode="lines",
                line=dict(color="#111827", width=3, dash="dash"),
                name="target rectangle",
            )
        )

    boundary = np.asarray(parameterization.omega_boundary, dtype=float)
    if boundary.size == 0:
        boundary = np.asarray(state.conformal_domain.boundary, dtype=float)
    fig.add_trace(
        go.Scatter(
            x=boundary[:, 0],
            y=boundary[:, 1],
            mode="lines",
            line=dict(color="#0f766e", width=3),
            name="final Omega boundary",
        )
    )

    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    uv_faces = np.asarray(parameterization.uv_faces, dtype=int)[:, :3]
    face_distortion = np.asarray(metrics.get("uv_face_angle_distortion_deg", []), dtype=float)
    if len(uv_faces) and len(face_distortion) == len(uv_faces):
        centers = np.mean(uv[uv_faces], axis=1)
        fig.add_trace(
            go.Scattergl(
                x=centers[:, 0],
                y=centers[:, 1],
                mode="markers",
                marker=dict(
                    size=7,
                    color=face_distortion,
                    colorscale="Viridis",
                    colorbar=dict(title="angle error (deg)"),
                    opacity=0.78,
                ),
                text=[f"triangle {index}<br>mean angle error={value:.4g} deg" for index, value in enumerate(face_distortion)],
                hoverinfo="text",
                name="angle distortion",
            )
        )

    boundary_loop = [int(value) for value in (metrics.get("boundary_loop", []) or [])]
    if boundary_loop and max(boundary_loop) < len(uv):
        boundary_vertices = uv[np.asarray(boundary_loop, dtype=int)]
        fig.add_trace(
            go.Scatter(
                x=boundary_vertices[:, 0],
                y=boundary_vertices[:, 1],
                mode="markers",
                marker=dict(
                    size=7,
                    color=np.arange(len(boundary_vertices)),
                    colorscale="Turbo",
                    showscale=False,
                    line=dict(color="#ffffff", width=0.5),
                ),
                text=[f"boundary order {index}<br>vertex {vertex_id}" for index, vertex_id in enumerate(boundary_loop)],
                hoverinfo="text",
                name="boundary correspondence order",
            )
        )
        corner_ids = [int(value) for value in (metrics.get("boundary_corner_vertex_ids", []) or [])]
        valid_corner_ids = [value for value in corner_ids if 0 <= value < len(uv)]
        if valid_corner_ids:
            corner_uv = uv[np.asarray(valid_corner_ids, dtype=int)]
            fig.add_trace(
                go.Scatter(
                    x=corner_uv[:, 0],
                    y=corner_uv[:, 1],
                    mode="markers+text",
                    marker=dict(size=14, color="#dc2626", symbol="star", line=dict(color="#7f1d1d", width=1)),
                    text=[f"C{index}" for index in range(len(valid_corner_ids))],
                    textposition="top center",
                    name="corner anchors",
                )
            )

    flip_ids = [int(value) for value in (metrics.get("uv_flip_triangle_ids", []) or [])]
    for display_index, triangle_id in enumerate(flip_ids):
        if not (0 <= triangle_id < len(uv_faces)):
            continue
        triangle = uv[uv_faces[triangle_id]]
        triangle = np.vstack([triangle, triangle[0]])
        fig.add_trace(
            go.Scatter(
                x=triangle[:, 0],
                y=triangle[:, 1],
                mode="lines",
                fill="toself",
                fillcolor="rgba(220,38,38,0.45)",
                line=dict(color="#991b1b", width=2),
                name="flipped UV triangle" if display_index == 0 else "flipped UV triangle",
                showlegend=display_index == 0,
            )
        )

    mesh = state.mesh_2d_initial
    grid_x: list[float | None] = []
    grid_y: list[float | None] = []
    for face in mesh.faces:
        pts = mesh.vertices[list(face) + [face[0]], :2]
        grid_x.extend([*pts[:, 0].tolist(), None])
        grid_y.extend([*pts[:, 1].tolist(), None])
    fig.add_trace(
        go.Scatter(
            x=grid_x,
            y=grid_y,
            mode="lines",
            line=dict(color="#14b8a6", width=1),
            name="regular quad grid",
            showlegend=False,
        )
    )
    for axis, value in state.conformal_domain.split_lines:
        if axis == "row":
            fig.add_hline(y=value, line=dict(color="#ef4444", dash="dash"))
        else:
            fig.add_vline(x=value, line=dict(color="#ef4444", dash="dash"))
    fig.update_layout(
        title="Omega boundary correspondence and distortion diagnostics",
        height=620,
        margin=dict(l=20, r=40, t=44, b=120),
        xaxis_title="u",
        yaxis_title="v",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", x=0.0, y=-0.18, xanchor="left", yanchor="top"),
    )
    return fig


def figure_tile_assembly(
    assembly: TileAssembly,
    title: str | None = None,
    gap_graph: GapGraph | None = None,
    hinge_graph: HingeGraph | None = None,
    string_path: StringPath | None = None,
    lift_gap_ids: list[int] | None = None,
) -> go.Figure:
    fig = go.Figure()
    add_tile_assembly(fig, assembly)
    if hinge_graph is not None:
        add_hinge_markers(fig, assembly, hinge_graph)
    if gap_graph is not None:
        add_gap_graph(fig, gap_graph, string_path=string_path, lift_gap_ids=lift_gap_ids)
    fig.update_layout(title=title or assembly.stage)
    _style_scene(fig)
    return fig


def figure_onestring_comparison(state: OneStringDesignState) -> go.Figure:
    fig = go.Figure()
    add_tile_assembly(fig, state.tiles_3d, color="#2563eb", opacity=0.32, name="T3D target")
    if state.simulation_result is not None:
        final = TileAssembly(
            vertices=state.simulation_result.final_tiles,
            top_faces=state.tiles_3d.top_faces,
            bottom_faces=state.tiles_3d.bottom_faces,
            side_faces=state.tiles_3d.side_faces,
            stage="Final simulated deployed",
        )
        add_tile_assembly(fig, final, color="#f97316", opacity=0.7, name="final simulated")
    _style_scene(fig)
    return fig


def add_tile_assembly(
    fig: go.Figure,
    assembly: TileAssembly,
    color: str = "#2dd4bf",
    opacity: float = 0.72,
    name: str = "tiles",
) -> None:
    metrics = getattr(assembly, "metrics", {}) or {}
    if bool(metrics.get("t3d_intersection_trim_applied", False)):
        render_vertices = np.asarray(metrics.get("t3d_trimmed_render_vertices", np.zeros((0, 3))), dtype=float)
        render_i = np.asarray(metrics.get("t3d_trimmed_render_i", np.zeros(0)), dtype=int)
        render_j = np.asarray(metrics.get("t3d_trimmed_render_j", np.zeros(0)), dtype=int)
        render_k = np.asarray(metrics.get("t3d_trimmed_render_k", np.zeros(0)), dtype=int)
        if len(render_vertices) and len(render_i) and len(render_i) == len(render_j) == len(render_k):
            lighting = dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0)
            fig.add_trace(
                go.Mesh3d(
                    x=render_vertices[:, 0],
                    y=render_vertices[:, 1],
                    z=render_vertices[:, 2],
                    i=render_i,
                    j=render_j,
                    k=render_k,
                    color=color,
                    opacity=opacity,
                    flatshading=True,
                    lighting=lighting,
                    name=f"{name} (trimmed)",
                    showlegend=True,
                )
            )
            _add_tile_edges(fig, assembly)
            return

    faces = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    side_face_edges = [None, None, 0, 1, 2, 3]
    hidden_side_edges = {
        (int(tile_id), int(edge_id))
        for tile_id, edge_id in metrics.get("split_contact_side_edges", [])
    }
    lighting = dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0)
    x: list[float] = []
    y: list[float] = []
    z: list[float] = []
    i_idx: list[int] = []
    j_idx: list[int] = []
    k_idx: list[int] = []
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    for tile_id, tile in enumerate(np.asarray(assembly.vertices, dtype=float)):
        for face_id, face in enumerate(faces):
            side_edge = side_face_edges[face_id]
            if side_edge is not None and (int(tile_id), int(side_edge)) in hidden_side_edges:
                continue
            base = len(x)
            pts = tile[list(face)]
            x.extend(pts[:, 0].tolist())
            y.extend(pts[:, 1].tolist())
            z.extend(pts[:, 2].tolist())
            i_idx.extend([base, base])
            j_idx.extend([base + 1, base + 2])
            k_idx.extend([base + 2, base + 3])
        for edge_id, edge in enumerate([(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]):
            pts = tile[list(edge)]
            edge_x.extend([pts[0, 0], pts[1, 0], None])
            edge_y.extend([pts[0, 1], pts[1, 1], None])
            edge_z.extend([pts[0, 2], pts[1, 2], None])
    fig.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i_idx,
            j=j_idx,
            k=k_idx,
            color=color,
            opacity=opacity,
            flatshading=True,
            lighting=lighting,
            name=name,
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(color="#111827", width=2),
            name="tile edges",
            showlegend=False,
        )
    )


def _add_tile_edges(fig: go.Figure, assembly: TileAssembly) -> None:
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    for tile in np.asarray(assembly.vertices, dtype=float):
        for edge in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
            pts = tile[list(edge)]
            edge_x.extend([pts[0, 0], pts[1, 0], None])
            edge_y.extend([pts[0, 1], pts[1, 1], None])
            edge_z.extend([pts[0, 2], pts[1, 2], None])
    fig.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(color="#111827", width=2),
            name="tile edges",
            showlegend=False,
        )
    )


def add_hinge_markers(fig: go.Figure, assembly: TileAssembly, hinge_graph: HingeGraph) -> None:
    top_pts: list[np.ndarray] = []
    bottom_pts: list[np.ndarray] = []
    for hinge in hinge_graph.hinges:
        p = 0.5 * (
            assembly.vertices[hinge.tile_a, hinge.local_vertex_a]
            + assembly.vertices[hinge.tile_b, hinge.local_vertex_b]
        )
        if hinge.surface == "top":
            top_pts.append(p)
        else:
            bottom_pts.append(p)
    for name, pts, color, symbol in [
        ("top hinges", top_pts, "#2563eb", "circle"),
        ("bottom hinges", bottom_pts, "#f97316", "diamond"),
    ]:
        if not pts:
            continue
        arr = np.asarray(pts)
        fig.add_trace(
            go.Scatter3d(
                x=arr[:, 0],
                y=arr[:, 1],
                z=arr[:, 2],
                mode="markers",
                marker=dict(size=2.2, color=color, symbol=symbol, opacity=0.85),
                name=name,
            )
        )


def add_gap_graph(
    fig: go.Figure,
    gap_graph: GapGraph,
    string_path: StringPath | None = None,
    lift_gap_ids: list[int] | None = None,
) -> None:
    lift_gap_ids = lift_gap_ids or []
    centroids = {gap.id: gap.centroid_2d for gap in gap_graph.gaps}
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    for a, b in gap_graph.edges:
        pa = centroids[a]
        pb = centroids[b]
        edge_x.extend([pa[0], pb[0], None])
        edge_y.extend([pa[1], pb[1], None])
        edge_z.extend([pa[2] + 0.03, pb[2] + 0.03, None])
    fig.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(color="#94a3b8", width=2),
            name="gap graph",
            showlegend=False,
        )
    )
    gaps = gap_graph.gaps
    colors = ["#ef4444" if gap.id in lift_gap_ids else ("#f97316" if gap.boundary else "#475569") for gap in gaps]
    fig.add_trace(
        go.Scatter3d(
            x=[gap.centroid_2d[0] for gap in gaps],
            y=[gap.centroid_2d[1] for gap in gaps],
            z=[gap.centroid_2d[2] + 0.05 for gap in gaps],
            mode="markers+text",
            marker=dict(size=3, color=colors, opacity=0.85),
            text=[str(gap.label) for gap in gaps],
            textposition="top center",
            name="gaps (orange=boundary, red=LiftPoint)",
        )
    )
    if string_path is not None and string_path.gap_ids:
        pts = np.asarray([centroids[gap_id] for gap_id in string_path.gap_ids if gap_id in centroids], dtype=float)
        if len(pts):
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2] + 0.09,
                    mode="lines+markers",
                    line=dict(color="#dc2626", width=5),
                    marker=dict(size=2.5, color="#dc2626"),
                    name="string path",
                )
            )


def _add_quad_mesh_surface(
    fig: go.Figure,
    vertices: np.ndarray,
    faces: np.ndarray,
    color: str,
    opacity: float,
    name: str,
    row: int | None = None,
    col: int | None = None,
) -> None:
    lighting = dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0)
    x: list[float] = []
    y: list[float] = []
    z: list[float] = []
    i_idx: list[int] = []
    j_idx: list[int] = []
    k_idx: list[int] = []
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    for face in faces:
        pts = vertices[list(face)]
        base = len(x)
        x.extend(pts[:, 0].tolist())
        y.extend(pts[:, 1].tolist())
        z.extend(pts[:, 2].tolist())
        i_idx.extend([base, base])
        j_idx.extend([base + 1, base + 2])
        k_idx.extend([base + 2, base + 3])
        closed = np.vstack([pts, pts[0]])
        edge_x.extend([*closed[:, 0].tolist(), None])
        edge_y.extend([*closed[:, 1].tolist(), None])
        edge_z.extend([*closed[:, 2].tolist(), None])
    mesh_trace = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i_idx,
        j=j_idx,
        k=k_idx,
        color=color,
        opacity=opacity,
        flatshading=True,
        lighting=lighting,
        name=name,
        showlegend=True,
    )
    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(color="#0f172a", width=2),
        name=f"{name} edges",
        showlegend=False,
    )
    if row is None or col is None:
        fig.add_trace(mesh_trace)
        fig.add_trace(edge_trace)
    else:
        fig.add_trace(mesh_trace, row=row, col=col)
        fig.add_trace(edge_trace, row=row, col=col)


def _stage_hover_text(state: OneStringDesignState, label: str) -> str:
    if label == "S":
        return f"target surface<br>vertices: {len(state.target_surface.vertices)}<br>faces: {len(state.target_surface.faces)}"
    if label == "Omega":
        return f"{state.conformal_domain.method}<br>max CSF: {state.conformal_domain.max_csf:.3f}"
    report_map = {
        "M3D": "M2D -> M3D",
        "K3D": "M3D -> K3D",
        "T3D": "K3D -> T3D",
        "K2D": "M2D -> K2D",
        "T2D top hinge": "K2D -> T2D top hinge",
        "T2D dual hinge": "T2D top hinge -> T2D dual hinge",
    }
    if label in report_map and report_map[label] in state.stage_reports:
        report = state.stage_reports[report_map[label]]
        return (
            f"{report.objective}<br>before: {report.before_error:.4g}<br>"
            f"after: {report.after_error:.4g}<br>violation: {report.constraint_violation:.4g}"
        )
    if label == "Lift Points":
        return f"lift points: {len(state.lift_points)}<br>max GPE: {state.gap_graph.metrics.get('max_gpe', 0):.4g}"
    if label == "String Path":
        return f"route gaps: {len(state.string_path.gap_ids)}<br>turn angle: {state.string_path.turn_angle_total:.4g}"
    if label == "Final" and state.simulation_result is not None:
        return f"error to T3D: {state.simulation_result.metrics['final_deployment_error_to_T3D']:.4g}"
    return label


def add_tiles(
    fig: go.Figure,
    tiles: np.ndarray,
    color: str = "#2dd4bf",
    opacity: float = 0.78,
    name: str = "rigid tiles",
) -> None:
    first = True
    lighting = dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0)
    for tile in np.asarray(tiles, dtype=float):
        fig.add_trace(
            go.Mesh3d(
                x=tile[:, 0],
                y=tile[:, 1],
                z=tile[:, 2],
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color=color,
                opacity=opacity,
                flatshading=False,
                lighting=lighting,
                name=name,
                showscale=False,
                showlegend=first,
            )
        )
        closed = np.vstack([tile, tile[0]])
        fig.add_trace(
            go.Scatter3d(
                x=closed[:, 0],
                y=closed[:, 1],
                z=closed[:, 2],
                mode="lines",
                line=dict(color="#111827", width=3),
                name="tile edges",
                showlegend=False,
            )
        )
        first = False


def add_hinges(fig: go.Figure, tiles: np.ndarray, design: DesignResult) -> None:
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for hinge in design.hinges:
        a = (tiles[hinge.tile_a, hinge.corner_a0] + tiles[hinge.tile_a, hinge.corner_a1]) * 0.5
        b = (tiles[hinge.tile_b, hinge.corner_b0] + tiles[hinge.tile_b, hinge.corner_b1]) * 0.5
        xs += [a[0], b[0], None]
        ys += [a[1], b[1], None]
        zs += [a[2], b[2], None]
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines+markers",
            line=dict(color="#4b5563", width=4),
            marker=dict(size=3, color="#4b5563"),
            name="hinges",
        )
    )


def add_rope(fig: go.Figure, rope: np.ndarray) -> None:
    rope = np.asarray(rope, dtype=float)
    closed = np.vstack([rope, rope[0]]) if len(rope) > 2 else rope
    fig.add_trace(
        go.Scatter3d(
            x=closed[:, 0],
            y=closed[:, 1],
            z=closed[:, 2],
            mode="lines+markers",
            line=dict(color="#dc2626", width=5),
            marker=dict(size=4, color="#ef4444"),
            name="rope",
        )
    )


def _style_scene(fig: go.Figure) -> None:
    fig.update_layout(
        height=620,
        margin=dict(l=0, r=0, b=0, t=38),
        uirevision="onestring-camera-preserved",
        scene=dict(
            aspectmode="data",
            uirevision="onestring-camera-preserved",
            dragmode="orbit",
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        ),
    )
