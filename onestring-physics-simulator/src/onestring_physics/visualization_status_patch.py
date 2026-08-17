"""Unambiguous recovery-status rendering for authoritative T3D solids.

Plotly's default directional lighting can make cyan/blue recovery states look
nearly gray.  This patch keeps the original geometry and diagnostics, but uses
flat unlit status colors and adds one legend entry per status actually present.
The emergency prism remains gray and is additionally marked with a dashed red
outline so that it cannot be confused with a clipped manufacturing solid.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import plotly.graph_objects as go


STATUS_COLORS: dict[str, str] = {
    "T3D_OK_NOMINAL_FRUSTUM": "#16a34a",
    "T3D_RECOVERED_CAPPED_FRUSTUM": "#22d3ee",
    "T3D_RECOVERED_HALFSPACE_CLIP": "#0891b2",
    "T3D_RECOVERED_WEDGE": "#eab308",
    "T3D_RECOVERED_PYRAMID": "#f97316",
    "T3D_RECOVERED_LOCAL_THICKNESS": "#a855f7",
    "T3D_RECOVERED_SYNCHRONIZED_PAIR": "#0284c7",
    "T3D_RECOVERED_JUNCTION_CAP": "#4f46e5",
    "T3D_RECOVERED_GLOBAL_CLIP": "#1d4ed8",
    "T3D_RECOVERED_MESH_CLEANUP": "#0d9488",
    "T3D_RECOVERED_LEGACY_EMERGENCY_PRISM": "#6b7280",
}

STATUS_LABELS: dict[str, str] = {
    "T3D_OK_NOMINAL_FRUSTUM": "Nominal frustum",
    "T3D_RECOVERED_CAPPED_FRUSTUM": "Recovered: capped frustum",
    "T3D_RECOVERED_HALFSPACE_CLIP": "Recovered: half-space clip",
    "T3D_RECOVERED_WEDGE": "Recovered: wedge",
    "T3D_RECOVERED_PYRAMID": "Recovered: pyramid",
    "T3D_RECOVERED_LOCAL_THICKNESS": "Recovered: reduced thickness",
    "T3D_RECOVERED_SYNCHRONIZED_PAIR": "Recovered: synchronized pair",
    "T3D_RECOVERED_JUNCTION_CAP": "Recovered: junction cap",
    "T3D_RECOVERED_GLOBAL_CLIP": "Recovered: global collision clip",
    "T3D_RECOVERED_MESH_CLEANUP": "Recovered: mesh cleanup",
    "T3D_RECOVERED_LEGACY_EMERGENCY_PRISM": "WARNING: emergency normal prism",
}


def _triangulate_faces(faces: list[list[int]]) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    for face in faces:
        for index in range(1, len(face) - 1):
            triangles.append((int(face[0]), int(face[index]), int(face[index + 1])))
    return np.asarray(triangles, dtype=int)


def _edge_lines(vertices: np.ndarray, faces: list[list[int]]) -> tuple[list[float | None], list[float | None], list[float | None]]:
    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    seen: set[tuple[int, int]] = set()
    for face in faces:
        for index, vertex_id in enumerate(face):
            other = face[(index + 1) % len(face)]
            edge = tuple(sorted((int(vertex_id), int(other))))
            if edge in seen:
                continue
            seen.add(edge)
            points = vertices[np.asarray(edge, dtype=int)]
            x.extend([float(points[0, 0]), float(points[1, 0]), None])
            y.extend([float(points[0, 1]), float(points[1, 1]), None])
            z.extend([float(points[0, 2]), float(points[1, 2]), None])
    return x, y, z


def install_status_visualization_patch() -> None:
    """Patch ``visualization.add_tile_assembly`` once per interpreter."""
    from . import visualization

    if getattr(visualization, "_status_visualization_patch_installed", False):
        return

    original_add_tile_assembly = visualization.add_tile_assembly

    def add_tile_assembly_status_safe(
        fig: go.Figure,
        assembly,
        color: str = "#2dd4bf",
        opacity: float = 0.72,
        name: str = "tiles",
        show_recovery_status: bool = True,
        show_generated_cap_faces: bool = True,
        show_fundamental_failures_only: bool = False,
    ) -> None:
        authoritative_solids = getattr(assembly, "authoritative_solids", None)
        if not authoritative_solids:
            original_add_tile_assembly(
                fig,
                assembly,
                color=color,
                opacity=opacity,
                name=name,
                show_recovery_status=show_recovery_status,
                show_generated_cap_faces=show_generated_cap_faces,
                show_fundamental_failures_only=show_fundamental_failures_only,
            )
            return

        statuses = [str(getattr(solid, "recovery_status", "")) for solid in authoritative_solids]
        counts = Counter(statuses)
        shown_legend_statuses: set[str] = set()
        unlit = dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0)

        for tile_id, solid in enumerate(authoritative_solids):
            status = str(getattr(solid, "recovery_status", ""))
            if show_fundamental_failures_only and not status.startswith("T3D_FAILED_"):
                continue

            vertices = np.asarray(solid.vertices, dtype=float)
            triangles = _triangulate_faces(solid.faces)
            if len(vertices) == 0 or len(triangles) == 0:
                continue

            report = dict(getattr(solid, "metrics", {}) or {})
            reasons = list(getattr(solid, "recovery_reasons", []) or [])
            authoritative = bool(report.get("manufacturing_authoritative", status != "T3D_RECOVERED_LEGACY_EMERGENCY_PRISM"))
            hover = (
                f"tile={tile_id}<br>status={status}<br>"
                f"manufacturing_authoritative={authoritative}<br>"
                f"recovery={', '.join(reasons) or 'none'}<br>"
                f"volume={float(report.get('volume', 0.0)):.6g}<br>"
                f"min depth={float(report.get('actual_min_depth', 0.0)):.6g}<br>"
                f"max depth={float(report.get('actual_max_depth', 0.0)):.6g}<br>"
                f"vertices={len(vertices)}, faces={len(solid.faces)}<br>"
                f"collisions after={int(report.get('collision_count_after', 0))}"
            )

            if show_recovery_status:
                tile_color = "#dc2626" if status.startswith("T3D_FAILED_") else STATUS_COLORS.get(status, color)
            else:
                tile_color = color

            showlegend = bool(show_recovery_status and status not in shown_legend_statuses)
            if showlegend:
                shown_legend_statuses.add(status)
            legend_name = STATUS_LABELS.get(status, status or "Unknown status")
            if showlegend:
                legend_name = f"{legend_name} ({counts[status]})"

            fig.add_trace(
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=triangles[:, 0],
                    j=triangles[:, 1],
                    k=triangles[:, 2],
                    color=tile_color,
                    opacity=1.0 if show_recovery_status else opacity,
                    flatshading=True,
                    lighting=unlit,
                    name=legend_name if showlegend else f"tile {tile_id}: {status}",
                    legendgroup=status,
                    text=[hover] * len(vertices),
                    hoverinfo="text",
                    showlegend=showlegend,
                )
            )

            edge_x, edge_y, edge_z = _edge_lines(vertices, solid.faces)
            emergency = status == "T3D_RECOVERED_LEGACY_EMERGENCY_PRISM"
            fig.add_trace(
                go.Scatter3d(
                    x=edge_x,
                    y=edge_y,
                    z=edge_z,
                    mode="lines",
                    line=dict(color="#dc2626" if emergency else "#0f172a", width=5 if emergency else 2, dash="dash" if emergency else "solid"),
                    hoverinfo="skip",
                    name="Emergency prism outline" if emergency else f"tile {tile_id} edges",
                    legendgroup=status,
                    showlegend=False,
                )
            )

            if show_generated_cap_faces and int(report.get("cap_face_count", 0)) > 0:
                excluded = set(int(value) for value in solid.top_face_ids) | set(int(value) for value in solid.contact_face_by_edge.values())
                for face_id, face in enumerate(solid.faces):
                    if face_id in excluded:
                        continue
                    points = vertices[np.asarray(face, dtype=int)]
                    loop = np.vstack([points, points[0]])
                    fig.add_trace(
                        go.Scatter3d(
                            x=loop[:, 0],
                            y=loop[:, 1],
                            z=loop[:, 2],
                            mode="lines",
                            line=dict(color="#f43f5e", width=6),
                            hoverinfo="skip",
                            name="generated cap face",
                            showlegend=False,
                        )
                    )

        fig.update_layout(
            legend=dict(
                title="T3D recovery status",
                itemsizing="constant",
                groupclick="toggleitem",
            )
        )

    visualization.add_tile_assembly = add_tile_assembly_status_safe
    visualization._status_visualization_patch_installed = True
