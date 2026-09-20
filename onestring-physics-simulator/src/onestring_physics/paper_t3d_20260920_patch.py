"""Install the paper-aligned K3D -> T3D route from OneString Sec. 4.2.

This switch is intentionally narrow: it replaces only pipeline._extrude_tiles.
All Omega/M2D/M3D/K3D/K2D/T2D logic stays on the branch baseline.
"""
from __future__ import annotations

def install_paper_t3d_20260920_patch(pipeline):
    if getattr(pipeline, "_onestring_paper_t3d_20260920_installed", False):
        return pipeline
    from .paper_extrusion import extrude_paper_face_planarity

    previous = pipeline._extrude_tiles

    def paper_extrude(mesh, thickness: float, stage: str):
        # Avoid the helper's empty-mesh fallback recursing through this wrapper.
        if len(getattr(mesh, "faces", ())) == 0:
            return previous(mesh, thickness, stage)
        assembly, report = extrude_paper_face_planarity(
            mesh, thickness, stage, pipeline
        )
        assembly.metrics["paper_t3d_version"] = "2026-09-20-paper-t3d"
        assembly.metrics["paper_t3d_source"] = (
            "One String to Pull Them All, Sec. 4.2: normal-offset extrusion "
            "followed by Eq.(2) face-planarity optimization"
        )
        assembly.metrics["t3d_variable_topology_enabled"] = False
        assembly.metrics["t3d_authoritative_geometry"] = "8-vertex quadrilateral frustum"
        return assembly, report

    paper_extrude._onestring_previous = previous
    pipeline._extrude_tiles = paper_extrude
    pipeline._onestring_paper_t3d_20260920_installed = True
    return pipeline

__all__ = ["install_paper_t3d_20260920_patch"]
