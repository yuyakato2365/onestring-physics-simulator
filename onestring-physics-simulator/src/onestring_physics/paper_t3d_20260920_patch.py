"""Install the paper-aligned K3D -> T3D route from OneString Sec. 4.2.

The patch also injects a visible MODEL_VERSIONS entry into the legacy Streamlit
app, so the active implementation is represented by the version selector rather
than only by the launcher name.
"""
from __future__ import annotations

VERSION_ID = "2026-09-20-paper-t3d"
VERSION_LABEL = "2026-09-20 — Paper-aligned T3D"
VERSION_DESCRIPTION = (
    "K3D→T3Dを元論文Sec.4.2方式へ変更: shared K3D mesh vertex normalで"
    "厚み分offsetし、固定8頂点/6 quad frustumのtop・bottom・contact facesを"
    "Eq.(2)のface-planarity optimizationで平面化します。"
)


def install_paper_t3d_20260920_patch(pipeline):
    if getattr(pipeline, "_onestring_paper_t3d_20260920_installed", False):
        return pipeline
    from .paper_extrusion import extrude_paper_face_planarity

    previous = pipeline._extrude_tiles

    def paper_extrude(mesh, thickness: float, stage: str):
        # Avoid the helper's empty-mesh fallback recursing through this wrapper.
        if len(getattr(mesh, "faces", ())) == 0:
            return previous(mesh, thickness, stage)
        assembly, report = extrude_paper_face_planarity(mesh, thickness, stage, pipeline)
        assembly.metrics["paper_t3d_version"] = VERSION_ID
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


def install_paper_t3d_20260920_version_ui():
    """Append/select the dated version when the legacy app defines MODEL_VERSIONS."""
    import streamlit as st

    if getattr(st, "_onestring_paper_t3d_20260920_version_ui_installed", False):
        return
    base_selectbox = st.selectbox

    def selectbox(label, options, *args, **kwargs):
        if str(label) != "version":
            return base_selectbox(label, options, *args, **kwargs)

        option_list = list(options)
        if option_list and all(isinstance(v, dict) for v in option_list):
            if not any(v.get("id") == VERSION_ID for v in option_list):
                option_list.append({
                    "id": VERSION_ID,
                    "label": VERSION_LABEL,
                    "description": VERSION_DESCRIPTION,
                    "t3d_extrusion_side": "negative_normal_from_k3d",
                    "t3d_variable_topology_enabled": False,
                    "allow_legacy_normal_prism_emergency_fallback": False,
                    "t3d_intersection_trim_enabled": False,
                })
            # This launcher is specifically the paper-T3D version. Select it by
            # default on first load while still leaving old versions visible.
            kwargs = dict(kwargs)
            kwargs["index"] = next(
                i for i, v in enumerate(option_list) if v.get("id") == VERSION_ID
            )
            kwargs.setdefault("key", "onestring_model_version_20260920")
            return base_selectbox(label, option_list, *args, **kwargs)

        return base_selectbox(label, options, *args, **kwargs)

    selectbox._onestring_base = base_selectbox
    st.selectbox = selectbox
    st._onestring_paper_t3d_20260920_version_ui_installed = True


__all__ = [
    "VERSION_ID", "VERSION_LABEL", "VERSION_DESCRIPTION",
    "install_paper_t3d_20260920_patch",
    "install_paper_t3d_20260920_version_ui",
]
