"""Bridge ``optcuts_test`` into seam smoothing and polygon-clipped M2D."""
from __future__ import annotations
from typing import Any
import streamlit as st
from .optcuts_seam_extraction_patch import extract_connected_seam_payload_robust
from .optcuts_test_simple_pipeline_patch import install_optcuts_test_simple_pipeline_patch
from .optcuts_test_boundary_clip_m2d_patch import install_optcuts_test_boundary_clip_m2d_patch
from .optcuts_test_polygon_visualization_patch import install_optcuts_test_polygon_visualization_patch
from .optcuts_test_performance_patch import install_optcuts_test_performance_patch
from .optcuts_test_k2d_relative_layout_patch import install_optcuts_test_k2d_relative_layout_patch
from .optcuts_test_k2d_hard_feasibility_patch import install_optcuts_test_k2d_hard_feasibility_patch
from .optcuts_test_k2d_hard_hinge_patch import install_optcuts_test_k2d_hard_hinge_patch
from .optcuts_test2_k2d_global_runtime_patch import install_optcuts_test2_k2d_global_runtime_patch
from .optcuts_test_k2d_overlap_visualization_patch import install_optcuts_test_k2d_overlap_visualization_patch
from .optcuts_k2d_global_history import install_k2d_history_recorder
from .optcuts_test_k3d_pre_al_validity_patch import install_optcuts_test_k3d_pre_al_validity_patch
from .optcuts_test_k3d_augmented_lagrangian_patch import install_optcuts_test_k3d_augmented_lagrangian_patch
from .optcuts_test_k3d_slsqp_polish_patch import install_optcuts_test_k3d_slsqp_polish_patch
from .optcuts_test_k3d_practical_planarity_tolerance_patch import install_optcuts_test_k3d_practical_planarity_tolerance_patch
from .optcuts_test2_surface_constrained_k3d_patch import install_optcuts_test2_surface_constrained_k3d_patch
from .optcuts_test2_surface_result_metadata_patch import install_optcuts_test2_surface_result_metadata_patch
from .optcuts_20260913_ui_patch import install_20260913_ui_patch


def install_optcuts_test_seam_metadata_bridge(pipeline: Any) -> None:
    if getattr(pipeline,"_onestring_optcuts_test_seam_metadata_bridge_installed",False): return

    # Display/default changes only.  app_optcuts captures st.selectbox after this
    # bridge is installed, so its internal optcuts_test2 identifier can stay
    # stable while the user sees the dated experiment name.
    install_20260913_ui_patch(st)

    install_optcuts_test_simple_pipeline_patch(pipeline)
    install_optcuts_test_boundary_clip_m2d_patch(pipeline)
    install_optcuts_test_performance_patch(pipeline)
    install_optcuts_test_k3d_pre_al_validity_patch(pipeline)
    install_optcuts_test_k3d_augmented_lagrangian_patch(pipeline)
    install_optcuts_test_k3d_slsqp_polish_patch(pipeline)
    install_optcuts_test_k3d_practical_planarity_tolerance_patch(pipeline)

    # 2026-09-13 variant=2: start from the existing hard-planarity result, then
    # alternate quad planarity and closest-point projection onto the original
    # target triangle mesh.  The final surface-projected geometry is explicitly
    # recorded as the experiment's authoritative K3D result.
    install_optcuts_test2_surface_constrained_k3d_patch(pipeline)
    install_optcuts_test2_surface_result_metadata_patch(pipeline)

    install_optcuts_test_k2d_hard_feasibility_patch()
    install_optcuts_test_k2d_relative_layout_patch(pipeline)

    # Capture the collision-free rigid K2D builder before the experimental
    # spanning-tree hard-hinge wrapper.  optcuts_test2 uses this path directly
    # so the old tree/loop solve is genuinely bypassed, not merely overwritten.
    from . import optcuts_test_k2d_relative_layout_patch as k2d_relative_mod

    install_k2d_history_recorder(pipeline)
    test2_global_base_builder = k2d_relative_mod._build_rigid_k2d_layout

    # Keep optcuts_test unchanged as the comparison baseline.
    install_optcuts_test_k2d_hard_hinge_patch(pipeline)

    # Outermost K2D numeric wrapper: optcuts_test2 -> global all-hinge Phase 1;
    # optcuts_test -> legacy spanning-tree/loop solver.
    install_optcuts_test2_k2d_global_runtime_patch(pipeline, test2_global_base_builder)
    install_optcuts_test_k2d_overlap_visualization_patch()
    install_optcuts_test_polygon_visualization_patch()

    base_flatten=pipeline._flatten_to_domain
    def flatten_with_test_seam(parameterization: Any, grid: Any, params: Any=None):
        domain=base_flatten(parameterization,grid,params)
        if str(getattr(parameterization,"method",""))!="optcuts_test": return domain
        payload=extract_connected_seam_payload_robust(parameterization)
        setattr(domain,"_optcuts_test_source_seam_payload",payload)
        setattr(parameterization,"_optcuts_test_source_seam_payload",payload)
        try:
            if hasattr(domain,"_optcuts_grid_seam_payload"): delattr(domain,"_optcuts_grid_seam_payload")
        except Exception: pass
        previous=list(getattr(domain,"split_lines",[]) or [])
        setattr(domain,"_optcuts_suppressed_legacy_split_lines",previous)
        try: domain.split_lines=[]
        except Exception: pass
        setattr(domain,"_optcuts_test_clip_boundary",True)
        setattr(domain,"_optcuts_test_smoothed_seam",True)
        setattr(domain,"_optcuts_test_rectilinear_seam_disabled",True)
        try: parameterization.metrics.update({"optcuts_test_rectilinear_seam_disabled":True,"optcuts_test_source_seam_kept_for_diagnostics_only":True})
        except Exception: pass
        return domain
    pipeline._flatten_to_domain=flatten_with_test_seam
    original=getattr(pipeline,"_original",None)
    if original is not None: original._flatten_to_domain=flatten_with_test_seam
    for fn in (getattr(pipeline,"build_onestring_design",None),getattr(pipeline,"_ORIGINAL_BUILD_ONESTRING_DESIGN",None),getattr(original,"build_onestring_design",None) if original is not None else None):
        glb=getattr(fn,"__globals__",None)
        if isinstance(glb,dict): glb["_flatten_to_domain"]=flatten_with_test_seam
    pipeline._onestring_optcuts_test_seam_metadata_bridge_installed=True

__all__=["install_optcuts_test_seam_metadata_bridge"]