"""Render the solved Eq.(5) K2D in the ordinary ``View stage = K2D`` slot.

UI-only patch.  It does not change the Eq.(5) solver, iteration history, K3D,
or downstream T2D/Dual-Hinge numerics.
"""
from __future__ import annotations


def install_final_k2d_view_stage_patch() -> None:
    try:
        import streamlit as st
    except Exception:
        return

    if getattr(st, "_onestring_final_k2d_view_stage_patch_installed", False):
        return

    previous = st.selectbox

    def selectbox_with_final_k2d(*args, **kwargs):
        label = args[0] if args else kwargs.get("label")
        selected = previous(*args, **kwargs)
        if label != "View stage" or str(selected).strip().upper() != "K2D":
            return selected

        history = st.session_state.get("eq5_history")
        if not history or not history.get("snapshots"):
            return selected

        try:
            from .eq5_iteration_history_view import render_final_k2d_result

            st.caption("Eq.(5) solver の最終 accepted K2D を表示しています。")
            if render_final_k2d_result(st, history):
                # Suppress the legacy K2D/FlatTileLayout renderer.  That renderer
                # belongs to the old independent-tile route and can be empty for
                # the reconstructed shared-topology Eq.(5) solver.
                return "__ONESSTRING_FINAL_EQ5_K2D_VIEW__"
        except Exception as exc:
            st.warning(f"Final Eq.(5) K2D rendering failed: {exc}")

        return selected

    st.selectbox = selectbox_with_final_k2d
    st._onestring_final_k2d_view_stage_patch_installed = True


__all__ = ["install_final_k2d_view_stage_patch"]
