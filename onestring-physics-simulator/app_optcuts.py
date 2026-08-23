"""OptCuts launcher.

``optcuts``
    Requirement-preserving path using the ordinary official OptCuts backend.

``optcuts_test``
    Experimental path: official OptCuts seam topology -> initial Omega -> the
    existing M2D quad-cell footprint -> grid-cell outer boundary -> Omega
    reparameterization while the OptCuts seam remains a hard boundary.

``optcuts_grid``
    Experimental native Grid-OptCuts V4 path kept available for comparison.
"""
from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import onestring_physics as package  # noqa: E402
from onestring_physics import onestring_pipeline as pipeline  # noqa: E402
from onestring_physics import simple_split_panel_patch as simple_split_module  # noqa: E402
from onestring_physics.fast_assembly_animation_patch import install_fast_assembly_animation_patch  # noqa: E402
from onestring_physics.optcuts_pipeline_patch import install_optcuts_pipeline_patch  # noqa: E402
from onestring_physics.optcuts_run_flag_patch import install_optcuts_run_flag_patch  # noqa: E402
from onestring_physics.optcuts_test_boundary_reparameterization_patch import (  # noqa: E402
    install_optcuts_test_boundary_reparameterization_patch,
)
from onestring_physics.optcuts_grid_native_pipeline_patch import install_native_grid_optcuts_pipeline_patch  # noqa: E402
from onestring_physics.optcuts_grid_native_lift_patch import install_native_grid_optcuts_lift_patch  # noqa: E402
from onestring_physics.optcuts_grid_constrained_m2d_patch import install_optcuts_grid_constrained_m2d_patch  # noqa: E402
from onestring_physics.optcuts_grid_consistency_patch import install_optcuts_grid_consistency_patch  # noqa: E402
from onestring_physics.optcuts_k3d_validity_patch import install_optcuts_k3d_validity_patch  # noqa: E402
from onestring_physics.optcuts_k3d_preflight_patch import install_optcuts_k3d_preflight_patch  # noqa: E402
from onestring_physics.optcuts_visualization_compat_patch import install_optcuts_visualization_compat_patch  # noqa: E402
from onestring_physics.optcuts_source_seam_visualization_patch import install_optcuts_source_seam_visualization_patch  # noqa: E402
from onestring_physics.optcuts_grid_seam_patch import install_optcuts_seam_metadata_patch  # noqa: E402
from onestring_physics.optcuts_seam_extraction_patch import install_robust_optcuts_seam_extraction  # noqa: E402
from onestring_physics.optcuts_rectilinear_seam_patch import install_optcuts_rectilinear_seam_patch  # noqa: E402
from onestring_physics.optcuts_seam_requirement_patch import (  # noqa: E402
    install_optcuts_seam_requirement_patch,
    install_strict_straight_grid_seam_verifier,
)


def _safe_float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _install_optcuts_selector() -> None:
    if getattr(st, "_onestring_optcuts_selector_installed", False):
        return
    original_selectbox = st.selectbox

    def patched_selectbox(label: str, options: Any, *args: Any, **kwargs: Any):
        option_list = list(options)
        if label != "Omega parameterization mode":
            return original_selectbox(label, options, *args, **kwargs)

        for mode in ("optcuts", "optcuts_test", "optcuts_grid"):
            if mode not in option_list:
                option_list.append(mode)
        value = original_selectbox(label, option_list, *args, **kwargs)
        if value not in {"optcuts", "optcuts_test", "optcuts_grid"}:
            return value

        if value == "optcuts":
            st.caption(
                "Official OptCuts baseline. The saved stable mode is not modified by optcuts_test."
            )
        elif value == "optcuts_test":
            st.caption(
                "OptCuts_test: keep the official OptCuts seam/cut topology, build the ordinary M2D "
                "quad-cell footprint from the initial Omega, use that footprint's outer boundary as "
                "the new Omega boundary target, and re-solve UV while the OptCuts seam remains fixed "
                "as a boundary."
            )
        else:
            st.caption(
                "Experimental Native Grid-OptCuts V4: split candidates are searched directly on "
                "the fixed fabrication lattice."
            )

        executable = st.text_input(
            "OptCuts executable",
            value=os.environ.get("ONESTRING_OPTCUTS_EXECUTABLE", ""),
            placeholder="third_party/OptCuts/build/OptCuts_bin",
            key="onestring_optcuts_executable",
        )
        distortion = st.number_input(
            "OptCuts distortion bound (Symmetric Dirichlet > 4)",
            min_value=4.0001,
            max_value=1000.0,
            value=max(4.0001, _safe_float_env("ONESTRING_OPTCUTS_DISTORTION_BOUND", 4.1)),
            step=0.05,
            format="%.4f",
            key="onestring_optcuts_distortion_bound",
        )
        lambda_init = st.number_input(
            "OptCuts initial lambda",
            min_value=0.0001,
            max_value=0.9999,
            value=min(0.9999, max(0.0001, _safe_float_env("ONESTRING_OPTCUTS_LAMBDA_INIT", 0.999))),
            step=0.001,
            format="%.4f",
            key="onestring_optcuts_lambda_init",
        )
        timeout = st.number_input(
            "OptCuts timeout [s]",
            min_value=10.0,
            max_value=7200.0,
            value=max(10.0, _safe_float_env("ONESTRING_OPTCUTS_TIMEOUT_SECONDS", 600.0)),
            step=30.0,
            key="onestring_optcuts_timeout",
        )

        if value in {"optcuts", "optcuts_test"}:
            use_bijectivity = st.checkbox(
                "OptCuts enforce bijectivity",
                value=os.environ.get("ONESTRING_OPTCUTS_USE_BIJECTIVITY", "1").lower()
                not in {"0", "false", "no", "off"},
                key="onestring_optcuts_use_bijectivity",
            )
            initial_cut_label = original_selectbox(
                "OptCuts initial cut",
                ["random one-point (0)", "farthest two-point (1)"],
                index=0 if os.environ.get("ONESTRING_OPTCUTS_INITIAL_CUT_OPTION", "0") != "1" else 1,
                key="onestring_optcuts_initial_cut",
            )
            os.environ["ONESTRING_OPTCUTS_USE_BIJECTIVITY"] = "1" if use_bijectivity else "0"
            os.environ["ONESTRING_OPTCUTS_INITIAL_CUT_OPTION"] = "1" if initial_cut_label.startswith("farthest") else "0"
            os.environ["ONESTRING_OPTCUTS_GRID_ANGLE_DEGREES"] = "0"
            os.environ["ONESTRING_OPTCUTS_GRID_PHASE_U"] = "0"
            os.environ["ONESTRING_OPTCUTS_GRID_PHASE_V"] = "0"
        else:
            angle = st.number_input(
                "Grid reference direction [deg]",
                min_value=-180.0,
                max_value=180.0,
                value=_safe_float_env("ONESTRING_OPTCUTS_GRID_ANGLE_DEGREES", 0.0),
                step=5.0,
                key="onestring_optcuts_grid_angle",
            )
            phase_u = st.number_input(
                "Grid phase U",
                value=_safe_float_env("ONESTRING_OPTCUTS_GRID_PHASE_U", 0.0),
                step=0.01,
                format="%.5f",
                key="onestring_optcuts_grid_phase_u",
            )
            phase_v = st.number_input(
                "Grid phase V",
                value=_safe_float_env("ONESTRING_OPTCUTS_GRID_PHASE_V", 0.0),
                step=0.01,
                format="%.5f",
                key="onestring_optcuts_grid_phase_v",
            )
            max_snap = st.number_input(
                "Max candidate displacement [Grid units]",
                min_value=0.5,
                max_value=8.0,
                value=max(0.5, _safe_float_env("ONESTRING_OPTCUTS_GRID_MAX_SNAP_STEPS", 2.0)),
                step=0.25,
                key="onestring_optcuts_grid_max_snap",
            )
            initial_candidates = st.number_input(
                "Initial Grid-cut exact candidates",
                min_value=1,
                max_value=16,
                value=max(1, min(16, int(round(_safe_float_env("ONESTRING_OPTCUTS_GRID_INITIAL_CANDIDATES", 2.0))))),
                step=1,
                key="onestring_optcuts_grid_initial_candidates",
                help=(
                    "Each candidate performs an expensive onePointCut + harmonic solve. "
                    "For large meshes, 2 is the recommended default; increase only when needed."
                ),
            )
            st.caption(
                "V4 is split-only: unconstrained OptCuts merge is disabled. The physical cut topology "
                "still follows existing input surface-mesh edges."
            )
            os.environ["ONESTRING_OPTCUTS_GRID_ANGLE_DEGREES"] = str(float(angle))
            os.environ["ONESTRING_OPTCUTS_GRID_PHASE_U"] = str(float(phase_u))
            os.environ["ONESTRING_OPTCUTS_GRID_PHASE_V"] = str(float(phase_v))
            os.environ["ONESTRING_OPTCUTS_GRID_MAX_SNAP_STEPS"] = str(float(max_snap))
            os.environ["ONESTRING_OPTCUTS_GRID_INITIAL_CANDIDATES"] = str(int(initial_candidates))
            os.environ["ONESTRING_OPTCUTS_USE_BIJECTIVITY"] = "1"
            os.environ["ONESTRING_OPTCUTS_INITIAL_CUT_OPTION"] = "0"

        if executable.strip():
            os.environ["ONESTRING_OPTCUTS_EXECUTABLE"] = executable.strip()
        else:
            os.environ.pop("ONESTRING_OPTCUTS_EXECUTABLE", None)
        os.environ["ONESTRING_OPTCUTS_DISTORTION_BOUND"] = str(float(distortion))
        os.environ["ONESTRING_OPTCUTS_LAMBDA_INIT"] = str(float(lambda_init))
        os.environ["ONESTRING_OPTCUTS_TIMEOUT_SECONDS"] = str(float(timeout))
        os.environ["ONESTRING_OPTCUTS_METHOD_TYPE"] = "0"
        return value

    st.selectbox = patched_selectbox
    st._onestring_optcuts_selector_installed = True


def _install_m2d_after_simple_split() -> None:
    """Install mode-specific M2D adapters without changing Simple Split numerics."""
    if getattr(simple_split_module, "_onestring_optcuts_requirement_hook_installed", False):
        return
    original_installer = simple_split_module.install_simple_split_panel_patch

    def install_then_optcuts_m2d(pipeline_module: Any, optimization_debug_module: Any) -> None:
        original_installer(pipeline_module, optimization_debug_module)
        install_optcuts_rectilinear_seam_patch(pipeline_module)
        install_strict_straight_grid_seam_verifier(pipeline_module)
        install_optcuts_grid_constrained_m2d_patch(pipeline_module)
        install_optcuts_grid_consistency_patch(pipeline_module)

    simple_split_module.install_simple_split_panel_patch = install_then_optcuts_m2d
    simple_split_module._onestring_optcuts_requirement_hook_installed = True


# Build wrappers from the stable official OptCuts path outward.  optcuts_test is
# installed after the run-flag wrapper so its internal ordinary-OptCuts call is
# seen as a normal active OptCuts run by downstream seam metadata patches.
install_optcuts_pipeline_patch(pipeline)
install_native_grid_optcuts_pipeline_patch(pipeline)
install_optcuts_run_flag_patch(pipeline)
install_optcuts_test_boundary_reparameterization_patch(pipeline)
install_native_grid_optcuts_lift_patch(pipeline)
install_robust_optcuts_seam_extraction()
install_optcuts_seam_metadata_patch(pipeline)
install_optcuts_seam_requirement_patch(pipeline)
install_fast_assembly_animation_patch()
install_optcuts_k3d_validity_patch(pipeline)
install_optcuts_k3d_preflight_patch(pipeline)
install_optcuts_visualization_compat_patch()
install_optcuts_source_seam_visualization_patch()
_install_m2d_after_simple_split()
package.onestring_pipeline = pipeline
package.build_onestring_design = pipeline.build_onestring_design
_install_optcuts_selector()

runpy.run_path(str(ROOT / "app_split_panels.py"), run_name="__main__")
