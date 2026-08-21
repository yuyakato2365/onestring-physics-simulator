"""Strict grid-constrained OptCuts launcher for OneString.

The ``optcuts`` mode implements one coherent flow:

1. official OptCuts chooses a distortion-relieving PHYSICAL cut topology on S;
2. OneString re-solves that SAME cut topology with hard fabrication constraints:
   every maximal physical seam chain is one straight line, all seam lines share
   one orthogonal frame, both UV copies of one physical seam coincide geometrically
   (zero-width cut), and seam endpoints/junctions lie on the fixed ``Tile size`` lattice;
3. M2D is generated on that exact same lattice;
4. M2D topology is disconnected exactly on those seam grid edges by vertex
   duplication, with zero geometric gap.

No second post-hoc seam is added, no free OptCuts seam is snapped afterwards,
and no seam-strip cell healing/deletion is used to fake alignment.
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
from onestring_physics.fast_assembly_animation_patch import (  # noqa: E402
    install_fast_assembly_animation_patch,
)
from onestring_physics.optcuts_pipeline_patch import install_optcuts_pipeline_patch  # noqa: E402
from onestring_physics.optcuts_run_flag_patch import install_optcuts_run_flag_patch  # noqa: E402
from onestring_physics.optcuts_grid_constrained_parameterization_patch import (  # noqa: E402
    install_optcuts_grid_constrained_parameterization_patch,
)
from onestring_physics.optcuts_grid_constrained_m2d_patch import (  # noqa: E402
    install_optcuts_grid_constrained_m2d_patch,
)
from onestring_physics.optcuts_grid_fusion_v2 import (  # noqa: E402
    install_strict_grid_seam_topology_patch,
    install_strict_optcuts_grid_fusion,
)
from onestring_physics.optcuts_k3d_validity_patch import (  # noqa: E402
    install_optcuts_k3d_validity_patch,
)
from onestring_physics.optcuts_k3d_preflight_patch import (  # noqa: E402
    install_optcuts_k3d_preflight_patch,
)
from onestring_physics.optcuts_visualization_compat_patch import (  # noqa: E402
    install_optcuts_visualization_compat_patch,
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

        if "optcuts" not in option_list:
            option_list.append("optcuts")
        value = original_selectbox(label, option_list, *args, **kwargs)

        if value == "optcuts":
            st.caption(
                "Strict Grid-constrained OptCuts: official OptCuts chooses the physical cut topology. "
                "That SAME topology is reparameterized so each physical seam chain is one straight "
                "horizontal/vertical line in a common orthogonal frame. Both UV copies of a physical "
                "seam coincide geometrically (zero-width) while remaining topologically separate. "
                "Tile size is the fixed lattice unit, and M2D is cut exactly on the same grid edges."
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
            timeout = st.number_input(
                "OptCuts timeout [s]",
                min_value=10.0,
                max_value=7200.0,
                value=max(10.0, _safe_float_env("ONESTRING_OPTCUTS_TIMEOUT_SECONDS", 600.0)),
                step=30.0,
                key="onestring_optcuts_timeout",
            )
            grid_opt_iters = st.number_input(
                "Grid-constrained UV optimization iterations",
                min_value=20,
                max_value=2000,
                value=max(20, int(os.environ.get("ONESTRING_OPTCUTS_GRID_OPT_ITERS", "180"))),
                step=20,
                key="onestring_optcuts_grid_opt_iters",
            )

            if executable.strip():
                os.environ["ONESTRING_OPTCUTS_EXECUTABLE"] = executable.strip()
            else:
                os.environ.pop("ONESTRING_OPTCUTS_EXECUTABLE", None)
            os.environ["ONESTRING_OPTCUTS_DISTORTION_BOUND"] = str(float(distortion))
            os.environ["ONESTRING_OPTCUTS_LAMBDA_INIT"] = str(float(lambda_init))
            os.environ["ONESTRING_OPTCUTS_USE_BIJECTIVITY"] = "1" if use_bijectivity else "0"
            os.environ["ONESTRING_OPTCUTS_INITIAL_CUT_OPTION"] = "1" if initial_cut_label.startswith("farthest") else "0"
            os.environ["ONESTRING_OPTCUTS_TIMEOUT_SECONDS"] = str(float(timeout))
            os.environ["ONESTRING_OPTCUTS_METHOD_TYPE"] = "0"
            os.environ["ONESTRING_OPTCUTS_GRID_OPT_ITERS"] = str(int(grid_opt_iters))
        return value

    st.selectbox = patched_selectbox
    st._onestring_optcuts_selector_installed = True


def _install_post_simple_split_grid_m2d_hook() -> None:
    """Keep stable Simple Split for other modes; replace only OptCuts M2D afterwards."""
    if getattr(simple_split_module, "_onestring_grid_constrained_optcuts_hook_installed", False):
        return
    original_installer = simple_split_module.install_simple_split_panel_patch

    def install_then_grid_constrained_m2d(pipeline_module: Any, optimization_debug_module: Any) -> None:
        original_installer(pipeline_module, optimization_debug_module)
        install_optcuts_grid_constrained_m2d_patch(pipeline_module)
        # Outermost OptCuts-only M2D step: the seam is ALREADY exactly on this
        # lattice, so only disconnect topology on those edges.  Never snap it.
        install_strict_grid_seam_topology_patch(pipeline_module)

    simple_split_module.install_simple_split_panel_patch = install_then_grid_constrained_m2d
    simple_split_module._onestring_grid_constrained_optcuts_hook_installed = True


# Order matters.
# First replace the earlier post-hoc axis behavior and hard-target builder with
# the strict one-physical-seam/one-line model.  Then install the official OptCuts
# proposal backend and the constrained reparameterization wrapper.
install_strict_optcuts_grid_fusion()
install_optcuts_pipeline_patch(pipeline)
install_optcuts_run_flag_patch(pipeline)
install_optcuts_grid_constrained_parameterization_patch(pipeline)
install_fast_assembly_animation_patch()
install_optcuts_k3d_validity_patch(pipeline)
install_optcuts_k3d_preflight_patch(pipeline)
install_optcuts_visualization_compat_patch()
_install_post_simple_split_grid_m2d_hook()
package.onestring_pipeline = pipeline
package.build_onestring_design = pipeline.build_onestring_design
_install_optcuts_selector()

runpy.run_path(str(ROOT / "app_split_panels.py"), run_name="__main__")
