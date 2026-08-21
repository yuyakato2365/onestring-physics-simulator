"""Globally constrained grid-aware OptCuts launcher for OneString.

The ``optcuts`` mode implements one coherent flow:

1. official OptCuts proposes a distortion-relieving physical cut topology on S;
2. OneString compiles the whole UV seam network as one constraint graph before
   assigning coordinates: same-axis chains sharing a UV junction share a lattice
   line, every junction is one shared lattice point, and every seam copy is a
   straight horizontal/vertical segment;
3. Tile size is the fixed lattice spacing h, while a single global lattice phase
   is fitted to reduce unnecessary motion;
4. the constrained UV map is solved by continuation with orientation checks;
5. M2D is generated on exactly the same phased lattice. No post-hoc seam snapping,
   extra central seam, or chain-local endpoint mutation is used.
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
from onestring_physics.optcuts_grid_fusion_v4 import (  # noqa: E402
    install_global_grid_constraint_fusion,
)
from onestring_physics.optcuts_grid_optimizer_v2 import (  # noqa: E402
    install_strict_optcuts_grid_optimizer,
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
                "Grid-constrained OptCuts: official OptCuts proposes the physical cut topology. "
                "The complete seam network is then solved globally: every seam side is one straight "
                "grid line, all lines use one orthogonal frame, shared junctions are single shared "
                "lattice points, and Tile size is the fixed lattice spacing. The lattice phase is "
                "fitted globally; no seam is snapped independently afterwards."
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
                min_value=40,
                max_value=2000,
                value=max(40, int(os.environ.get("ONESTRING_OPTCUTS_GRID_OPT_ITERS", "240"))),
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
    """Keep stable Simple Split for other modes; use phased fixed-lattice M2D for OptCuts."""
    if getattr(simple_split_module, "_onestring_grid_constrained_optcuts_hook_installed", False):
        return
    original_installer = simple_split_module.install_simple_split_panel_patch

    def install_then_grid_constrained_m2d(pipeline_module: Any, optimization_debug_module: Any) -> None:
        original_installer(pipeline_module, optimization_debug_module)
        install_optcuts_grid_constrained_m2d_patch(pipeline_module)

    simple_split_module.install_simple_split_panel_patch = install_then_grid_constrained_m2d
    simple_split_module._onestring_grid_constrained_optcuts_hook_installed = True


# Order matters: compile the seam graph globally before the official OptCuts
# wrapper is used, then solve the constrained UV and generate the same h-lattice.
install_global_grid_constraint_fusion()
install_strict_optcuts_grid_optimizer()
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
