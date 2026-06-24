from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from onestring_physics.animation import deployment_animation
from onestring_physics.design_optimizer import DesignParameters, optimize_design
from onestring_physics.input_shape import CLOSED_SHAPE_WARNING, create_builtin_shape, load_target_shape
from onestring_physics.physics_world import PhysicsParameters, simulate_deployment
from onestring_physics.visualization import figure_comparison, figure_loss, figure_target, figure_tiles


st.set_page_config(page_title="OneString Physics Simulator", layout="wide")
MODEL_VERSION = "lift-tendon-v2"

st.title("onestring-physics-simulator")

with st.sidebar:
    st.header("Target Input")
    target_kind = st.selectbox("target shape", ["dome", "flat", "saddle", "wave", "gaussian"])
    uploaded = st.file_uploader("mesh upload", type=["obj", "stl", "ply"])
    grid_size = st.selectbox("grid size", [3, 4, 5], index=0)
    tile_size = st.number_input("tile size", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    gap_size = st.number_input("gap size", min_value=0.0, max_value=0.4, value=0.08, step=0.01)
    amplitude = st.slider("target amplitude", 0.0, 2.0, 0.75, 0.05)

    st.header("Design Optimization")
    run_design = st.button("Run design optimization", type="primary")
    max_iterations = st.slider("max_iterations", 5, 120, 30, 5)
    w_target = st.slider("w_target", 0.0, 3.0, 1.0, 0.05)
    w_rigid = st.slider("w_rigid", 0.0, 1.0, 0.12, 0.01)
    w_hinge = st.slider("w_hinge", 0.0, 1.0, 0.2, 0.01)
    w_smooth = st.slider("w_smooth", 0.0, 1.0, 0.08, 0.01)

    st.header("Physics Simulation")
    run_physics = st.button("Run physical deployment")
    dt = st.number_input("dt", min_value=0.001, max_value=0.05, value=0.01, step=0.001, format="%.3f")
    substeps = st.slider("substeps", 1, 10, 3)
    solver_iterations = st.slider("solver_iterations", 1, 50, 12)
    damping = st.slider("damping", 0.0, 0.2, 0.04, 0.005)
    rope_stiffness = st.slider("rope stiffness", 0.0, 1.0, 0.95, 0.01)
    hinge_stiffness = st.slider("hinge stiffness", 0.0, 1.0, 0.92, 0.01)
    pull_speed = st.slider("pull speed", 0.1, 3.0, 1.0, 0.1)
    rope_rest_length_scale = st.slider("rope rest length scale", 0.2, 1.0, 0.55, 0.01)
    num_frames = st.slider("num_frames", 20, 300, 80, 10)
    debug_goal_attraction = st.toggle("debug goal attraction", value=False)


def build_target():
    if uploaded is None:
        radius = max(1.5, grid_size * tile_size * 0.7)
        return create_builtin_shape(target_kind, {"amplitude": amplitude, "radius": radius, "sigma": radius * 0.45})
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fh:
        fh.write(uploaded.getbuffer())
        temp_path = fh.name
    st.warning(CLOSED_SHAPE_WARNING)
    return load_target_shape(temp_path)


def current_design_key() -> tuple:
    return (
        MODEL_VERSION,
        target_kind,
        uploaded.name if uploaded else None,
        grid_size,
        tile_size,
        gap_size,
        amplitude,
        max_iterations,
        w_target,
        w_rigid,
        w_hinge,
        w_smooth,
    )


def current_physics_key() -> tuple:
    return (
        MODEL_VERSION,
        current_design_key(),
        dt,
        substeps,
        solver_iterations,
        damping,
        rope_stiffness,
        hinge_stiffness,
        pull_speed,
        rope_rest_length_scale,
        num_frames,
        debug_goal_attraction,
    )


design_key = current_design_key()
if run_design or "design" not in st.session_state or st.session_state.get("design_key") != design_key:
    with st.spinner("Optimizing assembled linkage"):
        st.session_state.design = optimize_design(
            build_target(),
            nx=grid_size,
            ny=grid_size,
            tile_size=tile_size,
            gap_size=gap_size,
            params=DesignParameters(max_iterations=max_iterations, w_target=w_target, w_rigid=w_rigid, w_hinge=w_hinge, w_smooth=w_smooth),
        )
        st.session_state.design_key = design_key
        st.session_state.pop("physics", None)
        st.session_state.pop("physics_key", None)

design = st.session_state.design

physics_key = current_physics_key()
estimated_work = grid_size * grid_size * num_frames * substeps * solver_iterations
if "physics" not in st.session_state:
    st.info("Run physical deployment to simulate the rope-driven physical deployment from the flat state.")
elif st.session_state.get("physics_key") != physics_key:
    st.warning("Physics settings changed. Run physical deployment again to refresh the simulation.")

if estimated_work > 150000:
    st.warning("This setting may take a while. For interactive use, try fewer frames or solver iterations first.")

if run_physics:
    with st.spinner("Running rope-driven physical deployment"):
        st.session_state.physics = simulate_deployment(
            design,
            PhysicsParameters(
                dt=dt,
                substeps=substeps,
                solver_iterations=solver_iterations,
                damping=damping,
                rope_stiffness=rope_stiffness,
                hinge_stiffness=hinge_stiffness,
                rope_pull_speed=pull_speed,
                rope_rest_length_scale=rope_rest_length_scale,
                debug_goal_attraction=debug_goal_attraction,
            ),
            num_frames=num_frames,
        )
        st.session_state.physics_key = physics_key

physics = st.session_state.get("physics")

tabs = st.tabs(
    [
        "Target shape",
        "Design optimization process",
        "Optimized assembled state",
        "Physical deployment simulation",
        "Comparison",
        "Metrics",
    ]
)

with tabs[0]:
    st.plotly_chart(figure_target(design), use_container_width=True, key="target_shape_chart")

with tabs[1]:
    st.plotly_chart(figure_loss(design.loss_history), use_container_width=True, key="loss_chart")
    st.plotly_chart(
        figure_tiles(
            design.flat_tiles,
            design=design,
            rope=design.boundary_string_path,
            title="flat initial configuration",
        ),
        use_container_width=True,
        key="design_flat_chart",
    )

with tabs[2]:
    st.plotly_chart(
        figure_tiles(
            design.assembled_tiles,
            design=design,
            rope=design.boundary_string_path,
            title="optimized assembled linkage",
        ),
        use_container_width=True,
        key="assembled_chart",
    )

with tabs[3]:
    if physics is None:
        st.plotly_chart(
            figure_tiles(
                design.flat_tiles,
                design=design,
                rope=design.boundary_string_path,
                title="flat initial configuration",
            ),
            use_container_width=True,
            key="physics_flat_placeholder_chart",
        )
    else:
        frame_index = st.slider("frame", 0, len(physics.frames) - 1, len(physics.frames) - 1)
        st.plotly_chart(
            figure_tiles(
                physics.frames[frame_index],
                design=design,
                rope=physics.rope_frames[frame_index],
                pull_handle=physics.pull_handle_frames[frame_index],
                title="physical deployment",
            ),
            use_container_width=True,
            key="physical_deployment_frame_chart",
        )
        st.plotly_chart(
            deployment_animation(design, physics.frames, physics.rope_frames, physics.pull_handle_frames),
            use_container_width=True,
            key="physical_deployment_animation_chart",
        )

with tabs[4]:
    if physics is None:
        st.plotly_chart(
            figure_tiles(
                design.assembled_tiles,
                design=design,
                rope=design.boundary_string_path,
                title="optimized assembled linkage",
            ),
            use_container_width=True,
            key="comparison_assembled_placeholder_chart",
        )
    else:
        st.plotly_chart(
            figure_comparison(design, physics.final_tiles),
            use_container_width=True,
            key="comparison_chart",
        )

with tabs[5]:
    if physics is None:
        st.json({"status": "physics simulation has not been run yet"})
    else:
        st.dataframe({k: [v] for k, v in physics.metrics.items()}, use_container_width=True)
