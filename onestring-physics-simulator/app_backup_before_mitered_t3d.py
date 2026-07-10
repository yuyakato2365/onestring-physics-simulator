from __future__ import annotations

import tempfile
import time
import os
from dataclasses import replace
from pathlib import Path
import sys
import importlib
from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent / "src"))

import onestring_physics.onestring_pipeline as _onestring_pipeline

_onestring_pipeline = importlib.reload(_onestring_pipeline)

from onestring_physics.animation import assembly_progress_animation, assembly_progress_frame_figure, tile_assembly_animation
from onestring_physics.input_shape import CLOSED_SHAPE_WARNING, create_builtin_shape, load_target_shape
from onestring_physics.onestring_pipeline import (
    ComputeConfig,
    DeploymentParameters,
    PipelineParameters,
    TileAssembly,
    build_onestring_design,
    complexity_metrics,
    compute_backend_info,
    gpu_self_test,
    nvidia_smi_probe,
    export_t2d_stl,
    run_simulator_gpu_benchmark,
    simulate_onestring_deployment,
    paper_consistency_report,
)
from onestring_physics.visualization import (
    figure_domain,
    figure_flat_tile_layout,
    figure_m3d_overlay,
    figure_onestring_comparison,
    figure_pipeline_overview,
    figure_quad_mesh,
    figure_split_mapping,
    figure_surface_mesh,
    figure_tile_assembly,
    add_tile_assembly,
)


st.set_page_config(page_title="OneString Paper-Faithful Simulator", layout="wide")
MODEL_VERSION = "2026-07-10-basic-implementation"
MODEL_VERSIONS = [
    {
        "id": MODEL_VERSION,
        "label": "2026-07-10 基礎実装",
        "description": "周囲との法線整合だけを使うT3D押し出しを含む基礎実装。",
        "t3d_intersection_trim_enabled": False,
    },
    {
        "id": "2026-07-10-t3d-large-panel-intersection-trim",
        "label": "2026-07-10 T3D大パネル交差除去",
        "description": "T3D押し出し後、ほかのパネルと交差する部分を大きい方のパネルの表示メッシュから除去します。",
        "t3d_intersection_trim_enabled": True,
    },
    {
        "id": "0.2.0-paper-reference-bff",
        "label": "0.2.0 Paper Reference BFF",
        "description": "公式BFF CLI、厳密Jacobian/CSF、規則grid、厳密barycentric inverse mapを追加。K3D以降は既存近似。",
        "t3d_intersection_trim_enabled": False,
    },
]
# Future version additions: append a new entry to MODEL_VERSIONS when the user
# asks to preserve another implementation version, then branch behavior from
# selected_model_version["id"] where version-specific behavior is needed.

st.title("onestring-physics-simulator")
st.caption("Strict Figure-5 order: S -> Omega -> M2D -> c^-1 M3D -> K3D/T3D and M2D -> K2D -> T2D Top -> T2D Dual -> lift/string -> PD snap/lift.")



# Short, always-visible parameter explanations shown next to sidebar controls.
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] .param-note {
        font-size: 0.74rem;
        line-height: 1.25;
        color: rgba(49, 51, 63, 0.72);
        border-left: 2px solid rgba(49, 51, 63, 0.16);
        padding-left: 0.45rem;
        margin-top: 1.9rem;
    }
    section[data-testid="stSidebar"] .param-note-tight {
        font-size: 0.74rem;
        line-height: 1.25;
        color: rgba(49, 51, 63, 0.72);
        border-left: 2px solid rgba(49, 51, 63, 0.16);
        padding-left: 0.45rem;
        margin-top: 0.2rem;
    }
    section[data-testid="stSidebar"] .param-section-note {
        font-size: 0.78rem;
        line-height: 1.35;
        color: rgba(49, 51, 63, 0.76);
        background: rgba(49, 51, 63, 0.045);
        padding: 0.45rem 0.55rem;
        border-radius: 0.35rem;
        margin: 0.2rem 0 0.45rem 0;
    }
    section[data-testid="stSidebar"] .meter-card {
        background: rgba(49, 51, 63, 0.035);
        border: 1px solid rgba(49, 51, 63, 0.08);
        border-radius: 0.45rem;
        padding: 0.45rem 0.55rem;
        margin: 0.35rem 0;
    }
    section[data-testid="stSidebar"] .meter-head {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 0.5rem;
        margin-bottom: 0.25rem;
    }
    section[data-testid="stSidebar"] .meter-label {
        font-weight: 600;
        font-size: 0.80rem;
    }
    section[data-testid="stSidebar"] .meter-value {
        font-weight: 700;
        font-size: 0.76rem;
        white-space: nowrap;
    }
    section[data-testid="stSidebar"] .meter-track {
        width: 100%;
        height: 0.45rem;
        background: rgba(49, 51, 63, 0.11);
        border-radius: 999px;
        overflow: hidden;
        margin-bottom: 0.25rem;
    }
    section[data-testid="stSidebar"] .meter-fill { height: 100%; border-radius: 999px; }
    section[data-testid="stSidebar"] .meter-fill-low { background: #2fb344; }
    section[data-testid="stSidebar"] .meter-fill-mid { background: #f08c00; }
    section[data-testid="stSidebar"] .meter-fill-high { background: #e03131; }
    section[data-testid="stSidebar"] .meter-low { color: #2b8a3e; }
    section[data-testid="stSidebar"] .meter-mid { color: #e67700; }
    section[data-testid="stSidebar"] .meter-high { color: #c92a2a; }
    section[data-testid="stSidebar"] .meter-note {
        font-size: 0.70rem;
        line-height: 1.25;
        color: rgba(49, 51, 63, 0.70);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _param_note(text: str, *, tight: bool = False) -> None:
    cls = "param-note-tight" if tight else "param-note"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def _param_row(note: str, render, *, tight_note: bool = False):
    left, right = st.columns([0.58, 0.42], gap="small")
    with left:
        value = render()
    with right:
        _param_note(note, tight=tight_note)
    return value


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _meter_level(score: float) -> tuple[str, str]:
    score = _clamp01(score)
    if score < 0.34:
        return "low", "低"
    if score < 0.67:
        return "mid", "中"
    return "high", "高"


def _meter_bar(label: str, score: float, note: str) -> None:
    score = _clamp01(score)
    level_class, level_text = _meter_level(score)
    percent = int(round(score * 100))
    st.markdown(
        f'''
        <div class="meter-card">
            <div class="meter-head">
                <span class="meter-label">{label}</span>
                <span class="meter-value meter-{level_class}">{percent}% / {level_text}</span>
            </div>
            <div class="meter-track"><div class="meter-fill meter-fill-{level_class}" style="width:{percent}%"></div></div>
            <div class="meter-note">{note}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def _compute_setting_meters(
    *,
    target_kind: str,
    grid_size: int,
    surface_mesh_subdivisions: int,
    tile_size: float,
    gap_size: float,
    thickness: float,
    amplitude: float,
    max_3d_iterations: int,
    max_2d_iterations: int,
    strict_k2d_time_budget_sec: float,
    hinge_layout_iterations: int,
    hinge_layout_connection_weight: float,
    hinge_layout_collision_weight: float,
    hinge_layout_anchor_weight: float,
    hinge_layout_initial_expansion: float,
    hinge_layout_max_center_drift_tiles: float,
    hinge_layout_max_candidate_pairs: int,
    hinge_layout_collision_sweeps_per_iteration: int,
    compute_backend: str,
    tensor_dtype: str,
    sim_steps: int,
    solver_iterations: int,
    solver_substeps: int,
    store_animation_frames: bool,
    high_fidelity: bool,
) -> dict[str, tuple[float, str]]:
    target_factor = {"flat": 0.15, "dome": 0.35, "gaussian": 0.55, "half_gourd": 0.62, "saddle": 0.70, "wave": 0.55}.get(target_kind, 0.55)
    grid_factor = _clamp01((grid_size / 40.0) ** 2)
    large_grid_factor = _clamp01((grid_size / 70.0) ** 2)
    subdiv_factor = _clamp01((surface_mesh_subdivisions - 1) / 7.0)
    amp_factor = _clamp01(amplitude / 1.5)
    thickness_factor = _clamp01(thickness / max(tile_size, 1e-6) / 0.18)
    gap_factor = _clamp01(gap_size / max(tile_size, 1e-6) / 0.18)
    k3d_factor = _clamp01(max_3d_iterations / 120.0)
    k2d_factor = _clamp01(max_2d_iterations / 120.0)
    strict_budget_factor = _clamp01(strict_k2d_time_budget_sec / 30.0)
    hinge_iter_factor = _clamp01(hinge_layout_iterations / 240.0)
    candidate_factor = _clamp01(hinge_layout_max_candidate_pairs / 12000.0)
    sweep_factor = _clamp01(hinge_layout_collision_sweeps_per_iteration / 6.0)
    sim_factor = _clamp01((sim_steps * solver_iterations * solver_substeps) / (120.0 * 60.0 * 4.0))
    dtype_factor = 1.15 if tensor_dtype == "float64" else 1.0
    backend_bonus = -0.08 if compute_backend == "cuda" else (0.08 if compute_backend == "cpu" else 0.0)

    processing = _clamp01((0.30 * large_grid_factor + 0.13 * subdiv_factor + 0.13 * k3d_factor + 0.13 * k2d_factor + 0.17 * hinge_iter_factor + 0.08 * candidate_factor + 0.06 * sweep_factor) * dtype_factor + backend_bonus)
    geometry = _clamp01(0.24 * target_factor + 0.28 * amp_factor + 0.20 * thickness_factor + 0.14 * subdiv_factor + 0.14 * grid_factor)
    k2d = _clamp01(0.32 * geometry + 0.25 * k2d_factor + 0.18 * strict_budget_factor + 0.15 * grid_factor + 0.10 * max(0.0, thickness_factor - gap_factor))
    hinge_conflict = _clamp01(0.28 * hinge_iter_factor + 0.20 * (hinge_layout_connection_weight / 8.0) + 0.22 * (hinge_layout_collision_weight / 4.0) + 0.12 * sweep_factor + 0.10 * candidate_factor + 0.08 * max(0.0, 0.12 - hinge_layout_anchor_weight) / 0.12)
    void = _clamp01(0.55 * ((hinge_layout_initial_expansion - 1.0) / 0.25) + 0.25 * (hinge_layout_max_center_drift_tiles / 5.0) + 0.20 * gap_factor)
    animation = _clamp01(0.60 * sim_factor + 0.18 * (1.0 if store_animation_frames else 0.0) + 0.12 * (1.0 if high_fidelity else 0.0) + 0.10 * grid_factor)

    notes = {
        "処理負荷": "grid・反復回数・衝突候補数から見た計算の重さ。高いほど待ち時間が増えます。",
        "形状難度": "Sの曲率/振幅/厚みから見た設計の難しさ。高いほどK2Dやヒンジ配置が破綻しやすいです。",
        "K2D辺長合わせ難度": "K3Dの対応辺長をK2Dで満たす難しさ。高いとedge errorが残りやすいです。",
        "ヒンジ配置競合": "E_ConnとE_Collisionが衝突する度合い。高いと接続維持と非衝突の両立が難しいです。",
        "空洞/展開量": "initial expansion・gap・center driftから見たタイル間の空きやすさ。高すぎると空洞が大きくなります。",
        "アニメーション負荷": "PD simulationのステップ数・反復数・保存フレーム量。高いほど再生/計算が重くなります。",
    }
    return {
        "処理負荷": (processing, notes["処理負荷"]),
        "形状難度": (geometry, notes["形状難度"]),
        "K2D辺長合わせ難度": (k2d, notes["K2D辺長合わせ難度"]),
        "ヒンジ配置競合": (hinge_conflict, notes["ヒンジ配置競合"]),
        "空洞/展開量": (void, notes["空洞/展開量"]),
        "アニメーション負荷": (animation, notes["アニメーション負荷"]),
    }


with st.sidebar:
    st.header("Version")
    selected_model_version = st.selectbox(
        "version",
        MODEL_VERSIONS,
        format_func=lambda version: version["label"],
        help="実装バージョンを選択します。今後バージョン追加指示があれば、この一覧に追記します。",
    )
    st.caption(selected_model_version["description"])

    st.header("Target Input")
    target_kind = _param_row(
        "目標曲面 S の種類。waveは標準で起伏を抑制。half_gourdは半割りヒョウタン状の非矩形メッシュで、Ω/M2D cropの検証用。",
        lambda: st.selectbox("target shape", ["dome", "flat", "half_gourd", "snowman_half", "snowman_full", "saddle", "wave", "gaussian"], help="Built-in target surface S. snowman_half isolates a single peak; snowman_full is a two-dome-with-neck stress case."),
    )
    uploaded = _param_row(
        "OBJ/STL/PLY を読み込む。閉じた形状では Ω の切断・境界条件が難しくなるので注意。",
        lambda: st.file_uploader("mesh upload", type=["obj", "stl", "ply"], help="Upload a target mesh instead of using a built-in height field."),
    )
    grid_size = int(_param_row(
        "Ω 上に重ねる quad grid の密度。大きいほど細かいが、K2D・ヒンジ配置・衝突判定が重くなる。",
        lambda: st.number_input("grid size", min_value=1, max_value=100, value=20, step=1, help="Number of grid cells per axis before Ω cropping."),
    ))
    omega_overlay_margin = int(_param_row(
        "Ω より何タイル分大きく grid を置くか。論文の『大きめに置いて外側 quad を削る』処理に対応。",
        lambda: st.number_input("Omega overlay margin tiles", min_value=0, max_value=4, value=1, step=1, help="Paper-style M2D: overlay a grid larger than Ω, then crop outside quads."),
    ))
    m2d_crop_policy = _param_row(
        "M2D の切り抜き条件。center は境界付近を残しやすく、strict_vertices は外にはみ出す quad を強く削る。",
        lambda: st.selectbox(
            "M2D crop policy",
            ["center", "strict_vertices"],
            index=1,
            help="center keeps boundary-crossing quads whose centers lie inside Ω. strict_vertices keeps only quads fully inside Ω.",
        ),
    )
    omega_boundary_mode = _param_row(
        "Omega boundary の扱い。paper_default は未実装なら明示的に停止し、PCAへ黙ってfallbackしない。",
        lambda: st.selectbox(
            "Omega boundary mode",
            ["paper_default", "shape_preserving_experimental", "rectangular_debug"],
            index=0,
            help="shape_preserving_experimental is PCA-based and not paper-default. paper_default fails explicitly when unimplemented.",
        ),
    )
    omega_parameterization_mode = _param_row(
        "S→Omega のパラメータ化。paper_reference_bff だけが公式BFF CLIを使用し、代替fallbackを禁止する。",
        lambda: st.selectbox(
            "Omega parameterization mode",
            ["rectangular_harmonic_legacy", "lscm_free_boundary", "paper_reference_bff", "bff", "boundary_sliding_lscm", "pca_debug"],
            index=2 if selected_model_version["id"] == "0.2.0-paper-reference-bff" else 0,
            help="bff is a deprecated alias of rectangular_harmonic_legacy. paper_reference_bff requires the official CLI.",
        ),
    )
    if omega_parameterization_mode in {"bff", "rectangular_harmonic_legacy"}:
        st.warning("This is not Boundary First Flattening. It is rectangular-boundary cotangent harmonic parameterization.")
    with st.expander("Paper-reference BFF settings", expanded=omega_parameterization_mode == "paper_reference_bff"):
        bff_boundary_policy = st.selectbox(
            "BFF boundary policy",
            ["automatic_reference", "boundary_scale_zero", "target_disk", "target_rectangle", "custom_boundary_curvature"],
            index=0,
            help="OneStringの正確な境界条件はUNSPECIFIED_IN_PAPER。CLI非対応のpolicyは明示失敗します。",
        )
        bff_executable = st.text_input(
            "official bff-command-line path",
            value=os.environ.get("ONESTRING_BFF_EXECUTABLE", ""),
            help="空ならONESTRING_BFF_EXECUTABLE、third_party、PATHの順に探索します。",
        )
        reference_grid_spacing_value = float(st.number_input("reference grid spacing (0 = derive hypothesis_a)", min_value=0.0, value=0.0, step=0.05))
        reference_grid_rotation_degrees = float(st.number_input("reference grid rotation (degrees)", value=0.0, step=1.0))
        reference_grid_origin_u = float(st.number_input("reference grid origin u", value=0.0, step=0.05))
        reference_grid_origin_v = float(st.number_input("reference grid origin v", value=0.0, step=0.05))
        reference_csf_normalization = st.selectbox(
            "lambda normalization",
            ["min_to_one_hypothesis_a", "none_unspecified"],
            index=0,
        )
        reference_stop_on_required_split = st.checkbox(
            "stop when lambda > 2 requires unspecified reparameterization",
            value=True,
        )
    with st.expander("Boundary-sliding LSCM advanced settings", expanded=False):
        boundary_target_aspect_mode = st.selectbox(
            "rectangle aspect mode",
            ["lscm_initial", "fixed"],
            index=0,
            help="Initialize from the free-LSCM bounding box or use the fixed ratio below.",
        )
        boundary_target_aspect_ratio = st.number_input(
            "fixed rectangle aspect ratio",
            min_value=0.2,
            max_value=5.0,
            value=1.0,
            step=0.1,
        )
        boundary_sliding_max_iterations = int(st.number_input("boundary slide max iterations", 0, 200, 40, 5))
        boundary_sliding_step_size = st.number_input("boundary slide step size", 0.001, 0.5, 0.08, 0.01, format="%.3f")
        boundary_sliding_energy_tolerance = st.number_input(
            "boundary slide energy tolerance",
            min_value=1e-10,
            max_value=1e-3,
            value=1e-7,
            format="%.1e",
        )
        boundary_sliding_min_spacing = st.number_input("minimum normalized boundary spacing", 0.0, 0.1, 0.001, 0.001, format="%.4f")
        boundary_sliding_length_weight = st.number_input("weak 3D boundary-length weight", 0.0, 1.0, 0.02, 0.005, format="%.4f")
        boundary_sliding_spacing_weight = st.number_input("weak spacing regularization weight", 0.0, 1.0, 0.002, 0.001, format="%.4f")
        boundary_sliding_flip_area_epsilon = st.number_input(
            "UV flip/degenerate area epsilon",
            min_value=1e-14,
            max_value=1e-4,
            value=1e-10,
            format="%.1e",
        )
        boundary_sliding_line_search_max_steps = int(st.number_input("line-search max steps", 1, 30, 14, 1))
    allow_experimental_pipeline = _param_row(
        "PCA/debug 経路を明示的に許可する。OFFなら論文未実装部分は停止する。",
        lambda: st.checkbox(
            "Allow non-paper experimental pipeline",
            value=False,
            help="Required for pca_debug / shape_preserving_experimental / rectangular_debug runs.",
        ),
    )
    gpu_first_mode = False
    st.markdown(
        '<div class="param-section-note">Audited mode: GPU-first / analytic height-field shortcuts are disabled unless explicitly selected. Ω parameterization reports whether it is debug, experimental, or unimplemented.</div>',
        unsafe_allow_html=True,
    )
    surface_mesh_subdivisions = _param_row(
        "S の表示・距離評価用メッシュ解像度。上げると表面近似は細かくなるが、逆写像や距離評価が遅くなる。",
        lambda: st.slider("surface mesh subdivisions", 1, 8, 2, 1, help="Sampling resolution for the target surface mesh."),
    )
    tile_size = _param_row(
        "平面上の基準タイル一辺。K2D の辺長・T2D の部品サイズの基準になる。",
        lambda: st.number_input("tile size", min_value=0.1, max_value=5.0, value=1.0, step=0.1, help="Base tile size before deformation/optimization."),
    )
    gap_size = _param_row(
        "加工・紐通し用の隙間幅の目安。大きいほど衝突しにくいが、空洞が目立つ。",
        lambda: st.number_input("gap size", min_value=0.0, max_value=0.4, value=0.08, step=0.01, help="Nominal fabrication/channel gap scale."),
    )
    thickness = _param_row(
        "タイルの厚み。厚いほど T2D footprint 衝突が増え、Dual Hinge 配置が難しくなる。",
        lambda: st.number_input("tile thickness", min_value=0.01, max_value=0.5, value=0.08, step=0.01, help="Extruded tile thickness."),
    )
    amplitude = _param_row(
        "built-in 形状の高さ振幅。大きいほど K3D/K2D 辺長差・ヒンジ角・衝突が厳しくなる。",
        lambda: st.slider("target amplitude", 0.0, 2.0, 0.75, 0.05, help="Height amplitude of built-in target surfaces."),
    )
    if target_kind == "wave":
        st.caption("Waveは内部で高さを約35%に抑え、波長も長めにしています。")
    elif target_kind == "half_gourd":
        st.caption("half_gourdは半割りヒョウタン状の非矩形Sです。Ω境界とM2D cropの検証に使えます。")

    st.header("Pipeline Optimization")
    run_pipeline = _param_row(
        "現在の設定で S→Ω→M2D→M3D→K3D/T3D と M2D→K2D→T2D を再構築する。",
        lambda: st.button("Run OneString pipeline", type="primary"),
        tight_note=True,
    )
    max_3d_iterations = _param_row(
        "M3D→K3D の反復回数。平面性・正方形性・表面追従を改善するが、重くなる。",
        lambda: st.slider("3D optimization iterations", 5, 120, 40, 5, help="Iterations for K3D optimization."),
    )
    max_2d_iterations = _param_row(
        "M2D→K2D の辺長合わせ反復回数。K3Dとの対応辺長誤差を下げるための回数。",
        lambda: st.slider("2D optimization iterations", 5, 120, 40, 5, help="Iterations for K2D edge-length matching."),
    )
    strict_k2d_time_budget_sec = _param_row(
        "K2D の厳密寄り解き直しに使える最大秒数。長くすると辺長一致は改善しやすいが待ち時間が増える。",
        lambda: st.slider("K2D strict solver time budget", 2.0, 30.0, 12.0, 1.0, help="Time budget for strict K2D edge-length projection."),
    )
    strict_k2d_scipy_vertex_limit = int(_param_row(
        "この頂点数以下なら SciPy の非線形 solver を許可。大きい形状では vectorized projection に逃がして固まりを防ぐ。",
        lambda: st.number_input("K2D SciPy vertex limit", min_value=0, max_value=1000, value=120, step=20, help="Above this, K2D uses bounded vectorized projection instead of slow nonlinear least_squares."),
    ))
    m3d_construction_mode = "mesh_harmonic"
    st.markdown(
        '<div class="param-section-note">M3D construction: mesh_harmonic only. Analytic height-field debug is not a paper implementation and remains disabled unless explicitly exposed as debug.</div>',
        unsafe_allow_html=True,
    )
    w_planar = _param_row(
        "K3D の各 quad を平面に近づける重み。高いほど面のねじれを嫌う。論文の EPlanar 相当。",
        lambda: st.number_input("w_planar / EPlanar", min_value=0.1, max_value=50000.0, value=10000.0, step=500.0, help="Weight for K3D quad planarity."),
    )
    w_square = _param_row(
        "K3D の quad を極端に歪ませない重み。高いほど正方形・均整なタイルを保つ。",
        lambda: st.number_input("w_square / ESquare", min_value=0.1, max_value=100.0, value=10.0, step=0.5, help="Weight for square-like quad shape."),
    )
    w_surface = _param_row(
        "K3D を目標曲面 S に近づける重み。上げすぎると平面性や辺長一致と衝突しやすい。",
        lambda: st.number_input("w_surface / ESurface", min_value=0.0, max_value=500.0, value=0.1, step=0.1, help="Weight for closeness to target surface S."),
    )
    lift_tau = _param_row(
        "lift point 選択の強さ。値が大きいほど曲率・高さ変化に強く反応して持ち上げ点が選ばれやすい。",
        lambda: st.slider("lift coupling tau", 0.1, 1.0, 0.8, 0.05, help="Threshold/coupling for choosing lift points."),
    )
    channel_friction = _param_row(
        "紐経路の曲がり・摩擦コストの重み。高いほど曲がりの多い経路が不利になる。",
        lambda: st.slider("string channel friction", 0.0, 1.0, 0.2, 0.01, help="Estimated string-channel friction penalty."),
    )

    st.header("Hinge Layout Optimization")
    st.markdown(
        '<div class="param-section-note">T2D Top→Dual Hinge の配置最適化。タイルは剛体のまま、ヒンジ接続 E_Conn・衝突 E_Collision・初期配置への拘束を調整する。</div>',
        unsafe_allow_html=True,
    )
    hinge_layout_iterations = _param_row(
        "E_Hinge 配置最適化の最大反復回数。増やすと改善余地は増えるが、時間予算に達すると途中終了する。",
        lambda: st.slider("hinge layout iterations", 20, 400, 80, 20, help="Maximum iterations for the T2D hinge layout optimizer."),
    )
    hinge_layout_connection_weight = _param_row(
        "ヒンジ点を一致させる重み。高いほど接続は保つが、衝突回避の自由度は減る。",
        lambda: st.slider("hinge connection weight", 0.1, 20.0, 8.0, 0.1, help="Weight for E_Conn: pairwise hinge vertex coincidence."),
    )
    hinge_layout_collision_weight = _param_row(
        "厚み付きタイル footprint の非衝突重み。高すぎるとタイルが散り、低すぎると重なりが残る。",
        lambda: st.slider("hinge collision weight", 0.0, 4.0, 4.0, 0.05, help="Weight for E_Collision: separating overlapping T2D footprints."),
    )
    hinge_layout_anchor_weight = _param_row(
        "初期展開配置に留める重み。高いほど散らばりにくいが、衝突から逃げにくくなる。",
        lambda: st.slider("hinge layout anchor weight", 0.0, 0.5, 0.0, 0.005, help="Anchor/trust weight toward the initial fabrication layout."),
    )
    hinge_layout_initial_expansion = _param_row(
        "最適化前にタイル中心を少し外へ逃がす量。1.03〜1.10程度が通常。大きいと空洞が広がりすぎる。",
        lambda: st.slider(
            "hinge layout initial expansion",
            1.0,
            10.0,
            1.6,
            0.01,
            help="Bounded additive tile-center expansion before E_Hinge. Use 1.03-1.12 normally; large values create oversized voids.",
        ),
    )
    hinge_layout_max_center_drift_tiles = _param_row(
        "展開後の基準配置から各タイル中心が移動できる最大距離。大きいほど解けるが散らばりやすい。",
        lambda: st.slider(
            "hinge layout max center drift / tile",
            0.25,
            5.0,
            5.0,
            0.25,
            help="Trust-region radius measured in tile-size units around the expanded layout.",
        ),
    )
    hinge_layout_time_budget_sec = _param_row(
        "Top Hinge / Dual Hinge 各段階の最大計算時間。超えると partial convergence として止める。",
        lambda: st.slider("hinge layout time budget / stage", 1.0, 30.0, 8.0, 1.0, help="Maximum seconds spent in each hinge layout stage."),
    )
    hinge_layout_max_candidate_pairs = int(_param_row(
        "衝突判定する候補ペア数の上限。上げると正確だが重く、下げると衝突を見逃しやすい。",
        lambda: st.number_input("hinge layout max collision candidate pairs", min_value=200, max_value=20000, value=3000, step=200, help="Cap on candidate tile pairs for SAT footprint collision checks."),
    ))
    hinge_layout_collision_sweeps_per_iteration = int(_param_row(
        "1反復内で衝突分離を何回掃くか。増やすと重なりは減りやすいが計算が重くなる。",
        lambda: st.slider("collision sweeps per layout iteration", 1, 6, 2, 1, help="Number of collision separation passes per layout iteration."),
    ))

    st.header("Compute Backend")
    compute_backend = _param_row(
        "計算バックエンド。cuda はK3D/K2D/PDの一部で使うが、描画や一部ジオメトリ処理はCPU。",
        lambda: st.selectbox("compute backend", ["auto", "cpu", "cuda"], index=0, help="Requested backend for supported numeric stages."),
    )
    tensor_dtype = _param_row(
        "torch tensorの精度。float32は速く、float64は安定だが重い。GPUでは通常float32推奨。",
        lambda: st.selectbox("tensor dtype", ["float32", "float64"], index=0, help="Numeric precision for torch-based stages."),
    )

    st.header("Actuation Simulation")
    run_actuation = _param_row(
        "現在のT2Dから snap/lift のProjective Dynamics風シミュレーションを実行する。",
        lambda: st.button("Run snap/lift actuation"),
        tight_note=True,
    )
    sim_steps = _param_row(
        "時間方向のステップ数。多いほど滑らかだが重くなる。",
        lambda: st.slider("actuation steps", 8, 120, 48, 4, help="Number of simulated actuation frames/steps."),
    )
    solver_iterations = _param_row(
        "各ステップ内の制約投影回数。多いほど剛体・snap・lift制約に近づくが重い。",
        lambda: st.slider("solver iterations", 1, 60, 16, 1, help="Constraint projection iterations per simulation step."),
    )
    solver_substeps = _param_row(
        "1ステップをさらに分割する数。安定性を上げるが計算時間も増える。",
        lambda: st.slider("solver substeps", 1, 8, 1, 1, help="Substeps per simulation step."),
    )
    damping_ratio = _param_row(
        "速度の減衰。高いほど振動が少なく、準静的に動く。",
        lambda: st.slider("damping ratio", 0.0, 0.95, 0.2, 0.05, help="Velocity damping for dynamic simulation."),
    )
    rigid_weight = _param_row(
        "タイル剛体性の重み。高いほどパネル形状を保つ。アニメーションでパネルが歪む場合は 0.95〜1.0 推奨。",
        lambda: st.slider("rigid constraint weight", 0.0, 1.0, 0.95, 0.01, help="Simulation weight for tile rigidity. Use 0.95-1.0 for strict rigid panels."),
    )
    rigid_projection_passes = int(_param_row(
        "1回の制約反復内で剛体投影を何回かけるか。上げるほどパネル形状が崩れにくいが、ヒンジ/snapとの妥協が遅くなる。",
        lambda: st.slider("rigid projection passes", 1, 12, 4, 1, help="Extra Kabsch rigid projections per solver iteration. Higher means stricter panel rigidity."),
    ))
    rigid_guard_final_projection = _param_row(
        "各反復の最後に完全剛体投影をかける。ONでアニメーション中のパネル変形を強制的に戻す。",
        lambda: st.toggle("strict final rigid projection", value=True, help="Project each tile exactly back to its rigid rest shape after constraints."),
    )
    hinge_weight = _param_row(
        "ヒンジ接続の重み。高いほど接続点を保つが、衝突解消と競合することがある。",
        lambda: st.slider("hinge constraint weight", 0.0, 1.0, 0.85, 0.01, help="Simulation weight for hinge connectivity."),
    )
    snap_weight = _param_row(
        "紐でgapを閉じるsnap制約の重み。高いほど急に締まる。",
        lambda: st.slider("snap constraint weight", 0.0, 1.0, 0.65, 0.01, help="Simulation weight for snap constraints along the string path."),
    )
    lift_weight = _param_row(
        "lift pointをT3D側の目標位置へ近づける重み。高いほど盛り上がりを強く誘導する。",
        lambda: st.slider("lift constraint weight", 0.0, 1.0, 0.9, 0.01, help="Simulation weight for lift constraints."),
    )
    collision_weight = _param_row(
        "シミュレーション中の衝突回避重み。高いほどタイル同士のめり込みを避けるが不安定になる場合がある。",
        lambda: st.slider("collision constraint weight", 0.0, 1.0, 0.25, 0.01, help="Simulation collision avoidance weight."),
    )
    target_fit_weight = _param_row(
        "T3D目標姿勢へ各タイルを剛体的に寄せる強さ。緑のパネルが青いT3Dからずれる/潜る場合に上げる。",
        lambda: st.slider("target T3D fit guard", 0.0, 1.0, 0.30, 0.01, help="Rigid per-tile pull toward the corresponding T3D pose during deployment."),
    )
    target_contact_guard_weight = _param_row(
        "T3D表面への一方向接触ガード。高いほど青い目標形状へ食い込む緑パネルを外側へ押し戻す。",
        lambda: st.slider("target penetration guard", 0.0, 1.0, 0.85, 0.01, help="One-sided guard that pushes tiles outward when they go inside the target T3D pose."),
    )
    target_contact_start_alpha = _param_row(
        "T3Dめり込み防止を開始する進行率。低くすると早い段階から表面に沿わせ、高くすると終盤だけ効く。",
        lambda: st.slider("target guard start progress", 0.0, 0.95, 0.60, 0.05, help="Actuation progress alpha at which target penetration guard begins."),
    )
    target_contact_clearance = _param_row(
        "T3D表面からの最小余白。少し上げると青い目標面に対して緑パネルが外側に残りやすい。",
        lambda: st.slider("target clearance", 0.0, 0.10, 0.0, 0.005, help="Small positive clearance from the target T3D surface."),
    )
    target_contact_projection_passes = int(_param_row(
        "1反復内でT3Dめり込み防止投影を何回行うか。上げるほど食い込みに強いが重くなる。",
        lambda: st.slider("target guard projection passes", 1, 6, 2, 1, help="Number of target contact guard projections per solver iteration."),
    ))
    debug_all_pair_collision = _param_row(
        "デバッグ用。全ペア衝突を調べるため重い。通常はOFF。",
        lambda: st.toggle("debug all-pairs collision", value=False, help="Debug mode; expensive for large grids."),
    )
    store_animation_frames = _param_row(
        "全フレーム保存。ONで再生しやすいがCPU転送・メモリ使用が増える。GPU確認時はOFF推奨。",
        lambda: st.toggle("store animation frames during solve", value=False, help="Off is faster and keeps deployment on GPU longer. Turn on only when you need the full animation frames."),
    )

    st.header("High-Fidelity Physical Mode")
    high_fidelity = _param_row(
        "より物理寄りの追加項を使う実験モード。重くなり、まだ近似的。",
        lambda: st.toggle("enable high-fidelity mode", value=False, help="Enable additional physical terms for experimental simulations."),
    )
    hinge_rotational_stiffness = _param_row(
        "ヒンジが曲がりに抵抗する強さ。高いほど折れにくくなる。",
        lambda: st.slider("hinge rotational stiffness", 0.0, 2.0, 0.25, 0.05, help="Rotational stiffness around hinges."),
    )
    hinge_damping = _param_row(
        "ヒンジ回転の減衰。高いほど振動しにくい。",
        lambda: st.slider("hinge damping", 0.0, 1.0, 0.2, 0.05, help="Rotational damping around hinges."),
    )
    tile_mass = _param_row(
        "タイル質量・慣性の代理値。動的応答や重力の効き方に影響する。",
        lambda: st.number_input("tile mass and inertia proxy", min_value=0.01, max_value=10.0, value=1.0, step=0.1, help="Mass/inertia proxy for high-fidelity mode."),
    )
    gravity = _param_row(
        "重力加速度。吊り下げ・持ち上げ時の下方向影響を調整する。",
        lambda: st.slider("gravity", 0.0, 20.0, 9.81, 0.1, help="Gravity acceleration used in high-fidelity mode."),
    )
    contact_friction = _param_row(
        "接触摩擦。高いほど接触時に滑りにくくなる。",
        lambda: st.slider("contact friction", 0.0, 2.0, 0.25, 0.05, help="Contact friction in high-fidelity mode."),
    )
    quasi_static_pulling_speed = _param_row(
        "紐を引く速度の代理値。低いほど準静的、高いほど動的効果が出やすい。",
        lambda: st.slider("quasi-static pulling speed", 0.1, 2.0, 1.0, 0.1, help="Pulling speed proxy for actuation."),
    )

    st.header("Setting Meters")
    st.markdown(
        '<div class="param-section-note">現在のスライダー設定から推定した負荷・難しさ・空洞の出やすさです。実測値ではなく、調整の目安として使います。</div>',
        unsafe_allow_html=True,
    )
    sidebar_meters = _compute_setting_meters(
        target_kind=target_kind,
        grid_size=grid_size,
        surface_mesh_subdivisions=surface_mesh_subdivisions,
        tile_size=tile_size,
        gap_size=gap_size,
        thickness=thickness,
        amplitude=amplitude,
        max_3d_iterations=max_3d_iterations,
        max_2d_iterations=max_2d_iterations,
        strict_k2d_time_budget_sec=strict_k2d_time_budget_sec,
        hinge_layout_iterations=hinge_layout_iterations,
        hinge_layout_connection_weight=hinge_layout_connection_weight,
        hinge_layout_collision_weight=hinge_layout_collision_weight,
        hinge_layout_anchor_weight=hinge_layout_anchor_weight,
        hinge_layout_initial_expansion=hinge_layout_initial_expansion,
        hinge_layout_max_center_drift_tiles=hinge_layout_max_center_drift_tiles,
        hinge_layout_max_candidate_pairs=hinge_layout_max_candidate_pairs,
        hinge_layout_collision_sweeps_per_iteration=hinge_layout_collision_sweeps_per_iteration,
        compute_backend=compute_backend,
        tensor_dtype=tensor_dtype,
        sim_steps=sim_steps,
        solver_iterations=solver_iterations,
        solver_substeps=solver_substeps,
        store_animation_frames=store_animation_frames,
        high_fidelity=high_fidelity,
    )
    for meter_label, (meter_score, meter_note) in sidebar_meters.items():
        _meter_bar(meter_label, meter_score, meter_note)


def _assembly_mesh_and_edge_traces(vertices: np.ndarray, *, color: str, opacity: float, name: str) -> tuple[go.Mesh3d, go.Scatter3d]:
    """Return stable mesh and edge traces for browser-side playback."""
    vertices = np.asarray(vertices, dtype=float)
    faces = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    x: list[float] = []
    y: list[float] = []
    z: list[float] = []
    i_idx: list[int] = []
    j_idx: list[int] = []
    k_idx: list[int] = []
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    for tile in vertices:
        for face in faces:
            base = len(x)
            pts = tile[list(face)]
            x.extend(pts[:, 0].tolist())
            y.extend(pts[:, 1].tolist())
            z.extend(pts[:, 2].tolist())
            i_idx.extend([base, base])
            j_idx.extend([base + 1, base + 2])
            k_idx.extend([base + 2, base + 3])
        for edge in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
            pts = tile[list(edge)]
            edge_x.extend([float(pts[0, 0]), float(pts[1, 0]), None])
            edge_y.extend([float(pts[0, 1]), float(pts[1, 1]), None])
            edge_z.extend([float(pts[0, 2]), float(pts[1, 2]), None])
    mesh = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i_idx,
        j=j_idx,
        k=k_idx,
        color=color,
        opacity=opacity,
        flatshading=True,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
        name=name,
        showlegend=True,
    )
    edges = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(color="#111827", width=2),
        name=f"{name} edges",
        showlegend=False,
    )
    return mesh, edges


def _smooth_browser_tile_animation(
    frames: list[np.ndarray],
    target_vertices: np.ndarray,
    *,
    max_tiles: int,
    show_target: bool,
    fps: int,
    title: str,
) -> go.Figure:
    """Browser-side Plotly animation for smoother playback and movable camera."""
    safe_frames = [np.asarray(frame, dtype=float) for frame in frames if len(frame)]
    if not safe_frames:
        safe_frames = [np.asarray(target_vertices, dtype=float)]
    limit = max(1, min(int(max_tiles), len(safe_frames[0]), len(target_vertices)))
    safe_frames = [frame[:limit] for frame in safe_frames]
    target = np.asarray(target_vertices[:limit], dtype=float)
    duration_ms = max(16, int(1000 / max(1, int(fps))))

    fig = go.Figure()
    tile_mesh, tile_edges = _assembly_mesh_and_edge_traces(safe_frames[0], color="#2dd4bf", opacity=0.74, name="animated tiles")
    fig.add_trace(tile_mesh)
    fig.add_trace(tile_edges)
    if show_target:
        target_mesh, target_edges = _assembly_mesh_and_edge_traces(target, color="#2563eb", opacity=0.16, name="T3D target")
        fig.add_trace(target_mesh)
        fig.add_trace(target_edges)

    plotly_frames: list[go.Frame] = []
    for frame_id, vertices in enumerate(safe_frames):
        mesh, edges = _assembly_mesh_and_edge_traces(vertices, color="#2dd4bf", opacity=0.74, name="animated tiles")
        plotly_frames.append(go.Frame(data=[mesh, edges], traces=[0, 1], name=str(frame_id)))
    fig.frames = plotly_frames
    fig.update_layout(
        title=title,
        height=720,
        margin=dict(l=0, r=0, b=0, t=42),
        uirevision="onestring-camera-preserved",
        scene=dict(
            aspectmode="data",
            uirevision="onestring-camera-preserved",
            dragmode="orbit",
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.02,
                y=0.02,
                xanchor="left",
                yanchor="bottom",
                buttons=[
                    dict(label="Smooth play", method="animate", args=[None, {"frame": {"duration": duration_ms, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": True, "mode": "immediate"}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}, "mode": "immediate"}]),
                ],
            )
        ],
        sliders=[
            dict(
                x=0.12,
                y=0.02,
                len=0.82,
                currentvalue=dict(prefix="frame "),
                steps=[
                    dict(method="animate", args=[[frame.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}], label=str(index + 1))
                    for index, frame in enumerate(plotly_frames)
                ],
            )
        ],
    )
    return fig


def build_target():
    if uploaded is None:
        radius = max(1.5, grid_size * tile_size * 0.7)
        shape_params = {"amplitude": amplitude, "radius": radius, "sigma": radius * 0.45}
        if target_kind == "wave":
            # Default wave was too steep for K2D/T2D linkage tests.
            # Use a longer wavelength and an internal damping factor while
            # preserving the existing amplitude slider semantics.
            shape_params.update({"wavelength": radius * 1.55, "wave_amplitude_scale": 0.35})
        elif target_kind == "half_gourd":
            # A non-rectangular upper half-shell of a hyotan/gourd.  It provides
            # an open surface boundary so Ω and M2D cropping can be checked with
            # something more interesting than a rectangular height field.
            shape_params.update({"gourd_neck": 0.42, "gourd_lobe_separation": 0.88})
        return create_builtin_shape(target_kind, shape_params)
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fh:
        fh.write(uploaded.getbuffer())
        temp_path = fh.name
    st.warning(CLOSED_SHAPE_WARNING)
    return load_target_shape(temp_path)


def current_pipeline_key() -> tuple:
    return (
        selected_model_version["id"],
        target_kind,
        uploaded.name if uploaded else None,
        grid_size,
        omega_overlay_margin,
        m2d_crop_policy,
        omega_boundary_mode,
        omega_parameterization_mode,
        bff_boundary_policy,
        bff_executable,
        reference_grid_spacing_value,
        reference_grid_rotation_degrees,
        reference_grid_origin_u,
        reference_grid_origin_v,
        reference_csf_normalization,
        reference_stop_on_required_split,
        boundary_target_aspect_mode,
        boundary_target_aspect_ratio,
        boundary_sliding_max_iterations,
        boundary_sliding_step_size,
        boundary_sliding_energy_tolerance,
        boundary_sliding_min_spacing,
        boundary_sliding_length_weight,
        boundary_sliding_spacing_weight,
        boundary_sliding_flip_area_epsilon,
        boundary_sliding_line_search_max_steps,
        allow_experimental_pipeline,
        strict_k2d_time_budget_sec,
        strict_k2d_scipy_vertex_limit,
        gpu_first_mode,
        surface_mesh_subdivisions,
        tile_size,
        gap_size,
        thickness,
        amplitude,
        max_3d_iterations,
        max_2d_iterations,
        m3d_construction_mode,
        w_planar,
        w_square,
        w_surface,
        lift_tau,
        channel_friction,
        hinge_layout_iterations,
        hinge_layout_connection_weight,
        hinge_layout_collision_weight,
        hinge_layout_anchor_weight,
        hinge_layout_initial_expansion,
        hinge_layout_max_center_drift_tiles,
        hinge_layout_time_budget_sec,
        hinge_layout_max_candidate_pairs,
        hinge_layout_collision_sweeps_per_iteration,
        compute_backend,
        tensor_dtype,
    )


def current_actuation_key() -> tuple:
    return (
        current_pipeline_key(),
        sim_steps,
        solver_iterations,
        solver_substeps,
        damping_ratio,
        rigid_weight,
        rigid_projection_passes,
        rigid_guard_final_projection,
        hinge_weight,
        snap_weight,
        lift_weight,
        collision_weight,
        target_fit_weight,
        target_contact_guard_weight,
        target_contact_start_alpha,
        target_contact_clearance,
        target_contact_projection_passes,
        high_fidelity,
        hinge_rotational_stiffness,
        hinge_damping,
        tile_mass,
        gravity,
        contact_friction,
        quasi_static_pulling_speed,
        debug_all_pair_collision,
    )


effective_m3d_construction_mode = "mesh_harmonic"
effective_surface_mesh_subdivisions = surface_mesh_subdivisions

pipeline_params = PipelineParameters(
    nx=grid_size,
    ny=grid_size,
    tile_size=tile_size,
    gap_size=gap_size,
    thickness=thickness,
    max_3d_iterations=max_3d_iterations,
    max_2d_iterations=max_2d_iterations,
    m3d_construction_mode=effective_m3d_construction_mode,
    surface_mesh_subdivisions=effective_surface_mesh_subdivisions,
    omega_overlay_margin=omega_overlay_margin,
    m2d_crop_policy=m2d_crop_policy,
    omega_boundary_mode=omega_boundary_mode,
    omega_parameterization_mode=omega_parameterization_mode,
    bff_executable=bff_executable or None,
    bff_boundary_policy=bff_boundary_policy,
    reference_grid_spacing=reference_grid_spacing_value if reference_grid_spacing_value > 0.0 else None,
    reference_grid_rotation_degrees=reference_grid_rotation_degrees,
    reference_grid_origin_u=reference_grid_origin_u,
    reference_grid_origin_v=reference_grid_origin_v,
    reference_csf_normalization=reference_csf_normalization,
    reference_stop_on_required_split=reference_stop_on_required_split,
    boundary_target_shape="rectangle",
    boundary_target_aspect_mode=boundary_target_aspect_mode,
    boundary_target_aspect_ratio=boundary_target_aspect_ratio,
    boundary_sliding_max_iterations=boundary_sliding_max_iterations,
    boundary_sliding_step_size=boundary_sliding_step_size,
    boundary_sliding_energy_tolerance=boundary_sliding_energy_tolerance,
    boundary_sliding_min_spacing=boundary_sliding_min_spacing,
    boundary_sliding_length_weight=boundary_sliding_length_weight,
    boundary_sliding_spacing_weight=boundary_sliding_spacing_weight,
    boundary_sliding_flip_area_epsilon=boundary_sliding_flip_area_epsilon,
    boundary_sliding_line_search_max_steps=boundary_sliding_line_search_max_steps,
    allow_experimental_pipeline=allow_experimental_pipeline,
    strict_k2d_time_budget_sec=strict_k2d_time_budget_sec,
    strict_k2d_scipy_vertex_limit=strict_k2d_scipy_vertex_limit,
    w_planar=w_planar,
    w_square=w_square,
    w_surface=w_surface,
    lift_tau=lift_tau,
    channel_friction=channel_friction,
    hinge_layout_iterations=hinge_layout_iterations,
    hinge_layout_connection_weight=hinge_layout_connection_weight,
    hinge_layout_collision_weight=hinge_layout_collision_weight,
    hinge_layout_anchor_weight=hinge_layout_anchor_weight,
    hinge_layout_initial_expansion=hinge_layout_initial_expansion,
    hinge_layout_max_center_drift_tiles=hinge_layout_max_center_drift_tiles,
    hinge_layout_time_budget_sec=hinge_layout_time_budget_sec,
    hinge_layout_max_candidate_pairs=hinge_layout_max_candidate_pairs,
    hinge_layout_collision_sweeps_per_iteration=hinge_layout_collision_sweeps_per_iteration,
    strict_paper_flow=True,
    model_version=selected_model_version["id"],
    t3d_intersection_trim_enabled=bool(selected_model_version.get("t3d_intersection_trim_enabled", False)),
    compute=ComputeConfig(backend=compute_backend, dtype=tensor_dtype),
)

pipeline_key = current_pipeline_key()
if "onestring_state" not in st.session_state and not run_pipeline:
    st.info("Set parameters, then click Run OneString pipeline. The paper-default path stops at unimplemented stages; enable the experimental pipeline explicitly to run the old prototype path.")
    st.stop()

if run_pipeline or st.session_state.get("pipeline_key") != pipeline_key:
    with st.spinner("Building OneString pipeline state"):
        progress_bar = st.progress(0.0, text="Preparing pipeline…")
        progress_status = st.empty()
        progress_log: list[dict[str, object]] = []
        progress_started = time.perf_counter()

        def _build_progress(stage: str, fraction: float, detail: str = "") -> None:
            fraction = max(0.0, min(1.0, float(fraction)))
            elapsed = time.perf_counter() - progress_started
            progress_bar.progress(fraction, text=f"{fraction * 100:5.1f}%  {stage}")
            progress_status.caption(f"{elapsed:7.2f}s  {stage}" + (f" — {detail}" if detail else ""))
            progress_log.append({"elapsed_sec": elapsed, "progress_%": fraction * 100.0, "stage": stage, "detail": detail})

        try:
            st.session_state.onestring_state = build_onestring_design(
                build_target(),
                pipeline_params,
                progress_callback=_build_progress,
            )
        except Exception as exc:
            st.session_state.pipeline_progress_log = progress_log
            st.exception(exc)
            st.stop()
        _build_progress("Done", 1.0, "Pipeline state stored")
        st.session_state.pipeline_progress_log = progress_log
        st.session_state.pipeline_key = pipeline_key
        st.session_state.pop("actuation_key", None)

state = st.session_state.onestring_state

deployment_params = DeploymentParameters(
    steps=sim_steps,
    solver_iterations=solver_iterations,
    rigid_weight=rigid_weight,
    rigid_projection_passes=rigid_projection_passes,
    rigid_guard_final_projection=rigid_guard_final_projection,
    hinge_weight=hinge_weight,
    snap_weight=snap_weight,
    lift_weight=lift_weight,
    collision_weight=collision_weight,
    target_fit_weight=target_fit_weight,
    target_contact_guard_weight=target_contact_guard_weight,
    target_contact_start_alpha=target_contact_start_alpha,
    target_contact_clearance=target_contact_clearance,
    target_contact_projection_passes=target_contact_projection_passes,
    damping_ratio=damping_ratio,
    quasi_static_pull_speed=quasi_static_pulling_speed,
    high_fidelity=high_fidelity,
    hinge_rotational_stiffness=hinge_rotational_stiffness,
    hinge_damping=hinge_damping,
    tile_mass=tile_mass,
    gravity=gravity,
    contact_friction=contact_friction,
    string_channel_friction=channel_friction,
    solver_substeps=solver_substeps,
    debug_all_pair_collision=debug_all_pair_collision,
    store_animation_frames=store_animation_frames,
    max_animation_frames=48,
    snap_scope="string_path_only",
    use_target_gap_contraction=True,
    compute=ComputeConfig(backend=compute_backend, dtype=tensor_dtype),
)

actuation_key = current_actuation_key()
if state.simulation_result is None:
    st.info("Run snap/lift actuation to simulate deployment against T3D. Central tendon and debug goal attraction are not used in this mode.")
elif st.session_state.get("actuation_key") != actuation_key:
    st.warning("Actuation settings changed. Run snap/lift actuation again to refresh the simulation.")

if run_actuation:
    with st.spinner("Solving Projective-Dynamics-style snap/lift actuation"):
        progress_bar = st.progress(0.0, text="Preparing deployment simulation…")
        progress_status = st.empty()
        progress_log: list[dict[str, object]] = []
        progress_started = time.perf_counter()

        def _actuation_progress(stage: str, fraction: float, detail: str = "") -> None:
            fraction = max(0.0, min(1.0, float(fraction)))
            elapsed = time.perf_counter() - progress_started
            progress_bar.progress(fraction, text=f"{fraction * 100:5.1f}%  {stage}")
            progress_status.caption(f"{elapsed:7.2f}s  {stage}" + (f" — {detail}" if detail else ""))
            progress_log.append({"elapsed_sec": elapsed, "progress_%": fraction * 100.0, "stage": stage, "detail": detail})

        try:
            state.simulation_result = simulate_onestring_deployment(
                state,
                deployment_params,
                progress_callback=_actuation_progress,
            )
        except RuntimeError as exc:
            st.session_state.actuation_progress_log = progress_log
            st.error(str(exc))
            st.stop()
        _actuation_progress("Done", 1.0, "Deployment simulation complete")
        st.session_state.actuation_progress_log = progress_log
        st.session_state.actuation_key = actuation_key

view_stage = st.selectbox(
    "View stage",
    [
        "Pipeline View",
        "S",
        "Split Map",
        "Omega",
        "Mode Comparison",
        "M2D",
        "M3D",
        "K3D",
        "T3D",
        "K2D",
        "T2D Top Hinge",
        "T2D Dual Hinge",
        "Lift Points",
        "String Path",
        "Assembly Animation",
        "Final Deployed",
        "Comparison",
        "Metrics",
        "Paper Consistency Audit",
        "Setting Meters",
        "Complexity / Backend",
        "Performance",
        "Approximations",
    ],
)

if view_stage == "Pipeline View":
    st.plotly_chart(figure_pipeline_overview(state), width="stretch", key="pipeline_overview")
    selected_report = st.selectbox("stage report", list(state.stage_reports.keys()))
    report = state.stage_reports[selected_report]
    cols = st.columns(4)
    cols[0].metric("before error", f"{report.before_error:.4g}")
    cols[1].metric("after error", f"{report.after_error:.4g}")
    cols[2].metric("constraint violation", f"{report.constraint_violation:.4g}")
    cols[3].metric("time", f"{report.computation_time:.3f}s")
    st.write({"objective": report.objective, "counts": report.counts, "failed_constraints": report.failed_constraints})
elif view_stage == "S":
    st.plotly_chart(figure_surface_mesh(state.target_surface), width="stretch", key="target_surface")
elif view_stage == "Split Map":
    st.plotly_chart(figure_split_mapping(state), width="stretch", key="split_mapping")
    st.write(
        {
            "meaning": "Red samples show the Omega split line mapped back onto S through the stored S->Omega parameterization.",
            "csf_split_threshold": state.mesh_2d_initial.metrics.get("csf_split_threshold"),
            "max_csf_before_split": state.mesh_2d_initial.metrics.get("max_csf_before_split"),
            "max_csf_after_split": state.mesh_2d_initial.metrics.get("max_csf_after_split"),
            "csf_split_step_analysis_model": state.mesh_2d_initial.metrics.get("csf_split_step_analysis_model"),
            "csf_split_residual_high_vertex_count_after_all": state.mesh_2d_initial.metrics.get("csf_split_residual_high_vertex_count_after_all"),
            "csf_split_additional_split_recommended_after_all": state.mesh_2d_initial.metrics.get("csf_split_additional_split_recommended_after_all"),
            "number_of_splits": state.mesh_2d_initial.metrics.get("number_of_splits"),
            "split_locations": state.mesh_2d_initial.metrics.get("split_locations"),
            "csf_split_model": state.mesh_2d_initial.metrics.get("csf_split_model"),
            "csf_split_removed_quad_count": state.mesh_2d_initial.metrics.get("csf_split_removed_quad_count"),
            "csf_split_duplicated_vertex_count": state.mesh_2d_initial.metrics.get("csf_split_duplicated_vertex_count"),
        }
    )
    split_steps = state.mesh_2d_initial.metrics.get("csf_split_step_analysis", [])
    if split_steps:
        st.caption("CSF residual analysis after each Split step")
        display_steps = []
        for row in split_steps:
            clean = dict(row)
            clean["next_recommended_split"] = str(clean.get("next_recommended_split", []))
            display_steps.append(clean)
        st.dataframe(display_steps, width="stretch")
elif view_stage == "Omega":
    st.plotly_chart(figure_domain(state), width="stretch", key="omega_domain")
    omega_info = {
        "parameterization": state.conformal_domain.method,
        "surface_parameterization_method": state.surface_parameterization.method,
        "m3d_construction": state.mesh_3d_initial.metrics.get("m3d_construction_method", ""),
        "height_field_shortcut_used": state.mesh_3d_initial.metrics.get("m3d_used_height_field_shortcut", False),
        "omega_corresponds_to_S": state.surface_parameterization.metrics.get("omega_corresponds_to_S", False),
        "omega_correspondence_model": state.surface_parameterization.metrics.get("omega_correspondence_model", ""),
        "flattening_backend": state.surface_parameterization.metrics.get("flattening_backend", ""),
        "bff_backend_used": state.surface_parameterization.metrics.get("bff_backend_used", ""),
        "bff_reference_backend_available": state.surface_parameterization.metrics.get("bff_reference_backend_available", False),
        "bff_implemented": state.surface_parameterization.metrics.get("bff_implemented", False),
        "omega_boundary_shape": state.surface_parameterization.metrics.get("omega_boundary_shape", ""),
        "omega_boundary_forced_rectangle": state.surface_parameterization.metrics.get("omega_boundary_forced_rectangle", False),
        "uv_triangle_flip_count": state.surface_parameterization.metrics.get("uv_triangle_flip_count", 0),
        "uv_degenerate_triangle_count": state.surface_parameterization.metrics.get("uv_degenerate_triangle_count", 0),
        "uv_min_triangle_area": state.surface_parameterization.metrics.get("uv_min_triangle_area", 0.0),
        "boundary_self_intersection_count": state.surface_parameterization.metrics.get("boundary_self_intersection_count", 0),
        "angle_distortion_mean_deg": state.surface_parameterization.metrics.get("angle_distortion_mean_deg", 0.0),
        "angle_distortion_max_deg": state.surface_parameterization.metrics.get("angle_distortion_max_deg", 0.0),
        "edge_stretch_median": state.surface_parameterization.metrics.get("edge_stretch_median", 0.0),
        "edge_stretch_p95": state.surface_parameterization.metrics.get("edge_stretch_p95", 0.0),
        "edge_stretch_max": state.surface_parameterization.metrics.get("edge_stretch_max", 0.0),
        "csf_median": state.surface_parameterization.metrics.get("csf_median", 0.0),
        "csf_p95": state.surface_parameterization.metrics.get("csf_p95", 0.0),
        "csf_max": state.surface_parameterization.metrics.get("csf_max", 0.0),
        "lambda_min": state.surface_parameterization.metrics.get("lambda_min", ""),
        "lambda_median": state.surface_parameterization.metrics.get("lambda_median", ""),
        "lambda_max": state.surface_parameterization.metrics.get("lambda_max", ""),
        "lambda_normalization_status": state.surface_parameterization.metrics.get("lambda_normalization_status", ""),
        "anisotropy_max": max(state.surface_parameterization.metrics.get("per_triangle_anisotropy", [0.0]) or [0.0]),
        "internal_triangle_overlap_count": state.surface_parameterization.metrics.get("internal_triangle_overlap_count", 0),
        "boundary_target_shape": state.surface_parameterization.metrics.get("boundary_target_shape", ""),
        "boundary_target_aspect_ratio": state.surface_parameterization.metrics.get("boundary_target_aspect_ratio", ""),
        "boundary_corner_vertex_ids": state.surface_parameterization.metrics.get("boundary_corner_vertex_ids", []),
        "boundary_sliding_iterations": state.surface_parameterization.metrics.get("boundary_sliding_iterations", ""),
        "boundary_sliding_converged": state.surface_parameterization.metrics.get("boundary_sliding_converged", ""),
        "boundary_sliding_stop_reason": state.surface_parameterization.metrics.get("boundary_sliding_stop_reason", ""),
        "lscm_energy_free_boundary_initial": state.surface_parameterization.metrics.get("lscm_energy_free_boundary_initial", ""),
        "lscm_energy_constrained_initial": state.surface_parameterization.metrics.get("lscm_energy_constrained_initial", ""),
        "lscm_energy_final": state.surface_parameterization.metrics.get("lscm_energy_final", ""),
        "boundary_length_energy_initial": state.surface_parameterization.metrics.get("boundary_length_energy_initial", ""),
        "boundary_length_energy_final": state.surface_parameterization.metrics.get("boundary_length_energy_final", ""),
        "boundary_spacing_energy_initial": state.surface_parameterization.metrics.get("boundary_spacing_energy_initial", ""),
        "boundary_spacing_energy_final": state.surface_parameterization.metrics.get("boundary_spacing_energy_final", ""),
        "boundary_target_rms_error": state.surface_parameterization.metrics.get("boundary_target_rms_error", ""),
        "boundary_target_max_error": state.surface_parameterization.metrics.get("boundary_target_max_error", ""),
        "boundary_order_violation_count": state.surface_parameterization.metrics.get("boundary_order_violation_count", ""),
        "parameterization_runtime_seconds": state.surface_parameterization.metrics.get("parameterization_runtime_seconds", ""),
        "omega_warning": state.surface_parameterization.metrics.get("omega_warning", ""),
        "m3d_uv_triangle_lookup_fail_count": state.mesh_3d_initial.metrics.get("m3d_uv_triangle_lookup_fail_count", 0),
        "m3d_outside_omega_count": state.mesh_3d_initial.metrics.get("m3d_outside_omega_count", 0),
        "m3d_surface_distance_mean": state.mesh_3d_initial.metrics.get("m3d_surface_distance_mean", 0.0),
        "m3d_surface_distance_max": state.mesh_3d_initial.metrics.get("m3d_surface_distance_max", 0.0),
        "m3d_surface_triangle_hit_fraction": state.mesh_3d_initial.metrics.get("m3d_surface_triangle_hit_fraction", 0.0),
        "m3d_round_trip_error_rms": state.mesh_3d_initial.metrics.get("m3d_round_trip_error_rms", ""),
        "fallbacks_used": state.surface_parameterization.metrics.get("fallbacks_used", []),
        "max_csf_before_split": state.mesh_2d_initial.metrics["max_csf_before_split"],
        "max_csf_after_split": state.mesh_2d_initial.metrics["max_csf_after_split"],
        "number_of_splits": state.mesh_2d_initial.metrics["number_of_splits"],
        "split_locations": state.mesh_2d_initial.metrics["split_locations"],
    }
    if not omega_info["omega_corresponds_to_S"]:
        st.error("Ω is not a paper conformal parameterization in this run. It is debug/experimental unless a real paper parameterization is implemented.")
    elif state.surface_parameterization.method == "boundary_sliding_lscm":
        st.info("Boundary-controlled LSCM approximation is active. This mode is intentionally separate from reference BFF.")
    elif not omega_info["bff_implemented"]:
        st.warning("Omega is not a reference Boundary First Flattening implementation. No BFF claim is made for this mode.")
    st.write(omega_info)
elif view_stage == "Mode Comparison":
    st.caption(
        "S→M3Dの3モード比較です。spacing / rotation / origin と fully-contained crop を共通化します。"
        "K3D以降は現行選択モードだけが下流ビューに表示され、paper-referenceとは扱いません。"
    )
    comparison_key = (
        "paper_reference_mode_comparison_v1",
        pipeline_key,
        reference_grid_spacing_value,
        reference_grid_rotation_degrees,
        reference_grid_origin_u,
        reference_grid_origin_v,
        bff_boundary_policy,
        bff_executable,
    )
    if st.session_state.get("mode_comparison_key") != comparison_key:
        comparison_states = {}
        comparison_target = build_target()
        comparison_grid = _onestring_pipeline.create_quad_grid(grid_size, grid_size, tile_size, gap_size)
        comparison_surface = _onestring_pipeline._build_surface_mesh(
            comparison_target,
            comparison_grid,
            effective_surface_mesh_subdivisions,
        )
        with st.spinner("Building shared-grid S→M3D comparison"):
            for comparison_mode in ("rectangular_harmonic_legacy", "lscm_free_boundary", "paper_reference_bff"):
                try:
                    comparison_params = replace(
                        pipeline_params,
                        omega_parameterization_mode=comparison_mode,
                        reference_stop_on_required_split=False,
                        enable_csf_splits=False,
                        enable_heuristic_csf_split=False,
                        enable_peak_guided_split=False,
                        enable_mirror_split=False,
                    )
                    parameterization = _onestring_pipeline._build_surface_parameterization(
                        comparison_surface,
                        comparison_target,
                        comparison_grid,
                        comparison_params,
                    )
                    if comparison_mode == "paper_reference_bff":
                        domain = _onestring_pipeline._flatten_to_domain(parameterization, comparison_grid, comparison_params)
                    else:
                        parameterization.metrics["per_triangle_lambda"] = np.ones(len(comparison_surface.faces), dtype=float).tolist()
                        domain = _onestring_pipeline._reference_flatten_to_domain(parameterization, comparison_grid, comparison_params)
                    mesh_2d = _onestring_pipeline._build_reference_m2d(comparison_grid, domain, comparison_params)
                    mesh_3d, _ = _onestring_pipeline._lift_m2d_to_m3d(
                        comparison_target,
                        mesh_2d,
                        parameterization,
                        comparison_params,
                    )
                    comparison_states[comparison_mode] = SimpleNamespace(
                        target_surface=comparison_surface,
                        surface_parameterization=parameterization,
                        conformal_domain=domain,
                        mesh_2d_initial=mesh_2d,
                        mesh_3d_initial=mesh_3d,
                    )
                except Exception as exc:
                    comparison_states[comparison_mode] = exc
        st.session_state.mode_comparison_states = comparison_states
        st.session_state.mode_comparison_key = comparison_key
    comparison_states = st.session_state.mode_comparison_states
    for column, comparison_mode in zip(
        st.columns(3),
        ("rectangular_harmonic_legacy", "lscm_free_boundary", "paper_reference_bff"),
    ):
        with column:
            st.subheader(comparison_mode)
            comparison = comparison_states[comparison_mode]
            if isinstance(comparison, Exception):
                st.error(f"{type(comparison).__name__}: {comparison}")
                continue
            st.plotly_chart(figure_domain(comparison), width="stretch", key=f"compare_omega_{comparison_mode}")
            st.plotly_chart(figure_quad_mesh(comparison.mesh_2d_initial, title="M2D"), width="stretch", key=f"compare_m2d_{comparison_mode}")
            st.plotly_chart(figure_m3d_overlay(comparison), width="stretch", key=f"compare_m3d_{comparison_mode}")
            metrics = comparison.surface_parameterization.metrics
            st.write(
                {
                    "backend": metrics.get("flattening_backend"),
                    "flip_count": metrics.get("uv_triangle_flip_count", 0),
                    "boundary_intersections": metrics.get("boundary_self_intersection_count", 0),
                    "internal_overlaps": metrics.get("internal_triangle_overlap_count", 0),
                    "lambda_max": metrics.get("lambda_max", "not available in legacy mode"),
                    "anisotropy_max": max(metrics.get("per_triangle_anisotropy", [0.0]) or [0.0]),
                    "M3D_round_trip_error": comparison.mesh_3d_initial.metrics.get("m3d_round_trip_error_rms", "legacy diagnostic unavailable"),
                    "M3D_surface_error": comparison.mesh_3d_initial.metrics.get("m3d_surface_distance_max", 0.0),
                }
            )
elif view_stage in {"M2D", "K3D"}:
    mesh = {"M2D": state.mesh_2d_initial, "M3D": state.mesh_3d_initial, "K3D": state.mesh_3d_optimized}[view_stage]
    st.plotly_chart(figure_quad_mesh(mesh, title=view_stage), width="stretch", key=f"mesh_{view_stage}")
    st.write(mesh.metrics)
elif view_stage == "M3D":
    st.plotly_chart(figure_m3d_overlay(state), width="stretch", key="mesh_M3D")
    if state.mesh_3d_initial.metrics.get("m3d_used_height_field_shortcut", False):
        st.warning("M3D is generated by analytic scaled height-field debug shortcut, not by mesh parameterization.")
    st.write(
        {
            "m3d_metrics": state.mesh_3d_initial.metrics,
            "parameterization_metrics": state.surface_parameterization.metrics,
        }
    )
elif view_stage == "K2D":
    st.plotly_chart(
        figure_flat_tile_layout(state.k2d_flat_layout, title="K2D flat independent tile layout", hinge_graph=state.hinge_graph),
        width="stretch",
        key="k2d",
    )
    st.write({"mesh_metrics": state.mesh_2d_optimized.metrics, "layout_metrics": state.k2d_flat_layout.metrics})
    st.subheader("K2D ↔ K3D edge-length diagnostics")
    st.write({
        "mean_edge_length_error_after": state.mesh_2d_optimized.metrics.get("mean_edge_length_error_after"),
        "max_edge_length_error_after": state.mesh_2d_optimized.metrics.get("max_edge_length_error_after"),
        "strict_k2d_solver_used": state.mesh_2d_optimized.metrics.get("strict_k2d_solver_used"),
        "strict_k2d_policy": state.mesh_2d_optimized.metrics.get("strict_k2d_policy"),
        "m2d_overlay_grid": state.mesh_2d_initial.metrics.get("m2d_grid_overlay"),
        "m2d_overlay_total_quad_count": state.mesh_2d_initial.metrics.get("m2d_overlay_total_quad_count"),
        "m2d_kept_quad_count": state.mesh_2d_initial.metrics.get("m2d_kept_quad_count"),
        "m2d_cropped_quad_count": state.mesh_2d_initial.metrics.get("m2d_cropped_quad_count"),
    })
elif view_stage == "T3D":
    st.plotly_chart(figure_tile_assembly(state.tiles_3d), width="stretch", key="t3d")
    st.write(state.tiles_3d.metrics)
elif view_stage == "T2D Top Hinge":
    st.plotly_chart(figure_tile_assembly(state.tiles_2d_top_hinge, hinge_graph=state.hinge_graph), width="stretch", key="t2d_top")
    t2d_top_stl, t2d_top_export_metrics = export_t2d_stl(state, stage="top_hinge", panel_size=0.1, solid_name="onestring_t2d_top_hinge")
    st.download_button(
        "Download T2D Top Hinge STL",
        data=t2d_top_stl,
        file_name="onestring_t2d_top_hinge.stl",
        mime="model/stl",
        key="download_t2d_top_hinge_stl",
    )
    state.tiles_2d_top_hinge.metrics.update(t2d_top_export_metrics)
    st.write(state.tiles_2d_top_hinge.metrics)
elif view_stage == "T2D Dual Hinge":
    st.plotly_chart(figure_tile_assembly(state.tiles_2d_dual_hinge, hinge_graph=state.hinge_graph), width="stretch", key="t2d_dual")
    t2d_dual_stl, t2d_dual_export_metrics = export_t2d_stl(state, stage="dual_hinge", panel_size=0.1, solid_name="onestring_t2d_dual_hinge")
    st.download_button(
        "Download T2D Dual Hinge STL",
        data=t2d_dual_stl,
        file_name="onestring_t2d_dual_hinge.stl",
        mime="model/stl",
        key="download_t2d_dual_hinge_stl",
    )
    state.tiles_2d_dual_hinge.metrics.update(t2d_dual_export_metrics)
    st.write(state.tiles_2d_dual_hinge.metrics)
elif view_stage in {"Lift Points", "String Path"}:
    st.plotly_chart(
        figure_tile_assembly(
            state.tiles_2d_dual_hinge,
            title="T2D with gap graph, paper-style lift points, and string path",
            gap_graph=state.gap_graph,
            hinge_graph=state.hinge_graph,
            string_path=state.string_path,
            lift_gap_ids=[lift.gap_id for lift in state.lift_points],
        ),
        width="stretch",
        key="gap_lift_string",
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("gaps", state.gap_graph.metrics["gap_count"])
    c2.metric("lift points", len(state.lift_points))
    c3.metric("turn angle", f"{state.string_path.turn_angle_total:.3f}")
    c4.metric("channel friction", f"{state.string_path.estimated_channel_friction:.3f}")
    st.write({"gap_graph": state.gap_graph.metrics, "string_path": state.string_path.metrics})
    lift_rows = state.gap_graph.metrics.get("lift_point_rows")
    if not lift_rows:
        lift_rows = [
            {
                "gap_id": int(lift.gap_id),
                "gpe": float(lift.gpe),
                "cluster_id": int(lift.cluster_id),
                "selection_reason": str(getattr(lift, "selection_reason", "")),
                "position_2d": [float(x) for x in np.asarray(lift.position_2d, dtype=float).reshape(-1)[:3]],
                "position_3d": [float(x) for x in np.asarray(lift.position_3d, dtype=float).reshape(-1)[:3]],
            }
            for lift in state.lift_points
        ]

    def _display_lift_value(value):
        if isinstance(value, (list, tuple, np.ndarray)):
            return ", ".join(f"{float(x):.5g}" for x in np.asarray(value, dtype=float).reshape(-1))
        return value

    st.dataframe([{k: _display_lift_value(v) for k, v in row.items()} for row in lift_rows], width="stretch")
elif view_stage == "Assembly Animation":
    st.subheader("Assembly Animation")
    total_tiles = int(state.tiles_2d_dual_hinge.tile_count)
    st.caption(
        "This view shows only the physical Projective-Dynamics-style deployment: rigid tiles + collision + simultaneous gap contraction + lift constraints. "
        "The design morph preview is disabled for physical checking."
    )

    frame_count = st.slider("animation frames", 8, 96, 40, 4)
    preview_default = min(total_tiles, 900)
    preview_upper = min(max(total_tiles, 0), 3000)
    if preview_upper <= 50:
        preview_tiles = total_tiles
        st.caption(f"preview tile limit: {preview_tiles} (all tiles)")
    else:
        preview_tiles = st.slider(
            "preview tile limit",
            min_value=50,
            max_value=preview_upper,
            value=min(max(50, preview_default), preview_upper),
            step=50,
            help="This only limits the displayed tiles. The simulation itself still uses the full linkage.",
        )
    if preview_tiles < total_tiles:
        st.warning(f"Animation preview is downsampled to about {preview_tiles} / {total_tiles} tiles. The simulation still used all tiles.")

    shape_rms = float(state.tiles_2d_dual_hinge.metrics.get("tile_shape_rms_error_to_T3D", 0.0))
    shape_max = float(state.tiles_2d_dual_hinge.metrics.get("tile_shape_max_error_to_T3D", 0.0))
    placement_rms = float(state.tiles_2d_top_hinge.metrics.get("top_vertices_rms_from_k2d_layout", 0.0))
    cols_shape = st.columns(3)
    cols_shape[0].metric("T2D/T3D tile shape RMS", f"{shape_rms:.3g}")
    cols_shape[1].metric("T2D/T3D tile shape max", f"{shape_max:.3g}")
    cols_shape[2].metric("K2D placement RMS", f"{placement_rms:.3g}")
    if shape_max > 1e-5:
        st.warning("T2D and T3D are not perfectly congruent. In the paper pipeline this means the K2D edge-matching / hinge-layout approximation is still imperfect; the app no longer hides it by shrinking or rigid-copying tiles.")
    else:
        st.success("T2D and T3D are nearly congruent under the current metric.")

    source_label = "Paper Projective Dynamics simulation"
    st.caption("Animation source: physical PD simulation only. Morph/interpolation preview is intentionally hidden.")

    if source_label.startswith("Paper"):
        st.info(
            "Physical model used here: the string is not simulated as rope particles. It is encoded as positional constraints. "
            "Only gaps on the computed string path receive snap constraints; selected lift gaps move toward prescribed 3D lift targets. "
            "Rigid and collision projections keep tiles rigid and non-overlapping."
        )
        c0, c1, c2, c3 = st.columns([1.3, 1.3, 1, 3])
        if "paper_pd_frame" not in st.session_state:
            st.session_state.paper_pd_frame = 0

        paper_sim_key = (
            "paper_pd_animation",
            actuation_key,
            frame_count,
            sim_steps,
            solver_iterations,
            solver_substeps,
            rigid_weight,
            rigid_projection_passes,
            rigid_guard_final_projection,
            snap_weight,
            lift_weight,
            collision_weight,
            hinge_weight,
            damping_ratio,
            high_fidelity,
            gravity,
            compute_backend,
            tensor_dtype,
        )
        need_sim_frames = (
            state.simulation_result is None
            or st.session_state.get("paper_pd_animation_key") != paper_sim_key
            or len(getattr(state.simulation_result, "frames", [])) < 2
        )
        refresh_sim = c0.button("Run / refresh paper simulation", type="primary")
        player_mode = c1.radio(
            "player mode",
            ["Smooth browser player", "Server frame player", "Scrubber"],
            horizontal=False,
            help="Smooth browser player preloads sampled frames and plays in Plotly without Streamlit reruns.",
        )
        if c2.button("⏮ Reset"):
            st.session_state.paper_pd_frame = 0

        if refresh_sim or need_sim_frames:
            paper_params = DeploymentParameters(
                steps=max(sim_steps, frame_count),
                solver_iterations=solver_iterations,
                rigid_weight=rigid_weight,
                rigid_projection_passes=rigid_projection_passes,
                rigid_guard_final_projection=rigid_guard_final_projection,
                hinge_weight=hinge_weight,
                snap_weight=snap_weight,
                lift_weight=lift_weight,
                collision_weight=collision_weight,
                damping_ratio=damping_ratio,
                quasi_static_pull_speed=quasi_static_pulling_speed,
                high_fidelity=high_fidelity,
                hinge_rotational_stiffness=hinge_rotational_stiffness,
                hinge_damping=hinge_damping,
                tile_mass=tile_mass,
                gravity=gravity,
                contact_friction=contact_friction,
                string_channel_friction=channel_friction,
                solver_substeps=solver_substeps,
                debug_all_pair_collision=debug_all_pair_collision,
                store_animation_frames=True,
                max_animation_frames=frame_count,
                snap_scope="all_internal_gaps",
                use_target_gap_contraction=True,
                compute=ComputeConfig(backend=compute_backend, dtype=tensor_dtype),
            )
            backend_info = compute_backend_info(paper_params.compute)
            st.write(
                {
                    "requested_backend": backend_info.get("requested_backend"),
                    "selected_backend": backend_info.get("current_backend"),
                    "cuda_available": backend_info.get("cuda_available"),
                    "gpu_name": backend_info.get("gpu_name"),
                    "use_gpu_for_simulation": bool(paper_params.compute.use_gpu_for_simulation),
                    "simulation_frames": int(paper_params.max_animation_frames),
                    "simulation_steps": int(paper_params.steps),
                    "solver_iterations": int(paper_params.solver_iterations),
                    "solver_substeps": int(paper_params.solver_substeps),
                }
            )
            progress_text = st.empty()
            progress_bar = st.progress(0.0)

            def _paper_sim_progress(stage: str, progress_value: float, detail: str) -> None:
                progress_bar.progress(max(0.0, min(1.0, float(progress_value))))
                progress_text.caption(f"{stage}: {detail}")

            with st.spinner("Running paper-style snap/lift Projective Dynamics simulation"):
                try:
                    state.simulation_result = simulate_onestring_deployment(
                        state,
                        paper_params,
                        progress_callback=_paper_sim_progress,
                    )
                except RuntimeError as exc:
                    st.error(str(exc))
                    st.stop()
            progress_bar.progress(1.0)
            progress_text.caption(
                f"Done: actual_backend={state.simulation_result.metrics.get('actual_backend')}, "
                f"elapsed={state.simulation_result.metrics.get('elapsed_time'):.3f}s"
            )
            st.session_state.paper_pd_animation_key = paper_sim_key
            st.session_state.actuation_key = actuation_key
            st.session_state.paper_pd_frame = 0

        result = state.simulation_result
        frames = result.frames if result is not None and result.frames else [state.tiles_2d_dual_hinge.vertices]
        max_frame = max(0, len(frames) - 1)
        current_frame = st.slider(
            "simulation frame",
            0,
            max_frame,
            min(int(st.session_state.paper_pd_frame), max_frame),
            1,
            key="paper_pd_frame_slider",
        )
        st.session_state.paper_pd_frame = current_frame

        show_target_preview = st.toggle("show translucent T3D target", value=True)
        show_hinges_preview = st.toggle("show hinge markers", value=False, help="Off is cleaner and faster. Markers are small when shown.")
        fps = st.slider("playback fps", 2, 24, 10, 1)

        def render_paper_pd_frame(frame_id: int):
            limit = min(preview_tiles, total_tiles)
            frame_vertices = frames[min(frame_id, max_frame)][:limit]
            frame_assembly = TileAssembly(
                vertices=frame_vertices,
                top_faces=state.tiles_3d.top_faces[:limit],
                bottom_faces=state.tiles_3d.bottom_faces[:limit],
                side_faces=state.tiles_3d.side_faces[:limit],
                stage="paper Projective Dynamics snap/lift simulation",
            )
            fig = figure_tile_assembly(
                frame_assembly,
                title=f"Paper PD snap/lift simulation frame {frame_id + 1}/{max_frame + 1}",
                hinge_graph=state.hinge_graph if show_hinges_preview and limit == total_tiles else None,
            )
            if show_target_preview:
                target_assembly = TileAssembly(
                    vertices=state.tiles_3d.vertices[:limit],
                    top_faces=state.tiles_3d.top_faces[:limit],
                    bottom_faces=state.tiles_3d.bottom_faces[:limit],
                    side_faces=state.tiles_3d.side_faces[:limit],
                    stage="T3D target",
                )
                add_tile_assembly(fig, target_assembly, color="#2563eb", opacity=0.18, name="T3D target")
            return fig

        if player_mode == "Smooth browser player":
            st.caption(
                "Smooth browser player: frames are played by Plotly in the browser. "
                "The Streamlit app does not rerun per frame, so the camera can be rotated/zoomed during playback."
            )
            smooth_fig = _smooth_browser_tile_animation(
                frames,
                state.tiles_3d.vertices,
                max_tiles=preview_tiles,
                show_target=show_target_preview,
                fps=fps,
                title=f"Smooth paper PD snap/lift simulation ({len(frames)} frames)",
            )
            st.plotly_chart(
                smooth_fig,
                width="stretch",
                key=f"paper_pd_smooth_{len(frames)}_{preview_tiles}_{int(show_target_preview)}",
                config={"scrollZoom": True, "displayModeBar": True, "responsive": True},
            )
            st.info("Use the Smooth play button inside the chart. You can rotate/zoom the view while it is playing.")
        elif player_mode == "Scrubber":
            st.plotly_chart(render_paper_pd_frame(current_frame), width="stretch", key="paper_pd_scrubber")
        else:
            autoplay = st.button("Play simulation")
            placeholder = st.empty()
            progress = st.progress((current_frame + 1) / (max_frame + 1))
            if autoplay:
                for frame_id in range(current_frame, max_frame + 1):
                    placeholder.plotly_chart(render_paper_pd_frame(frame_id), width="stretch", key=f"paper_pd_play_{frame_id}_{time.time_ns()}")
                    progress.progress((frame_id + 1) / (max_frame + 1))
                    st.session_state.paper_pd_frame = frame_id
                    time.sleep(max(0.01, 1.0 / fps))
            else:
                placeholder.plotly_chart(render_paper_pd_frame(current_frame), width="stretch", key="paper_pd_current")

        if result is not None:
            st.write(
                {
                    "simulation_model": "paper-style Projective Dynamics: E = w_rigid E_rigid + w_collision E_collision + w_actuation(E_snap + E_lift)",
                    "actual_backend": result.metrics.get("actual_backend"),
                    "dominant_backend": result.metrics.get("dominant_backend"),
                    "elapsed_time": result.metrics.get("elapsed_time"),
                    "gpu_kernel_time": result.metrics.get("gpu_kernel_time"),
                    "final_deployment_error_to_T3D": result.metrics.get("final_deployment_error_to_T3D"),
                    "snap_error": result.metrics.get("snap_error"),
                    "lift_error": result.metrics.get("lift_error"),
                    "rigid_error": result.metrics.get("rigid_error"),
                    "hinge_error": result.metrics.get("hinge_error"),
                    "collision_count": result.metrics.get("collision_count"),
                    "snap_scope": result.metrics.get("snap_scope"),
                    "actuated_snap_gap_count": result.metrics.get("actuated_snap_gap_count"),
                    "use_target_gap_contraction": result.metrics.get("use_target_gap_contraction"),
                }
            )

    else:
        st.warning("This is only a design morph preview. It is not the paper simulation and should not be used to judge physical actuation.")
        c_motion, c_mode, c_fps = st.columns([2, 2, 1])
        with c_motion:
            motion_label = st.radio(
                "preview motion model",
                ["Simultaneous hinge contraction", "Boundary string order debug"],
                horizontal=False,
            )
            motion_mode = "simultaneous_hinge_contraction" if motion_label.startswith("Simultaneous") else "boundary_string_order"
        with c_mode:
            animation_mode = st.radio(
                "player mode",
                ["Reliable player", "Scrubber", "Browser Plotly animation"],
                horizontal=False,
            )
        with c_fps:
            fps = st.slider("preview fps", 2, 24, 10, 1)

        show_target_preview = st.toggle("show translucent T3D target", value=True)
        show_string_preview = st.toggle("show string/gap overlay", value=False, help="This can be heavy. Keep it off for smooth playback. Markers are intentionally small.")

        if "assembly_player_frame" not in st.session_state:
            st.session_state.assembly_player_frame = 0

        if animation_mode == "Reliable player":
            controls = st.columns([1, 1, 1, 4])
            if controls[0].button("⏮ Reset preview"):
                st.session_state.assembly_player_frame = 0
            play_pressed = controls[1].button("▶ Play preview")
            if controls[2].button("⏭ End preview"):
                st.session_state.assembly_player_frame = frame_count - 1

            current_frame = st.slider(
                "current preview frame",
                0,
                frame_count - 1,
                int(st.session_state.assembly_player_frame),
                1,
                key="assembly_frame_slider",
            )
            st.session_state.assembly_player_frame = current_frame

            placeholder = st.empty()
            progress = st.progress((current_frame + 1) / frame_count)
            if play_pressed:
                for frame_id in range(current_frame, frame_count):
                    fig = assembly_progress_frame_figure(
                        state,
                        frame_id,
                        frame_count=frame_count,
                        max_tiles=preview_tiles,
                        show_target=show_target_preview,
                        show_path=show_string_preview,
                        motion_mode=motion_mode,
                    )
                    placeholder.plotly_chart(fig, width="stretch", key=f"assembly_reliable_play_{frame_id}_{time.time_ns()}")
                    progress.progress((frame_id + 1) / frame_count)
                    st.session_state.assembly_player_frame = frame_id
                    time.sleep(max(0.01, 1.0 / fps))
            else:
                fig = assembly_progress_frame_figure(
                    state,
                    current_frame,
                    frame_count=frame_count,
                    max_tiles=preview_tiles,
                    show_target=show_target_preview,
                    show_path=show_string_preview,
                    motion_mode=motion_mode,
                )
                placeholder.plotly_chart(fig, width="stretch", key="assembly_reliable_current")

        elif animation_mode == "Scrubber":
            frame_id = st.slider("preview frame", 0, frame_count - 1, 0, 1, key="assembly_scrubber_frame")
            fig = assembly_progress_frame_figure(
                state,
                frame_id,
                frame_count=frame_count,
                max_tiles=preview_tiles,
                show_target=show_target_preview,
                show_path=show_string_preview,
                motion_mode=motion_mode,
            )
            st.plotly_chart(fig, width="stretch", key="assembly_scrubber_plot")

        else:
            st.warning("This mode uses Plotly's browser-side animation. It may freeze for large grids because all frames are preloaded as JSON.")
            fig = assembly_progress_animation(
                state,
                frame_count=frame_count,
                max_tiles=preview_tiles,
                show_target=show_target_preview,
                show_path=show_string_preview,
                motion_mode=motion_mode,
            )
            html = fig.to_html(include_plotlyjs="cdn", full_html=False, auto_play=False, config={"responsive": True})
            components.html(html, height=720, scrolling=True)
            st.download_button(
                "Download playable HTML",
                data=fig.to_html(include_plotlyjs="cdn", full_html=True, auto_play=False),
                file_name="onestring_assembly_animation.html",
                mime="text/html",
            )

    st.write(
        {
            "paper_simulation": "Projective Dynamics with rigid, collision, snap, and lift constraints. String is abstracted as geometry constraints, not rope particles.",
            "snap": "paired side-face midpoints along the routed string path are pulled together",
            "lift": "selected lift gaps are pulled toward prescribed 3D lift targets",
            "preview_limit": preview_tiles,
            "tiles": total_tiles,
            "string_route_nodes": len(state.string_path.gap_ids),
            "lift_points": len(state.lift_points),
        }
    )

elif view_stage == "Final Deployed":
    if state.simulation_result is None:
        st.plotly_chart(figure_tile_assembly(state.tiles_2d_dual_hinge, title="T2D start state"), width="stretch", key="actuation_start")
    else:
        frame_index = st.slider("frame", 0, len(state.simulation_result.frames) - 1, len(state.simulation_result.frames) - 1)
        frame_assembly = TileAssembly(
            vertices=state.simulation_result.frames[frame_index],
            top_faces=state.tiles_2d_dual_hinge.top_faces,
            bottom_faces=state.tiles_2d_dual_hinge.bottom_faces,
            side_faces=state.tiles_2d_dual_hinge.side_faces,
            stage="snap/lift deployment frame",
        )
        st.plotly_chart(figure_tile_assembly(frame_assembly, title="snap/lift deployment frame"), width="stretch", key="actuation_frame")
        if st.button("Generate animation"):
            st.plotly_chart(
                tile_assembly_animation(state.tiles_2d_dual_hinge, state.simulation_result.frames, step=max(1, len(state.simulation_result.frames) // 24)),
                width="stretch",
                key="actuation_animation",
            )
elif view_stage == "Comparison":
    st.plotly_chart(figure_onestring_comparison(state), width="stretch", key="onestring_comparison")
elif view_stage == "Metrics":
    design_metrics = {
        "target_surface_fit_error_S": state.tiles_3d.metrics.get("surface_fit_error", 0.0),
        "k3d_planarity_error": state.mesh_3d_optimized.metrics.get("planarity_error_after", 0.0),
        "k3d_z_range_ratio": state.mesh_3d_optimized.metrics.get("z_range_ratio", 0.0),
        "k3d_fallback_used": state.mesh_3d_optimized.metrics.get("fallback_used", False),
        "k2d_edge_matching_error": state.mesh_2d_optimized.metrics.get("edge_matching_error", 0.0),
        "t3d_face_planarity_error": state.tiles_3d.metrics.get("face_planarity_error", 0.0),
        "hinge_connection_error": state.hinge_graph.metrics.get("hinge_connection_error", 0.0),
        "flat_collision_count": state.hinge_graph.metrics.get("flat_collision_count", 0),
        "turn_angle_total": state.string_path.turn_angle_total,
        "estimated_channel_friction": state.string_path.estimated_channel_friction,
    }
    if state.simulation_result is not None:
        design_metrics.update(state.simulation_result.metrics)
    st.dataframe({k: [v] for k, v in design_metrics.items()}, width="stretch")
elif view_stage == "Paper Consistency Audit":
    st.subheader("Paper Consistency Audit")
    st.caption("This table is intentionally strict. It compares the currently generated state with the paper's S→Ω→M2D→M3D→K3D/T3D and M2D→K2D→T2D→hinge optimization pipeline.")
    rows = paper_consistency_report(state)
    st.dataframe(rows, width="stretch")
    failed = [row for row in rows if not row.get("ok")]
    if failed:
        st.warning("Some rows are still approximations. The expected remaining red row is S↔Ω if BFF is unavailable; the rest of the Figure-5 flow should be green.")
    else:
        st.success("No critical consistency mismatch detected by the current audit.")
    st.write({
        "current_implementation_policy": [
            "Paper-default mode stops at unimplemented stages; experimental mode must be explicitly enabled for PCA/debug Ω.",
            "M2D is lifted to M3D through the stored surface parameterization.",
            "K3D optimizes planarity, square-like edge structure, and surface closeness.",
            "K2D matches K3D edge lengths in the flat shared-vertex mesh.",
            "T2D Top Hinge is generated directly from K2D top vertices and T3D top-to-bottom transforms; no hinge-layout optimization is applied yet.",
            "T2D Dual Hinge then selects top/bottom hinge surfaces and runs the E_Hinge-style rigid-pose tile placement optimization.",
            "Simulation applies snap constraints only along the computed string path and lift constraints at selected lift gaps.",
        ],
        "known_remaining_approximations": state.approximations,
        "important_current_metrics": {
            "m2d_grid_overlay": state.mesh_2d_initial.metrics.get("m2d_grid_overlay", None),
            "m2d_overlay_total_quad_count": state.mesh_2d_initial.metrics.get("m2d_overlay_total_quad_count", None),
            "m2d_kept_quad_count": state.mesh_2d_initial.metrics.get("m2d_kept_quad_count", None),
            "m2d_cropped_quad_count": state.mesh_2d_initial.metrics.get("m2d_cropped_quad_count", None),
            "k2d_mean_edge_length_error_after": state.mesh_2d_optimized.metrics.get("mean_edge_length_error_after", None),
            "k2d_max_edge_length_error_after": state.mesh_2d_optimized.metrics.get("max_edge_length_error_after", None),
            "strict_k2d_solver_used": state.mesh_2d_optimized.metrics.get("strict_k2d_solver_used", None),
            "m3d_used_height_field_shortcut": state.mesh_3d_initial.metrics.get("m3d_used_height_field_shortcut", False),
            "k2d_tile_shape_rms_error_after_layout": state.k2d_flat_layout.metrics.get("k2d_tile_shape_rms_error_after_layout", None),
            "k2d_shared_vertex_consistency_error": state.k2d_flat_layout.metrics.get("k2d_shared_vertex_consistency_error", None),
            "top_vertices_match_k2d_max_error": state.tiles_2d_top_hinge.metrics.get("top_vertices_match_k2d_max_error", None),
            "hinge_layout_optimizer_at_top_stage": state.k2d_flat_layout.metrics.get("hinge_layout_optimizer", None),
            "hinge_layout_deferred_to_dual_hinge": state.k2d_flat_layout.metrics.get("hinge_layout_deferred_to_dual_hinge", None),
            "dual_hinge_layout_optimizer": state.tiles_2d_dual_hinge.metrics.get("dual_hinge_layout_optimizer", None),
            "hinge_connection_error": state.hinge_graph.metrics.get("hinge_connection_error", None),
            "flat_collision_count": state.hinge_graph.metrics.get("flat_collision_count", None),
        },
    })
elif view_stage == "Setting Meters":
    st.subheader("Setting Meters")
    st.caption("現在のパラメータから推定した処理負荷・設計難度・配置の競合度です。これは実測プロファイルではなく、調整方向を見るためのメーターです。")
    meter_cols = st.columns(3)
    for idx, (meter_label, (meter_score, meter_note)) in enumerate(sidebar_meters.items()):
        col = meter_cols[idx % len(meter_cols)]
        with col:
            level_key, level_text = _meter_level(meter_score)
            st.metric(meter_label, f"{meter_score * 100:.0f}%", level_text)
            st.progress(float(_clamp01(meter_score)))
            st.caption(meter_note)

    st.divider()
    st.subheader("調整の目安")
    st.write(
        {
            "処理負荷が高い": [
                "grid size、hinge layout iterations、max collision candidate pairs、surface mesh subdivisions を下げる",
                "compute backend を cuda にする。tensor dtype は float32 推奨",
            ],
            "K2D辺長合わせ難度が高い": [
                "2D optimization iterations と K2D strict solver time budget を少し上げる",
                "amplitude / surface subdivisions / thickness を下げてまず流れを確認する",
            ],
            "ヒンジ配置競合が高い": [
                "hinge connection weight と hinge collision weight のどちらかが強すぎないか確認する",
                "anchor weight を少し上げると散らばりを抑えられるが、衝突から逃げにくくなる",
            ],
            "空洞/展開量が高い": [
                "hinge layout initial expansion を 1.03〜1.08 付近へ下げる",
                "gap size と max center drift / tile を下げる",
            ],
            "アニメーション負荷が高い": [
                "actuation steps、solver iterations、solver substeps を下げる",
                "store animation frames during solve をOFFにする",
            ],
        }
    )
    st.subheader("現在値")
    st.dataframe(
        [
            {"meter": label, "score_percent": round(score * 100, 1), "level": _meter_level(score)[1], "note": note}
            for label, (score, note) in sidebar_meters.items()
        ],
        width="stretch",
    )
elif view_stage == "Complexity / Backend":
    st.subheader("Complexity")
    st.dataframe({k: [v] for k, v in complexity_metrics(grid_size).items()}, width="stretch")
    st.subheader("Compute Backend")
    backend_info = compute_backend_info(pipeline_params.compute)
    st.write(backend_info)
    if not backend_info["cuda_available"]:
        st.warning(
            "CUDA is not available in the current Python environment. This usually means CPU-only PyTorch is installed, "
            "Streamlit is running in a different Python environment, the NVIDIA driver is not visible, or CUDA-enabled PyTorch was not installed correctly."
        )
    if st.button("Run GPU self-test"):
        try:
            st.write(gpu_self_test(ComputeConfig(backend="cuda", dtype=tensor_dtype)))
        except RuntimeError as exc:
            st.error(str(exc))
    if st.button("Run simulator GPU benchmark"):
        try:
            with st.spinner("Benchmarking K3D, K2D, and deployment on CPU vs CUDA"):
                st.dataframe(run_simulator_gpu_benchmark([5, 10, 15, 20]), width="stretch")
        except RuntimeError as exc:
            st.error(str(exc))
    st.subheader("nvidia-smi")
    st.write(nvidia_smi_probe())
    st.subheader("Stage Backend Reports")
    reports = dict(state.backend_reports)
    if state.simulation_result is not None:
        reports["deployment"] = {
            "requested_backend": compute_backend,
            "actual_backend": state.simulation_result.metrics.get("actual_backend", "cpu"),
            "gpu_memory_peak": state.simulation_result.metrics.get("gpu_memory_peak", 0),
        }
    st.dataframe([{**{"stage": key}, **value} for key, value in reports.items()], width="stretch")
elif view_stage == "Performance":
    st.subheader("Progress logs")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Latest pipeline build progress")
        st.dataframe(st.session_state.get("pipeline_progress_log", []), width="stretch")
    with c2:
        st.caption("Latest deployment simulation progress")
        st.dataframe(st.session_state.get("actuation_progress_log", []), width="stretch")

    st.subheader("Performance breakdown")
    rows = []
    for stage, report in state.backend_reports.items():
        elapsed = float(report.get("elapsed_time", 0.0) or 0.0)
        gpu = float(report.get("gpu_kernel_time", 0.0) or 0.0)
        cpu_pre = float(report.get("cpu_preprocess_time", 0.0) or 0.0)
        cpu_post = float(report.get("cpu_postprocess_time", 0.0) or 0.0)
        unaccounted = float(report.get("unaccounted_time", max(0.0, elapsed - gpu - cpu_pre - cpu_post)) or 0.0)
        rows.append(
            {
                "stage": stage,
                "elapsed_time": elapsed,
                "gpu_kernel_time": gpu,
                "cpu_measured_time": cpu_pre + cpu_post,
                "unaccounted_time": unaccounted,
                "gpu_time_ratio_%": 100.0 * gpu / max(elapsed, 1e-12),
                "actual_backend": report.get("actual_backend", ""),
                "dominant_backend": report.get("dominant_backend", ""),
                "gpu_memory_MB": float(report.get("gpu_memory_peak", 0) or 0) / (1024 * 1024),
            }
        )
    if state.simulation_result is not None:
        m = state.simulation_result.metrics
        elapsed = float(m.get("elapsed_time", m.get("gpu_kernel_time", 0.0)) or 0.0)
        gpu = float(m.get("gpu_kernel_time", 0.0) or 0.0)
        cpu_pre = float(m.get("cpu_preprocess_time", 0.0) or 0.0)
        cpu_post = float(m.get("cpu_postprocess_time", 0.0) or 0.0)
        rows.append(
            {
                "stage": "deployment",
                "elapsed_time": elapsed,
                "gpu_kernel_time": gpu,
                "cpu_measured_time": cpu_pre + cpu_post,
                "unaccounted_time": max(0.0, elapsed - gpu - cpu_pre - cpu_post),
                "gpu_time_ratio_%": 100.0 * gpu / max(elapsed, 1e-12),
                "actual_backend": m.get("actual_backend", ""),
                "dominant_backend": m.get("dominant_backend", ""),
                "gpu_memory_MB": float(m.get("gpu_memory_peak", 0) or 0) / (1024 * 1024),
            }
        )
    if not rows:
        st.info("Run the pipeline first.")
    else:
        try:
            import pandas as pd
            import plotly.express as px
            df = pd.DataFrame(rows)
            st.dataframe(df, width="stretch")
            long = df.melt(id_vars=["stage"], value_vars=["gpu_kernel_time", "cpu_measured_time", "unaccounted_time"], var_name="component", value_name="seconds")
            st.plotly_chart(px.bar(long, x="seconds", y="stage", color="component", orientation="h", title="Stage time breakdown"), width="stretch")
            st.plotly_chart(px.bar(df, x="gpu_time_ratio_%", y="stage", orientation="h", title="GPU kernel time ratio (%)"), width="stretch")
            st.plotly_chart(px.bar(df.sort_values("elapsed_time", ascending=True), x="elapsed_time", y="stage", orientation="h", title="Bottleneck ranking by elapsed time"), width="stretch")
            st.plotly_chart(px.bar(df, x="gpu_memory_MB", y="stage", orientation="h", title="Peak GPU memory by stage"), width="stretch")
            low_gpu = df[(df["actual_backend"] == "cuda") & (df["gpu_time_ratio_%"] < 50.0)]
            if len(low_gpu):
                st.warning("CUDA path is entered, but these stages are not GPU-dominant: " + ", ".join(low_gpu["stage"].astype(str).tolist()))
        except Exception as exc:
            st.exception(exc)
else:
    st.subheader("Current approximations")
    for item in state.approximations:
        st.write(f"- {item}")
    st.write(
        {
            "paper_faithful_baseline": "Fig. 5 intermediates plus snap/lift/rigid/hinge/collision actuation.",
            "simplified_approximation": state.approximations,
            "removed_from_default_mode": [
                "central tendon",
                "debug goal attraction",
                "target surface attraction",
                "rope-particle Verlet simulation as the primary actuator",
            ],
            "actuation_target": "T3D, not raw target surface S",
            "high_fidelity_extension": {
                "enabled": high_fidelity,
                "hinge_rotational_stiffness": hinge_rotational_stiffness,
                "hinge_damping": hinge_damping,
                "tile_mass": tile_mass,
                "gravity": gravity,
                "contact_friction": contact_friction,
            },
        }
    )
