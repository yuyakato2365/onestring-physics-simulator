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

# app_runtime_base.py is also the final legacy UI loaded by the dated launchers.
# When a user starts this file directly, bootstrap the current 2026-09-23
# launcher first so the OptCuts/paper-local-global patch stack is installed.
# The sentinel prevents recursion when that launcher eventually loads this file.
if __name__ == "__main__" and os.environ.get("ONESTRING_RUNTIME_BASE_BOOTSTRAPPED") != "1":
    import runpy
    os.environ["ONESTRING_RUNTIME_BASE_BOOTSTRAPPED"] = "1"
    runpy.run_path(
        str(Path(__file__).with_name("app_optcuts_20260923_extrusion_aware.py")),
        run_name="__main__",
    )
    st.stop()
from streamlit_plotly_events import plotly_events

sys.path.insert(0, str(Path(__file__).parent / "src"))

import onestring_physics.onestring_pipeline as _onestring_pipeline

# Do NOT reload the pipeline here.  Dedicated launchers install numerical
# patches (notably the paper-aligned T3D implementation) before this legacy UI
# is executed via runpy.  Reloading here silently discarded those patches.
# Keep the already-imported module object so the UI and launcher share exactly
# the same authoritative pipeline implementation.

from onestring_physics.animation import assembly_progress_animation, assembly_progress_frame_figure, tile_assembly_animation
from onestring_physics.input_shape import CLOSED_SHAPE_WARNING, create_builtin_shape, create_paper_bulb_neck_shape, load_target_shape
from onestring_physics.onestring_pipeline import (
    ComputeConfig,
    DeploymentParameters,
    PipelineParameters,
    TileAssembly,
    build_onestring_design,
    rerun_t2d_dual_hinge,
    complexity_metrics,
    compute_backend_info,
    gpu_self_test,
    nvidia_smi_probe,
    export_t2d_stl,
    export_t3d_stl,
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
    add_hinge_markers,
    correspondence_tile_colors,
)
from onestring_physics.panel_quality_visualization import (
    add_t2d_collision_overlay,
    figure_k3d_planarity,
    figure_t3d_planarized_k3d,
    planarized_k3d_from_t3d,
    quad_planarity_residuals,
)



def _mapping_distortion_values(surface_xyz: np.ndarray, uv: np.ndarray, faces: np.ndarray):
    """Per-triangle singular values of the local S->Omega map and area ratio."""
    xyz=np.asarray(surface_xyz,float); uv=np.asarray(uv,float); tri=np.asarray(faces,int)
    smax=np.ones(len(tri)); smin=np.ones(len(tri)); area=np.ones(len(tri))
    for k,(i,j,l) in enumerate(tri):
        p0,p1,p2=xyz[[i,j,l]]; q0,q1,q2=uv[[i,j,l],:2]
        e1=p1-p0; e2=p2-p0
        u=e1/max(np.linalg.norm(e1),1e-15)
        e2o=e2-u*np.dot(e2,u); n2=np.linalg.norm(e2o)
        if n2<1e-15: continue
        v=e2o/n2
        X=np.array([[np.dot(e1,u),np.dot(e2,u)],[np.dot(e1,v),np.dot(e2,v)]])
        Y=np.column_stack((q1-q0,q2-q0))
        try:
            J=Y@np.linalg.inv(X)
            sv=np.linalg.svd(J,compute_uv=False)
            smax[k],smin[k]=float(sv[0]),float(max(sv[-1],1e-15))
        except np.linalg.LinAlgError:
            continue
        a3=0.5*np.linalg.norm(np.cross(e1,e2))
        a2=0.5*abs(np.cross(q1-q0,q2-q0))
        area[k]=a2/max(a3,1e-15)
    return smax,smin,area

def _mapping_distortion_figures(parameterization, mode: str):
    xyz=np.asarray(parameterization.surface_vertices_3d,float)
    uv=np.asarray(parameterization.uv_vertices_2d,float)
    faces=np.asarray(parameterization.surface_faces,int)
    smax,smin,area=_mapping_distortion_values(xyz,uv,faces)
    if mode=="Stretch (sigma_max)":
        val=smax; label="sigma_max"
    elif mode=="Compression (1/sigma_min)":
        val=1.0/np.maximum(smin,1e-15); label="1 / sigma_min"
    elif mode=="Area ratio (A_Omega/A_S)":
        val=area; label="A_Omega / A_S"
    else:
        val=np.maximum(smax,1.0/np.maximum(smin,1e-15)); label="max(sigma_max, 1/sigma_min)"
    finite=val[np.isfinite(val)]
    vmax=float(np.percentile(finite,99)) if finite.size else 1.0
    vmax=max(vmax,1.0)
    common=dict(i=faces[:,0],j=faces[:,1],k=faces[:,2],intensity=val,intensitymode="cell",
                colorscale="Viridis",cmin=0.0,cmax=vmax,colorbar=dict(title=label),showscale=True,
                hovertemplate=f"{label}=%{{intensity:.4g}}<extra></extra>")
    fs=go.Figure(go.Mesh3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],**common))
    fs.update_layout(title=f"S — {label}",scene_aspectmode="data",margin=dict(l=0,r=0,t=40,b=0))
    fo=go.Figure(go.Mesh3d(x=uv[:,0],y=uv[:,1],z=np.zeros(len(uv)),**common))
    fo.update_layout(title=f"Omega — {label}",scene=dict(aspectmode="data",camera=dict(eye=dict(x=0,y=0,z=2.2))),margin=dict(l=0,r=0,t=40,b=0))
    stats={"min":float(np.min(finite)) if finite.size else 0.0,"median":float(np.median(finite)) if finite.size else 0.0,
           "p95":float(np.percentile(finite,95)) if finite.size else 0.0,"max":float(np.max(finite)) if finite.size else 0.0}
    return fs,fo,stats


st.set_page_config(page_title="OneString Paper-Faithful Simulator", layout="wide")
MODEL_VERSION = "2026-07-12-one-sided-t3d"
MODEL_VERSIONS = [
    {
        "id": MODEL_VERSION,
        "label": "2026-07-12 T3D one-sided 1t extrusion",
        "description": "K3Dを上面として固定し、法線の負方向の片側だけへ厚み1tを押し出します。隣接辺はmiter/contact planeで接続します。",
        "t3d_extrusion_side": "negative_normal_from_k3d",
        "t3d_variable_topology_enabled": True,
        "allow_legacy_normal_prism_emergency_fallback": True,
        "t3d_intersection_trim_enabled": False,
    },
    {
        "id": "2026-07-10-basic-implementation",
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
if os.environ.get("ONESTRING_PAPER_T3D_20260920", "0") == "1":
    MODEL_VERSIONS.append(
        {
            "id": "2026-09-18-optcuts-paper-local-global",
            "label": "2026-09-18 | OptCuts Ω + paper-aligned local/global K3D/K2D",
            "description": "Official OptCuts Ω with the paper-aligned local/global K3D/K2D route.",
            "t3d_extrusion_side": "negative_normal_from_k3d",
            "t3d_variable_topology_enabled": False,
            "allow_legacy_normal_prism_emergency_fallback": False,
            "t3d_intersection_trim_enabled": False,
        }
    )
    MODEL_VERSIONS.append(
        {
            "id": "2026-09-20-paper-t3d",
            "label": "2026-09-20 — Paper-aligned T3D",
            "description": (
                "K3D→T3Dを元論文Sec.4.2方式へ変更: shared K3D mesh vertex normalで"
                "厚み分offsetし、固定8頂点/6 quad frustumのtop・bottom・contact facesを"
                "Eq.(2) face-planarity optimizationで平面化。"
            ),
            "t3d_extrusion_side": "negative_normal_from_k3d",
            "t3d_variable_topology_enabled": False,
            "allow_legacy_normal_prism_emergency_fallback": False,
            "t3d_intersection_trim_enabled": False,
        }
    )
MODEL_VERSIONS.append({
    "id": "2026-09-23-extrusion-aware",
    "label": "2026-09-23 — Extrusion-aware experiments",
    "description": "独立した4フラグによる押し出し品質比較。Adaptive Gridは現在の格子制約により未対応。",
    "t3d_extrusion_side": "negative_normal_from_k3d",
    "t3d_variable_topology_enabled": False,
    "allow_legacy_normal_prism_emergency_fallback": False,
    "t3d_intersection_trim_enabled": False,
})
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
    void = _clamp01(0.75 * ((hinge_layout_initial_expansion - 1.0) / 0.25) + 0.25 * gap_factor)
    animation = _clamp01(0.60 * sim_factor + 0.18 * (1.0 if store_animation_frames else 0.0) + 0.12 * (1.0 if high_fidelity else 0.0) + 0.10 * grid_factor)

    notes = {
        "処理負荷": "grid・反復回数・衝突候補数から見た計算の重さ。高いほど待ち時間が増えます。",
        "形状難度": "Sの曲率/振幅/厚みから見た設計の難しさ。高いほどK2Dやヒンジ配置が破綻しやすいです。",
        "K2D辺長合わせ難度": "K3Dの対応辺長をK2Dで満たす難しさ。高いとedge errorが残りやすいです。",
        "ヒンジ配置競合": "E_ConnとE_Collisionが衝突する度合い。高いと接続維持と非衝突の両立が難しいです。",
        "空洞/展開量": "initial expansion・gapから見たタイル間の空きやすさ（移動距離制限はありません）。",
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
    # Experimental 2026-09-21 deployability-aware K3D variant.
    _deploy_version = {
        "id": "2026-09-21-deployability-k3d",
        "label": "2026-09-21 — Deployability-aware hard-planar K3D",
        "description": "Experimental K3D: minimize wSquare*ESquare + wSurface*ESurface + wDeploy*EDeployability under hard quad-planarity tolerance.",
    }
    if not any(v.get("id") == _deploy_version["id"] for v in MODEL_VERSIONS):
        MODEL_VERSIONS = list(MODEL_VERSIONS) + [_deploy_version]
    _default_version_index = next(
        (i for i,v in enumerate(MODEL_VERSIONS) if v.get("id") == "2026-09-21-deployability-k3d"), 0
    )
    selected_model_version = st.selectbox(
        "version",
        MODEL_VERSIONS,
        index=_default_version_index,
        format_func=lambda version: version["label"],
        help="実装バージョンを選択します。今後バージョン追加指示があれば、この一覧に追記します。",
    )
    st.caption(selected_model_version["description"])
    experiment_options = {}
    if selected_model_version["id"] == "2026-09-23-extrusion-aware":
        from onestring_physics.extrusion_aware import FEATURES, ADAPTIVE_LIMITATION
        labels = ("Principal-curvature Grid", "Extrusion-aware K3D", "Curvature-adaptive Grid Size", "Extrusion-aware Split")
        for name, label in zip(FEATURES, labels):
            experiment_options[name] = st.checkbox(label, value=False, key="0923_"+name)
        experiment_options["extrusion_weight"] = st.number_input("Extrusion objective weight", min_value=0.0, value=1.0, step=0.1)
        experiment_options["extrusion_split_weight"] = st.number_input("Extrusion split cost weight", min_value=0.0, value=1.0, step=0.1)
        experiment_options["extrusion_max_nfev"] = st.number_input("Extrusion K3D evaluation budget", min_value=1, value=12, step=1)
        st.caption("Mask: " + ''.join('1' if experiment_options[k] else '0' for k in FEATURES))
        st.caption("Principal grid: confidence-weighted global UV rotation. Local varying grid fields are not remeshed.")
        if experiment_options["use_curvature_adaptive_grid"]:
            st.warning(ADAPTIVE_LIMITATION)

    # Expose the official OptCuts distortion setting on the current
    # 2026-09-20 Paper-aligned T3D version selected by the user.
    optcuts_distortion_bound = 4.1
    if str(selected_model_version.get("id", "")) in {"2026-09-20-paper-t3d", "2026-09-21-deployability-k3d", "2026-09-23-extrusion-aware"}:
        optcuts_distortion_bound = float(st.number_input(
            "OptCuts distortion bound (Symmetric Dirichlet > 4)",
            min_value=4.0001,
            max_value=1000.0,
            value=max(4.0001, float(os.environ.get("ONESTRING_OPTCUTS_DISTORTION_BOUND", "4.1"))),
            step=0.05,
            format="%.4f",
            help="このバージョンの公式OptCuts Ωへ渡す distortion_bound。小さいほど歪みに厳しい。",
            key="onestring_0920_optcuts_distortion_bound",
        ))
        os.environ["ONESTRING_OPTCUTS_DISTORTION_BOUND"] = str(optcuts_distortion_bound)

        k3d_planarity_polish = bool(st.checkbox(
            "K3D: minimum-displacement planarity polish",
            value=str(os.environ.get("ONESTRING_K3D_PLANARITY_POLISH", "1")).strip().lower()
                  not in {"0", "false", "no", "off"},
            help="K3Dの通常最適化後に、元のK3Dからの移動量を抑えながらquadの共平面性だけを追加最適化します。OFFでは通常のK3DをそのままK2D/T3Dへ渡します。",
            key="onestring_0920_k3d_planarity_polish",
        ))
        os.environ["ONESTRING_K3D_PLANARITY_POLISH"] = "1" if k3d_planarity_polish else "0"

        k3d_iterations = int(st.number_input(
            "K3D local/global iterations",
            min_value=1,
            max_value=1000,
            value=max(1, int(float(os.environ.get("ONESTRING_PAPER_K3D_ITERATIONS", "40")))),
            step=10,
            help="M3D→K3D の paper-aligned local/global 最適化の反復回数。デフォルトは40回です。",
            key="onestring_0920_k3d_iterations",
        ))
        os.environ["ONESTRING_PAPER_K3D_ITERATIONS"] = str(k3d_iterations)

        k3d_planarity_mode = st.selectbox(
            "K3D planarity constraint",
            ["Soft (weighted EPlanar)", "Hard-priority (planarity first, square second)"],
            index=1 if str(os.environ.get("ONESTRING_K3D_PLANARITY_MODE", "soft")).strip().lower() == "hard" else 0,
            help="Softは従来どおりEPlanar/ESquare/ESurfaceを重み付きで同時最適化。Hard-priorityは通常K3Dで正方性等を整えた後、その解からの移動を最小化しながら共平面性を許容誤差まで満たし、満たせない解は受理しません。",
            key="onestring_0920_k3d_planarity_mode",
        )
        os.environ["ONESTRING_K3D_PLANARITY_MODE"] = "hard" if k3d_planarity_mode.startswith("Hard") else "soft"
        if str(selected_model_version.get("id", "")) == "2026-09-21-deployability-k3d":
            # This experimental version is defined by the constrained formulation.
            os.environ["ONESTRING_K3D_PLANARITY_MODE"] = "hard"
            st.info("2026-09-21 mode: K3D uses hard planarity and adds an experimental deployability proxy to Square + Surface.")
            k3d_w_deploy = float(st.number_input(
                "w_deploy / EDeployability",
                min_value=0.0, value=float(os.environ.get("ONESTRING_K3D_W_DEPLOY", "100.0")),
                step=10.0, format="%.2f",
                help="Experimental panel-feasibility weight. Penalizes quads whose shortest-edge ratio or normalized area deteriorates.",
                key="onestring_0921_k3d_w_deploy",
            ))
            k3d_deploy_edge = float(st.number_input(
                "Deployability: min edge / mean edge threshold",
                min_value=0.05, max_value=0.99,
                value=float(os.environ.get("ONESTRING_K3D_DEPLOY_EDGE_RATIO", "0.70")),
                step=0.05, format="%.2f", key="onestring_0921_deploy_edge",
            ))
            k3d_deploy_area = float(st.number_input(
                "Deployability: area / mean edge² threshold",
                min_value=0.05, max_value=2.0,
                value=float(os.environ.get("ONESTRING_K3D_DEPLOY_AREA_RATIO", "0.55")),
                step=0.05, format="%.2f", key="onestring_0921_deploy_area",
            ))
            os.environ["ONESTRING_K3D_W_DEPLOY"] = str(k3d_w_deploy)
            os.environ["ONESTRING_K3D_DEPLOY_EDGE_RATIO"] = str(k3d_deploy_edge)
            os.environ["ONESTRING_K3D_DEPLOY_AREA_RATIO"] = str(k3d_deploy_area)

        if k3d_planarity_mode.startswith("Hard"):
            k3d_hard_planarity_rel_tol = float(st.number_input(
                "Hard planarity tolerance / bbox diagonal",
                min_value=1e-7, max_value=1e-2,
                value=float(os.environ.get("ONESTRING_K3D_HARD_PLANARITY_REL_TOL", "1e-4")),
                step=1e-4, format="%.7f",
                help="best-fit planeからの最大頂点距離 / K3D bbox対角長。この値以下にならない場合はK3Dを失敗扱いにします。",
                key="onestring_0920_k3d_hard_planarity_rel_tol",
            ))
            os.environ["ONESTRING_K3D_HARD_PLANARITY_REL_TOL"] = str(k3d_hard_planarity_rel_tol)

        k3d_degeneracy_barrier = bool(st.checkbox(
            "K3D: steep anti-degeneracy barrier",
            value=str(os.environ.get("ONESTRING_K3D_DEGENERACY_BARRIER", "0")).strip().lower()
                  not in {"0", "false", "no", "off"},
            help="通常のK3D最適化中、quadの最短辺比または面積比が閾値を下回ったときだけ急激な追加ペナルティを有効化します。健康なquadには追加項を入れません。",
            key="onestring_0920_k3d_degeneracy_barrier",
        ))
        os.environ["ONESTRING_K3D_DEGENERACY_BARRIER"] = "1" if k3d_degeneracy_barrier else "0"
        if k3d_degeneracy_barrier:
            st.caption("Anti-degeneracy barrier settings")
            k3d_deg_edge_ratio = float(st.number_input(
                "Barrier start: min edge / mean edge",
                min_value=0.05, max_value=0.95,
                value=float(os.environ.get("ONESTRING_K3D_DEGENERACY_EDGE_RATIO", "0.65")),
                step=0.05, format="%.2f",
                help="この比率を下回ったquadでbarrierを発火。大きいほど早く三角形化を抑えます。",
                key="onestring_0920_k3d_deg_edge_ratio",
            ))
            k3d_deg_area_ratio = float(st.number_input(
                "Barrier start: area / mean edge²",
                min_value=0.05, max_value=2.0,
                value=float(os.environ.get("ONESTRING_K3D_DEGENERACY_AREA_RATIO", "0.45")),
                step=0.05, format="%.2f",
                help="面積が潰れるタイプの退化を検出する閾値です。",
                key="onestring_0920_k3d_deg_area_ratio",
            ))
            k3d_deg_weight = float(st.number_input(
                "Barrier base weight",
                min_value=0.0, max_value=1000000.0,
                value=float(os.environ.get("ONESTRING_K3D_DEGENERACY_BARRIER_WEIGHT", "100.0")),
                step=25.0, format="%.1f",
                help="発火直後から掛かる最低スケール。大きいほど退化を強く押し戻します。",
                key="onestring_0920_k3d_deg_weight",
            ))
            k3d_deg_power = float(st.number_input(
                "Barrier growth power",
                min_value=1.0, max_value=8.0,
                value=float(os.environ.get("ONESTRING_K3D_DEGENERACY_BARRIER_POWER", "2.0")),
                step=0.5, format="%.1f",
                help="退化が深くなるほど重みをどれだけ急増させるか。weight × (1 + severity)^power。",
                key="onestring_0920_k3d_deg_power",
            ))
            os.environ["ONESTRING_K3D_DEGENERACY_EDGE_RATIO"] = str(k3d_deg_edge_ratio)
            os.environ["ONESTRING_K3D_DEGENERACY_AREA_RATIO"] = str(k3d_deg_area_ratio)
            os.environ["ONESTRING_K3D_DEGENERACY_BARRIER_WEIGHT"] = str(k3d_deg_weight)
            os.environ["ONESTRING_K3D_DEGENERACY_BARRIER_POWER"] = str(k3d_deg_power)

    st.header("Target Input")
    target_kind = _param_row(
        "目標曲面 S。Remeshed_Bunny.stl はリポジトリ内または一般的なローカル配置から自動探索します。",
        lambda: st.selectbox("target shape", ["Remeshed_Bunny.stl", "vessel.stl", "paper_bulb_neck", "dome", "flat", "half_gourd", "snowman_half", "snowman_full", "saddle", "wave", "gaussian"], index=0, help="Default target is Remeshed_Bunny.stl. You can still upload another OBJ/STL/PLY below."),
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
            format_func=lambda mode: {
                "bff":"bff — LEGACY harmonic, NOT official BFF",
                "paper_reference_bff":"paper_reference_bff — official BFF CLI",
                "rectangular_harmonic_legacy":"rectangular_harmonic_legacy — NOT BFF",
            }.get(mode,mode),
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
        reference_stop_on_required_split = False
    if selected_model_version["id"] in {"2026-09-20-paper-t3d", "2026-09-23-extrusion-aware"}:
        csf_split_threshold = float(
            st.number_input(
                "Split threshold (area expansion)",
                min_value=1.0,
                value=2.0,
                step=0.05,
                format="%.3f",
                help="論文 Supplement Fig. 18 の面積拡張率。論文の上限は2。各成分を再評価して完全分割します。",
            )
        )
    else:
        csf_split_threshold = 2.0
    max_csf_splits = int(st.number_input("Maximum complete splits", min_value=0, value=64, step=1))
    with st.expander(
        "Bijective free-boundary settings",
        expanded=omega_parameterization_mode == "bijective_free_boundary",
    ):
        bijective_free_boundary_initial_boundary_shape = st.selectbox(
            "initial Omega boundary shape",
            ["circle", "rectangle"],
            index=0,
            help=(
                "Both choices use equal-area convex Floater initialization. "
                "Compare their final boundaries to measure initialization dependence."
            ),
        )
        bijective_free_boundary_max_iterations = int(
            st.number_input(
                "bijective max iterations",
                min_value=1,
                max_value=2000,
                value=1000,
                step=50,
                help="Lower this for a quick safe preview. Final local and global validity checks still run.",
            )
        )
        bijective_free_boundary_line_search_max_steps = int(
            st.number_input(
                "bijective line-search max steps",
                min_value=1,
                max_value=30,
                value=20,
                step=1,
            )
        )
        bijective_free_boundary_conformal_weight = float(
            st.number_input(
                "bijective conformal weight",
                min_value=0.0,
                max_value=50.0,
                value=4.0,
                step=0.5,
                help="Penalty for scale-invariant angular distortion. Zero disables the explicit conformal condition.",
            )
        )
        bijective_free_boundary_minimum_isotropic_scale = float(
            st.number_input(
                "minimum isotropic scale (s_min)",
                min_value=0.0,
                max_value=2.0,
                value=0.90,
                step=0.05,
                help="Triangles below this local isotropic scale receive the shrink penalty.",
            )
        )
        bijective_free_boundary_initial_step_scale = float(
            st.number_input(
                "bijective initial step scale",
                min_value=0.1,
                max_value=10.0,
                value=3.0,
                step=0.25,
                help="Larger values request stronger boundary/interior updates; validity and energy line search still limit unsafe moves.",
            )
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
    adaptive_planarity_penalty = st.checkbox(
        "Adaptive planarity penalty",
        value=str(os.environ.get("ONESTRING_K3D_ADAPTIVE_PLANARITY_PENALTY", "0")).strip().lower() in {"1", "true", "yes", "on"},
        help="ON: below the threshold, the ordinary EPlanar penalty is used; at or above the threshold, w_planar is multiplied by the strong-penalty multiplier.",
        key="onestring_k3d_adaptive_planarity_penalty",
    )
    os.environ["ONESTRING_K3D_ADAPTIVE_PLANARITY_PENALTY"] = "1" if adaptive_planarity_penalty else "0"
    if adaptive_planarity_penalty:
        adaptive_planarity_threshold = float(st.number_input(
            "Planarity strong-penalty threshold",
            min_value=1e-6, max_value=1.0,
            value=float(os.environ.get("ONESTRING_K3D_ADAPTIVE_PLANARITY_THRESHOLD", "0.01")),
            step=0.001, format="%.6f",
            key="onestring_k3d_adaptive_planarity_threshold",
        ))
        adaptive_planarity_multiplier = float(st.number_input(
            "Planarity strong-penalty multiplier",
            min_value=1.0, max_value=100000.0,
            value=float(os.environ.get("ONESTRING_K3D_ADAPTIVE_PLANARITY_MULTIPLIER", "100.0")),
            step=10.0,
            key="onestring_k3d_adaptive_planarity_multiplier",
        ))
        os.environ["ONESTRING_K3D_ADAPTIVE_PLANARITY_THRESHOLD"] = str(adaptive_planarity_threshold)
        os.environ["ONESTRING_K3D_ADAPTIVE_PLANARITY_MULTIPLIER"] = str(adaptive_planarity_multiplier)

    st.caption("K3D 正方形性（shape-preserving AL 段階）：平均誤差と最大歪みLSEの両方を抑制")
    st.latex(r"E_{\mathrm{Square}}=\frac{w_s}{2}\left[\frac{1}{N}\sum_{i=1}^{N}d_i^2+\alpha\left(\frac{1}{\beta}\log\left(\frac{1}{N}\sum_{i=1}^{N}e^{\beta d_i}\right)\right)^2\right]")
    square_tail_alpha = float(st.number_input(
        "Square 最大歪み α", min_value=0.0, max_value=1000.0,
        value=float(os.environ.get("ONESTRING_K3D_SQUARE_TAIL_ALPHA", "4.0")),
        step=0.5, format="%.2f", key="onestring_k3d_square_tail_alpha",
        help="0で最大歪みペナルティを無効化。大きいほど極端に歪むパネルを強く罰する。",
    ))
    square_tail_beta = float(st.number_input(
        "Square 最大歪み β", min_value=0.001, max_value=1000.0,
        value=float(os.environ.get("ONESTRING_K3D_SQUARE_TAIL_BETA", "12.0")),
        step=1.0, format="%.3f", key="onestring_k3d_square_tail_beta",
        help="大きいほどLSEが最大歪みに近づく。0より大きい値を指定する。",
    ))
    os.environ["ONESTRING_K3D_SQUARE_TAIL_ALPHA"] = str(square_tail_alpha)
    os.environ["ONESTRING_K3D_SQUARE_TAIL_BETA"] = str(square_tail_beta)
    st.caption("d_iは4辺の隣接長さ差と対角線長さ差（計5残差）の無次元RMS。通常のlocal/global段階は従来のESquareを使用。")
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
        lambda: st.number_input("hinge layout iterations", min_value=20, max_value=10000, value=80, step=100, help="Maximum L-BFGS iterations for the T2D dual-hinge layout optimizer. Values above 1000 are supported; convergence may stop earlier."),
    )
    hinge_layout_connection_weight = _param_row(
        "ヒンジ点を一致させる重み。高いほど接続は保つが、衝突回避の自由度は減る。",
        lambda: st.slider("hinge connection weight", 0.1, 20.0, 8.0, 0.1, help="Weight for E_Conn: pairwise hinge vertex coincidence."),
    )
    hinge_layout_collision_weight = _param_row(
        "厚み付きタイル footprint の非衝突重み。高すぎるとタイルが散り、低すぎると重なりが残る。",
        lambda: st.number_input(
            "hinge collision weight",
            min_value=0.0,
            value=4.0,
            step=1.0,
            format="%.2f",
            help="Weight for E_Collision: separating overlapping T2D footprints. No UI upper bound; larger values prioritize collision separation more strongly.",
        ),
    )
    hinge_layout_anchor_weight = _param_row(
        "初期展開配置に留める重み。高いほど散らばりにくいが、衝突から逃げにくくなる。",
        lambda: st.slider("hinge layout anchor weight", 0.0, 0.5, 0.0, 0.005, help="Anchor/trust weight toward the initial fabrication layout."),
    )
    hinge_layout_initial_expansion_enabled = _param_row(
        "独自拡張: E_Hinge 最適化前にタイル中心を外向きへ広げる処理。OFFでは元の T2D Top Hinge 配置をそのまま初期状態に使う。",
        lambda: st.checkbox(
            "Enable hinge layout initial expansion (project extension)",
            value=False,
            help="OFF: start Eq.6 from the unexpanded T2D Top Hinge layout. ON: apply the project-specific radial center expansion before optimization.",
        ),
    )
    hinge_layout_initial_expansion_requested = _param_row(
        "ON時だけ使う外向き拡大倍率。1.0は拡大なし。",
        lambda: st.slider(
            "hinge layout initial expansion",
            1.0,
            10.0,
            1.6,
            0.01,
            disabled=not hinge_layout_initial_expansion_enabled,
            help="Project-specific radial tile-center expansion before E_Hinge.",
        ),
    )
    hinge_layout_initial_expansion = (
        hinge_layout_initial_expansion_requested if hinge_layout_initial_expansion_enabled else 1.0
    )
    st.caption("T2D Dual Hinge: パネル中心の移動距離制限なし（Soft / Hard共通）。各パネルの形状は剛体のままです。")
    hinge_layout_time_budget_sec = _param_row(
        "通常パイプラインのT2Dでは、各L-BFGS-B試行にこの秒数を割り当てます。"
        "Multi-startなら入力した全試行に個別の制限時間を適用します（回数分だけ計算時間が増えます）。",
        lambda: st.slider(
            "hinge layout time budget / trial (full pipeline)",
            1.0, 120.0, 8.0, 1.0,
            help="Full pipeline: seconds PER T2D attempt, not total across restarts. Each requested trial is executed. Other hinge stages use their existing stage budget.",
        ),
    )
    hinge_layout_max_candidate_pairs = int(_param_row(
        "衝突判定する候補ペア数の上限。上げると正確だが重く、下げると衝突を見逃しやすい。",
        lambda: st.number_input("hinge layout max collision candidate pairs", min_value=200, max_value=20000, value=3000, step=200, help="Cap on candidate tile pairs for SAT footprint collision checks."),
    ))
    hinge_layout_collision_sweeps_per_iteration = int(_param_row(
        "1反復内で衝突分離を何回掃くか。増やすと重なりは減りやすいが計算が重くなる。",
        lambda: st.slider("collision sweeps per layout iteration", 1, 6, 2, 1, help="Number of collision separation passes per layout iteration."),
    ))

    # Both Run OneString pipeline and the cached T2D-only rerun honor these
    # controls; only the time-budget policy differs.
    t2d_search_mode = _param_row(
        "通常パイプラインでもT2D単独再実行でも同じ最適化方式を使う。"
        "Multi-startは一度求めた最良解と元のTop Hinge配置の両方から再探索する。",
        lambda: st.selectbox(
            "T2D optimization method (full run + rerun)",
            ["multi_start", "single"],
            format_func=lambda value: "Multi-start + perturbation (70% best / 30% initial)" if value == "multi_start" else "Original L-BFGS-B (single trial)",
            index=0,
        ),
    )
    t2d_collision_mode = _param_row(
        "Soft: 従来どおり衝突ペナルティを最小化。Hard: ソフト探索後にXY footprintの非貫通へ修復し、実行可能な配置のみHard成功と判定。ヒンジ接続は引き続きソフト制約。",
        lambda: st.selectbox(
            "T2D collision constraint (soft / hard)",
            ["soft", "hard"],
            format_func=lambda value: "Soft collision (original)" if value == "soft" else "Hard nonpenetration (repair + feasible refinement)",
            index=0,
        ),
    )
    t2d_restart_count = int(_param_row(
        "初回の通常最適化に加えて実行する再探索回数。例えば10なら初回＋10回＝計11回。"
        "通常パイプライン実行時にも必ずこの回数を試行する。",
        lambda: st.slider("T2D additional search trials (full run + rerun)",
                          1, 30, 6, 1, disabled=t2d_search_mode=="single"),
    ))
    st.caption(
        f"T2D実行予定: {1+t2d_restart_count if t2d_search_mode=='multi_start' else 1} 回"
        "（初回のL-BFGS-Bを含む）。"
    )
    t2d_perturb_translation = float(_param_row(
        "摂動の平行移動標準偏差（タイル代表辺長単位）。",
        lambda: st.slider("T2D translation perturbation / tile", 0.0, 2.0, 0.35, 0.05, disabled=t2d_search_mode=="single"),
    ))
    t2d_perturb_rotation_deg = float(_param_row(
        "摂動の回転角の標準偏差（度）。",
        lambda: st.slider("T2D rotation perturbation (degrees)", 0.0, 90.0, 20.0, 1.0, disabled=t2d_search_mode=="single"),
    ))
    t2d_search_seed = int(_param_row(
        "比較実験の再現性のための乱数シード。同じ形状・パラメータなら同じ初期候補を生成。",
        lambda: st.number_input("T2D multi-start random seed", min_value=0, max_value=2147483647, value=0, step=1, disabled=t2d_search_mode=="single"),
    ))
    st.markdown("#### Collision-driven Split（逐次追加・実験機能）")
    collision_split_enabled = _param_row(
        "Run OneString pipeline時のみ動作。重なりが残ると、衝突深さと境界距離で"
        "完全な格子カットを順位付けし、1本だけ加えてM2D以降を再計算します。"
        "改善がなければ元に戻し、最大本数まで1本ずつ反復します。既存CSF Splitとは別です。",
        lambda: st.checkbox(
            "Enable sequential collision-driven Split (full pipeline)",
            value=False,
            help="EXPERIMENTAL: can rebuild the full pipeline several times. "
                 "Requires a paper-style M2D Split topology. No T2D-only fake hinge deletion.",
        ),
    )
    collision_split_max_additional = int(_param_row(
        "追加できる衝突回避用Splitの最大本数。0なら追加しません。",
        lambda: st.slider("Maximum collision Split cuts", 0, 5, 2, 1,
                          disabled=not collision_split_enabled),
    ))
    collision_split_boundary_weight = float(_param_row(
        "境界から遠い内部の衝突ほどSplit候補を優先する度合い。0は衝突深さのみを重視。",
        lambda: st.slider("Collision Split boundary-distance weight", 0.0, 4.0,
                          1.0, 0.1, disabled=not collision_split_enabled),
    ))
    collision_split_gap_multiplier = float(_param_row(
        "Collision-driven Splitのみ切断領域間の初期離隔を増やす倍率。"
        "CSF Splitは元の距離のまま。後続のK2D/T2D最適化では"
        "再び近づくことがあるため、診断表で各工程を比較してください。",
        lambda: st.slider("Collision Split gap multiplier (experimental)",
                          1.0, 8.0, 1.0, 0.5,
                          disabled=not collision_split_enabled),
    ))
    collision_split_candidate_attempts = int(_param_row(
        "1本のSplitを決める前に試せる候補数。各候補でM2D以降を再計算し、"
        "衝突改善が確認できなければ棄却します。",
        lambda: st.slider("Split candidates tested per cut", 1, 5, 2, 1,
                          disabled=not collision_split_enabled),
    ))
    collision_split_min_improvement = float(_param_row(
        "Split採用に必要な貫通エネルギー改善率。接続誤差が大幅に悪化した候補は棄却します。",
        lambda: st.slider("Minimum collision improvement fraction", 0.0, 0.5,
                          0.03, 0.01, disabled=not collision_split_enabled),
    ))
    collision_split_lookahead_enabled = _param_row(
        "通常候補がすべて棄却された場合のみ、衝突数と総貫通エネルギーが"
        "改善した棄却候補を1件保留し、その状態から2本目を仮追加します。"
        "最終的な2本の配置をSplit前と比較し、既存の厳しい採用基準を"
        "満たす場合だけ両方採用します。中間状態だけをSTLへ出力しません。",
        lambda: st.checkbox(
            "Enable two-cut lookahead from rejected Split (experimental)",
            value=False,disabled=not collision_split_enabled,
        ),
    )
    collision_split_lookahead_children = int(_param_row(
        "保留した1本目から試す2本目の候補数。いずれも全工程を再計算します。",
        lambda: st.slider("Second-cut candidates per lookahead",1,4,2,1,
                          disabled=not collision_split_enabled or not collision_split_lookahead_enabled),
    ))
    if collision_split_lookahead_enabled and collision_split_max_additional < 2:
        st.info("Two-cut lookaheadにはMaximum collision Split cutsが2以上必要です。")
    if collision_split_enabled:
        st.warning("Collision-driven Splitは全工程の再計算を繰り返します。"
                   "T2D-only rerunボタンではSplitを追加しません。")
    t2d_rerun_time_budget_sec = _param_row(
        "T2D単独再実行の全試行に共有する時間予算（秒）。試行ごとに時間を分配。0で時間制限なし。",
        lambda: st.number_input(
            "T2D-only TOTAL search time budget (sec; 0 = unlimited)",
            min_value=0, max_value=7200, value=360, step=60,
            help="T2D-only rerun: split this total equally among every requested soft L-BFGS-B attempt (55% for soft search when Hard mode is selected). All attempts will run.",
        ),
    )

    # Keep the rerun action immediately after its parameters, rather than at
    # the very bottom of the long sidebar.
    st.markdown("### ↻ T2D Dual Hingeのみ再最適化")
    st.caption("上のヒンジ設定を変更後に押してください。保存済みのT2D Top Hingeから計算し直し、K3D・K2D・T3Dは再計算しません。")
    rerun_t2d_sidebar = st.button(
        "↻ T2Dだけ再最適化（K3D再計算なし）",
        type="primary",
        use_container_width=True,
        disabled="onestring_state" not in st.session_state,
        key="rerun_t2d_sidebar",
        help="現在のT2D Top Hingeを元にDual Hinge配置と3Dアニメーション履歴を更新します。",
    )
    if "onestring_state" not in st.session_state:
        st.caption("最初に「Run OneString pipeline」を実行してください。")

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
    physics_backend = st.selectbox(
        "physics backend",
        ["paper_pd", "native", "legacy", "abd"],
        index=0,
        format_func=lambda name: {
            "paper_pd": "Paper Sec 5.4 / Appendix E — Snap + Lift PD (reference)",
            "native": "Native rigid-body + routed string (experimental, CPU)",
            "legacy": "Legacy snap/lift + target guidance (not physical string)",
            "abd": "Autodesk ABD + IPC/CCD (external executable required)",
        }[name],
        help="Native: T2D-start rigid tile dynamics, point hinges, gravity, "
             "one tension-only routed string, approximate discrete 3D OBB contact. "
             "This is an experimental solver, not the paper's validated implementation.",
    )
    paper_pd_steps = 120
    paper_pd_timestep = 0.015
    if physics_backend == "paper_pd":
        st.info(
            "元論文 Section 5.4 / Supplement Appendix E の Snap（紐が通る"
            "ギャップ側面中央の一致）と Lift（選択ギャップを目標高さへ）を"
            "使います。T3Dパネル姿勢への追加吸引や仮想紐長短縮は使いません。"
            "剛体PDの反復と3D接触には独立実装の近似が残ります。"
        )
        with st.expander("Paper PD timing / numerical approximation",expanded=True):
            paper_pd_steps = st.slider(
                "Paper PD integration steps",20,480,120,20,
                help="Physical integration steps, independent from saved display frames.",
            )
            paper_pd_timestep = st.number_input(
                "Paper PD timestep (s)",min_value=0.001,max_value=0.050,
                value=0.015,step=0.005,format="%.3f",
            )
    native_pull_end_ratio = 0.45
    native_anchor_height_tiles = 2.0
    native_timestep = 0.02
    native_solver_steps = 180
    native_pull_fraction = 0.70
    native_extra_settle_steps = 40
    if physics_backend == "native":
        st.info(
            "T2D Dualのパネルをそのまま剛体として使用し、ヒンジを保ちながら"
            "紐の全長を短縮して引き上げます。T3Dへの直接吸引はしません。"
            "ただし3D接触は近似OBB、時間積分はPBDで、元論文の物理エンジンを"
            "忠実に再現したものではありません。"
        )
        with st.expander("Native string-pull parameters", expanded=True):
            native_pull_end_ratio = st.slider(
                "end string length / initial length", 0.1, 1.0, 0.45, 0.05,
                help="1.0 means no pulling; lower values reel in more string.",
            )
            native_anchor_height_tiles = st.slider(
                "string anchor height (tile edges)", 0.5, 10.0, 2.0, 0.5,
                help="Fixed world anchor directly above first string guide.",
            )
            native_timestep = st.number_input(
                "native timestep (seconds)", min_value=0.001,
                max_value=0.05, value=0.02, step=0.005, format="%.3f",
            )
            native_solver_steps = st.slider(
                "Native PHYSICS steps (not animation frames)",
                min_value=40, max_value=600, value=180, step=20,
                help="Increases the physical duration: steps × timestep. "
                     "Display frames are independent samples of this trajectory. "
                     "Larger values can be much slower on large grids.",
            )
            native_pull_fraction = st.slider(
                "Pull phase fraction (rest is settling)",
                min_value=0.25, max_value=1.0, value=0.70, step=0.05,
                help="0.70 means 70% of time reeling in the string and "
                     "30% holding the command at its final length.",
            )
            native_extra_settle_steps = st.slider(
                "Extra settling steps after scheduled physics",
                min_value=0, max_value=400, value=40, step=20,
                help="Bounded additional physical iterations, holding the final "
                     "commanded string length. This does not force the T3D target.",
            )
            st.caption(
                f"物理時間: {(native_solver_steps+native_extra_settle_steps)*native_timestep:.2f} 秒 "
                f"(引く {native_solver_steps*native_timestep*native_pull_fraction:.2f} 秒、"
                f"紐長保持 {(native_solver_steps*(1-native_pull_fraction)+native_extra_settle_steps)*native_timestep:.2f} 秒)。"
                "表示フレーム数を増やしても、この計算時間は増えません。"
            )
            st.caption(
                "Nativeに反映する共通パラメータ: actuation steps、solver iterations/substeps、"
                "damping ratio、hinge/collision weight、gravity、contact friction。"
                "Legacy専用のsnap/lift weight、target T3D fit/guardなどはNativeに効きません。"
            )
    abd_executable = ""
    abd_timestep = 0.01
    abd_density = 1000.0
    abd_gravity_z = -9.81
    abd_use_desk = True
    abd_desk_clearance = 5.0e-3
    abd_stiffness = 1e9
    abd_nthreads = 0
    abd_timeout_seconds = 600.0
    abd_pull_end_ratio = 0.75
    abd_shake_amplitude = 0.0
    abd_shake_frequency_hz = 0.0
    abd_shake_direction = (1.0, 0.0, 0.0)
    abd_shake_start_time = 0.0
    abd_shake_end_time = 0.0
    if physics_backend == "abd":
        st.warning(
            "ABD requires a Release build of Autodesk/affine-body-dynamics plus the OneString unilateral-string extension. "
            "If unavailable, execution stops explicitly; legacy SAT is not used as fallback."
        )
        with st.expander("Autodesk ABD settings", expanded=True):
            abd_executable = st.text_input(
                "abd_sim executable",
                value=os.environ.get("ONESTRING_ABD_EXECUTABLE", ""),
                help="Path to the Autodesk ABD Release executable. The full OneString run requires --onestring-manifest capability.",
            )
            abd_timestep = st.number_input("ABD timestep", min_value=0.0001, max_value=0.1, value=0.01, step=0.001, format="%.4f")
            abd_density = st.number_input("tile density", min_value=1.0, max_value=50000.0, value=1000.0, step=100.0)
            abd_gravity_z = st.number_input(
                "ABD gravity Z",
                min_value=-20.0,
                max_value=20.0,
                value=-9.81,
                step=0.1,
                help="Gravity remains enabled. The desk supplies an upward one-sided support reaction at each panel's bottom vertices.",
            )
            abd_use_desk = st.toggle(
                "support panels on fixed desk plane",
                value=True,
                help="Keeps panels from falling below a fixed horizontal desk while allowing horizontal sliding.",
            )
            abd_desk_clearance = st.number_input(
                "desk initial clearance",
                min_value=0.000001,
                max_value=0.1,
                value=0.005,
                step=0.0001,
                format="%.6f",
            )
            abd_stiffness = st.number_input("ABD orthogonality stiffness", min_value=1e5, max_value=1e12, value=1e9, step=1e8, format="%.3e")
            abd_nthreads = int(st.number_input("ABD CPU threads (0 = automatic)", min_value=0, max_value=128, value=0, step=1))
            abd_timeout_seconds = float(st.number_input("ABD timeout (seconds)", min_value=60, max_value=14400, value=600, step=60))
            abd_pull_end_ratio = st.slider("final string command / initial length", 0.05, 1.0, 0.75, 0.01)
            abd_shake_amplitude = st.number_input("shake amplitude", min_value=0.0, max_value=10.0, value=0.0, step=0.01)
            abd_shake_frequency_hz = st.number_input("shake frequency (Hz)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
            shake_cols = st.columns(3)
            abd_shake_direction = (
                float(shake_cols[0].number_input("shake dir X", value=1.0, step=0.1)),
                float(shake_cols[1].number_input("shake dir Y", value=0.0, step=0.1)),
                float(shake_cols[2].number_input("shake dir Z", value=0.0, step=0.1)),
            )
            time_cols = st.columns(2)
            abd_shake_start_time = float(time_cols[0].number_input("shake start time", min_value=0.0, value=0.0, step=0.1))
            abd_shake_end_time = float(time_cols[1].number_input("shake end time", min_value=0.0, value=0.0, step=0.1))
    run_actuation = _param_row(
        "現在のT2Dから選択中の物理方式を実行。Nativeは紐全長の片側拘束による引き上げ。",
        lambda: st.button({
            "paper_pd": "Run Paper Snap/Lift PD",
            "native": "Run native rigid string-pull",
            "legacy": "Run snap/lift actuation",
            "abd": "Run ABD OneString assembly",
        }[physics_backend]),
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
    string_path_frames: list[np.ndarray] | None = None,
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
    show_native_string=bool(string_path_frames and
                            len(string_path_frames)==len(safe_frames))
    if show_native_string:
        polyline=np.asarray(string_path_frames[0],dtype=float)
        fig.add_trace(go.Scatter3d(
            x=polyline[:,0],y=polyline[:,1],z=polyline[:,2],
            mode="lines+markers",
            line=dict(color="#dc2626",width=6),
            marker=dict(color="#f97316",size=2.5),
            name="routed string (constraint)",
            showlegend=True,
        ))
    if show_target:
        target_mesh, target_edges = _assembly_mesh_and_edge_traces(target, color="#2563eb", opacity=0.16, name="T3D target")
        fig.add_trace(target_mesh)
        fig.add_trace(target_edges)

    plotly_frames: list[go.Frame] = []
    for frame_id, vertices in enumerate(safe_frames):
        mesh, edges = _assembly_mesh_and_edge_traces(vertices, color="#2dd4bf", opacity=0.74, name="animated tiles")
        data=[mesh, edges]
        indices=[0,1]
        if show_native_string:
            polyline=np.asarray(string_path_frames[frame_id],dtype=float)
            data.append(go.Scatter3d(
                x=polyline[:,0],y=polyline[:,1],z=polyline[:,2],
                mode="lines+markers",
                line=dict(color="#dc2626",width=6),
                marker=dict(color="#f97316",size=2.5),
                name="routed string (constraint)",showlegend=True,
            ))
            indices.append(2)
        plotly_frames.append(go.Frame(data=data, traces=indices, name=str(frame_id)))
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


def _default_bunny_path():
    candidates = [
        Path(__file__).resolve().parent / "assets" / "Remeshed_Bunny.stl",
        Path(__file__).resolve().parent / "Remeshed_Bunny.stl",
        Path.home() / "Downloads" / "Remeshed_Bunny.stl",
        Path.home() / "Documents" / "Remeshed_Bunny.stl",
    ]
    return next((p for p in candidates if p.exists()), None)

def _default_vessel_path():
    candidates = [
        Path(__file__).resolve().parent / "assets" / "vessel.stl",
        Path(__file__).resolve().parent / "vessel.stl",
        Path.home() / "Downloads" / "vessel.stl",
        Path.home() / "Documents" / "vessel.stl",
    ]
    return next((p for p in candidates if p.exists()), None)


def build_target():
    if uploaded is None and target_kind == "paper_bulb_neck":
        st.caption("Built-in closed bulb + neck target (procedural approximation of the paper reference figure).")
        st.warning(CLOSED_SHAPE_WARNING)
        return create_paper_bulb_neck_shape()
    if uploaded is None and target_kind == "Remeshed_Bunny.stl":
        bunny_path = _default_bunny_path()
        if bunny_path is None:
            raise FileNotFoundError(
                "Default Remeshed_Bunny.stl was not found. Put it in repository assets/, "
                "repository root, ~/Downloads, or ~/Documents; or upload it in the sidebar."
            )
        st.caption(f"Default mesh: {bunny_path}")
        st.warning(CLOSED_SHAPE_WARNING)
        return load_target_shape(str(bunny_path))
    if uploaded is None and target_kind == "vessel.stl":
        vessel_path = _default_vessel_path()
        if vessel_path is None:
            raise FileNotFoundError(
                "vessel.stl was not found. Put it in repository assets/, repository root, "
                "~/Downloads, or ~/Documents; or upload it in the sidebar."
            )
        st.caption(f"Target mesh: {vessel_path}")
        return load_target_shape(str(vessel_path))
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
        tuple(sorted(experiment_options.items())),
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
        csf_split_threshold,
        max_csf_splits,
        bijective_free_boundary_initial_boundary_shape,
        bijective_free_boundary_max_iterations,
        bijective_free_boundary_line_search_max_steps,
        bijective_free_boundary_conformal_weight,
        bijective_free_boundary_minimum_isotropic_scale,
        bijective_free_boundary_initial_step_scale,
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
        hinge_layout_time_budget_sec,
        t2d_search_mode,
        t2d_collision_mode,
        t2d_restart_count,
        t2d_perturb_translation,
        t2d_perturb_rotation_deg,
        t2d_search_seed,
        collision_split_enabled,
        collision_split_max_additional,
        collision_split_boundary_weight,
        collision_split_gap_multiplier,
        collision_split_candidate_attempts,
        collision_split_min_improvement,
        collision_split_lookahead_enabled,
        collision_split_lookahead_children,
        hinge_layout_max_candidate_pairs,
        hinge_layout_collision_sweeps_per_iteration,
        compute_backend,
        tensor_dtype,
    )


def current_actuation_key() -> tuple:
    return (
        current_pipeline_key(),
        physics_backend,
        paper_pd_steps,
        paper_pd_timestep,
        native_pull_end_ratio,
        native_anchor_height_tiles,
        native_timestep,
        native_solver_steps,
        native_pull_fraction,
        native_extra_settle_steps,
        abd_executable,
        abd_timestep,
        abd_density,
        abd_gravity_z,
        abd_use_desk,
        abd_desk_clearance,
        abd_stiffness,
        abd_nthreads,
        abd_timeout_seconds,
        abd_pull_end_ratio,
        abd_shake_amplitude,
        abd_shake_frequency_hz,
        abd_shake_direction,
        abd_shake_start_time,
        abd_shake_end_time,
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
    **experiment_options,
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
    optcuts_distortion_bound=optcuts_distortion_bound,
    bff_boundary_policy=bff_boundary_policy,
    reference_grid_spacing=reference_grid_spacing_value if reference_grid_spacing_value > 0.0 else None,
    reference_grid_rotation_degrees=reference_grid_rotation_degrees,
    reference_grid_origin_u=reference_grid_origin_u,
    reference_grid_origin_v=reference_grid_origin_v,
    reference_csf_normalization=reference_csf_normalization,
    reference_stop_on_required_split=reference_stop_on_required_split,
    csf_split_threshold=csf_split_threshold,
    max_csf_splits=max_csf_splits,
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
    bijective_free_boundary_max_iterations=bijective_free_boundary_max_iterations,
    bijective_free_boundary_line_search_max_steps=bijective_free_boundary_line_search_max_steps,
    bijective_free_boundary_conformal_weight=bijective_free_boundary_conformal_weight,
    bijective_free_boundary_minimum_isotropic_scale=bijective_free_boundary_minimum_isotropic_scale,
    bijective_free_boundary_initial_step_scale=bijective_free_boundary_initial_step_scale,
    bijective_free_boundary_initial_boundary_shape=bijective_free_boundary_initial_boundary_shape,
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
    hinge_layout_time_budget_sec=hinge_layout_time_budget_sec,
    hinge_layout_search_mode=t2d_search_mode,
    hinge_layout_collision_mode=t2d_collision_mode,
    hinge_layout_budget_mode="per_trial",
    hinge_layout_restart_count=t2d_restart_count,
    hinge_layout_perturb_translation=t2d_perturb_translation,
    hinge_layout_perturb_rotation_deg=t2d_perturb_rotation_deg,
    hinge_layout_random_seed=t2d_search_seed,
    collision_split_enabled=collision_split_enabled,
    collision_split_max_additional=collision_split_max_additional,
    collision_split_boundary_weight=collision_split_boundary_weight,
    collision_split_gap_multiplier=collision_split_gap_multiplier,
    collision_split_candidate_attempts=collision_split_candidate_attempts,
    collision_split_min_relative_improvement=collision_split_min_improvement,
    collision_split_lookahead_enabled=collision_split_lookahead_enabled,
    collision_split_lookahead_children=collision_split_lookahead_children,
    hinge_layout_max_candidate_pairs=hinge_layout_max_candidate_pairs,
    hinge_layout_collision_sweeps_per_iteration=hinge_layout_collision_sweeps_per_iteration,
    strict_paper_flow=True,
    model_version=selected_model_version["id"],
    t3d_extrusion_side=str(selected_model_version.get("t3d_extrusion_side", "negative_normal_from_k3d")),
    t3d_variable_topology_enabled=bool(selected_model_version.get("t3d_variable_topology_enabled", False)),
    allow_legacy_normal_prism_emergency_fallback=bool(
        selected_model_version.get("allow_legacy_normal_prism_emergency_fallback", False)
    ),
    t3d_intersection_trim_enabled=bool(selected_model_version.get("t3d_intersection_trim_enabled", False)),
    compute=ComputeConfig(backend=compute_backend, dtype=tensor_dtype),
)

pipeline_key = current_pipeline_key()
if "onestring_state" not in st.session_state and not run_pipeline:
    st.info("Set parameters, then click Run OneString pipeline. The paper-default path stops at unimplemented stages; enable the experimental pipeline explicitly to run the old prototype path.")
    st.stop()

stored_pipeline_key = st.session_state.get("pipeline_key")
if stored_pipeline_key is not None and stored_pipeline_key != pipeline_key and not run_pipeline:
    st.warning(
        "Calculation settings changed. The existing result is still displayed; "
        "click Run OneString pipeline when you want to recompute it."
    )

# Streamlit reruns this script for display-only widgets such as View stage.
# Rebuild only on the explicit Run button; the completed state remains in the
# session and changing the view never repeats the expensive pipeline.
if run_pipeline:
    # Keep the previous *successful* state intact until the replacement
    # pipeline has actually returned. The old grid is used to reset stale
    # view/widget state after a successful topology-changing rebuild.
    _previous_successful_state = st.session_state.get("onestring_state")
    # Prefer the previously submitted UI grid size. mesh.grid.nx is only a
    # fallback: some Omega overlay builders change the effective grid count.
    _previous_grid = st.session_state.get("onestring_completed_grid_size")
    if _previous_grid is None:
        _previous_grid = getattr(
            getattr(getattr(_previous_successful_state, "mesh_2d_initial", None),
                    "grid", None),
            "nx", None,
        )
    with st.spinner("Building OneString pipeline state"):
        progress_bar = st.progress(0.0, text="Preparing pipeline…")
        progress_status = st.empty()
        progress_log: list[dict[str, object]] = []
        progress_started = time.perf_counter()
        last_progress_rendered = [-1.0]

        def _build_progress(stage: str, fraction: float, detail: str = "") -> None:
            fraction = max(0.0, min(1.0, float(fraction)))
            elapsed = time.perf_counter() - progress_started
            if fraction < 1.0 and elapsed - last_progress_rendered[0] < 0.25:
                return
            last_progress_rendered[0] = elapsed
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
        st.session_state.onestring_last_completed_params = pipeline_params
        from onestring_physics.ui_result_recovery import prepare_completed_pipeline_display
        _grid_view_reset = prepare_completed_pipeline_display(
            st.session_state,
            old_grid_size=_previous_grid,
            new_grid_size=grid_size,
        )
        st.session_state.pop("actuation_key", None)
        # A completed rebuild must not continue rendering inside the same
        # Streamlit script run that still owns the "Building..." spinner.
        # Persist the results and show them in a CLEAN display-only rerun.
        st.session_state["onestring_just_completed_pipeline"] = True
        progress_bar.empty()
        progress_status.empty()

    print(
        f"[ONESTRING-UI] Completed and stored grid={grid_size}; "
        "switching to display-only rerun without rebuilding.",
        flush=True,
    )
    st.rerun()

state = st.session_state.onestring_state
_just_completed_pipeline = bool(
    st.session_state.pop("onestring_just_completed_pipeline", False)
)
# A small, persistent result status is shown on EVERY subsequent UI rerun,
# not just on the Run button event. This helps distinguish a missing rendered
# figure from a lost session state after a grid change.
_saved_grid_size = st.session_state.get("onestring_completed_grid_size")
if _saved_grid_size is not None:
    _quad_count = len(state.mesh_2d_initial.faces)
    _tile_count = len(state.tiles_2d_dual_hinge.vertices)
    _result_text = (
        f"セッション内の計算結果: grid={_saved_grid_size}×{_saved_grid_size}, "
        f"M2D四角形={_quad_count}, T2Dパネル={_tile_count}。"
    )
    if _just_completed_pipeline:
        st.success("計算結果をセッションに保持しました。 " + _result_text +
                   " 下の「View stage」で各工程を確認できます。")
    else:
        st.caption(_result_text)
    if _just_completed_pipeline and st.session_state.get("onestring_grid_view_reset", False):
        st.info(
            "gridサイズが変わったため、古いパネル番号と表示状態をリセットし、"
            "軽量なPipeline Viewを表示しています。"
            "T2D Dual Hingeなどの表示はView stageから選び直せます。"
        )

# The main-page action is shown after the initial pipeline is available, so
# it is enabled immediately after the first successful full build.
st.markdown("### ↻ T2D Dual Hingeのみ再最適化")
st.caption(
    "左サイドバーの「Hinge Layout Optimization」で条件を変更してから実行。"
    "保存済みのT2D Top Hingeを使い、K3D・K2D・T3Dを再計算せずT2D配置と3Dアニメーションを更新します。"
)
rerun_t2d_main = st.button(
    "↻ T2Dだけ再最適化（K3D再計算なし）",
    type="primary",
    use_container_width=True,
    key="rerun_t2d_main",
    help="初期配置は保存済みのT2D Top Hingeです。上流設定を変更した場合は先にRun OneString pipelineを実行してください。",
)
rerun_t2d_only = rerun_t2d_sidebar or rerun_t2d_main

if rerun_t2d_only and not run_pipeline:
    # Do not mutate the parameters shared with a future full pipeline run.
    import copy as _copy
    t2d_rerun_params = _copy.copy(pipeline_params)
    t2d_rerun_params.hinge_layout_time_budget_sec = float(t2d_rerun_time_budget_sec)
    t2d_rerun_params.hinge_layout_budget_mode = "total_shared"
    t2d_rerun_params.collision_split_enabled = False  # requires complete M2D rebuild
    t2d_rerun_params.hinge_layout_search_mode = t2d_search_mode
    t2d_rerun_params.hinge_layout_collision_mode = t2d_collision_mode
    t2d_rerun_params.hinge_layout_restart_count = t2d_restart_count
    t2d_rerun_params.hinge_layout_perturb_translation = t2d_perturb_translation
    t2d_rerun_params.hinge_layout_perturb_rotation_deg = t2d_perturb_rotation_deg
    t2d_rerun_params.hinge_layout_random_seed = t2d_search_seed
    rerun_progress = st.empty()
    def _t2d_rerun_progress(stage, fraction, detail=""):
        rerun_progress.caption(f"T2D rerun: {stage}" + (f" — {detail}" if detail else ""))
    with st.spinner("Re-optimizing cached T2D Dual Hinge (K3D/K2D unchanged)…"):
        try:
            rerun_t2d_dual_hinge(state, t2d_rerun_params, progress_callback=_t2d_rerun_progress)
        except Exception as exc:
            st.exception(exc)
            st.stop()
    # A fresh solver call also regenerates dual_hinge_pose_history for animation.
    st.session_state.pop("actuation_key", None)
    st.session_state.pop("dual_hinge_inspection_trial", None)
    rerun_metrics = state.tiles_2d_dual_hinge.metrics
    rerun_iterations = rerun_metrics.get("dual_hinge_iterations", "unknown")
    rerun_reason = rerun_metrics.get("dual_hinge_solver_message", "No solver status available")
    winner_id = rerun_metrics.get("dual_hinge_selected_trial", 0)
    winner_source = rerun_metrics.get("dual_hinge_selected_source", "baseline")
    n_trials = rerun_metrics.get("dual_hinge_trials_completed", 1)
    improvement = rerun_metrics.get("dual_hinge_best_improvement", 0.0)
    if rerun_metrics.get("dual_hinge_collision_constraint_mode") == "hard":
        if rerun_metrics.get("hard_collision_feasible", False):
            st.success(
                f"Hard Collision: **非貫通を確認**（XY footprint）。"
                f"ソフト候補の試行 #{rerun_metrics.get('hard_collision_repair_source_trial')} を修復・再最適化して採用しました。"
                f"最終 E_Hinge={rerun_metrics.get('EHinge', 0.):.6g}。"
                "ヒンジ接続とその他の製作条件は別途検証してください。"
            )
        else:
            st.error(
                "Hard Collision: **非貫通配置を発見できませんでした。製作可能な解ではありません。** "
                f"試行した修復候補 {len(rerun_metrics.get('hard_collision_repair_attempts', []))} 件。"
                f"修復途中の最少残存衝突数 {rerun_metrics.get('hard_collision_best_unresolved_count')}。"
                "画面の配置は診断用のソフト最良解であり、Hard制約の成功解ではありません。"
            )
        st.caption(f"ソフト探索 {n_trials} / 予定 {rerun_metrics.get('dual_hinge_trials_requested', n_trials)} 試行。"
                   f" 元の初回試行に対する最終エネルギー変化 {-improvement:.6g}（Hardでは増加する場合があります）。")
    else:
        st.info(f"採用された試行: **#{winner_id}（{winner_source}）** / 実行{n_trials}回。初回通常試行からのエネルギー改善: {improvement:.6g}。")
        if rerun_metrics.get("dual_hinge_timed_out", False):
            st.warning(f"採用された試行は時間制限で途中終了しました（反復 {rerun_iterations} 回）。停止理由: {rerun_reason}")
        elif rerun_metrics.get("dual_hinge_solver_success") is False:
            st.warning(f"採用された試行は収束成功を報告していません（反復 {rerun_iterations} 回）。停止理由: {rerun_reason}")
        else:
            st.success(f"T2D Dual Hingeを再最適化しました。採用試行 #{winner_id}、反復 {rerun_iterations} 回。停止理由: {rerun_reason}")
    if rerun_metrics.get("dual_hinge_search_global_budget_exhausted", False):
        st.warning("指定されたソフト探索試行がすべては完了していません（時間予算に達したか、Hard修復用に時間を確保しました）。")


deployment_params = DeploymentParameters(
    physics_backend=physics_backend,
    paper_pd_timestep=paper_pd_timestep,
    native_pull_end_ratio=native_pull_end_ratio,
    native_anchor_height_tiles=native_anchor_height_tiles,
    native_timestep=native_timestep,
    native_pull_fraction=native_pull_fraction,
    native_extra_settle_steps=native_extra_settle_steps,
    abd_executable=abd_executable or None,
    abd_timestep=abd_timestep,
    abd_density=abd_density,
    abd_gravity_z=abd_gravity_z,
    abd_use_desk=abd_use_desk,
    abd_desk_clearance=abd_desk_clearance,
    abd_orthogonality_stiffness=abd_stiffness,
    abd_nthreads=abd_nthreads,
    abd_timeout_seconds=abd_timeout_seconds,
    abd_pull_end_ratio=abd_pull_end_ratio,
    abd_shake_amplitude=abd_shake_amplitude,
    abd_shake_frequency_hz=abd_shake_frequency_hz,
    abd_shake_direction_x=abd_shake_direction[0],
    abd_shake_direction_y=abd_shake_direction[1],
    abd_shake_direction_z=abd_shake_direction[2],
    abd_shake_start_time=abd_shake_start_time,
    abd_shake_end_time=abd_shake_end_time,
    steps=(paper_pd_steps if physics_backend == "paper_pd"
           else native_solver_steps if physics_backend == "native"
           else sim_steps),
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
    st.info("Run the selected deployment solver to animate T2D. Native mode uses the routed string without T3D target guidance.")
elif st.session_state.get("actuation_key") != actuation_key:
    st.warning("Actuation settings changed. Run snap/lift actuation again to refresh the simulation.")

if run_actuation:
    run_label = {
        "paper_pd": "Paper Section 5.4 Snap/Lift actuation PD",
        "native": "Native rigid panels + unilateral routed-string pulling",
        "legacy": "Projective-Dynamics-style snap/lift actuation",
        "abd": "Autodesk ABD OneString assembly",
    }[physics_backend]
    with st.spinner(f"Solving {run_label}"):
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
        "T3D-planarized K3D",
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
    key="onestring_view_stage",
)

# One stable Omega-space color field is reused downstream so a location keeps
# the same color through M2D -> M3D -> K3D and the corresponding panel stages.
_corr_uv=np.asarray(state.mesh_2d_initial.vertices,dtype=float)[:,:2]
_corr_bounds=np.asarray(state.surface_parameterization.uv_vertices_2d,dtype=float)[:,:2]
_corr_tile_colors=correspondence_tile_colors(state.mesh_2d_initial,_corr_bounds)
if hasattr(state, "extrusion_experiment"):
    import json
    from onestring_physics.extrusion_aware import difficulty_colors
    st.caption("Experiment " + state.extrusion_experiment["feature_mask"] + " · " + getattr(state, "extrusion_experiment_path", ""))
    st.download_button("Download experiment JSON", json.dumps(state.extrusion_experiment, ensure_ascii=False, indent=2), "extrusion-experiment.json", "application/json")
    color_mode = st.radio("Panel color", ["Correspondence gradient", "Extrusion difficulty"], horizontal=True)
    if color_mode == "Extrusion difficulty":
        _corr_tile_colors = difficulty_colors(state.extrusion_experiment["panel_extrusion_difficulty"])
        st.caption("Final K3D normal-extrusion difficulty: blue=0, yellow=0.5, red≥1. Selected panel remains pink.")

# Cross-stage linked panel inspector.
# Dedicated per-panel figures: each panel is exactly one clickable Mesh3d trace.
# This avoids relying on Mesh3d triangle pointNumber/customdata behavior.
_PANEL_HIGHLIGHT = "#ff006e"

def _panel_quad_figure(quads, colors, selected, title, label):
    quads=np.asarray(quads,dtype=float)
    fig=go.Figure()
    xyz=[]; ti=[]; tj=[]; tk=[]; facecolors=[]; triangle_panel_ids=[]
    ex=[]; ey=[]; ez=[]
    for panel_id,q in enumerate(quads):
        if len(q)<4: continue
        base=len(xyz)
        xyz.extend(q[:4].tolist())
        ti.extend([base,base]); tj.extend([base+1,base+2]); tk.extend([base+2,base+3])
        color=_PANEL_HIGHLIGHT if panel_id==selected else colors[panel_id]
        facecolors.extend([color,color])
        triangle_panel_ids.extend([int(panel_id),int(panel_id)])
        loop=np.vstack([q[:4],q[0]])
        ex.extend(loop[:,0].tolist()+[None]); ey.extend(loop[:,1].tolist()+[None]); ez.extend(loop[:,2].tolist()+[None])
    if xyz:
        xyz=np.asarray(xyz,dtype=float)
        # A single Mesh3d lets WebGL perform depth testing within one trace.
        # Raw events from Plotly 5 confirmed Mesh3d pointNumber is the picked
        # triangle index, so meta maps that triangle back to its panel.
        fig.add_trace(go.Mesh3d(
            x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],i=ti,j=tj,k=tk,
            facecolor=facecolors,opacity=1.0,flatshading=True,
            meta={"triangle_panel_ids":triangle_panel_ids},
            name=label,showlegend=False,hovertemplate=f"{label}<extra></extra>",
        ))
    fig.add_trace(go.Scatter3d(
        x=ex,y=ey,z=ez,mode="lines",line=dict(color="#111827",width=2),
        hoverinfo="skip",showlegend=False,name=f"{label} edges",
    ))
    fig.update_layout(title=title)
    fig.update_scenes(aspectmode="data")
    return fig

def _panel_solid_figure(assembly, colors, selected, title, label, hinge_graph=None):
    verts=np.asarray(assembly.vertices,dtype=float)
    fig=go.Figure()
    tri_i0=[0,0,4,4,0,0,1,1,2,2,3,3]
    tri_j0=[1,2,7,6,1,5,2,6,3,7,0,4]
    tri_k0=[2,3,6,5,5,4,6,5,7,6,4,7]
    edges=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
    xyz=[]; ti=[]; tj=[]; tk=[]; facecolors=[]; triangle_panel_ids=[]
    ex=[]; ey=[]; ez=[]
    for panel_id,v in enumerate(verts):
        if len(v)<8: continue
        base=len(xyz)
        xyz.extend(v[:8].tolist())
        ti.extend([base+i for i in tri_i0]); tj.extend([base+j for j in tri_j0]); tk.extend([base+k for k in tri_k0])
        color=_PANEL_HIGHLIGHT if panel_id==selected else colors[panel_id]
        facecolors.extend([color]*len(tri_i0))
        triangle_panel_ids.extend([int(panel_id)]*len(tri_i0))
        for u,w in edges:
            ex.extend([v[u,0],v[w,0],None]); ey.extend([v[u,1],v[w,1],None]); ez.extend([v[u,2],v[w,2],None])
    if xyz:
        xyz=np.asarray(xyz,dtype=float)
        fig.add_trace(go.Mesh3d(
            x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],i=ti,j=tj,k=tk,
            facecolor=facecolors,opacity=0.90,flatshading=True,
            meta={"triangle_panel_ids":triangle_panel_ids},
            name=label,showlegend=False,hovertemplate=f"{label}<extra></extra>",
        ))
    fig.add_trace(go.Scatter3d(
        x=ex,y=ey,z=ez,mode="lines",line=dict(color="#111827",width=2),
        hoverinfo="skip",showlegend=False,name=f"{label} edges",
    ))
    if hinge_graph is not None:
        add_hinge_markers(fig,assembly,hinge_graph)
    fig.update_layout(title=title)
    fig.update_scenes(aspectmode="data")
    return fig

def _clicked_panel(fig, events):
    if not events: return None
    ev=events[-1]
    try: curve=int(ev.get("curveNumber",-1))
    except (TypeError,ValueError): return None
    if not (0<=curve<len(fig.data)): return None
    meta=getattr(fig.data[curve],"meta",None)
    if not isinstance(meta,dict): return None
    if "triangle_panel_ids" in meta:
        try:
            point=int(ev.get("pointNumber",-1))
            ids=meta["triangle_panel_ids"]
            return int(ids[point]) if 0<=point<len(ids) else None
        except (TypeError,ValueError,IndexError):
            return None
    if "panel_id" in meta:
        try: return int(meta["panel_id"])
        except (TypeError,ValueError): return None
    return None

def _render_click_chart(fig,key,height=620):
    fig.update_layout(dragmode="orbit",clickmode="event")
    events=plotly_events(
        fig,click_event=True,hover_event=False,select_event=False,
        override_height=height,key=key,
    )
    if events:
        print(f"[PANEL-CLICK-RAW] key={key} event={events[-1]!r}",flush=True)
    return _clicked_panel(fig,events)

def _render_cross_stage_panel_inspector(t2d_assembly,*,t2d_label,hinge_graph,key_prefix):
    kfaces=np.asarray(state.mesh_3d_optimized.faces,dtype=int)
    kverts=np.asarray(state.mesh_3d_optimized.vertices,dtype=float)
    kquads=kverts[kfaces]
    t3verts=np.asarray(state.tiles_3d.vertices,dtype=float)
    t2verts=np.asarray(t2d_assembly.vertices,dtype=float)
    counts={"K3D":len(kquads),"T3D":len(t3verts),t2d_label:len(t2verts)}
    common=min(counts.values()) if counts else 0
    if common<=0:
        st.error("No corresponding K3D/T3D/T2D panels are available."); return
    if len(set(counts.values()))!=1:
        st.warning(f"Panel correspondence count mismatch: {counts}. Linking is limited to 0..{common-1}.")

    # The linked inspector creates one selectable 3D trace PER tile on EACH of
    # K3D, T3D and T2D. For large grids, automatically preloading all of them
    # can monopolize the Streamlit WebSocket or crash the browser.
    if common > 200:
        st.warning(
            f"連動パネル検査には{common}枚×3ステージの3D表示が必要です。"
            "大きなgridではブラウザ負荷を避けるため初期状態では描画しません。"
        )
        if not st.checkbox(
            "大量パネルのK3D/T3D/T2D連動3D検査を描画する（高負荷）",
            value=False,
            key=f"{key_prefix}_show_large_panel_inspector",
        ):
            return

    colors=list(_corr_tile_colors[:common])
    if len(colors)<common:
        colors += ["#2dd4bf"]*(common-len(colors))

    shared=max(0,min(int(st.session_state.get("cross_stage_panel_id",0)),common-1))
    widget_key=f"{key_prefix}_panel_id"
    pending_key=f"{key_prefix}_pending_panel_id"
    # Apply click-driven widget synchronization before number_input exists.
    # This is the only legal point in the Streamlit run to mutate widget_key.
    if pending_key in st.session_state:
        shared=max(0,min(int(st.session_state.pop(pending_key)),common-1))
        st.session_state["cross_stage_panel_id"]=shared
        st.session_state[widget_key]=shared
    elif widget_key not in st.session_state:
        st.session_state[widget_key]=shared
    manual=max(0,min(int(st.session_state[widget_key]),common-1))
    if manual!=shared:
        shared=manual
        st.session_state["cross_stage_panel_id"]=shared
    selected=shared

    st.number_input("Panel ID",min_value=0,max_value=common-1,step=1,key=widget_key)
    st.caption("左クリック = パネル選択、ドラッグ = 3D回転。対応グラデーションを全パネルに表示し、選択パネルだけを不透明なピンクで表示します。")

    kfig=_panel_quad_figure(kquads[:common],colors,selected,f"K3D — selected panel {selected}","K3D")
    t3fig=_panel_solid_figure(state.tiles_3d,colors,selected,f"T3D — selected panel {selected}","T3D")
    t2fig=_panel_solid_figure(t2d_assembly,colors,selected,f"{t2d_label} — selected panel {selected}",t2d_label,hinge_graph=hinge_graph)

    for label,fig,suffix in (("K3D",kfig,"k3d"),("T3D",t3fig,"t3d"),(t2d_label,t2fig,"t2d")):
        st.subheader(label)
        picked=_render_click_chart(fig,key=f"{key_prefix}_click_{suffix}_sel_{selected}",height=620)
        if picked is not None and 0<=picked<common and picked!=selected:
            # Streamlit forbids mutating a widget's own key after that widget
            # has been instantiated in the current run.  Store only the
            # authoritative shared selection here; on the next rerun, sync
            # the number-input key *before* constructing the widget.
            st.session_state["cross_stage_panel_id"]=picked
            st.session_state[f"{key_prefix}_pending_panel_id"]=picked
            print(f"[PANEL-CLICK] stage={label} panel={picked}",flush=True)
            st.rerun()

# Keep a compact correspondence reference visible for every downstream stage.
# This is deliberately separate from the selected-stage renderer.
if view_stage not in {"Pipeline View", "S", "Split Map", "Mode Comparison", "Metrics", "Paper Consistency Audit", "Setting Meters", "Complexity / Backend", "Performance", "Approximations"}:
    with st.expander("Mapping reference: original M2D / Omega colors", expanded=True):
        st.caption("同じ色 = Ω/M2D上の同じ場所。下の各stageでもこの色を引き継ぎます。")
        from copy import copy as _copy_mapping_mesh
        _mapping_mesh = _copy_mapping_mesh(state.mesh_2d_initial)
        _mapping_mesh.vertices = np.asarray(
            getattr(state.mesh_2d_initial, "_split_panel_source_vertices", state.mesh_2d_initial.vertices),
            dtype=float,
        ).copy()
        st.caption("元のΩ座標で表示しています。Split後の接続を保持し、分割片を離す表示用移動は適用しません。切断線は Split Map で確認できます。")
        st.plotly_chart(
            figure_quad_mesh(
                _mapping_mesh,
                title="Correspondence reference — M2D at original Ω coordinates",
                correspondence_uv=_corr_uv,
                correspondence_bounds=_corr_bounds,
            ),
            use_container_width=True,
            key=f"mapping_reference_{view_stage}",
        )

if view_stage == "Pipeline View":
    # Always provide a lightweight useful result BEFORE any Plotly rendering.
    st.subheader("計算済み工程の概要")
    st.write({
        "M2D四角形数": len(state.mesh_2d_initial.faces),
        "K3D四角形数": len(state.mesh_3d_optimized.faces),
        "T3Dパネル数": len(state.tiles_3d.vertices),
        "T2Dパネル数": len(state.tiles_2d_dual_hinge.vertices),
        "T2D残存衝突ペア": state.tiles_2d_dual_hinge.metrics.get("collision_pair_count", "unknown"),
        "衝突回避Split採用数": state.tiles_2d_dual_hinge.metrics.get("collision_split_accepted", 0),
    })
    # The overview graph is small but can still trip browser-side Plotly
    # rendering in some reruns. During a changed-grid rebuild show only
    # non-Plotly results until the user explicitly requests the graphic.
    _render_overview = st.checkbox(
        "パイプライン図を描画する",
        value=not st.session_state.get("onestring_grid_view_reset", False),
        key="onestring_render_overview",
    )
    if _render_overview:
        try:
            st.plotly_chart(figure_pipeline_overview(state),
                            use_container_width=True, key="pipeline_overview")
        except Exception as exc:
            st.error("計算は完了していますが、Pipeline Viewの描画に失敗しました。"
                     "View stageのMetricsなどから保存結果を確認できます。")
            st.exception(exc)
    if state.stage_reports:
        selected_report = st.selectbox("stage report", list(state.stage_reports.keys()))
        report = state.stage_reports[selected_report]
        cols = st.columns(4)
        cols[0].metric("before error", f"{report.before_error:.4g}")
        cols[1].metric("after error", f"{report.after_error:.4g}")
        cols[2].metric("constraint violation", f"{report.constraint_violation:.4g}")
        cols[3].metric("time", f"{report.computation_time:.3f}s")
        st.write({"objective": report.objective, "counts": report.counts, "failed_constraints": report.failed_constraints})
    else:
        st.warning("工程ごとのレポートが空です。パネル数は上に表示しています。"
                   "View stageのMetricsから詳細を確認できます。")
elif view_stage == "S":
    st.plotly_chart(figure_surface_mesh(state.target_surface), use_container_width=True, key="target_surface")
elif view_stage == "Split Map":
    from onestring_physics.split_process_view import figure_split_process
    st.subheader("Splitを追加する過程")
    _split_records = state.mesh_2d_initial.metrics.get("csf_split_step_analysis", [])
    _split_count = len(_split_records)
    _split_step = st.slider("追加済みのSplit数", 0, _split_count, 0, key=f"split_process_step_{_split_count}") if _split_count else 0
    _split_figure, _split_rows = figure_split_process(state.mesh_2d_initial, _split_step)
    if _split_figure is None:
        st.info("この結果には途中状態の記録がありません。対応するSplit経路で再生成すると表示できます。")
    else:
        st.caption("0は追加Split前です。赤＝今回追加、黒＝それまでの切断。元のΩ座標で表示し、分割片を移動させません。初期状態にある切れ目は追加Splitに含みません。")
        st.plotly_chart(_split_figure, use_container_width=True, key="split_process_figure")
        if _split_step:
            _record = _split_records[_split_step-1]
            st.caption(f"切断した領域のCSF: {_record['sigma_before']:.4g} → " + ", ".join(f"{value:.4g}" for value in _record['sigma_after']))
        else:
            st.caption("追加Split前の状態" if _split_count else "追加されたSplitはありません。")
        st.dataframe(_split_rows, hide_index=True, width="stretch")
        st.caption("CSFは現在のUV写像に基づく評価値です。")
    if state.mesh_2d_initial.metrics.get("csf_split_parameterization_repair_required"):
        st.warning(
            "UVに退化した三角形があるため、有効な領域で処理を継続しています。"
            "表示CSFは部分評価であり、閾値達成は未確認です。"
            f" 対象: {state.mesh_2d_initial.metrics.get('csf_split_degenerate_uv_face_count', 0)} 三角形。"
        )
    st.plotly_chart(figure_split_mapping(state), use_container_width=True, key="split_mapping")
    st.write(
        {
            "meaning": "Red samples show the Omega split line mapped back onto S through the stored S->Omega parameterization.",
            "csf_split_threshold": state.mesh_2d_initial.metrics.get("csf_split_threshold"),
            "max_csf_before_split": state.mesh_2d_initial.metrics.get("max_csf_before_split"),
            "max_csf_after_split": state.mesh_2d_initial.metrics.get("max_csf_after_split"),
            "csf_split_step_analysis_model": state.mesh_2d_initial.metrics.get("csf_split_step_analysis_model"),
            "csf_split_residual_high_face_count": state.mesh_2d_initial.metrics.get("csf_split_residual_high_face_count"),
            "csf_split_status": state.mesh_2d_initial.metrics.get("csf_split_status"),
            "parameterization_method": state.mesh_2d_initial.metrics.get("csf_split_parameterization_method"),
            "components_before_additional_split": state.mesh_2d_initial.metrics.get("m2d_connected_component_count_before_csf_split"),
            "components_after_additional_split": state.mesh_2d_initial.metrics.get("m2d_connected_component_count_after_csf_split"),
            "split_stopping_rule": state.mesh_2d_initial.metrics.get("csf_split_stopping_rule"),
            "csf_split_budget_exhausted": state.mesh_2d_initial.metrics.get("csf_split_budget_exhausted"),
            "csf_split_component_sigma": state.mesh_2d_initial.metrics.get("csf_split_component_sigma"),
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
    st.plotly_chart(figure_domain(state), use_container_width=True, key="omega_domain")
    st.subheader("S ↔ Omega Mapping Distortion")
    st.caption("The same per-triangle S→Omega distortion values are painted on the original surface S and on Omega, so corresponding distorted regions have exactly the same color.")
    distortion_mode = st.selectbox(
        "Distortion metric",
        ["Total distortion", "Stretch (sigma_max)", "Compression (1/sigma_min)", "Area ratio (A_Omega/A_S)"],
        key="omega_mapping_distortion_metric",
    )
    distortion_s, distortion_o, distortion_stats = _mapping_distortion_figures(state.surface_parameterization, distortion_mode)
    distortion_cols = st.columns(2)
    with distortion_cols[0]:
        st.plotly_chart(distortion_s, use_container_width=True, key="mapping_distortion_S")
    with distortion_cols[1]:
        st.plotly_chart(distortion_o, use_container_width=True, key="mapping_distortion_Omega")
    st.caption("Distortion statistics: " + ", ".join(f"{k}={v:.4g}" for k,v in distortion_stats.items()))
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
        "optimization_requested_max_iterations": state.surface_parameterization.metrics.get("optimization_requested_max_iterations", ""),
        "optimization_iteration_count": state.surface_parameterization.metrics.get("optimization_iteration_count", ""),
        "optimization_termination_reason": state.surface_parameterization.metrics.get("optimization_termination_reason", ""),
        "optimization_converged": state.surface_parameterization.metrics.get("optimization_converged", ""),
        "initial_energy": state.surface_parameterization.metrics.get("initial_energy", ""),
        "final_energy": state.surface_parameterization.metrics.get("final_energy", ""),
        "initial_conformal_energy": state.surface_parameterization.metrics.get("initial_conformal_energy", ""),
        "final_conformal_energy": state.surface_parameterization.metrics.get("final_conformal_energy", ""),
        "conformal_energy_weight": state.surface_parameterization.metrics.get("conformal_energy_weight", ""),
        "line_search_initial_step_scale": state.surface_parameterization.metrics.get("line_search_initial_step_scale", ""),
        "boundary_displacement_rms": state.surface_parameterization.metrics.get("boundary_displacement_rms", ""),
        "boundary_displacement_max": state.surface_parameterization.metrics.get("boundary_displacement_max", ""),
        "initial_boundary_radius_cv": state.surface_parameterization.metrics.get("initial_boundary_radius_cv", ""),
        "final_boundary_radius_cv": state.surface_parameterization.metrics.get("final_boundary_radius_cv", ""),
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
    if state.surface_parameterization.method == "bijective_free_boundary":
        accepted = int(state.surface_parameterization.metrics.get("optimization_iteration_count", 0))
        requested = int(state.surface_parameterization.metrics.get("optimization_requested_max_iterations", 0))
        boundary_rms = float(state.surface_parameterization.metrics.get("boundary_displacement_rms", 0.0))
        initial_shape = str(
            state.surface_parameterization.metrics.get("initialization_boundary_shape", "circle")
        )
        final_circle_residual = float(
            state.surface_parameterization.metrics.get("final_boundary_circle_fit_relative_rms", 0.0)
        )
        st.caption(
            f"Final optimized Ω: initial shape={initial_shape}; "
            f"accepted iterations {accepted}/{requested}; "
            f"boundary RMS displacement from initialization {boundary_rms:.6g}. "
            f"Final best-circle residual={100.0 * final_circle_residual:.4f}%. "
            "The gray dashed outline is the initial boundary and the thick green outline is the final boundary."
        )
        if accepted == 0:
            st.error("Ω optimization accepted no update; the displayed final boundary equals the initialization.")
        elif boundary_rms <= 1.0e-5:
            st.warning(
                "Ω interior energy improved, but the optimized boundary moved only a negligible amount. "
                "The final outline can therefore remain visually indistinguishable from the initialization."
            )
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
            st.plotly_chart(figure_domain(comparison), use_container_width=True, key=f"compare_omega_{comparison_mode}")
            st.plotly_chart(figure_quad_mesh(comparison.mesh_2d_initial, title="M2D"), use_container_width=True, key=f"compare_m2d_{comparison_mode}")
            st.plotly_chart(figure_m3d_overlay(comparison), use_container_width=True, key=f"compare_m3d_{comparison_mode}")
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
    mesh = {"M2D": state.mesh_2d_initial, "K3D": state.mesh_3d_optimized}[view_stage]
    if view_stage == "K3D":
        st.subheader("K3D")
        st.caption(
            "M3D → K3D optimization の最終結果を表示しています。"
            f" · vertices={len(mesh.vertices)} · quads={len(mesh.faces)}"
        )
    if view_stage == "K3D":
        residuals = quad_planarity_residuals(mesh.vertices, mesh.faces)
        st.plotly_chart(figure_k3d_planarity(mesh), use_container_width=True, key="mesh_K3D_planarity")
        if len(residuals):
            st.caption(f"K3D panel planarity: max={float(np.max(residuals)):.6g}, mean={float(np.mean(residuals)):.6g}. Color = max point-to-best-fit-plane distance.")
    else:
        st.plotly_chart(figure_quad_mesh(mesh, title=view_stage, correspondence_uv=_corr_uv, correspondence_bounds=_corr_bounds), use_container_width=True, key=f"mesh_{view_stage}")
    if view_stage == "K3D":
        # Geometry first, then the authoritative solver's optimization trajectory.
        try:
            from onestring_physics.k3d_iteration_history_view import render_k3d_iteration_history
            render_k3d_iteration_history(st, state)
        except Exception as exc:
            st.warning(f"K3D history plot skipped: {exc}")
        st.write({k: v for k, v in mesh.metrics.items() if k != "k3d_iteration_history"})
    else:
        st.write(mesh.metrics)
elif view_stage == "M3D":
    st.plotly_chart(figure_m3d_overlay(state, correspondence_uv=_corr_uv, correspondence_bounds=_corr_bounds), use_container_width=True, key="mesh_M3D")
    if state.mesh_3d_initial.metrics.get("m3d_used_height_field_shortcut", False):
        st.warning("M3D is generated by analytic scaled height-field debug shortcut, not by mesh parameterization.")
    st.write(
        {
            "m3d_metrics": state.mesh_3d_initial.metrics,
            "parameterization_metrics": state.surface_parameterization.metrics,
        }
    )
elif view_stage == "K2D":
    from onestring_physics.eq5_iteration_history_view import render_final_k2d_result, render_eq5_iteration_history
    if not render_final_k2d_result(st):
        st.plotly_chart(
            figure_flat_tile_layout(state.k2d_flat_layout, title="K2D flat independent tile layout", hinge_graph=state.hinge_graph, tile_colors=_corr_tile_colors),
            use_container_width=True,
            key="k2d",
        )
    with st.expander("K2D optimization history", expanded=False):
        render_eq5_iteration_history(st)
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
elif view_stage == "T3D-planarized K3D":
    st.subheader("T3D-planarized K3D")
    st.caption("T3D Eq.(2) planarity solve 後の上面4頂点を、K3Dと同じface IDで表示します。")
    st.plotly_chart(figure_t3d_planarized_k3d(state.mesh_3d_optimized, state.tiles_3d), use_container_width=True, key="t3d_planarized_k3d")
    solved_vertices = planarized_k3d_from_t3d(state.mesh_3d_optimized, state.tiles_3d)
    before_vertices = np.asarray(state.mesh_3d_optimized.vertices, dtype=float)
    delta = np.linalg.norm(solved_vertices - before_vertices, axis=1)
    st.write({"vertex_correction_rms": float(np.sqrt(np.mean(delta * delta))) if len(delta) else 0.0, "vertex_correction_max": float(np.max(delta)) if len(delta) else 0.0})
elif view_stage == "T3D":
    emergency_tile_ids = state.tiles_3d.metrics.get("t3d_emergency_normal_prism_tile_ids", [])
    if emergency_tile_ids:
        st.warning(
            "Geometric recovery was exhausted for the gray emergency tiles. "
            f"A one-sided normal prism is shown for inspection only: tile ids={emergency_tile_ids}. "
            "These tiles are not certified manufacturing geometry."
        )
    t3d_view_columns = st.columns(3)
    show_recovery_status = t3d_view_columns[0].toggle("Show recovery status", value=True)
    show_generated_caps = t3d_view_columns[1].toggle("Show generated cap faces", value=True)
    show_fundamental_only = t3d_view_columns[2].toggle("Show fundamental failures only", value=False)
    st.plotly_chart(
        figure_tile_assembly(
            state.tiles_3d,
            show_recovery_status=show_recovery_status,
            show_generated_cap_faces=show_generated_caps,
            show_fundamental_failures_only=show_fundamental_only,
            tile_colors=_corr_tile_colors,
        ),
        use_container_width=True,
        key="t3d",
    )
    if getattr(state.tiles_3d, "authoritative_solids", None):
        t3d_stl_bytes, t3d_stl_metrics = export_t3d_stl(state.tiles_3d)
        st.download_button(
            "Download authoritative T3D STL",
            data=t3d_stl_bytes,
            file_name="onestring_t3d_authoritative.stl",
            mime="model/stl",
        )
        st.caption(
            f"Authoritative variable-topology export: {t3d_stl_metrics['t3d_export_tile_count']} tiles, "
            f"{t3d_stl_metrics['t3d_export_triangle_count']} triangles."
        )
    st.write(state.tiles_3d.metrics)
elif view_stage == "T2D Top Hinge":
    top_hinge_graph = _onestring_pipeline._build_hinge_graph(
        state.mesh_2d_optimized.grid, state.mesh_3d_optimized.faces,
        state.tiles_2d_top_hinge, state.tiles_3d, dual=False,
    )
    _render_cross_stage_panel_inspector(
        state.tiles_2d_top_hinge,
        t2d_label="T2D Top Hinge",
        hinge_graph=top_hinge_graph,
        key_prefix="t2d_top",
    )
    collision_fig = figure_tile_assembly(state.tiles_2d_top_hinge, hinge_graph=top_hinge_graph)
    collision_fig, collision_ids, collision_pairs = add_t2d_collision_overlay(collision_fig, state.tiles_2d_top_hinge, hinge_graph=top_hinge_graph)
    st.plotly_chart(collision_fig, use_container_width=True, key="t2d_top_collision_map")
    st.caption(f"Collision map: {len(collision_ids)} panels / {len(collision_pairs)} SAT-overlap pairs (hinge-connected pairs excluded). Red = collision participant.")
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
    from onestring_physics.paper_reproduction_audit import audit_paper_reproduction
    _audit_params = st.session_state.get(
        "onestring_last_completed_params", pipeline_params
    )
    _paper_audit = audit_paper_reproduction(state, _audit_params)
    with st.expander(
        "論文との一致条件：Ω / CSF Split / T2D配置診断",
        expanded=(_paper_audit["verdict"]=="NOT_PAPER_REPRODUCTION"),
    ):
        st.write(
            "この診断は計算を変更しません。"
            "論文のFigure 6に対応するCSF Splitと、"
            "T2D Dual Hingeの非貫通配置を別々に評価します。"
        )
        if "onestring_last_completed_params" not in st.session_state:
            st.caption(
                "この結果は以前のUIで計算されたため、Ω方式の表示は現在の"
                "サイドバー設定から推定しています。パイプライン再計算で確定してください。"
            )
        a1,a2,a3,a4=st.columns(4)
        a1.metric("CSF Split本数",str(_paper_audit["csf_split_count"]))
        a2.metric("Collision Split本数",str(_paper_audit["collision_split_count"]))
        a3.metric("Split前CSF",str(_paper_audit["csf_before"]))
        a4.metric("Split後CSF",str(_paper_audit["csf_after"]))
        st.write({
            "Ω parameterization":_paper_audit["omega_mode"],
            "CSF limit":_paper_audit["csf_limit"],
            "CSF split status":_paper_audit["csf_split_status"],
            "T2D collision mode":_paper_audit["collision_mode"],
            "T2D SAT collision pairs":_paper_audit["t2d_collision_count"],
            "T2D hinge RMS":_paper_audit["t2d_hinge_rms"],
        })
        if _paper_audit["warnings"]:
            for item in _paper_audit["warnings"]:
                st.warning(f"[{item['stage']}] {item['code']}：{item['detail']}")
        else:
            st.info(
                "この簡易チェックでは明白な設定不一致は検出しませんでした。"
                "論文の同一メッシュ・境界条件・ソルバー・紐経路である証明ではありません。"
            )
        st.caption(
            "重要：Splitを入れても、同じUV上の点ごとのCSFは変わりません。"
            "現行実装に追加の再パラメータ化がない場合、切断による変形率改善は"
            "実証されていません。Split本数だけで論文への一致を判定しないでください。"
        )
    audit_metrics = state.tiles_2d_dual_hinge.metrics
    # Animate accepted SE(2) poses as actual 3D rigid solids. Z is fixed by
    # the solver; this visualization must not suggest out-of-plane motion.
    trial_summaries = audit_metrics.get("dual_hinge_trial_summary", [])
    trial_trajectories = audit_metrics.get("dual_hinge_trial_pose_histories", [])
    selected_trial_id = int(audit_metrics.get("dual_hinge_selected_trial", 0))
    pose_history = audit_metrics.get("dual_hinge_pose_history", [])
    hard_mode_active = audit_metrics.get("dual_hinge_collision_constraint_mode") == "hard"
    hard_ok = audit_metrics.get("hard_collision_feasible", False)
    hard_history = audit_metrics.get("hard_collision_history", [])
    split_history = audit_metrics.get("collision_split_feedback_steps", [])
    if audit_metrics.get("collision_split_feedback_enabled", False):
        status = audit_metrics.get("collision_split_status", "unknown")
        ncuts = len(audit_metrics.get("collision_split_accepted_lines", []))
        st.subheader("Collision-driven Split — 逐次フィードバック")
        st.write(
            f"状態: **{status}** / 採用したSplit: **{ncuts}本** / "
            f"境界距離の重み: {audit_metrics.get('collision_split_boundary_weight', 0.):.2g}"
        )
        st.caption("候補はM2Dの境界距離とT2DのSAT貫通深さから選択。"
                   "1本ごとにM2D→M3D→K3D→K2D→T3D→T2Dを再計算し、"
                   "改善しない候補は棄却します。"
                   f" Collision Split初期離隔倍率: {audit_metrics.get('collision_split_gap_multiplier', 1.):.2g}。")
        if audit_metrics.get("collision_split_lookahead_enabled", False):
            st.info(
                "2本先読み探索はONです。単独採用候補がない場合、衝突数が"
                "2%程度増えても貫通エネルギーが改善した候補から、もう1本試します。"
                "2本セットで従来の採用条件を満たしたときだけ正式採用します。"
            )
            if (status=="no_candidate_improved_or_rebuilt"
                    and not any(row.get("acceptance_mode")=="lookahead_second_cut"
                                for row in split_history)):
                st.caption(
                    "今回は2本目の全面再計算が記録されていません。"
                    "候補が中間条件を満たさないか、2本目の有効な切断線を生成できなかった可能性があります。"
                )
        else:
            st.caption(
                "2本先読み探索はOFFです。2本セットの改善を試す場合は、"
                "サイドバーの Enable two-cut lookahead from rejected Split をONにし、"
                "Maximum collision Split cutsを2以上にして再実行してください。"
            )
        if status == "T2D_reoptimized_on_existing_split_topology":
            st.info(
                "この画面のSplit候補プレビューは、最後にRun OneString pipelineで"
                "各Split候補を仮計算した時点の形状です。"
                "その後にT2Dのみ再最適化した現在の最終配置とは異なる場合があります。"
            )
        with st.expander(
            "棄却・採用した仮Split候補を3Dで描画して比較",
            expanded=True,
        ):
            from onestring_physics.paper_collision_split_preview_view import (
                render_split_candidate_previews,
            )
            render_split_candidate_previews(st, state, split_history)
        final_component_audit = audit_metrics.get("collision_split_final_component_audit", {})
        if final_component_audit:
            with st.expander("最終採用T2Dの領域間・領域内衝突（工程別）", expanded=True):
                summary_rows=[]
                for label in ("M2D","K2D","T2D Top","T2D Dual"):
                    stage=final_component_audit.get(label,{})
                    summary_rows.append({
                        "工程":label,
                        "衝突総数":stage.get("collision_pairs"),
                        "別領域間衝突":stage.get("between_split_component_pairs"),
                        "同領域内衝突":stage.get("within_split_component_pairs"),
                        "別領域間貫通エネルギー":stage.get("between_component_depth_squared"),
                        "同領域内貫通エネルギー":stage.get("within_component_depth_squared"),
                        "状態":stage.get("status"),
                    })
                st.dataframe(summary_rows,use_container_width=True,hide_index=True)
                st.caption("M2D/K2Dには厚みがなく、T2Dはパネル全8頂点のXY凸包で判定。"
                           "M2Dで分離できてもT2Dの厚みにより再衝突する場合があります。")
        if split_history:
            st.markdown("**Split前後の工程別衝突診断（同じ連結領域内 / 別々の連結領域間）**")
            rows=[]
            for entry in split_history:
                for period in ("before","after"):
                    audit=entry.get(period,{}).get("component_audit",{})
                    for stage in ("M2D","K2D","T2D Top","T2D Dual"):
                        stage_audit=audit.get(stage,{})
                        rows.append({
                            "Split候補":f"#{entry['cut_number']} {entry['proposal']['axis']}={entry['proposal']['value']:.4g}",
                            "段階":"追加前" if period=="before" else "仮Split再計算後",
                            "工程":stage,
                            "連結領域数":audit.get("component_count"),
                            "衝突総数":stage_audit.get("collision_pairs"),
                            "別領域間衝突":stage_audit.get("between_split_component_pairs"),
                            "同領域内衝突":stage_audit.get("within_split_component_pairs"),
                            "別領域間貫通エネルギー":stage_audit.get("between_component_depth_squared"),
                            "同領域内貫通エネルギー":stage_audit.get("within_component_depth_squared"),
                            "診断状態":stage_audit.get("status"),
                        })
            with st.expander("どの工程でSplit間の衝突が発生したか", expanded=True):
                st.dataframe(rows,use_container_width=True,hide_index=True)
                st.caption(
                    "M2D・K2Dは薄い四角形、T2Dは厚みを持つパネルのXY射影で判定。"
                    "別領域間の衝突が増えていれば離隔不足を疑えます。"
                    "同領域内だけの衝突ならSplit間隔の増加だけでは解消しません。"
                )
            def _format_split_result(row):
                before = row["before"]
                after = row["after"]
                energy_before = before.get("collision_energy")
                energy_after = after.get("collision_energy")
                pairs_before = before.get("collision_pairs")
                pairs_after = after.get("collision_pairs")
                improvement = (
                    100.0 * (1.0 - energy_after / energy_before)
                    if isinstance(energy_before, (int, float))
                    and isinstance(energy_after, (int, float))
                    and energy_before > 0 else None
                )
                reason_labels = {
                    "collision_free": "衝突ゼロを確認",
                    "penetration_energy_reduced_without_new_pairs": "貫通深刻度改善（衝突数も非増加）",
                    "penetration_severity_reduced_with_minor_pair_increase": "深い衝突を大幅に軽減（浅い衝突が少数増加）",
                    "too_many_new_collision_pairs": "新しく増えた衝突ペアが多すぎる",
                    "max_penetration_worsened": "最大貫通深さが悪化",
                    "hinge_connection_regression": "ヒンジ接続誤差が悪化",
                    "insufficient_penetration_energy_reduction": "貫通エネルギーの改善が不十分",
                    "insufficient_reduction_for_extra_pairs": "新規衝突の増加を許容するほどの改善ではない",
                    "missing_collision_or_hinge_diagnostics": "貫通・ヒンジ診断値が欠落",
                }
                return {
                    "Split番号": row["cut_number"],
                    "採否": "採用" if row["accepted"] else "棄却",
                    "方向": row["proposal"]["axis"],
                    "格子座標": row["proposal"]["value"],
                    "境界距離/タイル": row["proposal"]["boundary_distance_tiles"],
                    "衝突強度": row["proposal"]["hotspot_severity"],
                    "候補スコア": row["proposal"]["score"],
                    "衝突ペア 前": pairs_before,
                    "衝突ペア 後": pairs_after,
                    "衝突ペア 増減": (
                        pairs_after - pairs_before
                        if isinstance(pairs_before, int) and isinstance(pairs_after, int)
                        else None
                    ),
                    "貫通エネルギー 前": energy_before,
                    "貫通エネルギー 後": energy_after,
                    "貫通改善率 (%)": round(improvement, 2) if improvement is not None else None,
                    "最大貫通深さ 前": before.get("penetration_max"),
                    "最大貫通深さ 後": after.get("penetration_max"),
                    "ヒンジRMS 前": before.get("hinge_rms"),
                    "ヒンジRMS 後": after.get("hinge_rms"),
                    "理由": reason_labels.get(row["decision_reason"], row["decision_reason"]),
                }
            st.dataframe([_format_split_result(row) for row in split_history],
                         use_container_width=True, hide_index=True)
            st.caption(
                "Splitの採用は貫通エネルギー改善、最大貫通深さの非悪化、"
                "ヒンジ接続誤差で判定します。衝突ペア数が増える場合は、"
                "増加が少数（従来数の2%、最低2ペア以内）で、"
                "貫通エネルギーが20%以上改善した場合のみ許容します。"
                "衝突ゼロに到達したこととは区別してください。"
            )
    if hard_mode_active:
        if hard_ok:
            st.success(
                f"Hard Collision: XY footprint非貫通を検証済み。"
                f"採用元のソフト試行 #{audit_metrics.get('hard_collision_repair_source_trial')}。"
                f"最終ヒンジRMS {audit_metrics.get('hinge_rms', 0.):.5g}。"
                f"製作可能性: {audit_metrics.get('fabrication_feasible', False)}"
            )
        else:
            st.error("Hard Collision未達成：このT2Dは非貫通の製作可能配置ではありません。表示中の配置は診断用です。")
        if audit_metrics.get("hard_collision_repair_attempts"):
            with st.expander("Hard Collision修復試行の診断", expanded=True):
                st.dataframe(audit_metrics["hard_collision_repair_attempts"], use_container_width=True, hide_index=True)
    if trial_summaries:
        st.subheader("T2D探索試行の比較")
        st.markdown(
            f"**最終採用元: 試行 #{selected_trial_id} "
            f"（{audit_metrics.get('dual_hinge_selected_source', 'baseline')}）** "
            f"｜Soft探索 {len(trial_summaries)} / 予定 {audit_metrics.get('dual_hinge_trials_requested', len(trial_summaries))} 回 "
            f"｜初回比エネルギー変化 {-audit_metrics.get('dual_hinge_best_improvement', 0.):.6g}"
        )
        st.caption(
            "計算予算の方式: "
            + ("各試行に独立した時間制限" if audit_metrics.get("dual_hinge_budget_mode")=="per_trial"
               else "総予算を全試行へ均等に分配")
            + f"。L-BFGS-Bの試行ごとの上限: {audit_metrics.get('dual_hinge_trial_time_budget_sec', 0.):.3g} 秒。"
        )
        if not audit_metrics.get("dual_hinge_all_requested_trials_run", True):
            st.error("要求されたL-BFGS-B試行数と実行した試行数が一致しません。")
        if hard_mode_active and hard_ok:
            st.caption("以下の表はHard修復前のSoft探索結果です。最終採用配置はHard修復・非貫通制約下の再最適化後のものです。")
        st.dataframe(trial_summaries, use_container_width=True, hide_index=True)
    if trial_trajectories:
        trial_ids = ([-1] if hard_ok and hard_history else []) + [
            int(item["trial_id"]) for item in trial_trajectories
        ]
        display_trial = st.selectbox(
            "3Dアニメーションで表示する探索試行",
            trial_ids,
            index=0 if hard_ok and hard_history else (
                trial_ids.index(selected_trial_id) if selected_trial_id in trial_ids else 0
            ),
            format_func=lambda trial_id: (
                "★ Hard修復・最終採用配置" if trial_id == -1
                else f"試行 #{trial_id} — {next((item['initial_source'] for item in trial_summaries if item['trial_id']==trial_id), '')}"
                + (" ★ Soft最良" if trial_id==selected_trial_id else "")
            ),
            key="dual_hinge_inspection_trial",
        )
        if display_trial == -1:
            pose_history = hard_history
        else:
            inspected_trial = next(item for item in trial_trajectories if item["trial_id"]==display_trial)
            pose_history = inspected_trial["pose_history"]
            if hard_ok or display_trial != selected_trial_id:
                st.caption("選択した試行の途中経過を表示しています。最終T2D結果はHard修復の採用配置です（Hard成功時）。")
    _history_tile_count = len(state.tiles_2d_top_hinge.vertices)
    _history_frame_count = len(pose_history)
    _large_history = _history_tile_count * _history_frame_count > 12000
    _render_history = bool(pose_history)
    if pose_history and _large_history:
        st.warning(
            f"このgridでは最適化履歴が大きくなっています"
            f"（{_history_tile_count}パネル × {_history_frame_count}フレーム）。"
            "全フレームの3D描画はブラウザが重くなり、結果が表示されない原因になるため、"
            "初期状態では無効にしています。"
        )
        _render_history = st.checkbox(
            "大きなDual Hinge 3D最適化履歴を描画する（高負荷）",
            value=False,
            key="dual_hinge_show_large_history",
        )
    if pose_history and _render_history:
        import plotly.graph_objects as go
        import numpy as _np
        rest_tiles = _np.asarray(state.tiles_2d_top_hinge.vertices, dtype=float)
        centers = rest_tiles[:, :4, :2].mean(axis=1)
        local_xy = rest_tiles[:, :, :2] - centers[:, None, :]
        tile_scale = float(_np.median(_np.linalg.norm(
            _np.roll(rest_tiles[:, :4, :2], -1, axis=1) - rest_tiles[:, :4, :2], axis=2)))
        hinges = state.hinge_graph.hinges
        # Each rigid panel is an eight-vertex prism: top/bottom quads and sides.
        triangles = ((0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
                     (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
                     (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7))
        edges = ((0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7))
        ii, jj, kk = [], [], []
        for tile_idx in range(len(rest_tiles)):
            for x, y, z in triangles:
                ii.append(8 * tile_idx + x)
                jj.append(8 * tile_idx + y)
                kk.append(8 * tile_idx + z)
        def _hinge_frame(entry):
            pose = _np.asarray(entry["poses"], dtype=float)
            angle = pose[:, 0]
            c, sn = _np.cos(angle)[:, None], _np.sin(angle)[:, None]
            xy = _np.stack((c * local_xy[:, :, 0] - sn * local_xy[:, :, 1],
                            sn * local_xy[:, :, 0] + c * local_xy[:, :, 1]), axis=2)
            xy += centers[:, None, :] + tile_scale * pose[:, None, 1:]
            xyz = _np.concatenate((xy, rest_tiles[:, :, 2:3]), axis=2)
            flat = xyz.reshape(-1, 3)
            ex, ey, ez, hx, hy, hz = [], [], [], [], [], []
            for tile in xyz:
                for v1, v2 in edges:
                    a, b = tile[v1], tile[v2]
                    ex.extend((a[0], b[0], None))
                    ey.extend((a[1], b[1], None))
                    ez.extend((a[2], b[2], None))
            for hinge in hinges:
                a = xyz[hinge.tile_a, hinge.local_vertex_a]
                b = xyz[hinge.tile_b, hinge.local_vertex_b]
                hx.extend((a[0], b[0], None))
                hy.extend((a[1], b[1], None))
                hz.extend((a[2], b[2], None))
            return [
                go.Mesh3d(x=flat[:, 0], y=flat[:, 1], z=flat[:, 2],
                          i=ii, j=jj, k=kk, color="#26c6b7",
                          opacity=0.80, flatshading=True, name="Rigid panels",
                          hoverinfo="skip", showscale=False),
                go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                             line=dict(color="#087f75", width=2),
                             name="Panel edges", hoverinfo="skip"),
                go.Scatter3d(x=hx, y=hy, z=hz, mode="lines",
                             line=dict(color="#e05b36", width=4),
                             name="Hinge connection gaps", hoverinfo="skip"),
            ]
        frames = [go.Frame(data=_hinge_frame(entry), name=str(entry["iteration"]))
                  for entry in pose_history]
        all_xyz = rest_tiles.reshape(-1, 3)
        # Fix axes across the entire trajectory, avoiding misleading auto-zoom.
        all_centers = _np.asarray([_np.asarray(entry["poses"])[:, 1:]
                                   for entry in pose_history])
        all_trans = centers[None, :, :] + tile_scale * all_centers
        max_radius = float(_np.max(_np.linalg.norm(local_xy, axis=2)))
        low = _np.min(all_trans, axis=(0, 1)) - max_radius
        high = _np.max(all_trans, axis=(0, 1)) + max_radius
        zlo = float(_np.min(all_xyz[:, 2]))
        zhi = float(_np.max(all_xyz[:, 2]))
        zpad = max((zhi - zlo) * 0.1, tile_scale * 0.05)
        fig = go.Figure(data=frames[0].data, frames=frames)
        fig.update_layout(
            title="T2D Dual Hinge — 3D rigid panel optimization history",
            scene=dict(xaxis=dict(title="X", range=[float(low[0]), float(high[0])]),
                       yaxis=dict(title="Y", range=[float(low[1]), float(high[1])]),
                       zaxis=dict(title="Z (fixed)", range=[zlo-zpad, zhi+zpad]),
                       aspectmode="data", camera=dict(eye=dict(x=1.5, y=1.4, z=1.1))),
            uirevision="dual_hinge_3d_camera", height=720, showlegend=True,
            updatemenus=[dict(type="buttons", showactive=False, buttons=[
                dict(label="▶ Play", method="animate", args=[None, dict(
                    frame=dict(duration=100, redraw=True), transition=dict(duration=0),
                    fromcurrent=True, mode="immediate")]),
                dict(label="❚❚ Pause", method="animate", args=[[None], dict(
                    frame=dict(duration=0, redraw=False), mode="immediate")])])],
            sliders=[dict(steps=[dict(label=str(entry["iteration"]), method="animate",
                    args=[[str(entry["iteration"])], dict(frame=dict(duration=0, redraw=True),
                    mode="immediate")]) for entry in pose_history],
                    currentvalue=dict(prefix="Iteration "), len=0.95)])
        with st.expander("▶ Dual Hinge 3D最適化アニメーション", expanded=True):
            shown_trial = next((item for item in trial_summaries if item["trial_id"] == (display_trial if trial_trajectories else selected_trial_id)), None)
            if shown_trial and shown_trial.get("timed_out", False):
                st.warning("この試行は時間制限による途中結果です。")
            st.caption(f"記録されたL-BFGS採用反復 {len(frames)} フレーム（間引き設定により実際の反復数とは異なることがあります）。"
                       "青緑＝厚みを持つ剛体パネル、橙＝ヒンジ接続誤差。"
                       " パネルはXY平面内でのみ剛体運動し、Z位置は固定です。")
            st.plotly_chart(fig, use_container_width=True, key="dual_hinge_iteration_animation_3d")
    elif pose_history:
        st.info("最適化履歴は保存済みです。上のチェックをONにすると3Dアニメーションを描画します。"
                "アニメーションを描画しなくても最終T2D配置・衝突マップは確認できます。")
    else:
        st.info("この実行にはヒンジ配置履歴がありません。修正後に再計算してください。")
    if audit_metrics.get("dual_hinge_history"):
        with st.expander("DualHinge エネルギー・接続・貫通の監査", expanded=True):
            st.caption("各反復の生エネルギーと重み付き寄与。XY footprint の貫通と3D solidの貫通は別の指標です。")
            st.dataframe(audit_metrics["dual_hinge_history"], use_container_width=True)
            st.write({key: audit_metrics.get(key) for key in (
                "dual_hinge_weights", "dual_hinge_solver_success", "dual_hinge_solver_message",
                "dual_hinge_timed_out", "dual_hinge_stop_reason",
                "dual_hinge_iterations", "dual_hinge_time_budget_sec",
                "dual_hinge_budget_mode", "dual_hinge_trial_time_budget_sec",
                "dual_hinge_hard_phase_time_budget_sec", "dual_hinge_all_requested_trials_run",
                "dual_hinge_search_mode", "dual_hinge_collision_constraint_mode",
                "hard_collision_status", "hard_collision_feasible",
                "hard_collision_repair_source_trial", "hard_collision_tolerance",
                "hard_collision_repair_elapsed_sec", "hard_collision_refinement_elapsed_sec",
                "hard_collision_best_unresolved_count", "hard_collision_best_unresolved_max_depth",
                "dual_hinge_selected_trial",
                "dual_hinge_selected_source", "dual_hinge_trials_completed",
                "dual_hinge_trials_requested", "dual_hinge_best_improvement",
                "dual_hinge_search_global_budget_exhausted",
                "dual_hinge_setup_elapsed_sec", "dual_hinge_optimization_elapsed_sec",
                "hinge_mean", "hinge_rms", "hinge_p95", "hinge_max",
                "collision_pair_count", "penetration_max", "penetration_rms",
                "solid_collision_pair_count", "solid_penetration_max", "solid_penetration_rms",
                "dual_hinge_fixed_z_infeasible", "fabrication_feasible")})
    _render_cross_stage_panel_inspector(
        state.tiles_2d_dual_hinge,
        t2d_label="T2D Dual Hinge",
        hinge_graph=state.hinge_graph,
        key_prefix="t2d_dual",
    )
    collision_fig = figure_tile_assembly(state.tiles_2d_dual_hinge, hinge_graph=state.hinge_graph)
    # Match the optimizer: hinged neighbours still contribute to footprint SAT
    # penetration. Do not silently hide those pairs in the audit overlay.
    collision_fig, collision_ids, collision_pairs = add_t2d_collision_overlay(collision_fig, state.tiles_2d_dual_hinge, hinge_graph=None)
    st.plotly_chart(collision_fig, use_container_width=True, key="t2d_dual_collision_map")
    st.caption(f"Collision map: {len(collision_ids)} panels / {len(collision_pairs)} SAT-overlap pairs (hinge-connected pairs included, as in ECollision). Red = collision participant.")

    # Diagnostic counterfactual: rigidly put ACTUAL T2D fabricated solids at
    # their corresponding T3D tile poses. This does not claim that a string
    # can move through a collision-free path to these poses.
    st.subheader("T2Dのパネルそのものを剛体移動・回転した3Dモデル")
    st.markdown(
        "**青緑色の全パネルはT2D由来です。** "
        "T2D Dual Hingeの8頂点を各パネルにつき同じ回転・平行移動で変換し、"
        "**曲げず・伸ばさず・変形させず**3Dへ配置します。"
    )
    st.caption(
        "どの位置・方向へ動かすかだけを対応するT3Dパネルの位置・姿勢から決めます。"
        "T3Dのパネル形状は一切流用していません。"
        "ヒンジを物理的に閉じながら移動させた軌道ではなく、"
        "設計した3D姿勢へT2Dパネルを剛体再配置した結果です。"
    )
    from onestring_physics.t2d_rigid_reassembly import rigid_reassemble_t2d
    try:
        _reassembled_tiles, _reassembled_metrics = rigid_reassemble_t2d(state)
    except (ValueError, np.linalg.LinAlgError) as _reassembly_exc:
        st.error(f"T2D剛体再配置に失敗: {_reassembly_exc}")
    else:
        c_r1, c_r2, c_r3, c_r4 = st.columns(4)
        c_r1.metric("T3Dとの頂点RMS", f"{_reassembled_metrics['total_target_vertex_rms']:.4g}")
        c_r2.metric("ヒンジRMS", f"{_reassembled_metrics['hinge_rms']:.4g}")
        c_r3.metric("再組立て3D交差ペア(近似)", str(_reassembled_metrics["reassembled_3d_obb_pairs"]))
        c_r4.metric("T2D初期3D交差ペア(近似)", str(_reassembled_metrics["initial_t2d_3d_obb_pairs"]))
        st.caption(
            "形状不変の確認：パネル内頂点間距離の最大変化 "
            f"{_reassembled_metrics['max_internal_distance_change']:.3e}。"
            "（理論上は0、数値誤差のみ）"
        )
        _reassembly_view = TileAssembly(
            vertices=_reassembled_tiles,
            top_faces=state.tiles_2d_dual_hinge.top_faces,
            bottom_faces=state.tiles_2d_dual_hinge.bottom_faces,
            side_faces=state.tiles_2d_dual_hinge.side_faces,
            stage="Only moved and rotated T2D rigid panels (no T3D geometry)",
        )
        _reassembly_fig = figure_tile_assembly(
            _reassembly_view,
            title="T2D rigid panels moved into 3D (no deformation)",
        )
        if st.checkbox(
            "比較用にT3D設計形状を半透明で重ねる（初期状態は非表示）",
            value=False,
            key="rigid_reassembly_show_target",
        ):
            add_tile_assembly(
                _reassembly_fig, state.tiles_3d,
                color="#2563eb", opacity=0.16, name="T3D target (reference)",
            )
        st.plotly_chart(_reassembly_fig, use_container_width=True,
                        key="t2d_rigid_reassembly_3d")
        if _reassembled_metrics["reassembled_3d_obb_pairs"]:
            st.warning(
                "再配置後の3Dパネル同士にも近似交差が存在します。"
                "T2Dで衝突0件でもT3Dの配置で3D衝突が起こり得ます。"
                "接触判定は15軸OBB SAT近似であり厳密な厚肉タイルの衝突証明ではありません。"
            )
        if _reassembled_metrics["hinge_count"] and _reassembled_metrics["hinge_rms"] > 1e-3:
            st.warning("剛体再配置後にもヒンジの接続ギャップが残っています。")
        with st.expander("剛体再組立ての数値診断", expanded=False):
            st.json({k:v for k,v in _reassembled_metrics.items()
                     if k!="reassembled_3d_obb_participants"})
    if hard_mode_active and not hard_ok:
        st.error("Hard非貫通未達成のため、製作用T2D STLのダウンロードを無効化しています。")
    else:
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
        use_container_width=True,
        key="gap_lift_string",
    )
    if physics_backend == "paper_pd":
        with st.expander("Paper PD: 実際にSnap拘束するT2D側面とLiftの確認",expanded=False):
            from onestring_physics.paper_pd_deployment import (
                _validate_route, PaperPDInputError,
            )
            try:
                _snap_faces,_lift_faces,_route_check=_validate_route(state)
            except (PaperPDInputError,ValueError) as exc:
                st.error(
                    f"論文準拠の拘束幾何を作れません: {exc}。"
                    "T2D・Gap Graph・String Pathを更新した最新版で"
                    "Run OneString pipelineから再計算してください。"
                )
            else:
                st.caption(
                    "赤線は各ギャップで一致させる2枚のT2D側面中央同士を結ぶ"
                    "診断線です。物理的にその線に紐が通るという意味ではありません。"
                    "経路の順序自体は上のString Pathで確認してください。"
                )
                _constrained_view=figure_tile_assembly(
                    state.tiles_2d_dual_hinge,
                    title="Actual T2D material side-face Snap pairs (red)",
                )
                _snap_coords=np.asarray(state.tiles_2d_dual_hinge.vertices,float)
                _xx,_yy,_zz=[],[],[]
                for snap in _snap_faces:
                    pa=_snap_coords[snap.a,list(snap.face_a)].mean(axis=0)
                    pb=_snap_coords[snap.b,list(snap.face_b)].mean(axis=0)
                    _xx.extend([pa[0],pb[0],None])
                    _yy.extend([pa[1],pb[1],None])
                    _zz.extend([pa[2],pb[2],None])
                _constrained_view.add_trace(go.Scatter3d(
                    x=_xx,y=_yy,z=_zz,mode="lines+markers",
                    line=dict(color="#ef4444",width=5),
                    marker=dict(color="#ef4444",size=2),
                    name="Snap material face midpoints",
                ))
                st.plotly_chart(_constrained_view,use_container_width=True,
                                key="paper_snap_actual_material_faces")
                st.write({
                    **_route_check,
                    "snap_material_face_pairs":[{
                        "gap_id":int(q.gap_id),
                        "tile_a":int(q.a),"tile_b":int(q.b),
                        "face_a":list(q.face_a),"face_b":list(q.face_b),
                    } for q in _snap_faces],
                })
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
    selected_simulation_name = {
        "paper_pd": "Paper Appendix E Snap/Lift projective reference (3D contact approximation)",
        "native": "Native rigid-body + tension-only string (experimental PBD)",
        "legacy": "Legacy Projective Dynamics + 2D SAT/target guidance",
        "abd": "Autodesk ABD + IPC/CCD + unilateral OneString constraint",
    }[physics_backend]
    st.caption(f"Selected simulation: {selected_simulation_name}. This view displays solver output frames, not design interpolation.")
    if physics_backend == "abd":
        st.info(
            "ABD mode exports the thick T3D panel meshes, hinge pin joints, guide points and pull schedule to abd_sim. "
            "Only frames returned by that executable are shown here; legacy PD/SAT is not used as fallback."
        )

    frame_count = st.slider(
        "animation frames (saved display samples)", 8, 160, 40, 4,
        help="Only the number of saved visual frames. Native PHYSICS steps "
             "and timestep in the sidebar control simulated time.",
    )
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

    source_label = {
        "paper_pd": "Section 5.4 / Supplement E Snap+Lift (independent approximate solver)",
        "native": "T2D-origin rigid-body PBD with one routed string (experimental)",
        "legacy": "Legacy target-guided Projective Dynamics",
        "abd": "Autodesk ABD simulation",
    }[physics_backend]
    st.caption(f"Animation source: {source_label}.")

    if physics_backend in {"paper_pd", "native", "legacy", "abd"}:
        if physics_backend == "paper_pd":
            st.info(
                "論文準拠の駆動拘束：MILPで最適化された有効な閉経路上の"
                "実パネル側面中央をSnapし、指定Lift Pointを目標位置へ動かす。"
                "全パネルのT3D姿勢への吸引はありません。"
                "PDの反復解法と3D接触は近似で、著者の数値実装の完全再現ではありません。"
            )
        elif physics_backend == "native":
            st.info(
                "Native: 各タイルは6自由度の剛体、ヒンジは点接続。"
                "床面支持・重力・3D OBB接触を使い、紐経路の総長だけを"
                "一方向に短縮します。T3Dは結果比較専用です。"
                "近似PBDであり、CCD/IPC・滑車摩擦はありません。"
            )
        elif physics_backend == "abd":
            st.info(
                "Physical model used here: affine bodies with IPC/CCD contact, pin-joint hinges, gravity and friction. "
                "The routed guide points are coupled by a unilateral total-length constraint, and its commanded length decreases over time."
            )
        else:
            st.info(
                "Physical model used here: the string is not simulated as rope particles. It is encoded as positional constraints. "
                "Only gaps on the computed string path receive snap constraints; selected lift gaps move toward prescribed 3D lift targets. "
                "Rigid and collision projections keep tiles rigid and non-overlapping."
            )
        c0, c1, c2, c3 = st.columns([1.3, 1.3, 1, 3])
        if "paper_pd_frame" not in st.session_state:
            st.session_state.paper_pd_frame = 0

        paper_sim_key = (
            "assembly_solver_animation",
            actuation_key,
            frame_count,
        )
        current_result_backend = str(getattr(state.simulation_result, "metrics", {}).get("physics_backend", "")) if state.simulation_result is not None else ""
        need_sim_frames = (
            state.simulation_result is None
            or current_result_backend != physics_backend
            or len(getattr(state.simulation_result, "frames", [])) < 2
            or (physics_backend != "abd" and st.session_state.get("paper_pd_animation_key") != paper_sim_key)
        )
        refresh_label = {
            "paper_pd": "Run / refresh Paper Appendix E PD",
            "native": "Run / refresh native string simulation",
            "legacy": "Run / refresh legacy PD",
            "abd": "Run / refresh ABD simulation",
        }[physics_backend]
        refresh_sim = c0.button(refresh_label, type="primary")
        player_mode = c1.radio(
            "player mode",
            ["Smooth browser player", "Server frame player", "Scrubber"],
            horizontal=False,
            help="Smooth browser player preloads sampled frames and plays in Plotly without Streamlit reruns.",
        )
        if c2.button("⏮ Reset"):
            st.session_state.paper_pd_frame = 0

        if need_sim_frames and physics_backend == "abd" and not refresh_sim:
            st.warning("No ABD result matching the current settings is loaded. Press 'Run / refresh ABD simulation'.")

        if refresh_sim or (need_sim_frames and physics_backend != "abd"):
            paper_params = replace(
                deployment_params,
                steps=(native_solver_steps if physics_backend == "native"
                       else paper_pd_steps if physics_backend == "paper_pd"
                       else max(sim_steps, frame_count)),
                store_animation_frames=True,
                max_animation_frames=frame_count,
            )
            if physics_backend == "abd":
                st.write(
                    {
                        "physics_backend": "abd",
                        "executable": paper_params.abd_executable or "auto-detect project Release build",
                        "timestep": paper_params.abd_timestep,
                        "simulation_steps": paper_params.steps,
                        "cpu_threads": paper_params.abd_nthreads if paper_params.abd_nthreads > 0 else "automatic (ABD default)",
                        "timeout_seconds": paper_params.abd_timeout_seconds,
                        "pull_end_ratio": paper_params.abd_pull_end_ratio,
                        "collision_model": "Autodesk ABD IPC barrier + CCD",
                        "legacy_fallback": False,
                    }
                )
            elif physics_backend == "paper_pd":
                st.write({
                    "physics_backend":"paper_pd",
                    "string_route_model":state.string_path.metrics.get("string_path_model"),
                    "route_status":state.string_path.metrics.get("string_path_status"),
                    "drive":"side-face midpoint Snap + selected Lift targets",
                    "no_direct_target_pose_fit":True,
                    "integration_steps":paper_params.steps,
                    "integration_timestep":paper_params.paper_pd_timestep,
                    "contact":"approximate discrete 3D OBB SAT (not authors exact collision solver)",
                })
            elif physics_backend == "native":
                st.write({
                    "physics_backend": "native",
                    "initial_geometry": "exact T2D Dual rigid panels",
                    "string_model": "unilateral routed path length",
                    "initial_anchor_height_tiles": paper_params.native_anchor_height_tiles,
                    "end_string_length_ratio": paper_params.native_pull_end_ratio,
                    "timestep": paper_params.native_timestep,
                    "steps": paper_params.steps,
                    "simulated_seconds": (
                        (paper_params.steps+paper_params.native_extra_settle_steps)
                        * paper_params.native_timestep
                    ),
                    "extra_settle_steps":paper_params.native_extra_settle_steps,
                    "pull_fraction": paper_params.native_pull_fraction,
                    "pull_seconds": (
                        paper_params.steps * paper_params.native_timestep
                        * paper_params.native_pull_fraction
                    ),
                    "settling_seconds": (
                        (paper_params.steps * (1.0 - paper_params.native_pull_fraction)
                         + paper_params.native_extra_settle_steps)
                        * paper_params.native_timestep
                    ),
                    "display_frames_requested": frame_count,
                    "solver_iterations": paper_params.solver_iterations,
                    "substeps": paper_params.solver_substeps,
                    "contact": "discrete 3D OBB-SAT approximation (not CCD/IPC)",
                })
            else:
                backend_info = compute_backend_info(paper_params.compute)
                st.write(
                    {
                        "physics_backend": "legacy",
                        "requested_backend": backend_info.get("requested_backend"),
                        "selected_backend": backend_info.get("current_backend"),
                        "cuda_available": backend_info.get("cuda_available"),
                        "gpu_name": backend_info.get("gpu_name"),
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

            with st.spinner(f"Running {selected_simulation_name}"):
                try:
                    state.simulation_result = simulate_onestring_deployment(
                        state,
                        paper_params,
                        progress_callback=_paper_sim_progress,
                    )
                except (RuntimeError, ValueError) as exc:
                    st.error(str(exc))
                    st.stop()
            progress_bar.progress(1.0)
            runtime = state.simulation_result.metrics.get("simulation_runtime_seconds", state.simulation_result.metrics.get("elapsed_time"))
            runtime_text = f"{float(runtime):.3f}s" if runtime is not None else "not reported"
            progress_text.caption(
                f"Done: physics_backend={state.simulation_result.metrics.get('physics_backend')}, "
                f"actual_backend={state.simulation_result.metrics.get('actual_backend', physics_backend)}, elapsed={runtime_text}"
            )
            st.session_state.paper_pd_animation_key = paper_sim_key
            st.session_state.actuation_key = actuation_key
            st.session_state.paper_pd_frame = 0

        result = state.simulation_result
        if result is not None and str(result.metrics.get("physics_backend", "")) != physics_backend:
            result = None
        if result is None and physics_backend == "abd":
            st.stop()
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
                stage=source_label,
            )
            fig = figure_tile_assembly(
                frame_assembly,
                title=f"{source_label} frame {frame_id + 1}/{max_frame + 1}",
                hinge_graph=state.hinge_graph if show_hinges_preview and limit == total_tiles else None,
            )
            if physics_backend == "native" and result is not None:
                strings=getattr(result,"string_path_frames",[])
                if 0 <= frame_id < len(strings):
                    route=np.asarray(strings[frame_id],dtype=float)
                    fig.add_trace(go.Scatter3d(
                        x=route[:,0],y=route[:,1],z=route[:,2],
                        mode="lines+markers",
                        line=dict(color="#dc2626",width=6),
                        marker=dict(color="#f97316",size=2.5),
                        name="routed string (constraint)",
                    ))
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
                title=f"Smooth {source_label} ({len(frames)} frames)",
                string_path_frames=(
                    getattr(result,"string_path_frames",None)
                    if physics_backend=="native" and result is not None
                    else None
                ),
            )
            st.plotly_chart(
                smooth_fig,
                use_container_width=True,
                key=f"paper_pd_smooth_{len(frames)}_{preview_tiles}_{int(show_target_preview)}",
                config={"scrollZoom": True, "displayModeBar": True, "responsive": True},
            )
            st.info("Use the Smooth play button inside the chart. You can rotate/zoom the view while it is playing.")
        elif player_mode == "Scrubber":
            st.plotly_chart(render_paper_pd_frame(current_frame), use_container_width=True, key="paper_pd_scrubber")
        else:
            autoplay = st.button("Play simulation")
            placeholder = st.empty()
            progress = st.progress((current_frame + 1) / (max_frame + 1))
            if autoplay:
                for frame_id in range(current_frame, max_frame + 1):
                    placeholder.plotly_chart(render_paper_pd_frame(frame_id), use_container_width=True, key=f"paper_pd_play_{frame_id}_{time.time_ns()}")
                    progress.progress((frame_id + 1) / (max_frame + 1))
                    st.session_state.paper_pd_frame = frame_id
                    time.sleep(max(0.01, 1.0 / fps))
            else:
                placeholder.plotly_chart(render_paper_pd_frame(current_frame), use_container_width=True, key="paper_pd_current")

        if result is not None:
            if physics_backend == "paper_pd":
                vals=result.metrics
                m0,m1,m2,m3=st.columns(4)
                m0.metric("Snap側面ギャップRMS",f"{vals.get('snap_error',0.):.5g}")
                m1.metric("Lift位置RMS",f"{vals.get('lift_error',0.):.5g}")
                m2.metric("ヒンジRMS",f"{vals.get('hinge_error',0.):.5g}")
                m3.metric("残存3D交差 (近似)",str(vals.get("collision_count","?")))
                st.caption(
                    "Snap対象 {0}組、Lift対象 {1}箇所。紐の実体・滑車摩擦はPD内で"
                    "解かず、Appendix Eの幾何拘束としてモデル化。"
                    .format(vals.get("actuated_snap_gap_count",0),
                            vals.get("actuated_lift_gap_count",0))
                )
                if vals.get("status") != "constraints_satisfied":
                    st.warning(
                        "計算は終了しましたが、Snap/Lift/ヒンジ/非衝突の"
                        "拘束残差が残っています。論文の組立て成功と解釈しないでください。"
                    )
                st.line_chart({
                    "Snap midpoint RMS":getattr(result,"snap_error_history",[]),
                    "Lift RMS":getattr(result,"lift_error_history",[]),
                    "Hinge RMS":getattr(result,"hinge_error_history",[]),
                })
            if physics_backend == "native":
                vals=result.metrics
                m0,m1,m2,m3=st.columns(4)
                m0.metric("T2Dからの最高点の上昇",f"{vals.get('rise_from_t2d',0.):.5g}")
                m1.metric("ヒンジRMS",f"{vals.get('hinge_error',0.):.5g}")
                m2.metric("残存3D接触ペア (OBB)",str(vals.get('collision_count','?')))
                m3.metric("紐長の未達",f"{vals.get('constraint_violation',0.):.5g}")
                initial_overlap=int(vals.get("initial_collision_count",0))
                if initial_overlap:
                    st.warning(
                        f"初期T2Dに {initial_overlap} 件の3D近似接触があります。"
                        "物理的には貫通した剛体配置から開始しているため、"
                        "途中の軌道を信頼できる状態にするには、先にT2Dの衝突を修正する必要があります。"
                        "計算は診断目的で継続しています。"
                    )
                st.caption(
                    f"物理時間 {vals.get('total_physical_duration_seconds',0.):.2f}秒："
                    f"紐を引く {vals.get('pull_duration_seconds',0.):.2f}秒、"
                    f"引いた後の保持 {vals.get('settle_duration_seconds',0.):.2f}秒。"
                    f"保存した表示フレーム {len(frames)}枚。"
                )
                st.metric(
                    "T3D設計位置との最終頂点RMS（目標に吸引しない）",
                    f"{vals.get('final_deployment_error_to_T3D',float('nan')):.4g}",
                )
                if not vals.get("stable_state",False):
                    st.error(
                        "計算時間が終了しましたが、物理拘束は未収束です。"
                        "これは再生フレーム数の不足ではありません。"
                        "ヒンジ誤差・残存3D衝突・紐長未達を下のグラフで確認してください。"
                        "T2Dの2D衝突0件は、立ち上がった後の3D非貫通を保証しません。"
                    )
                else:
                    st.info(
                        "指定の近似拘束の残差は許容範囲ですが、"
                        "T3D設計形状に達したことや、製作物で組み立て可能なことは証明されていません。"
                    )
                _contact_hist=getattr(result,"collision_counts",[])
                _hinge_hist=getattr(result,"hinge_error_history",[])
                if _contact_hist and _hinge_hist and len(_contact_hist)==len(_hinge_hist):
                    c1,c2=st.columns(2)
                    with c1:
                        st.caption("各保存フレームの3D接触ペア数（OBB近似）")
                        st.line_chart({"3D contact pairs": _contact_hist})
                    with c2:
                        st.caption("各保存フレームのヒンジRMS")
                        st.line_chart({"hinge RMS": _hinge_hist})
                history_actual=getattr(result,"string_actual_history",[])
                history_cmd=getattr(result,"string_command_history",[])
                if history_actual and len(history_actual)==len(history_cmd):
                    st.markdown("**紐長の時間変化（赤線は上の3Dビューで描画）**")
                    st.line_chart({
                        "実際の紐長":history_actual,
                        "指定紐長（片側制約）":history_cmd,
                    })
                st.caption(
                    "厳密な衝突判定・摩擦評価・製造可能性の証明ではありません。"
                    "剛体PBDの近似シミュレーションです。元論文の著者実装とは異なります。"
                )
            st.write(
                {
                    "physics_backend": result.metrics.get("physics_backend"),
                    "simulation_model": {
                        "paper_pd": "Section 5.4 Appendix E PD: route Snap and selected Lift; discrete 3D contact approximation",
                        "native": "Approximate rigid-body PBD: hinge + 3D OBB contact + floor + unilateral total string length; T3D NOT an actuator",
                        "legacy": "Legacy target-guided snap/lift Projective Dynamics",
                        "abd": "Autodesk ABD affine bodies + IPC/CCD + unilateral total-path-length string",
                    }[physics_backend],
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
                    "legacy_sat_collision_projection_used": result.metrics.get("legacy_sat_collision_projection_used"),
                }
            )
            if physics_backend == "abd":
                st.subheader("ABD OneString solver log")
                st.write(
                    {
                        "result_json": getattr(result, "result_json_path", None),
                        "gltf": getattr(result, "gltf_path", None),
                        "npz": getattr(result, "npz_path", None),
                        "frame_count": len(frames),
                        "final_active_contacts": result.metrics.get("final_collision_count"),
                        "initial_layout": result.metrics.get("abd_initial_layout"),
                        "rest_collision_geometry": result.metrics.get("abd_rest_collision_geometry"),
                        "collision_skin": result.metrics.get("abd_collision_skin"),
                        "legacy_fallback_used": bool(result.metrics.get("legacy_sat_collision_projection_used", False)),
                    }
                )
                if getattr(result, "frame_logs", None):
                    st.dataframe(result.frame_logs, width="stretch")

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
                    placeholder.plotly_chart(fig, use_container_width=True, key=f"assembly_reliable_play_{frame_id}_{time.time_ns()}")
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
                placeholder.plotly_chart(fig, use_container_width=True, key="assembly_reliable_current")

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
            st.plotly_chart(fig, use_container_width=True, key="assembly_scrubber_plot")

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
            "selected_physics_backend": physics_backend,
            "simulation": "Autodesk ABD affine-body dynamics with IPC/CCD contact and unilateral guided string." if physics_backend == "abd" else "Projective Dynamics with rigid, collision, snap, and lift constraints.",
            "actuation": "unilateral guided-string total length command" if physics_backend == "abd" else "paired side-face midpoint snap plus selected lift targets",
            "preview_limit": preview_tiles,
            "tiles": total_tiles,
            "string_route_nodes": len(state.string_path.gap_ids),
            "lift_points": len(state.lift_points),
        }
    )

elif view_stage == "Final Deployed":
    displayed_result_backend = str(getattr(state.simulation_result, "metrics", {}).get("physics_backend", "")) if state.simulation_result is not None else ""
    if state.simulation_result is None or displayed_result_backend != physics_backend:
        if state.simulation_result is not None:
            st.warning(f"Loaded result uses '{displayed_result_backend}', but '{physics_backend}' is selected. Run the selected simulation before viewing its result.")
        st.plotly_chart(figure_tile_assembly(state.tiles_2d_dual_hinge, title="T2D start state"), use_container_width=True, key="actuation_start")
    else:
        st.caption(f"Result source: {displayed_result_backend}")
        frame_index = st.slider("frame", 0, len(state.simulation_result.frames) - 1, len(state.simulation_result.frames) - 1)
        frame_assembly = TileAssembly(
            vertices=state.simulation_result.frames[frame_index],
            top_faces=state.tiles_2d_dual_hinge.top_faces,
            bottom_faces=state.tiles_2d_dual_hinge.bottom_faces,
            side_faces=state.tiles_2d_dual_hinge.side_faces,
            stage="snap/lift deployment frame",
        )
        st.plotly_chart(figure_tile_assembly(frame_assembly, title="snap/lift deployment frame"), use_container_width=True, key="actuation_frame")
        if st.button("Generate animation"):
            st.plotly_chart(
                tile_assembly_animation(state.tiles_2d_dual_hinge, state.simulation_result.frames, step=max(1, len(state.simulation_result.frames) // 24)),
                use_container_width=True,
                key="actuation_animation",
            )
elif view_stage == "Comparison":
    st.plotly_chart(figure_onestring_comparison(state), use_container_width=True, key="onestring_comparison")
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
    if state.simulation_result is not None and getattr(state.simulation_result, "frame_logs", None):
        st.subheader("ABD per-frame solver log")
        st.dataframe(state.simulation_result.frame_logs, width="stretch")
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
                "gap size を下げる",
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
            st.plotly_chart(px.bar(long, x="seconds", y="stage", color="component", orientation="h", title="Stage time breakdown"), use_container_width=True)
            st.plotly_chart(px.bar(df, x="gpu_time_ratio_%", y="stage", orientation="h", title="GPU kernel time ratio (%)"), use_container_width=True)
            st.plotly_chart(px.bar(df.sort_values("elapsed_time", ascending=True), x="elapsed_time", y="stage", orientation="h", title="Bottleneck ranking by elapsed time"), use_container_width=True)
            st.plotly_chart(px.bar(df, x="gpu_memory_MB", y="stage", orientation="h", title="Peak GPU memory by stage"), use_container_width=True)
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
