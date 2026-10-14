#!/usr/bin/env python3
"""Interactive, REAL post-cut official-BFF experiment (not a full T2D pipeline).

Run on Mac: streamlit run app_postcut_rebff.py
Original OneString design/split algorithms are left untouched.
"""
from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from onestring_physics.input_shape import load_target_shape
from onestring_physics.onestring_pipeline import (
    PipelineParameters, build_paper_reference_initialization,
    create_quad_grid, _reference_flatten_to_domain,
    _build_reference_m2d, _lift_m2d_to_m3d,
)
from onestring_physics.legacy_pipeline_base import SurfaceParameterization
from onestring_physics.paper_fidelity_mode import configure_paper_fidelity
from onestring_physics.reference_bff import (
    run_official_bff, find_official_bff_executable,
    find_rectangular_bff_executable, ReferenceBFFUnavailableError,
)
from scripts.test_lilium_postcut_bff_hypothesis import field
from postcut_rebff_iterative import iterative_postcut_rebff


def _find_bff_in_other_mac_worktrees(*, rectangle: bool) -> Path | None:
    """Reuse separately built official BFF executables from another worktree.

    git worktree includes tracked sources but not ignored CMake build binaries.
    Avoid changing the existing research worktree or recompiling needlessly.
    """
    relative = (
        Path("third_party/bff_rectangle_build/bff-rect-command-line")
        if rectangle else
        Path("third_party/boundary-first-flattening/build/bff-command-line")
    )
    documents = Path.home() / "Documents"
    candidates = [
        ROOT / relative,
        documents / "OneString-large-steps-mesh-conditioning" /
        "onestring-physics-simulator" / relative,
    ]
    if documents.is_dir():
        # Look only one folder below Documents, not every file in home.
        candidates.extend(
            p / relative for p in documents.glob("*/onestring-physics-simulator")
            if p.is_dir()
        )
        candidates.extend(
            p / relative for p in documents.glob("onestring-physics-simulator")
            if p.is_dir()
        )
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
    return None


def _locate_bff_cli(explicit: str, *, rectangle: bool) -> Path:
    finder = (find_rectangular_bff_executable if rectangle
              else find_official_bff_executable)
    if explicit.strip():
        # Respect a manually entered path, including a misspelled one.
        return finder(explicit.strip())
    try:
        return finder()
    except ReferenceBFFUnavailableError as original_error:
        candidate = _find_bff_in_other_mac_worktrees(rectangle=rectangle)
        if candidate is not None:
            return finder(str(candidate))
        raise ReferenceBFFUnavailableError(
            str(original_error)
            + "\nChecked sibling OneString worktrees under ~/Documents for an "
              "existing official BFF executable; none was available. "
              "Build with: bash scripts/repair_postcut_bff_mac.sh"
        ) from original_error


def _jacobian(xyz: np.ndarray, uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    points3 = np.asarray(xyz, float)[faces]
    points2 = np.asarray(uv, float)[faces]
    a3, b3 = points3[:, 1] - points3[:, 0], points3[:, 2] - points3[:, 0]
    a2, b2 = points2[:, 1] - points2[:, 0], points2[:, 2] - points2[:, 0]
    j = np.linalg.norm(np.cross(a3, b3), axis=1) / np.maximum(
        np.abs(a2[:, 0] * b2[:, 1] - a2[:, 1] * b2[:, 0]), 1e-300
    )
    return np.sqrt(j)


def _figure(xyz: np.ndarray, uv: np.ndarray, faces: np.ndarray, *,
            surface: bool, title: str) -> go.Figure:
    coords = np.asarray(xyz if surface else np.column_stack(
        (uv, np.zeros(len(uv)))), float)
    lam = _jacobian(xyz, uv, faces)
    fig = go.Figure(go.Mesh3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=np.log(np.maximum(lam, 1e-12)), intensitymode="cell",
        colorscale="Viridis", opacity=1, flatshading=True, showscale=True,
        colorbar=dict(title="log(CSF)", len=0.7),
        hovertemplate="log(CSF): %{intensity:.3f}<extra></extra>",
    ))
    camera = dict(eye=dict(x=0, y=0, z=2.4)) if not surface else None
    fig.update_layout(
        title=title, margin=dict(l=0, r=0, t=48, b=0), height=440,
        scene=dict(aspectmode="data", camera=camera,
                   xaxis_title="u" if not surface else "X",
                   yaxis_title="v" if not surface else "Y",
                   zaxis_title="" if not surface else "Z"),
    )
    return fig



def _rebff_piece_m2d_m3d(patch: dict, params: PipelineParameters,
                         grid_resolution: int, spacing: float) -> dict:
    """Actually retile one newly BFF-mapped source patch, then lift M2D to M3D.

    The two results are separate UV islands and are not wired into K3D/T2D.
    """
    from onestring_physics.reference_bff import validate_reference_mesh
    from copy import copy
    xyz, uv, faces = patch["xyz"], patch["new_uv"], patch["faces"]
    topo = validate_reference_mesh(xyz, faces)
    loop = [int(v) for v in topo["boundary_loops"][0]]
    chart = SurfaceParameterization(
        method="paper_reference_bff",
        surface_vertices_3d=xyz,
        surface_faces=faces,
        uv_vertices_2d=uv,
        uv_faces=faces,
        omega_boundary=uv[loop + [loop[0]]],
        metrics={"boundary_loop": loop,
                 "per_triangle_lambda": _jacobian(xyz, uv, faces).tolist()},
    )
    piece_params = copy(params)
    piece_params.reference_grid_spacing = float(spacing)
    piece_params.reference_grid_rotation_degrees = 0.0
    piece_params.reference_grid_origin_u = 0.0
    piece_params.reference_grid_origin_v = 0.0
    piece_grid = create_quad_grid(
        int(grid_resolution), int(grid_resolution),
        piece_params.tile_size, piece_params.gap_size,
    )
    domain = _reference_flatten_to_domain(chart, piece_grid, piece_params)
    m2d = _build_reference_m2d(piece_grid, domain, piece_params)
    m3d, _ = _lift_m2d_to_m3d(None, m2d, chart, piece_params)
    return {"uv": np.asarray(m2d.vertices, float)[:, :2],
            "xyz": np.asarray(m3d.vertices, float),
            "quads": np.asarray(m2d.faces, int),
            "roundtrip_max": float(m3d.metrics.get("m3d_round_trip_error_max", 0.0))}


def _quad_preview_obj(xyz: np.ndarray, uv: np.ndarray, quads: np.ndarray) -> bytes:
    lines = ["# Post-cut BFF quad preview with 3D vertices and new UV coordinates"]
    lines.extend(f"v {v[0]:.17g} {v[1]:.17g} {v[2]:.17g}" for v in xyz)
    lines.extend(f"vt {v[0]:.17g} {v[1]:.17g}" for v in uv)
    for face in quads:
        ids = [int(i)+1 for i in face]
        lines.append("f " + " ".join(f"{i}/{i}" for i in ids))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _obj_bytes(xyz: np.ndarray, uv: np.ndarray, faces: np.ndarray) -> bytes:
    out = ["# Post-cut BFF experiment. v = source 3D positions, vt = new BFF UV."]
    out.extend(f"v {p[0]:.17g} {p[1]:.17g} {p[2]:.17g}" for p in xyz)
    out.extend(f"vt {q[0]:.17g} {q[1]:.17g}" for q in uv)
    for a, b, c in np.asarray(faces, int):
        a, b, c = int(a) + 1, int(b) + 1, int(c) + 1
        out.append(f"f {a}/{a} {b}/{b} {c}/{c}")
    return ("\n".join(out) + "\n").encode("utf-8")


def _download_zip(result: dict) -> bytes:
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("comparison.json", json.dumps(result["report"], indent=2,
                                                        ensure_ascii=False))
        for patch in result["patches"]:
            name = patch["side"]
            archive.writestr(
                f"{name}_postcut_rebff.obj",
                _obj_bytes(patch["xyz"], patch["new_uv"], patch["faces"]),
            )
            archive.writestr(
                f"{name}_fixed_original_uv.obj",
                _obj_bytes(patch["xyz"], patch["original_uv"], patch["faces"]),
            )
            if "quad_preview" in patch:
                q = patch["quad_preview"]
                archive.writestr(
                    f"{name}_new_M3D_tiles.obj",
                    _quad_preview_obj(q["xyz"], q["uv"], q["quads"]),
                )
    return data.getvalue()


def _execute(mesh_path: Path, grid: int, standard_exe: str, rect_exe: str,
             phase: float, cut_mode: str, manual_axis: str,
             manual_fraction: float, post_policy: str,
             max_splits: int = 8, progress=None) -> dict:
    standard_path = str(_locate_bff_cli(standard_exe, rectangle=False))
    rectangle_path = str(_locate_bff_cli(rect_exe, rectangle=True))

    params = PipelineParameters(
        nx=grid, ny=grid, omega_parameterization_mode="paper_reference_bff",
        bff_boundary_policy="target_rectangle", paper_fidelity_mode=True,
        max_csf_splits=1,
        bff_executable=standard_path, bff_rectangle_executable=rectangle_path,
    )
    params, _ = configure_paper_fidelity(params)
    params.bff_executable = standard_path
    params.bff_rectangle_executable = rectangle_path
    params.bff_rectangle_corner_phase_fraction = float(phase)
    params.max_csf_splits = 1
    initial = build_paper_reference_initialization(load_target_shape(mesh_path), params)
    param = initial.surface_parameterization
    xyz = np.asarray(param.surface_vertices_3d, float)
    uv = np.asarray(param.uv_vertices_2d, float)
    faces = np.asarray(param.surface_faces, int)
    initial_report = field(xyz, uv, faces)

    cut_records = initial.mesh_2d_initial.metrics.get("csf_split_step_analysis") or []
    if cut_mode == "曲率優先の最初のSplit":
        if not cut_records:
            raise RuntimeError(
                "A curvature-guided Split was not produced by the current M2D. "
                "Use manual cut mode; no implicit center-cut fallback is performed."
            )
        first = cut_records[0]
        axis = str(first["axis"])
        coordinate = float(first["snapped_value"])
        source_vertex = first.get("curvature_vertex_id")
        curvature = first.get("gaussian_curvature")
    else:
        axis = manual_axis
        dim = {"row": 1, "col": 0}[axis]
        coordinate = float(np.min(uv[:, dim]) +
                           manual_fraction * np.ptp(uv[:, dim]))
        source_vertex, curvature = None, None

    def _rebff(x, f):
        bff = run_official_bff(
            x, f, executable=standard_path,
            rectangle_executable=rectangle_path, boundary_policy=post_policy,
        )
        return np.asarray(bff.uv_vertices, float), bff.metrics

    trial = iterative_postcut_rebff(
        xyz, uv, faces, grid=int(grid), rebff=_rebff, max_splits=int(max_splits),
        csf_limit=2.0, progress=progress,
        first_cut=dict(
            axis=axis, coordinate=coordinate,
            origin=cut_mode, source_vertex=source_vertex,
            curvature=curvature,
        ),
    )
    patches = trial["patches"]
    # Regenerate real square M2D panels and 3D inverse-map lifts on the NEW UV.
    # Keep one shared spacing across the two patches for meaningful comparison.
    maximum_span = max(float(np.max(np.ptp(p["new_uv"], axis=0))) for p in patches)
    common_spacing = max(maximum_span / max(int(grid), 1), 1e-8)
    for patch in patches:
        try:
            patch["quad_preview"] = _rebff_piece_m2d_m3d(
                patch, params, int(grid), common_spacing
            )
        except Exception as exc:
            patch["quad_preview_error"] = f"{type(exc).__name__}: {exc}"

    worst = max(p["post"]["linear_span"] for p in patches)
    report = {
        "experiment": "Iterative high-curvature cuts and independent official BFF until CSF <= 2",
        "not_paper_author_verified": True,
        "not_applied_to_K3D_or_T2D": True,
        "input_mesh": str(mesh_path), "grid": int(grid),
        "official_bff_executable": standard_path,
        "rectangular_bff_executable": rectangle_path,
        "initial_bff": "target_rectangle", "postcut_bff": post_policy,
        "rectangle_corner_phase": float(phase),
        "split_count": trial["split_count"],
        "split_history": trial["split_history"],
        "rejected_candidates": trial["rejected_candidates"],
        "stop_reason": trial["stop_reason"],
        "csf_limit": trial["csf_limit"],
        "max_splits_budget": trial["max_splits"],
        "split": {
            "axis": axis, "uv_coordinate": coordinate, "origin": cut_mode,
            "source_curvature_vertex": source_vertex, "gaussian_curvature": curvature,
        },
        "initial_source_triangle_csf": initial_report,
        "initial_m2d": {
            "quads": int(len(initial.mesh_2d_initial.faces)),
            "initial_linear_csf": initial.mesh_2d_initial.metrics.get("max_csf_before_split"),
            "after_single_cut_linear_csf": initial.mesh_2d_initial.metrics.get("max_csf_after_split"),
        },
        "patches": [
            {
                "side": p["side"], "triangles": int(len(p["faces"])),
                "fixed_original_chart": p["pre"],
                "postcut_rebff": p["post"],
            } for p in patches
        ],
        "new_piece_m2d_quad_counts": [
            int(len(p["quad_preview"]["quads"])) if "quad_preview" in p else None
            for p in patches
        ],
        "new_piece_m2d_grid_spacing": float(common_spacing),
        "new_piece_m2d_errors": [p.get("quad_preview_error") for p in patches],
        "max_postcut_patch_linear_span": float(worst),
        "linear_limit_two_satisfied": trial["all_pass"],
        "caution": "Independent per-patch M2D/M3D previews are computed on "
                   "the new BFF UVs. These UV islands are not globally packed "
                   "or integrated into K3D/T2D. Fabrication not certified.",
    }
    return {"report": report, "patches": patches, "initial": {
        "xyz": xyz, "uv": uv, "faces": faces,
    }}


def _mesh_top_stage_figure(vertices, faces, *, title: str):
    """Visualize REAL mesh vertices/faces returned by the original solver."""
    xyz = np.asarray(vertices, float)
    f = np.asarray(faces, int)
    tris = np.concatenate([f[:, [0, 1, 2]], f[:, [0, 2, 3]]])
    fig = go.Figure(go.Mesh3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
        color="#3995aa", opacity=0.9, flatshading=True,
    ))
    fig.update_layout(title=title, scene=dict(aspectmode="data"),
                      height=480, margin=dict(l=0, r=0, t=40, b=0))
    return fig


def _real_tile_stage_figure(tiles: np.ndarray, title: str, flat: bool = False):
    """Plot per-tile outlines without fabricating shared-vertex connections."""
    verts = np.asarray(tiles, float)
    if verts.ndim != 3 or verts.shape[1] < 4 or verts.shape[2] < 3:
        raise ValueError(f"Expected actual (tiles,>=4,3) output, got {verts.shape}")
    fig = go.Figure()
    for i, polygon in enumerate(verts[:, :4, :]):
        closed = np.vstack((polygon, polygon[0]))
        fig.add_trace(go.Scatter3d(
            x=closed[:, 0], y=closed[:, 1],
            z=np.zeros(5) if flat else closed[:, 2],
            mode="lines", line=dict(width=3),
            showlegend=False, hovertemplate=f"Tile {i}<extra></extra>",
        ))
    fig.update_layout(title=title, showlegend=False, height=560,
                      scene=dict(aspectmode="data",
                                 camera=dict(eye=dict(x=0, y=0, z=2.6)) if flat else None),
                      margin=dict(l=0, r=0, t=40, b=0))
    return fig


def _show_existing_pipeline_state(state, audit):
    st.success("既存のOneStringソルバーでT2D Dual Hingeまで計算が完了しました。")
    st.json({
        "M2D/M3D shared tile count": len(state.mesh_2d_initial.faces),
        "K3D tile count": len(state.mesh_3d_optimized.faces),
        "T3D tile count": len(state.tiles_3d.vertices),
        "K2D tile count": len(state.mesh_2d_optimized.faces),
        "T2D Dual Hinge tile count": len(state.tiles_2d_dual_hinge.vertices),
        "Dual Hinge count": len(state.hinge_graph.hinges),
        **audit,
    })
    st.subheader("既存の最適化器が出力した全段階の形状")
    a, b = st.columns(2)
    with a:
        st.plotly_chart(_mesh_top_stage_figure(
            state.mesh_3d_optimized.vertices, state.mesh_3d_optimized.faces,
            title="K3D：平面性・四角形性・曲面追従最適化後",
        ), use_container_width=True)
    with b:
        st.plotly_chart(_real_tile_stage_figure(
            state.tiles_3d.vertices, title="T3D：実タイル（上面輪郭）",
        ), use_container_width=True)
    a, b = st.columns(2)
    with a:
        st.plotly_chart(_real_tile_stage_figure(
            state.tiles_2d_top_hinge.vertices,
            title="T2D Top Hinge：実計算結果", flat=True,
        ), use_container_width=True)
    with b:
        st.plotly_chart(_real_tile_stage_figure(
            state.tiles_2d_dual_hinge.vertices,
            title="T2D Dual Hinge：実計算結果", flat=True,
        ), use_container_width=True)
    with st.expander("Dual Hingeソルバーの評価値"):
        st.json(state.tiles_2d_dual_hinge.metrics)


def main() -> None:
    st.set_page_config(page_title="OneString Split → re-BFF", layout="wide")
    st.title("OneString: Split後のBFF再計算（Mac実験モード）")
    st.caption("矩形BFF → 初回Split → CSF>2の領域を追加Split → 各領域で公式BFFを反復再計算")
    st.warning(
        "研究用の独立した実験画面です。新しいUVとOBJを実際に生成しますが、"
        "各領域で再BFF後のM2D/M3Dも生成しますがK3D/T2Dは未接続です。"
        "通常のシミュレーターの計算経路は変更しません。"
    )
    with st.sidebar:
        st.header("実験条件")
        mesh = st.text_input("OBJのパス", value="~/Downloads/lilium.obj")
        grid = st.number_input("グリッド（nx=ny）", min_value=2, max_value=60, value=14)
        rect = st.text_input("矩形BFF実行ファイル（空欄なら自動検出）", value="")
        standard = st.text_input("通常BFF実行ファイル（空欄なら自動検出）", value="")
        phase = st.select_slider(
            "矩形BFFの四隅の位相",
            options=[0.0, 0.0625, 0.125, 0.1875], value=0.0,
        )
        cut_mode = st.radio(
            "切断位置", ["曲率優先の最初のSplit", "手動指定"], index=0,
        )
        axis = "row"
        fraction = 0.5
        if cut_mode == "手動指定":
            axis = st.selectbox("切断方向", ["row", "col"])
            fraction = st.slider("UV境界幅に対する切断位置", 0.05, 0.95, 0.5, 0.01)
        post_policy = st.selectbox(
            "Split後のBFF境界条件",
            ["automatic_reference", "target_rectangle"],
            index=0,
            help="自由境界BFF（推奨の検証候補）か、再度矩形BFFにするかを比較",
        )
        max_splits = int(st.number_input(
            "最大Split数（初回を含む）", min_value=1, max_value=64, value=8, step=1,
            help="CSF>2の領域だけを切断。切れない・予算到達なら停止理由を表示。",
        ))
        st.caption("CSF判定は全領域の sqrt(max(J)/min(J)) ≤ 2（J=3D/2D面積倍率）")
        start = st.button("Split → BFF再計算を収束まで繰り返す", type="primary")

    if start:
        st.session_state.pop("postcut_rebff_full_pipeline_state", None)
        st.session_state.pop("postcut_rebff_full_pipeline_audit", None)
        path = Path(mesh).expanduser().resolve()
        if not path.is_file():
            st.error(f"OBJが存在しません: {path}")
        else:
            try:
                live = st.empty()
                with st.spinner("各Split後にBFFを再計算し、CSFを再判定中…"):
                    st.session_state["postcut_rebff_data"] = _execute(
                        path, int(grid), standard, rect, float(phase),
                        cut_mode, axis, float(fraction), post_policy,
                        max_splits=max_splits,
                        progress=lambda message: live.info(message),
                    )
                live.empty()
            except ReferenceBFFUnavailableError as exc:
                st.error("公式BFF CLIが見つからないか、起動に失敗しました。"
                         "OptCutsではなく、実験用の公式BFF実行ファイルが必要です。")
                st.code("bash scripts/repair_postcut_bff_mac.sh", language="bash")
                st.caption("ターミナルで上のコマンドを実行した後、実験ページを再実行してください。"
                           "既存のOneString・OptCutsのコードは変更しません。")
                st.exception(exc)
            except Exception as exc:
                st.exception(exc)
    data = st.session_state.get("postcut_rebff_data")
    if data is None:
        st.info(
            "公式BFFが未セットアップの場合はターミナルで "
            "bash scripts/repair_postcut_bff_mac.sh を実行してください。"
            "その後、実行ボタンを押します。"
        )
        return

    report = data["report"]
    passed = bool(report["linear_limit_two_satisfied"])
    summary = (
        f"Split={report['split_count']}本（上限{report['max_splits_budget']}本）、"
        f"領域={len(data['patches'])}、"
        f"最大CSF比={report['max_postcut_patch_linear_span']:.4f}、"
        f"停止理由={report['stop_reason']}"
    )
    if passed:
        st.success("全領域のCSFが2以下になりました。" + summary)
    else:
        st.error("全領域のCSFを2以下にできませんでした。" + summary)
    left, middle, right = st.columns(3)
    left.metric("元BFF（元3Dメッシュ）CSF比",
                f"{report['initial_source_triangle_csf']['linear_span']:.4f}")
    middle.metric("最終の最大CSF比",
                  f"{report['max_postcut_patch_linear_span']:.4f}")
    right.metric("Split本数", str(report["split_count"]))
    st.subheader("Split履歴")
    for item in report["split_history"]:
        st.write(
            f"**#{item['step']}**　{item['parent']} → {item['axis']}="
            f"{item['uv_coordinate']:.6f}　|　"
            f"全領域最大CSF={item['all_patches_max_linear_span']:.4f}　|　"
            f"CSF>2の領域={item['violating_patch_count']}"
        )
    if report["rejected_candidates"]:
        with st.expander("採用できなかった切断候補"):
            st.json(report["rejected_candidates"])

    st.divider()
    st.subheader("従来OneStringに接続してT2D Dual Hingeまで実計算")
    st.caption("従来のOptCuts対応OneStringパイプラインのK3D・T3D・K2D・"
               "T2D Top Hinge・Dual Hinge最適化器をそのまま使用します。"
               "再BFF後は全領域のタイルを対応付きで統合し、元M2Dに近い"
               "タイル数になる共通格子間隔を探索します。"
               "K3D以降の制約違反や最適化失敗は隠さず表示します。")
    if not report["linear_limit_two_satisfied"]:
        st.warning("まだCSFが2以下になっていないため、下流段階は実行できません。")
    elif st.button("従来の全段階を計算してT2D Dual Hingeを表示", type="primary"):
        try:
            from postcut_rebff_full_pipeline import run_existing_pipeline_through_dual_hinge
            live_progress = st.empty()
            def on_full_progress(stage, value, description=""):
                live_progress.info(
                    f"{stage}（{float(value)*100:.1f}%）: {description}"
                )
            with st.spinner("既存のOneStringでK3D→T3D→K2D→T2D Dual Hingeを計算中…"):
                state, audit = run_existing_pipeline_through_dual_hinge(
                    Path(mesh).expanduser().resolve(), data,
                    grid=int(grid), progress=on_full_progress,
                )
                st.session_state["postcut_rebff_full_pipeline_state"] = state
                st.session_state["postcut_rebff_full_pipeline_audit"] = audit
            live_progress.empty()
        except Exception as exc:
            st.error("元のOneStringソルバーが停止しました。"
                     "T2Dを偽の結果で代替していません。")
            st.exception(exc)
    final_state = st.session_state.get("postcut_rebff_full_pipeline_state")
    final_audit = st.session_state.get("postcut_rebff_full_pipeline_audit")
    if final_state is not None and final_audit is not None:
        _show_existing_pipeline_state(final_state, final_audit)

    st.subheader("2D展開の比較")
    st.caption(
        "各パッチの位置・回転・拡大率は独立です。左＝切断前のUVを保持、"
        "右＝実際に再計算されたBFFのUV。色は各三角形のlog(CSF)。"
    )
    for patch in data["patches"]:
        st.markdown(f"**{patch['side']} 領域（{len(patch['faces'])} triangles）**")
        m1, m2 = st.columns(2)
        m1.metric("既存UVのCSF比", f"{patch['pre']['linear_span']:.4f}")
        m2.metric("再BFFのCSF比", f"{patch['post']['linear_span']:.4f}")
        left, right = st.columns(2)
        with left:
            st.plotly_chart(_figure(
                patch["xyz"], patch["original_uv"], patch["faces"],
                surface=False, title="切断前のUVを維持",
            ), use_container_width=True)
        with right:
            st.plotly_chart(_figure(
                patch["xyz"], patch["new_uv"], patch["faces"],
                surface=False, title="Split後にBFFを再計算",
            ), use_container_width=True)
    st.subheader("切断後の3D形状")
    for patch in data["patches"]:
        with st.expander(f"{patch['side']}（{len(patch['faces'])} triangles）"):
            st.plotly_chart(_figure(
                patch["xyz"], patch["new_uv"], patch["faces"],
                surface=True, title=f"{patch['side']}：最終3D形状",
            ), use_container_width=True)

    st.subheader("中間検証のみ：再BFF後の各領域のM2D / M3D")
    st.warning("この領域別プレビューは最終設計ではありません。"
               "最終タイル配置とT2D Dual Hingeは上の「従来の全段階を計算」から確認してください。")
    st.caption("各領域に共通間隔の正方形タイルを作り、新しいBFFの逆写像で"
               "3Dへリフトしました。K3D/T2Dにはまだ接続していません。")
    for patch in data["patches"]:
        if "quad_preview" not in patch:
            st.error(f"{patch['side']}: M2D→M3D生成失敗: {patch.get('quad_preview_error')}")
            continue
        q = patch["quad_preview"]
        st.markdown(f"**{patch['side']}：{len(q['quads'])}枚の新M2D/M3Dタイル**")
        triangles = np.concatenate((
            q["quads"][:, [0, 1, 2]], q["quads"][:, [0, 2, 3]],
        ), axis=0)
        left, right = st.columns(2)
        with left:
            st.plotly_chart(_figure(
                q["xyz"], q["uv"], triangles,
                surface=False, title="再BFF後に生成したM2D",
            ), use_container_width=True)
        with right:
            st.plotly_chart(_figure(
                q["xyz"], q["uv"], triangles,
                surface=True, title="再BFFからリフトしたM3D",
            ), use_container_width=True)

    st.download_button(
        "最終全領域のBFF UV・M3D四角形・Split履歴をZIP保存",
        data=_download_zip(data), file_name="onestring_postcut_rebff.zip",
        mime="application/zip",
    )
    with st.expander("詳細CSFと実験条件"):
        st.json(report)


if __name__ == "__main__":
    main()
