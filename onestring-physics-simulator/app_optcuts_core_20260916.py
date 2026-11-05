"""OptCuts launcher.

``optcuts``
    Requirement-preserving path using the ordinary official OptCuts backend.

``optcuts_test``
    Experimental path: official OptCuts seam topology -> initial Omega -> the
    existing M2D quad-cell footprint -> grid-cell outer boundary -> Omega
    reparameterization while the OptCuts seam remains a hard boundary.

``optcuts_test2``
    Same numerical path as optcuts_test, but with separate faster K3D/K2D
    runtime budgets for profiling and iteration.  optcuts_test itself is kept
    unchanged for direct comparison.

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
from onestring_physics.optcuts_test_seam_metadata_bridge import (  # noqa: E402
    install_optcuts_test_seam_metadata_bridge,
)
from onestring_physics.optcuts_test2_acceleration_patch import (  # noqa: E402
    install_optcuts_test2_acceleration_patch,
)
from onestring_physics.optcuts_grid_native_pipeline_patch import install_native_grid_optcuts_pipeline_patch  # noqa: E402
from onestring_physics.optcuts_grid_native_lift_patch import install_native_grid_optcuts_lift_patch  # noqa: E402
from onestring_physics.optcuts_grid_constrained_m2d_patch import install_optcuts_grid_constrained_m2d_patch  # noqa: E402
from onestring_physics.optcuts_planar_quad_20261026 import install_planar_quad_20261026  # noqa: E402
from onestring_physics.optcuts_planar_quad_downstream_20261026 import install_unstructured_quad_downstream_20261026  # noqa: E402
from onestring_physics.optcuts_zhang2021_quad_20261027 import install_zhang2021_quad_20261027  # noqa: E402
from onestring_physics.optcuts_grid_consistency_patch import install_optcuts_grid_consistency_patch  # noqa: E402
from onestring_physics.optcuts_test_k3d_augmented_lagrangian_patch import (  # noqa: E402
    install_optcuts_test_k3d_augmented_lagrangian_patch,
)
from onestring_physics.optcuts_k3d_validity_patch import install_optcuts_k3d_validity_patch  # noqa: E402
from onestring_physics.optcuts_k3d_preflight_patch import install_optcuts_k3d_preflight_patch  # noqa: E402
from onestring_physics.optcuts_visualization_compat_patch import install_optcuts_visualization_compat_patch  # noqa: E402
from onestring_physics.optcuts_source_seam_visualization_patch import install_optcuts_source_seam_visualization_patch  # noqa: E402
from onestring_physics.optcuts_seam_exact_closure import install_optcuts_exact_seam_closure  # noqa: E402
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

        experimental = (
            "optcuts_area_grid_phase_search",
            "optcuts_area_grid_tail_sd",
            "optcuts_area_grid_adaptive_remesh",
            "optcuts_area_grid_combined",
            "optcuts_grid_seam_native",
            "optcuts_planar_quad_20261026",
            "optcuts_zhang2021_quad_20261027",
            "optcuts_structured_grid_20261028",
            "optcuts_structured_grid_valley_fix_align_20261029",
            "optcuts_structured_grid_strong_lines_20261029",
            "optcuts_structured_grid_slide_shape_keep_20261030",
            "optcuts_structured_grid_align_even_keep_20261030",
            "optcuts_structured_grid_align_even_neighbor_keep_20261030",
            "optcuts_structured_grid_soft_fit_20261104",
            "optcuts_structured_grid_fit_20261104",
        )
        for mode in ("optcuts", "optcuts_test", "optcuts_test2",
                     "optcuts_grid_aware", "optcuts_area_grid_boundary",
                     *experimental, "optcuts_grid"):
            if mode not in option_list:
                option_list.append(mode)
        original_format=kwargs.get("format_func")
        def label_for_mode(mode):
            if mode=="optcuts_planar_quad_20261026":
                return "2026-10-26 | OptCuts → boundary-conforming Planar Quad M2D (Gmsh prototype)"
            if mode=="optcuts_zhang2021_quad_20261027":
                return ("2026-10-27 | OptCuts → Zhang et al. 2021 Quad M2D "
                        "(authors' reference code) + OneString seam/3D adapter")
            if mode=="optcuts_structured_grid_valley_fix_align_20261029":
                return "2026-10-29 | 10-28 構造化グリッド＋谷線に格子線を通して固定＋格子の向きを曲がる方向にそろえる"
            if mode=="optcuts_structured_grid_slide_shape_keep_20261030":
                return "2026-10-30 | 10-28 構造化グリッド＋境界の点をすべらせる＋形を優先（大きさは写像に合わせる）＋T3D で上面パネルの形を保つ"
            if mode=="optcuts_structured_grid_align_even_keep_20261030":
                return "2026-10-30 | 10-28 構造化グリッド＋格子の向きを曲がる方向にそろえる＋形を優先＋小さいパネルを防ぎ大きさをそろえる＋T3D で上面パネルの形を保つ"
            if mode=="optcuts_structured_grid_align_even_neighbor_keep_20261030":
                return "2026-10-30 | 上の向きそろえの版＋隣り合うパネルの大きさの比をそろえる"
            if mode=="optcuts_structured_grid_fit_20261104":
                return "2026-11-04 | 境界に添わせる経路をアーティファクトと同じ手順で（OptCuts の制約・seam の始点の探索・対称リメッシュを選べる。各最適化は収束まで）"
            if mode=="optcuts_structured_grid_soft_fit_20261104":
                return "2026-11-04 | OptCuts（σ1/σ2 のソフト制約）→ 構造格子を境界に添わせる（seam を 4 枚で閉じる＋格子の角を立体の縁の曲がりで決める）＋格子の向きそろえ・形優先・T3D で上面パネルの形を保つ"
            if mode=="optcuts_structured_grid_strong_lines_20261029":
                return "2026-10-29 | 10-28 構造化グリッド＋強く曲がる谷線・山線を Ω に描いて格子を沿わせる"
            if mode=="optcuts_structured_grid_20261028":
                return ("2026-10-28 | OptCuts → piecewise smooth Ω (line+spline) → structured quad grid "
                        "(Brackbill–Saltzman, interior valence 4) + 09-18 K3D…T2D")
            if mode=="optcuts_grid_aware":
                return "2026-10-23 | Grid-aware OptCuts Ω + paper-aligned K3D/K2D (UV refinement)"
            if mode=="optcuts_area_grid_boundary":
                return "2026-10-23 | OptCuts Ω + Area-Threshold Grid Boundary → Reparameterization"
            variants = {
                "optcuts_area_grid_phase_search": "① Rotate/phase search only",
                "optcuts_area_grid_tail_sd": "③ Boundary-tail SD optimization only",
                "optcuts_area_grid_adaptive_remesh": "⑤ Boundary-corner adaptive remeshing only",
                "optcuts_area_grid_combined": "①+③+⑤ Combined",
                "optcuts_grid_seam_native": "Native C++ seam penalty + actual-grid boundary fitting",
            }
            if mode in variants:
                return "EXPERIMENTAL 10/23 OptCuts Ω | " + variants[mode]
            return str(original_format(mode)) if original_format else str(mode)
        kwargs["format_func"]=label_for_mode
        value = original_selectbox(label, option_list, *args, **kwargs)
        # Preserve the chosen frontend mode even though the numerical carrier
        # for this mode (same as the working 2026-09-18 setup) is optcuts_test.
        # Never use a process-global env flag as the sole mode source.
        st.session_state["_onestring_grid_aware_omega_selected"] = (
            value == "optcuts_grid_aware"
        )
        st.session_state["_onestring_area_grid_boundary_omega_selected"] = (
            value == "optcuts_area_grid_boundary" or (
                value in experimental
                and value not in {"optcuts_planar_quad_20261026",
                                  "optcuts_zhang2021_quad_20261027",
                                  "optcuts_structured_grid_20261028",
                                  "optcuts_structured_grid_valley_fix_align_20261029",
                                  "optcuts_structured_grid_strong_lines_20261029",
                                  "optcuts_structured_grid_slide_shape_keep_20261030",
                                  "optcuts_structured_grid_align_even_keep_20261030",
                                  "optcuts_structured_grid_align_even_neighbor_keep_20261030",
                                  "optcuts_structured_grid_soft_fit_20261104", "optcuts_structured_grid_fit_20261104"})
        )
        st.session_state["_onestring_planar_quad_20261026_selected"] = (value == "optcuts_planar_quad_20261026")
        st.session_state["_onestring_zhang2021_quad_20261027_selected"] = (
            value == "optcuts_zhang2021_quad_20261027")
        st.session_state["_onestring_structured_grid_20261028_selected"] = (
            value in {"optcuts_structured_grid_20261028",
                      "optcuts_structured_grid_valley_fix_align_20261029",
                      "optcuts_structured_grid_strong_lines_20261029",
                                  "optcuts_structured_grid_slide_shape_keep_20261030",
                                  "optcuts_structured_grid_align_even_keep_20261030",
                                  "optcuts_structured_grid_align_even_neighbor_keep_20261030",
                                  "optcuts_structured_grid_soft_fit_20261104", "optcuts_structured_grid_fit_20261104"})
        st.session_state["_onestring_structured_grid_feature_preset"] = {
            "optcuts_structured_grid_valley_fix_align_20261029": "valley_fix_align",
            "optcuts_structured_grid_strong_lines_20261029": "strong_lines",
            "optcuts_structured_grid_slide_shape_keep_20261030": "slide_shape_keep_20261030",
            "optcuts_structured_grid_align_even_keep_20261030": "align_shape_even_keep_20261030",
            "optcuts_structured_grid_align_even_neighbor_keep_20261030": "align_shape_even_neighbor_keep_20261030",
            "optcuts_structured_grid_soft_fit_20261104": "soft_fit_20261104",
            "optcuts_structured_grid_fit_20261104": "soft_fit_20261104",
        }.get(value, "off")
        # 2026-11-05: the app maps every OptCuts mode to "optcuts_test" afterwards; remember this one explicitly
        st.session_state["_onestring_fit_artifact_selected"] = (value == "optcuts_structured_grid_fit_20261104")
        st.session_state["_onestring_grid_seam_native_selected"] = (
            value == "optcuts_grid_seam_native")
        st.session_state["_onestring_area_grid_optimizations"] = {
            "phase_search": value in {
                "optcuts_area_grid_phase_search", "optcuts_area_grid_combined"},
            "tail_sd": value in {
                "optcuts_area_grid_tail_sd", "optcuts_area_grid_combined"},
            "adaptive_remesh": value in {
                "optcuts_area_grid_adaptive_remesh", "optcuts_area_grid_combined"},
        }
        if value not in {"optcuts", "optcuts_test", "optcuts_test2",
                         "optcuts_grid_aware", "optcuts_area_grid_boundary",
                         *experimental, "optcuts_grid"}:
            os.environ["ONESTRING_OPTCUTS_TEST_VARIANT"] = "0"
            return value

        if value == "optcuts":
            st.caption("Official OptCuts baseline. The saved stable mode is not modified by optcuts_test.")
        elif value == "optcuts_test":
            st.caption(
                "OptCuts_test: keep the official OptCuts seam/cut topology, build the ordinary M2D "
                "quad-cell footprint from the initial Omega, use that footprint's outer boundary as "
                "the new Omega boundary target, and re-solve UV while the OptCuts seam remains fixed "
                "as a boundary."
            )
        elif value == "optcuts_test2":
            st.caption(
                "OptCuts_test2: same geometry/constraint pipeline as OptCuts_test, but with a shorter "
                "K3D AL budget and an accelerated K2D kinematic solve. OptCuts_test remains unchanged "
                "for comparison."
            )
        elif value == "optcuts_grid_aware":
            st.caption(
                "Grid-aware OptCuts Ω: working OptCuts_test2 / paper-aligned K3D-K2D path "
                "+ experimental grid-aligned UV refinement after official OptCuts. "
                "The original OptCuts topology and K3D/K2D solvers remain unchanged. "
                "Use OptCuts_test2 as the baseline comparison."
            )
        elif value == "optcuts_area_grid_boundary":
            st.caption(
                "NEW EXPERIMENT: official OptCuts Omega -> exact area-overlap "
                "threshold for grid cells (default 50%) -> rectilinear cell-union "
                "boundary -> order-preserving boundary UV fit and harmonic "
                "interior reparameterization. The ORIGINAL lattice is held fixed. "
                "Non-disk outlines, insufficient boundary vertices, UV foldovers "
                "and excessive distortion fail explicitly; no silent fallback."
            )
        elif value == "optcuts_planar_quad_20261026":
            st.caption(
                "2026-10-26 EXPERIMENT: official OptCuts UV triangle chart -> "
                "Gmsh boundary-conforming all-quad mesh. The original UV↔3D "
                "piecewise-linear inverse is checked; 50% masked-grid clipping is "
                "not used. This is NOT the Zhang et al. 2021 implementation. "
                "Invalid quad geometry, outside-UV points and unsupported "
                "CSF Split fail explicitly; irregular-grid downstream compatibility "
                "has not been validated end-to-end."
            )
            st.number_input(
                "Target Planar Quad panel count (approximate)",
                min_value=4, max_value=10000, value=35, step=1,
                key="onestring_planar_quad_target_faces",
                help="Target edge length is sqrt(OptCuts Omega area / requested panels). "
                     "Final number of quads can differ."
            )
            st.checkbox("Low-distortion Ω boundary + interior UV reparameterization",
                        value=True, key="onestring_planar_quad_uv_reparam")
            st.checkbox("M3D-aware quad refinement when panels are distorted", value=True,
                        key="onestring_planar_quad_refine_m3d")
            st.caption("Optional dependencies: python3 -m pip install gmsh shapely")
        elif value == "optcuts_zhang2021_quad_20261027":
            st.caption(
                "2026-10-27: official OptCuts UV chart → Zhang, Chai, Liu, Fu, "
                "'Quad meshing with coarse layouts for planar domains' (CAD 2021) — "
                "the AUTHORS' reference programs Straighten/Segment/QMSim, run exactly "
                "as their WholePipeline.bat (portable build, no algorithmic change). "
                "OneString extension (separate, reported as zhang2021_onestring_*): "
                "seam-aware boundary correspondence + bijective SLIM chart so every "
                "quad corner lifts exactly onto S and the OptCuts cut. No Gmsh, no "
                "cropped grid, no silent fallback. Setup: "
                "python3 scripts/setup_zhang2021_quad_official.py"
            )
            st.number_input(
                "Zhang 2021: density input N_tar (paper default 4000; NOT a panel-count goal)",
                min_value=4, max_value=200000, value=35, step=1,
                key="onestring_zhang2021_target_quads",
                help="Only the paper's density input: its target edge length is l_tar = sqrt(area / N_tar). "
                     "Several densities are tried (factors below) and the final panel count is whatever the "
                     "best feasible candidate (by panel quality, never by count) has.",
            )
            st.number_input(
                "Zhang 2021: or 3D panel edge length → N_tar = area(S)/l² (0 = use N_tar)",
                min_value=0.0, value=0.0, step=0.01, format="%.4f",
                key="onestring_zhang2021_target_edge_length",
            )
            st.text_input("Density factors tried (× N_tar)", value="1,2,4",
                          key="onestring_zhang2021_candidate_density_factors")
            st.number_input(
                "Zhang 2021: Hausdorff bound ε (× bbox diagonal; paper default 0.01)",
                min_value=0.001, max_value=0.5, value=0.05, step=0.005, format="%.3f",
                key="onestring_zhang2021_hausdorff_ratio",
                help="Used when the ε candidate list below is empty.",
            )
            st.text_input("Zhang ε candidates (× bbox diagonal)",
                          value="0.02,0.03,0.05", key="onestring_zhang2021_candidate_hausdorff_ratios")
            st.number_input(
                "Zhang 2021: poly-chord collapse width limit l_collapse (× bbox diagonal; paper default 0.05)",
                min_value=0.001, max_value=0.5, value=0.05, step=0.005, format="%.3f",
                key="onestring_zhang2021_collapse_width_ratio",
            )
            st.checkbox("Zhang 2021: keep boundary feature points (paper default)",
                        value=True, key="onestring_zhang2021_keep_features")
            st.text_input("Zhang 2021 reference binaries directory (optional)",
                          value="", key="onestring_zhang2021_bin_dir",
                          placeholder="third_party/Zhang2021_QuadCoarseLayout_official/build/onestring_port")
            st.markdown("**OneString extension (not part of Zhang 2021)**")
            st.checkbox(
                "Hard OptCuts seams: sample both sides of every cut at the same 3D points (matched / nested), "
                "exact closure in K3D (never soft)",
                value=True, key="onestring_zhang2021_hard_seams")
            st.checkbox("Repair cut-tip / junction valence (≥ 3 panels around an interior seam node in 3D)",
                        value=True, key="onestring_zhang2021_repair_node_valence")
            st.checkbox("3D panel-quality optimisation of the interior M2D vertices (size/shape uniformity)",
                        value=True, key="onestring_zhang2021_panel_optimization")
            st.checkbox("Equalize quad counts on both sides of every cut by poly-chord insertion (optional; "
                        "adds quads, otherwise unequal counts are closed exactly as nested partitions)",
                        value=False, key="onestring_zhang2021_seam_matching")
            st.checkbox(
                "Pre-condition Ω before Zhang (simplify ∂Ω keeping seam-node directions + bijective SLIM)",
                value=True, key="onestring_zhang2021_precondition_enabled",
                help="OptCuts → h (Douglas–Peucker per seam run with Hausdorff ≤ tol, seam nodes and their "
                     "local directions kept, symmetric-Dirichlet SLIM inside) → Zhang 2021 → adapter. "
                     "Candidates (tol × ε × density) are filtered by hard constraints and ranked by panel "
                     "quality (no panel-count term); all candidates are reported.",
            )
            st.text_input("Pre-conditioning tolerances (× bbox diagonal, 0 = none)",
                          value="0,0.02", key="onestring_zhang2021_precondition_tolerances")
            st.number_input("Pre-conditioning: keep seam-node directions over (× bbox diagonal)",
                            min_value=0.0, max_value=0.5, value=0.06, step=0.01, format="%.3f",
                            key="onestring_zhang2021_precondition_node_keep_ratio")
            st.number_input("Pre-conditioning limit: SD mean ratio (design 1.035 / 1.08)",
                            min_value=1.0, max_value=3.0, value=1.08, step=0.01, format="%.3f",
                            key="onestring_zhang2021_precondition_sd_mean_ratio_max")
            st.number_input("Pre-conditioning limit: SD p95 ratio",
                            min_value=1.0, max_value=5.0, value=1.30, step=0.05, format="%.2f",
                            key="onestring_zhang2021_precondition_sd_p95_ratio_max")
            st.text_input("M3D gate (empty = defaults: min_angle_deg=30, max_angle_deg=150, max_edge_ratio=3, "
                          "max_size_ratio=4, max_size_spread_p95_p5=2.5, max_e_size=0.15, max_e_angle=0.15, "
                          "max_area_cv=0.75)", value="", key="onestring_zhang2021_m3d_gate")
            st.text_input("K3D gate (empty = M3D defaults + max_nonplanarity=0.001)", value="",
                          key="onestring_zhang2021_k3d_gate")
            st.text_input("M2D (UV) validity gate (empty = min_angle_deg=3, max_angle_deg=177, max_edge_ratio=25)",
                          value="", key="onestring_zhang2021_m2d_gate")
            st.text_input("Quality score weights (empty = size=1,shape=1,area=0.5,distortion=1,fit=2,worst=0.5,"
                          "m2d=0.1; no count term)", value="", key="onestring_zhang2021_selection_weights")
            st.number_input("Downstream verification attempts (next-ranked candidate after a K3D/T2D failure)",
                            min_value=1, max_value=20, value=3, step=1,
                            key="onestring_zhang2021_verification_attempts")
            st.checkbox("Compare all verified candidates (Phase 4) instead of adopting the first that passes",
                        value=False, key="onestring_zhang2021_compare_verified_candidates")
            st.number_input("T2D limit: hinge connection error / mean K3D edge",
                            min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.3f",
                            key="onestring_zhang2021_t2d_max_hinge_error_ratio")
            st.number_input("T3D limit: panel collisions", min_value=0, max_value=100000, value=0, step=1,
                            key="onestring_zhang2021_t3d_max_panel_collisions")
            st.selectbox(
                "OneString downstream route for the Zhang M2D (K3D/K2D/T2D)",
                ["paper_local_global_0918", "optcuts_test2"],
                index=0, key="onestring_zhang2021_downstream_route",
                format_func=lambda r: {
                    "optcuts_test2": "optcuts_test2: AL K3D with the hard OptCuts seam projector, kinematic K2D, "
                                     "legacy Dual Hinge T2D",
                    "paper_local_global_0918": "paper_local_global_0918: K3D/K2D/T2D solvers of the 2026-09-18 Grid "
                                               "route, OptCuts seams as exact K3D variables; needs even interior valences",
                }[r],
                help="Not part of Zhang et al. 2021. Both routes close the OptCuts seams exactly in K3D; the "
                     "09-18 route's alternating corner linkage additionally needs every interior vertex to have "
                     "even valence (checked in Phase 1).",
            )
        elif value in {"optcuts_structured_grid_20261028",
                       "optcuts_structured_grid_valley_fix_align_20261029",
                       "optcuts_structured_grid_strong_lines_20261029",
                       "optcuts_structured_grid_slide_shape_keep_20261030",
                       "optcuts_structured_grid_align_even_keep_20261030",
                       "optcuts_structured_grid_align_even_neighbor_keep_20261030",
                       "optcuts_structured_grid_soft_fit_20261104", "optcuts_structured_grid_fit_20261104"}:
            if value == "optcuts_structured_grid_fit_20261104":
                # 2026-11-04: the choices of the cloud comparison runs (the artifacts)
                st.radio("OptCuts の σ1/σ2 の制約", ["hard", "soft", "none"], index=0, key="onestring_fit_constraint",
                         format_func=lambda c: {"hard": "ハード制約（重み 4096、上限の判定は SD）",
                                                "soft": "ソフト制約（重み 16）", "none": "なし（普通の OptCuts）"}[c])
                st.checkbox("seam の始点を探索する（OptCuts 自身の上位 N^0.8 個の境界の頂点を候補に、topology の変更回数を倍々に"
                            "増やして良い半分を残す。対称面があれば対称性の罰則を入れ、ハード制約では枝分かれを禁止）",
                            value=True, key="onestring_fit_multistart")
                st.checkbox("入力メッシュを対称リメッシュする（対称面で半分に切り、Botsch–Kobbelt の等方リメッシュを形が変わらなく"
                            "なるまでかけて鏡に映し貼り合わせる。読み込んだメッシュファイルにだけ効く。gpytoolbox が必要）",
                            value=False, key="onestring_fit_symremesh")
            st.caption(
                "2026-10-28: official OptCuts chart Ω → Ω boundary replaced by straight lines / smoothing "
                "splines between features (seam nodes, sharp corners), bijective SLIM inside → regular quad "
                "grid of an orthogonal logical domain (corners = features, cut tips = reflex corners; interior "
                "valence 4, no T-junctions; seam copies get equal counts) → boundary points by 3D arc length, "
                "interior harmonic + Brackbill–Saltzman-type optimisation (orthogonality, size, smoothness in "
                "the 3D metric) → M3D → unchanged 2026-09-18 K3D…T2D with OptCuts seams closed exactly. "
                "Initial Ω CSF Split stays OFF. Quality is reported (console, metrics, "
                "output/structured_grid_diagnostics), never a reason to stop."
            )
            st.radio("Panel size from", ["grid size (sidebar)", "panel edge / panel count below"], index=0,
                     key="onestring_structured_grid_size_source",
                     help="2026-11-04: grid size = number of cells across the longer side of Omega's bounding box "
                          "(3D metric), as in the paper route's overlay grid.")
            st.number_input("Target 3D panel edge length (0 = from panel count below)",
                            min_value=0.0, value=0.0, step=0.01, format="%.4f",
                            key="onestring_structured_grid_panel_edge")
            st.number_input("Approximate panel count when edge length = 0 (sets the edge as sqrt(area/N))",
                            min_value=4, max_value=100000, value=150, step=10,
                            key="onestring_structured_grid_auto_panels")
            st.text_input("Ω boundary smoothing tolerances tried in order (× bbox diagonal; 0 = no smoothing)",
                          value="0.01,0.005,0", key="onestring_structured_grid_smoothing_tolerances")
            st.number_input("Corner detection scale (× bbox diagonal)", min_value=0.005, max_value=0.5,
                            value=0.04, step=0.005, format="%.3f", key="onestring_structured_grid_corner_scale")
            st.number_input("Grid-line smoothness weight (Brackbill–Saltzman smoothness term)",
                            min_value=0.0, max_value=100.0, value=1.0, step=0.1, format="%.2f",
                            key="onestring_structured_grid_smoothness_weight")
            st.checkbox("Close OptCuts seams exactly in K3D (seam vertices as shared variables)",
                        value=True, key="onestring_structured_grid_seam_closure")
        elif value in experimental:
            st.caption(
                "Isolated 10/23 OptCuts area-grid experiment; exact 50% mask, "
                "paired 3D/UV geometry and fail-closed distortion checks remain. "
                "Only the method(s) named in this mode are enabled."
            )
        else:
            st.caption(
                "Experimental Native Grid-OptCuts V4: split candidates are searched directly on "
                "the fixed fabrication lattice."
            )

        if value == "optcuts_grid_seam_native":
            st.markdown("**Native OptCuts Grid-Seam penalty (C++ Split/Merge)**")
            st.number_input(
                "Grid seam penalty strength",
                min_value=0.0, max_value=20.0,
                value=0.10, step=.05, format="%.2f",
                key="onestring_native_grid_seam_weight",
                help="0 reproduces original OptCuts C++ split/merge scoring; "
                     "higher values discourage diagonally oriented new seam edges. "
                     "Requires separately compiled patched C++ OptCuts.",
            )
            st.caption(
                "Uses a dedicated patched OptCuts binary; no fallback to "
                "Official OptCuts. The original 10/23 mask and distortion "
                "validity checks remain active."
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

        if value in {"optcuts", "optcuts_test", "optcuts_test2",
                      "optcuts_grid_aware","optcuts_area_grid_boundary",*experimental}:
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

            if value in {"optcuts_test", "optcuts_test2",
                          "optcuts_grid_aware","optcuts_area_grid_boundary",*experimental}:
                st.markdown("**K3D hard-planarity (Augmented Lagrangian)**")
                anchor_weight = st.number_input(
                    "K3D AL anchor weight (w_a)",
                    min_value=0.001,
                    max_value=10.0,
                    value=max(0.001, min(10.0, _safe_float_env("ONESTRING_K3D_AL_ANCHOR_WEIGHT", 1.0))),
                    step=0.05,
                    format="%.3f",
                    key="onestring_k3d_al_anchor_weight",
                    help=(
                        "Weight for staying close to the validity-repaired ordinary K3D. "
                        "Smaller values allow vertices to move more freely to satisfy planarity. "
                        "Try 1.0, 0.3, 0.1, or 0.03."
                    ),
                )
                os.environ["ONESTRING_K3D_AL_ANCHOR_WEIGHT"] = str(float(anchor_weight))
                st.caption(
                    "Smaller w_a = stronger freedom to planarize; larger w_a = stronger preservation "
                    "of the original K3D shape."
                )
                if value in {"optcuts_test2","optcuts_grid_aware",
                             "optcuts_area_grid_boundary"}:
                    st.caption(
                        "Test2 speed profile: K3D AL outer=8; K2D kinematic outer<=2, nfev<=40, "
                        "collision candidates<=900; SciPy finite-difference workers are used when supported."
                    )
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
                value=max(0.5, _safe_float_env("ONESTRING_OPTCUTS_GRID_MAX_SNAP_STEPS", 2.0),),
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

        # Keep the visible selector value distinct while routing test2 through the
        # mature optcuts_test patch stack.  The environment flag is the only
        # behavioral difference and is read by the acceleration wrapper.
        if value in {"optcuts_test2","optcuts_grid_aware",
                      "optcuts_area_grid_boundary",*experimental}:
            # Both keep the same proven K3D/K2D and paper-aligned M2D route.
            # Grid awareness is a separate Omega-only parameter.
            os.environ["ONESTRING_OPTCUTS_TEST_VARIANT"] = "2"
            return "optcuts_test"
        if value == "optcuts_test":
            os.environ["ONESTRING_OPTCUTS_TEST_VARIANT"] = "1"
        else:
            os.environ["ONESTRING_OPTCUTS_TEST_VARIANT"] = "0"
        return value

    st.selectbox = patched_selectbox
    st._onestring_optcuts_selector_installed = True


def _install_optcuts_test_k3d_ui_params(pipeline_module: Any) -> None:
    """Inject optcuts_test K3D AL UI values into the runtime params object."""
    if getattr(pipeline_module, "_onestring_optcuts_test_k3d_ui_params_installed", False):
        return

    base_optimize = pipeline_module._optimize_k3d

    def optimize_with_ui_params(target: Any, mesh: Any, parameterization: Any, params: Any):
        if str(getattr(params, "omega_parameterization_mode", "")) == "optcuts_test":
            anchor_weight = max(1e-6, _safe_float_env("ONESTRING_K3D_AL_ANCHOR_WEIGHT", 1.0))
            try:
                setattr(params, "k3d_al_anchor_weight", float(anchor_weight))
            except Exception:
                try:
                    object.__setattr__(params, "k3d_al_anchor_weight", float(anchor_weight))
                except Exception:
                    pass
            setattr(pipeline_module, "_optcuts_test_k3d_al_anchor_weight_ui", float(anchor_weight))
            print(f"[OPTCUTS-TEST-K3D-UI] anchor_weight={anchor_weight:.6g}")
        return base_optimize(target, mesh, parameterization, params)

    pipeline_module._optimize_k3d = optimize_with_ui_params
    original = getattr(pipeline_module, "_original", None)
    if original is not None:
        original._optimize_k3d = optimize_with_ui_params
    for fn in (
        getattr(pipeline_module, "build_onestring_design", None),
        getattr(pipeline_module, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_optimize_k3d"] = optimize_with_ui_params

    pipeline_module._onestring_optcuts_test_k3d_ui_params_installed = True


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
# Install before the generic seam metadata wrapper.  The generic wrapper will
# then call through this bridge and leave optcuts_test metadata intact.
install_optcuts_test_seam_metadata_bridge(pipeline)
# IMPORTANT: this was missing, silently leaving the newer K3D solver inert
# even with Omega=optcuts_test and the AL controls visible in the sidebar.
# Keep the AL wrapper inside UI-parameter, validity and preflight wrappers.
install_optcuts_test_k3d_augmented_lagrangian_patch(pipeline)
# Install immediately outside the AL wrapper so UI values are injected before
# the AL implementation reads params.k3d_al_anchor_weight.
_install_optcuts_test_k3d_ui_params(pipeline)
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
# Outermost runtime-budget wrapper.  It is a no-op unless the visible selector
# chose optcuts_test2 (ONESTRING_OPTCUTS_TEST_VARIANT=2).
install_optcuts_test2_acceleration_patch(pipeline)
# Outermost K3D and M3D hooks: enforce exact *paired* source-Seam edge
# equality after every other K3D optimization wrapper. T3D consumes the
# same hard equality groups in its shared top/bottom Eq.(2) solve.
install_optcuts_exact_seam_closure(pipeline)
# Outermost adapter: intercept only the explicit 2026-10-26 mode.
install_planar_quad_20261026(pipeline)
# Independent 2026-10-27 Zhang2021 mode (inactive unless selected).
install_zhang2021_quad_20261027(pipeline)
install_unstructured_quad_downstream_20261026()
package.onestring_pipeline = pipeline
package.build_onestring_design = pipeline.build_onestring_design
_install_optcuts_selector()

runpy.run_path(str(ROOT / "app_split_panels.py"), run_name="__main__")
