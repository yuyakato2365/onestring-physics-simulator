"""On-demand Streamlit 3D comparison of accepted and REJECTED Split designs.

The geometry comes from temporary, fully rebuilt trial states. Displaying it
must never mutate the authoritative state, replace a manufacturing export, or
run a new optimization. The preview data intentionally have no STL export.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import plotly.colors
import plotly.graph_objects as go

from .paper_collision_split_preview import (
    collision_tile_severity, hinge_gap_segments, rejection_comparison,
    acceptance_checklist,
)
from .visualization import figure_tile_assembly, figure_quad_mesh
from .panel_quality_visualization import add_t2d_collision_overlay


def _severity_colors(severity: np.ndarray, shared_max: float) -> list[str]:
    # Plotly's per-tile vertex colors share ONE normalizer across the before
    # and after previews, so an "improved" trial cannot look artificially red
    # simply because it has a different individual color scale.
    return [
        str(plotly.colors.sample_colorscale(
            "YlOrRd", max(.16, float(np.sqrt(v/shared_max))),
        )[0]) if v > 0 and shared_max > 0 else "#92c8c2"
        for v in severity
    ]


def _add_largest_hinge_gaps(fig, snapshot: dict, vertices: np.ndarray,
                            *, max_segments: int = 50):
    largest = hinge_gap_segments(
        vertices, snapshot.get("hinges", np.zeros((0, 4), dtype=int)),
        max_segments=max_segments,
    )
    if len(largest["gap"]) == 0:
        return
    errors = largest["gap"]
    positive = errors > 1e-9
    # Show the worst nonzero gaps; these are hinge endpoints, NOT SAT contact
    # points. The true SAT intersection footprint is encoded in panel color.
    if not np.any(positive):
        return
    x, y, z, hover = [], [], [], []
    for p, q, length, idx in zip(
        largest["start"][positive], largest["end"][positive],
        errors[positive], largest["index"][positive],
    ):
        x.extend((float(p[0]), float(q[0]), None))
        y.extend((float(p[1]), float(q[1]), None))
        z.extend((float(p[2]), float(q[2]), None))
        msg=f"hinge #{int(idx)} gap={float(length):.6g}"
        hover.extend((msg,msg,None))
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode="lines",
        line=dict(color="#f97316", width=5),
        text=hover, hoverinfo="text",
        name=f"Large hinge gaps (up to {max_segments})",
    ))


def _preview_figure(snapshot: dict, stage: str, *,
                    title: str, severity_max: float,
                    include_hinges: bool, red_collision_overlay: bool):
    geometry=snapshot["stages"][stage]
    vertices=np.asarray(geometry["vertices"], dtype=float)
    if stage in ("M2D", "K2D"):
        mesh=SimpleNamespace(
            vertices=vertices, faces=np.asarray(geometry["faces"],dtype=int),
            stage=stage, metrics={},
        )
        fig=figure_quad_mesh(mesh,title=title)
        return fig, None

    assembly=SimpleNamespace(vertices=vertices,stage=stage,metrics={})
    severity=collision_tile_severity(vertices)
    colors=_severity_colors(severity["strength"],severity_max)
    fig=figure_tile_assembly(assembly,title=title,tile_colors=colors)
    if red_collision_overlay:
        fig,_,_=add_t2d_collision_overlay(fig,assembly,hinge_graph=None)
    if include_hinges:
        hinge_snapshot=(snapshot if stage=="T2D Dual" else
                        dict(snapshot,hinges=snapshot.get("top_hinges",snapshot.get("hinges",[]))))
        _add_largest_hinge_gaps(fig,hinge_snapshot,vertices)
    fig.update_layout(
        scene=dict(aspectmode="data"),
        margin=dict(l=8,r=8,t=52,b=8),
        uirevision=title,
    )
    return fig,severity


def render_split_candidate_previews(st, state, step_rows: list[dict]):
    """Render the selected trial, with equal before/after severity color scale."""
    previews=list(getattr(state,"collision_split_candidate_previews",()) or ())
    if not previews:
        st.caption(
            "この実行には棄却候補の描画データがありません。"
            "新しいコードでRun OneString pipelineを再実行すると、"
            "仮Splitを含む候補ごとの形状を保存します。"
        )
        return

    selectable=list(range(len(previews)))
    first_rejected=next((i for i,p in enumerate(previews)
                         if not p.get("accepted",False)),0)

    def _display_candidate(i):
        p=previews[i]
        prop=p["proposal"]
        flag="採用" if p.get("accepted",False) else "棄却"
        return (f"候補 {i+1} / Split #{p['cut_number']} "
                f"{prop.get('axis','?')}={prop.get('value',0):.5g} — {flag}"
                f"（{p.get('decision_reason','no reason')}）")

    selection=st.selectbox(
        "描画する仮Split候補（棄却された候補も選択可能）",
        selectable,index=first_rejected,
        format_func=_display_candidate,
        key="collision_split_preview_selected",
    )
    p=previews[selection]
    old,new=p.get("before"),p.get("after")
    if new is None:
        st.warning("この候補は再構築されたT2D形状が存在しないため描画できません。")
        return
    old_stages=old.get("stages",{}) if old else {}
    new_stages=new.get("stages",{})
    order=("T2D Dual","T2D Top","K2D","M2D")
    available=[stage for stage in order
               if stage in new_stages and (not old or stage in old_stages)]
    if not available:
        st.warning("この候補には比較可能なステージの形状がありません。")
        return

    before=p.get("before_metrics",{})
    after=p.get("after_metrics",{})
    decision=p.get("decision_reason","not evaluated")
    facts=rejection_comparison(before,after)
    mode=p.get("acceptance_mode")
    if p.get("accepted",False):
        st.success(f"正式採用したSplit候補：{decision}")
    else:
        st.warning(
            "この候補は仮Splitの再最適化まで実行済みですが、"
            f"最終設計には採用されていません。棄却理由：{decision}"
        )

    if mode=="lookahead_first_cut":
        st.info(
            "この1本は単独では棄却でした（理由："
            f"{p.get('single_step_decision_reason','unknown')}）。"
            "追加の2本目と合わせて元の採用条件を満たしたため、"
            "2本まとめて正式採用されました。"
        )
    if p.get("lookahead_baseline_metrics"):
        st.info(
            "2本先読み候補です。下の前後比較は「1本目の仮配置→2本目の仮配置」。"
            "正式採否は「最後の正式採用状態→2本目の仮配置」で評価しています。"
        )
    d0=before.get("penetration_max")
    d1=after.get("penetration_max")
    h0=before.get("hinge_rms")
    h1=after.get("hinge_rms")
    pair_delta=facts.get("collision_delta")
    st.dataframe([{
        "評価項目":"SAT衝突ペア数",
        "Split前":before.get("collision_pairs"),
        "仮Split後":after.get("collision_pairs"),
        "差":pair_delta,
    },{
        "評価項目":"貫通エネルギー（二乗深さ総和）",
        "Split前":before.get("collision_energy"),
        "仮Split後":after.get("collision_energy"),
        "差":(after["collision_energy"]-before["collision_energy"])
             if isinstance(before.get("collision_energy"),(int,float))
             and isinstance(after.get("collision_energy"),(int,float)) else None,
    },{
        "評価項目":"最大SAT貫通深さ",
        "Split前":d0,"仮Split後":d1,
        "差":facts.get("max_penetration_delta"),
    },{
        "評価項目":"残存ヒンジRMS",
        "Split前":h0,"仮Split後":h1,
        "差":facts.get("hinge_rms_delta"),
    }],hide_index=True,use_container_width=True)
    improvement=facts.get("penetration_energy_improvement_percent")
    if improvement is not None:
        st.caption(f"貫通エネルギー改善率：{improvement:+.2f}%（正なら改善）。"
                   f" 衝突ペア数増加許容：{facts.get('extra_pairs_allowance')}件。"
                   f" ヒンジRMS棄却閾値：{facts.get('hinge_rms_rejection_threshold'):.5g}"
                   if facts.get("hinge_rms_rejection_threshold") is not None
                   else f"貫通エネルギー改善率：{improvement:+.2f}%")
    minimum=float(state.tiles_2d_dual_hinge.metrics.get(
        "collision_split_min_relative_improvement",.03
    ))
    checklist=acceptance_checklist(
        before,after,min_relative_improvement=minimum,
    )
    if checklist:
        st.markdown("**採否に使われる閾値と実測値の比較**")
        st.dataframe([{
            "判定項目":item["criterion"],
            "実測値":item["observed"],
            "許容閾値":item["limit"],
            "閾値を満たすか":"✓" if item["ok"] else "✕",
        } for item in checklist],hide_index=True,use_container_width=True)
        st.caption(
            "この表は数値条件の評価です。"
            "切断線が本当にメッシュへ適用されたか等のトポロジー監査は"
            "別途行い、その結果も棄却理由に反映されます。"
        )
    baseline=p.get("lookahead_baseline_metrics")
    if baseline:
        st.markdown("**2本セット全体の評価：正式採用前 → 2本追加後**")
        st.dataframe([{
            "評価項目":name,"2本追加前":baseline.get(key),
            "2本追加後":after.get(key),
        } for name,key in (
            ("SAT衝突ペア数","collision_pairs"),
            ("貫通エネルギー（二乗深さ総和）","collision_energy"),
            ("最大SAT貫通深さ","penetration_max"),
            ("残存ヒンジRMS","hinge_rms"),
        )],hide_index=True,use_container_width=True)
        final_checks=acceptance_checklist(
            baseline,after,min_relative_improvement=minimum,
        )
        if final_checks:
            st.dataframe([{
                "2本セットの正式判定条件":item["criterion"],
                "実測値":item["observed"],"許容閾値":item["limit"],
                "合否":"✓" if item["ok"] else "✕",
            } for item in final_checks],hide_index=True,use_container_width=True)
    st.caption(
        "評価表は実際の仮Split再計算時に記録した値です。"
        "以下の形状は診断専用コピーであり、棄却候補は最終採用STLに反映しません。"
    )

    stage=st.selectbox(
        "比較する再最適化工程",
        available, key="collision_split_preview_stage",
    )
    red=st.checkbox(
        "衝突するパネルを赤で強調する（色の濃淡より衝突箇所を優先）",
        value=False,key="collision_split_preview_red_overlay",
        disabled=stage not in ("T2D Dual","T2D Top"),
    )
    hinges=st.checkbox(
        "接続誤差のあるヒンジ上位50本を橙色の線で表示",
        value=True,key="collision_split_preview_hinges",
        disabled=stage not in ("T2D Dual","T2D Top"),
    )

    ntiles=int(new.get("tile_count",0))
    # Match the existing performance guard in the cross-stage 3D inspector:
    # two live Plotly meshes with hundreds of tiles are expensive.
    render=ntiles<=250
    if not render:
        st.warning(
            f"{ntiles}パネル×前後2画面の3D描画は高負荷です。"
            "形状は保存済みです。以下のチェックで選択中の候補だけ描画できます。"
        )
        render=st.checkbox(
            "高負荷でも選択候補の3D形状を表示",
            value=False,key="collision_split_preview_show_large",
        )
    if not render:
        return

    severity_max=0.
    if stage in ("T2D Dual","T2D Top"):
        for snapshot in (old,new):
            if snapshot and stage in snapshot.get("stages",{}):
                info=collision_tile_severity(snapshot["stages"][stage]["vertices"])
                severity_max=max(severity_max,float(np.max(info["strength"],initial=0.)))
        st.caption(
            "薄緑＝衝突なし、黄〜赤＝SAT貫通深さの二乗和が大きいパネル。"
            "2画面で共通の色スケールを使います。橙線＝実際のヒンジ接続ギャップ。"
        )

    left,right=st.columns(2)
    for column,snapshot,side in ((left,old,"before"),(right,new,"after")):
        with column:
            st.markdown("**Split追加前**" if side=="before"
                        else "**仮Split後（再最適化完了）**")
            if not snapshot:
                st.info("この工程のSplit追加前形状は保存されていません。")
                continue
            try:
                fig,info=_preview_figure(
                    snapshot,stage,
                    title=f"{stage} — {'Before Split' if side=='before' else 'Trial after Split'}",
                    severity_max=severity_max,
                    include_hinges=hinges,red_collision_overlay=red,
                )
                st.plotly_chart(
                    fig,use_container_width=True,
                    key=f"split_trial_{selection}_{stage.replace(' ','_')}_{side}_{int(red)}_{int(hinges)}",
                )
                if info is not None:
                    st.caption(
                        f"描画形状のSAT再計算：衝突 {info['pair_count']}ペア、"
                        f"最大貫通深さ {info['max_depth']:.5g}、"
                        f"貫通二乗和 {info['energy']:.6g}"
                    )
            except Exception as exc:
                st.error(
                    f"{side}の{stage}描画に失敗しました。"
                    "仮Split候補の採否・最終結果への影響はありません。"
                )
                st.exception(exc)

    st.caption(
        "2画面は独立に回転・ズームできます。"
        "Split後の配置は採用していない仮の計算結果そのもので、"
        "最適化をもう一度実行したり、後から形状を引き離したりしていません。"
    )
