# View Stage復旧とK2D→T2D監査（2026-09-20）

作業元: `agent/paper-t3d-20260920` / `ecdae19`。作業branch: `codex/fix-view-and-paper-t2d-20260920`。

## 1. 表示が一瞬出て消える原因

K3D/T3Dの3D canvasを生成した後、外側の`app_optcuts.py`が選択中のViewに関係なくK2D最終図と全反復履歴を追加していた。履歴の各図が`go.Scattergl`を使い、ブラウザーのWebGL contextを多数消費する。先に描かれた3D contextが失われるため、枠・凡例は残ってもメッシュが消える。

修正前の実画面でK2D履歴27枚、全30 Plotly図、canvas82個を確認。T3D切替後にブラウザーログの`Cannot read properties of null (reading 'gl')`を確認した。先行する9/19の独立最小再現でも「3Dのみは正常→多数のScattergl履歴追加で消失→SVG化で復旧」を確認済み。今回も同じScattergl実装が残っていた。

導入箇所は`91a1bd5`（三列のK2D履歴）、`6c50b0d`（固定スケールの履歴）と、その後のmain結果追加で追跡できる。先行修正`260fd54`は今回の対象branchの祖先ではなかった。runpy自体によるメッシュの消去やK3D計算失敗が主因ではない。

## 2. 正常時との差分、実行経路

`app.py`および`src/onestring_physics/visualization.py`は、作業開始時点ですでに`agent/large-steps-mesh-conditioning`と同一だった。したがって3Dの`figure_quad_mesh`、`figure_tile_assembly`とcamera wrapperは変更しない。

実行経路:

```
app_optcuts_20260920_paper_t3d.py  (paper T3Dをinstall)
 → app_optcuts_20260916.py
 → app_optcuts.py               (paper UIをinstall)
 → app_optcuts_core_20260916.py
 → app_split_panels.py          (数値split patchとanimation選択肢)
 → app.py                      (既存camera対応plotly wrapper)
 → app_backup_before_mitered_t3d.py  (実UI、state、View stage)
```

`st.plotly_chart`はunified UIのOmega figure captureと`app.py`のcamera wrapperを通る。View stageの所有者は最後のlegacy UI。unified selectorは明示的なanimation選択時だけ描画し、K3D/T3D等のstatic stageはそのまま返す。新しいrendererは追加していない。

`onestring_state`はsessionに保持され、Runボタンだけが再計算する。View stageに明示的な`onestring_view_stage` keyを付けた。`app.py`のrunpy結果にlegacyのstate変数が直接返るという古い前提も除去した。

## 3. 削除・整理した処理

- 外側launcherから無条件で追加されるK2D最終図・全履歴。
- K2Dをselectboxの途中で横取りするfinal-result selector patchのinstall。
- `app_split_panels.py`のstatic persistence / postrun描画 / synthetic fallback呼出し。
- 同launcherの不要な`importlib.reload`抑止。package側の数値patch再install guardは維持。
- 呼出元がなくなったunified UIのpostrun診断・fallback関数と環境変数。
- 計算途中でK2D履歴を描く副作用。今はsessionへ記録するだけ。

`K3D pipeline check`とXY診断scatterは元HEADですでに削除済みで、再導入していない。明示的に選ぶ既存のprocess animationは維持。K2D履歴はK2D viewのexpanderへ移し、全図をSVGの`go.Scatter`に変更。Streamlit 1.50に合わせPlotlyの幅指定を`use_container_width=True`に統一した。

## 4. 表示の確認

指定launcherをポート8509で実起動。簡易確認としてbuilt-in Dome、grid size 4、その他既定値（2026-09-18 local/global mode、paper T3D有効）を使用。121 source vertices、20 quads、K3D 41 checkpointsで完了。

- 実ChromeでK3D→T3D→K3Dを操作し、数秒以上経過後のメッシュ表示を目視確認。
- K3Dの3D本体→EAssembled履歴→planarity履歴の順を確認。
- 実launcherのStreamlit AppTestを別プロセスで起動。K3D、T3D、M2D、M3D、K2D、T2D Top Hinge、T2D Dual Hinge、K3Dの順で切替し、各段階でさらにrerun。
- state、K3D mesh、authoritative historyの同一オブジェクト保持を検証。3D図は各stage一つ、K3Dだけ計3図。K2D以外へ履歴が追加されない。全stageでScatterglゼロ。

これは小規模Domeによる表示・接続確認であり、大規模Domeの製造可能性や全入力形状での最適化収束の証明ではない。

## 5. K3D履歴

`paper_local_global_solvers.optimize_paper_local_global_k3d`内で、実際に返す頂点のcheckpointを記録している。既存の記録処理・目的関数・重みは変更していない。初期状態を含む41 checkpoints。表示は`mesh_3d_optimized.metrics['k3d_iteration_history']`を読むだけ。

EAssembled、weighted Planar/Square/Surface、best-fit planeへの頂点距離のmax/RMS/meanを表示。最終頂点から各量を独立に再計算するテストを維持し、表示側でoptimizerを呼べないテストを追加した。重みはUIラベルの定数を推測せず、authoritative historyに記録された実値を表示する。

## 6. 一次資料で確認したK2D→T2D

参照:

- [著者版本文](https://onestringtopullthemall.github.io/static/pdfs/onestringpull_authors_version_compressed.pdf): Sec.4.3、4.4、5.2、5.3。
- [Supplement](https://onestringtopullthemall.github.io/static/pdfs/OneStringPull-supplement.pdf): Appendix A、Figures 18–19。
- [公式project](https://onestringtopullthemall.github.io/)。確認時点で著者の実装コードへのリンクは見つからなかった。非公開実装を推測して同一と称していない。

1. M2DとK3Dに対応するcorner-connected quad linkageを用意する。
2. K3Dの辺長をtargetとするEEdge、非貫通ECollision、開口角制約EFabを合わせたEq.(5)でK2Dを求める。Appendix AのEFabは角度範囲`[theta_min,90°]`へ二つのベクトルを最近接射影する。
3. T3Dの対応する上面→下面の回転・平行移動を求め、K2Dへ適用して下面を作り、側面を接続する。
4. Top Hinge方式は重なる領域を切除する。Dual Hinge方式は開口方向の負の法曲率でbottom hinge、その他でtop hingeを選び、Eq.(6)で剛体性・衝突・ヒンジ接続を調整する。
5. タイル間の実際のvoidをgap nodeとする。Sec.5.2は共有tileによる隣接、Sec.5.3はstring routingのラベル・制約を定義する。

## 7. 現在実装との差分と限界

従来はT3Dの8頂点すべてを剛体コピーして配置したため、一般には最適化済みK2D上面からずれていた。今回、K2D頂点を上面としてそのまま使う構成へ変更した。

ただし一般のfrustumの上下4点は合同ではなく、一つの剛体変換で厳密対応できない。本文はその場合のfit手順を明示していない。proper Kabschの最小二乗fitを使い、対応tile frameで変換を適用する。fit残差、T3Dとの全頂点間距離誤差を必ず報告する。**K2D保持とT3D剛体形状の両方を厳密に満たしたと偽らない。**

既存Eq.(6)はSE(2) pose最適化でT2Dの初期形状を保持する。T3Dとの非合同を直す自由度はない。また3D frustum同士の厳密衝突ではなく、8頂点の凸XY footprintを使う保守的な近似。Top Hinge方式の切除、hinge頂点の実際のmerge、split boundaryの-2ラベル、論文のrouting MILPは未実装である。既存STL出力もthin-plate近似であり、今回のT2D solidの厳密な製造exportとは別。

Dome/grid4の既定重みではK2D最終時に31 collision pairs、28 fabrication violationsが残る。新しいT2D validatorはこの入力を製造可能と認定しない。単にペナルティが有限、または衝突ゼロになっただけで成功とは扱わない。

## 8. 分類

| 処理 | 分類 |
|---|---|
| 対応tile identity、K2D上面を保持して上面→下面変換を適用、側面を接続 | Paper-exact（操作・対応関係の範囲） |
| EEdgeのtarget辺長、Appendix Aの角度射影、voidをnodeとする定義 | Paper-exact（数学的定義） |
| 非合同面のKabsch fitとframe合わせ、離散法線差による曲率判定 | Paper-consistent approximation |
| K2DのSAT局所非貫通射影、Eq.6のSE(2)・凸XY footprint | Paper-consistent approximation |
| void boundary抽出の実装、幾何学的な縦横ラベル | Paper-consistent approximation |
| 未切除Top Hinge preview、任意anchor、境界node sampling、既存string route heuristic | Project-specific extension |

分類と未実装範囲はT2D/hinge/gap metricsにも保存する。アルゴリズム全体をPaper-exactとは称していない。

## 9. T2D変更内容とstate対応

| state | 意味 |
|---|---|
| `mesh_2d_optimized` | Eq.(5)のK2D、物理jointを共有する頂点 |
| `k2d_flat_layout` | K2Dのtile別corner gather。再配置・packingなし |
| `tiles_2d_top_hinge` | K2Dを上面にした厚み付きT2D。未切除preview |
| `tiles_2d_dual_hinge` | 同じtileとcorner対応でtop/bottom hingeを選んだEq.(6)結果 |

- K2Dの明示的hinge pairsを最後まで使用。grid幅やface番号から再推定しない。
- unsigned dihedralから離散的な符号付き法曲率へ変更。
- 最適化後の個別平行移動による非貫通cleanupは、このexplicit linkage経路では実行しない。接続を壊すため。
- top/dualそれぞれの衝突数、角度違反数、最大hinge接続誤差、K2D辺長誤差、T3D合同性を検証。
- tile edgeの外側boundaryをhingeでたどり、実際のvoidを抽出。従来の二枚のtile間edge-midpointをgapとする処理をこの経路では使用しない。
- splitを再溶接せず、非連結graphを勝手に橋渡ししない。
- K2DのSAT包含ケース、global更新後のcollision再評価、初期snapshotとtheta_minの記録を修正。
- OptCuts、Ω、M2D/M3D、K3D、paper T3D、scale、split計算は変更していない。

## 10. テスト結果

基準実行（修正前）: `182 passed, 13 failed, 1 skipped`。
全体実行（収集不能の既存ファイル一つを除外）: `197 passed, 13 failed, 1 skipped`、64.82秒。失敗した13テストのID集合は基準と完全一致。全体実行後に追加した2テストを含め、最終の対象4ファイルは`33 passed`、11.06秒。

コマンド:

```
python3 -m pytest tests -q --ignore=tests/test_optcuts_test_boundary_reparameterization.py
python3 -m pytest tests/test_view_rerun.py tests/test_paper_t2d.py tests/test_paper_eq5_k2d.py tests/test_k3d_iteration_history.py -q
```

除外ファイル単独も実行し、既存の`_build_test_targets` import不能によるcollection errorを確認。隠してskipしたものではない。

既存の失敗13件:

```
test_bijective_free_boundary.py::test_pipeline_mode_is_explicit_and_does_not_replace_bff
test_dynamic_free_boundary.py::test_boundary_direction_is_local_not_global_polynomial_mode
test_lscm_latest_omega_hybrid_20260914.py::test_hybrid_routes_only_omega_and_k3d_to_latest
test_optcuts_20260914_component_k2d_layout.py::test_latest_flat_layout_calls_exact_lscm_builder_once_on_whole_mesh
test_optcuts_20260914_component_k2d_layout.py::test_latest_flat_layout_does_not_change_common_branch_input_size
test_optcuts_grid_native.py::test_internal_optcuts_seam_detected_from_surface_uv_index_mismatch
test_optcuts_grid_native.py::test_setup_uses_verified_v4_installer_and_cpp14
test_optcuts_grid_seam_detection.py::test_geometrically_separated_uv_copies_are_a_seam
test_paper_reference_initialization.py::test_reference_flat_disk_matches_input_plane_up_to_similarity_and_round_trips
test_paper_reference_initialization.py::test_reference_gaussian_bump_saves_flip_singular_value_and_lambda_diagnostics
test_paper_reference_initialization.py::test_reference_saddle_completes_with_negative_gaussian_curvature_and_reports_overlap
test_paper_reference_initialization.py::test_reference_half_snowman_user_shape_produces_comparable_initialization
test_reference_bff.py::test_official_bff_matches_committed_golden_without_skip
```

これらはBFF executable不在、既存OptCuts/seam判定・古いmock前提等。今回禁止された上流数値処理の範囲へ修正を広げていない。

T2Dテストは解析的な3×3 rotating-square linkage（角度40°、剛体prism）でtile identity、辺長、接続、非衝突、加工角、4つの四枚tile gap、後段string path生成を確認する。壊れたhinge・非合同frustumを合格にしない負例と、非連結tileの橋渡し禁止も含む。

## 11–12. Git

branch: `codex/fix-view-and-paper-t2d-20260920`。この報告を含むcommitが対応する実装。hashは最終回答に記載する（ファイル自身へ自己参照hashを埋め込まない）。
