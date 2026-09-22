# 2026-09-23 extrusion-aware experiment

Branch: `codex/extrusion-aware-pipeline-20260923`  
Base: `ea6ca5f` (`codex/deployability-k3d-20260921`, originと一致を確認後に分岐)  
Version: `2026-09-23-extrusion-aware`

## 実装前に確認したデータフロー・制約

- launcher → `app_optcuts_20260916.py` → OptCuts/common solverのpatch群 → `app_split_panels.py` → `app.py` → `app_backup_before_mitered_t3d.py`。
- 実際のstage driverは、`src/onestring_physics/onestring_pipeline.py`が動的importする`src_backup_before_sideface_contact/onestring_physics/onestring_pipeline.py`にある。名前がbackupでも稼働コード。
- `SurfaceParameterization`は元曲面の三角形とUV三角形を対で持つ。Ωから`PlanarDomain`のoverlayを作り、`QuadMesh M2D`をcrop/splitし、対応三角形内の逆写像で別の`QuadMesh M3D`を生成する。
- Simple Splitは`_split_panel_source_vertices`にgap導入前のUVを保存。M3Dへのliftはこのcanonical UVを使う。CSFはopen surfaceにも切れ目を追加する。OptCuts seamとCSFは別段階。
- paper local/global solverはM3Dから別のK3Dを生成。T3Dは共有top/bottom頂点で平面化した後に`(panel,8,3)`へ展開する。同じK3Dオブジェクトの`vertices`を`top_solved`で更新する。
- K2DはM2Dのcut topologyと**T3D更新後のK3D**を受け取る。corner linkageを作り、T2Dはpanelごとの剛体変換で生成。panel順序を変えない。
- virtual weldを無効化した既存処理は、flat layout、hinge/gap graphで意図したcutが再接続されるのを防いでいる。その処理は変更していない。
- `QuadGrid._vertex_id`、`TileSpec.row/col`、格子hinge、`tile_corners_from_vertices`の固定reshape、軸に沿ったCSFのsnap、native OptCutsのh/phase、crop後のtile対応が規則格子を前提とする。

## 論文と今回の独自実装

[原論文・著者版](https://akibzaman.github.io/data/onestringpull_authors_version_compressed.pdf)のSec.4.1–4.3を確認。主曲率方向を使える場合に格子を揃える考え方、K3DのPlanar/Square/Surface、法線offset、top/bottom/contact facesの平面化、flat側への対応関係は論文に由来する。

area-weighted quad normals、local/global LS、weak rest、反復上限、polish、OptCuts統合、CSF candidate heuristicはrepository側の具体化・拡張。今回の曲率fit・信頼度・fourfold場・UV回転、押し出し残差・重み・sparse有限差分、candidate cost、source同値制約、実験JSONも新規研究実装であり「論文の指定数値手法」とはしない。

## 機能と挿入位置

| bit | checkbox / flag | 実装場所と範囲 |
|---|---|---|
| 1 | Principal-curvature Grid / `use_principal_curvature_grid` | `extrusion_curvature.py`、parameterization直後。近傍法線差から対称shape operatorをfit、2主方向・曲率・信頼度を計算。`exp(4iθ)`で符号/90°交換を同一視して近傍平滑化。安全な最小範囲として**格子全体のUV回転**へ集約。局所方向を完全追従するquad remeshingではない。球面/臍点/flat/noisy領域は連続的に既存方向へ戻す。native Grid-OptCutsでは固定seam格子と矛盾するため明示停止。 |
| 2 | Extrusion-aware K3D / `use_extrusion_aware_k3d` | `extrusion_aware.refine_k3d`を`paper_local_global_solvers.py`の既存solveとhard/polishの間に挿入。既存Planar/Shape/Length/Surfaceを再評価し追加残差を最小化。anti-degeneracyの最終local制約も保持。hard-mode目的にも追加。既存polishを維持するため、最終品質の単調改善を保証するものではない。paper local/global以外のsolverを選んだ場合は明示停止。 |
| 3 | Curvature-adaptive Grid Size / `use_curvature_adaptive_grid` | **未対応**。`extrusion_aware.validate`で理由付き`NotImplementedError`。T-junctionや暗黙ID破壊を避けるにはconforming quad remesherとpanel/hinge対応の一般化が必要。均一gridを細かくする偽の代替は実装していない。 |
| 4 | Extrusion-aware Split / `use_extrusion_aware_split` | `onestring_pipeline._flatten_to_domain`の既存CSF候補生成・選択。現在のoverlayを仮lift/押し出しし、共通評価関数のdifficultyをCSF costへ加算。既存peak/residual候補poolを`D_existing + λD_extrusion`で順位付けし、既存budget/mirror/snap/duplicateへ渡す。OptCutsのC++ seam objectiveは改変しない。対応するCSF経路を使わないmodeでは明示停止。 |

4 checkboxは独立したdataclass field。adaptiveを除く8 maskを同じpipelineで扱う。`0010`、`1111`を含むadaptive ONの8 maskは**実行成功ではなく明示的な未対応**。全16通りを完了したとの主張はしない。

## 共通押し出し評価

`extrusion_quality.evaluate_extrusion`がK3D/Split/記録を共有する。T3Dと同じ`paper_extrusion._vertex_normals`と実際のthicknessを使用。

- 側面best-fit plane距離
- top/bottomの向き付き法線差・角度
- panel法線に沿った厚みのばらつき、正の深さの逸脱
- tangential shear
- top/bottom辺ベクトルの差（自然な厚板からの変形）、rigid-fit RMS

dimensionless残差二乗和の平方根をpanel difficultyとする。目的関数へは元M3D平均辺長の二乗を掛けて長さ単位を揃える。単純な法線offsetではcorner間のoffset長が一定になるため、厚みばらつきはpanel法線への射影深さで評価する。自己交差・製造可能性の完全な証明ではない。

## Splitの3Dとflatの区別

非zeroの対応maskでは、元曲面vertex IDとbarycentric weightの一致から同じ元位置を識別する。座標が同じという理由だけで別sheetを接続しない。canonical UVはCSF splitの入力由来情報として保持する。

K3DとT3Dのsolve中だけ幾何自由度を共有し、元の頂点配列長・face IDへ展開し直す。K2D/T2Dには元のcut face IDを渡す。T3Dの`top_solved`によるK3D更新も維持する。

**0000はbaseline一致を優先して従来の境界処理をそのまま使用する**。元版でsplit両側が離れる場合、その問題を0000だけ修復すると回帰条件を満たさないため。非zero maskでの同値制約自体が比較に影響する点をJSONへ明記。完全一致の対応sampleがないOptCuts両境界の連続曲線拘束は未実装。

## 記録と表示

各実行は`output/extrusion-aware/*.json`へ保存（git対象外）。UIからもJSONをdownload可能。version/mask、パラメータ、solver環境設定、入力mesh SHA256、panel count、split edge count/length、K3D各誤差、仮押し出し/最終K3D/実際T3D品質、panel difficulty、T2D collision/fabrication、6 stage runtimeを記録する。

`split_count`は実際の対応付きboundary **edge segment数**で、連結cutの本数とは異なる。`split_requested_line_count`も併記。`split_total_length`はM3D上の長さ。同じsource端点を持つedgeを一度数える。対応samplingの不一致による連続seam長は別途評価が必要。利用中の経路が提供しないcollision/fabrication値はnullとして記録し、0と偽らない。

linked-panel viewerの`Panel color → Extrusion difficulty`はfinal K3Dの仮押し出しdifficultyを0=青、0.5=黄、1以上=赤の固定尺度で表示する。panel click対応とselectedのピンクを維持し、元のcorrespondence gradientへ戻せる。通常の別メッシュ表示すべてへ色モードを追加したわけではない。

## 簡潔な検証と未確認範囲

- `tests/test_extrusion_aware.py`: plane/cylinder/sphere/saddle/neck、低信頼度fallback、source同値と別sheet非結合、共通評価、cost加算、gradientを検証。
- `0000`は小規模deterministic pipelineで旧versionとM2D/M3D/K3D/K2D/T3D/T2D top/dualの配列をbitwise比較。split ON/OFF両方、hinge数とsplit linesも比較（ON側はsynthetic CSF=2.5を与え、実際にcutが生成されることも確認）。
- `1000/0100/0001/1100/0101/1001/1101`は小規模neckで機能→K3D→T3D→K2D→T2D topの経路を確認。`0010/1111`などadaptive maskは理由付き停止を検証。
- paper T3D、K3D history、T2D、Eq.5の関連既存テストも実行。合計60件。これらはOptCuts外部processを使わない。
- Streamlit AppTestによるheadless起動、新version初期選択、4 checkbox表示を確認。実ブラウザの描画・panel click操作は未確認。
- **Bunny full run、実OptCuts再実行、Bunnyの首とcollision改善の因果関係、物理組立、性能比較は未検証**。hard K3Dの新gradientは数値差分と比較したが、大規模hard-mode収束は未確認。
- 対応するmaskのcode path成功は品質改善の証明ではない。局所方向追従grid・adaptive remeshing・OptCuts native objectiveへのdifficulty導入は今後の課題。

実行コマンド:

```bash
python3 -m pytest -q tests/test_extrusion_aware.py tests/test_paper_t3d_20260920.py tests/test_k3d_iteration_history.py tests/test_paper_t2d.py tests/test_paper_eq5_k2d.py
```

## 起動

```bash
cd ~/Documents/OneString-large-steps-mesh-conditioning/onestring-physics-simulator
git fetch origin
git switch codex/extrusion-aware-pipeline-20260923
git pull --ff-only
python3 -m streamlit run app_optcuts_20260923_extrusion_aware.py
```

旧launcherも残してある。新launcherのversionは`2026-09-23 — Extrusion-aware experiments`、Ωは既存の`2026-09-18 | OptCuts Ω + paper-aligned local/global K3D/K2D`を使う。まず4 OFFでbaseline保存。次に1機能ずつON。adaptive ONは現在停止する。比較時は入力、厚さ、格子、既存K3D重み/反復/polish/hard設定、OptCuts/CSF設定を同じに保つ。

## 変更ファイル

- `app_optcuts_20260923_extrusion_aware.py`: 新launcher
- `app_backup_before_mitered_t3d.py`: 新version、checkbox/weight、cache key、JSON download、linked panel coloring
- `src/onestring_physics/extrusion_quality.py`: 共通評価
- `src/onestring_physics/extrusion_curvature.py`: 曲率・信頼度・主方向Grid
- `src/onestring_physics/extrusion_aware.py`: flags/制約、K3D追加目的、Split cost、source同値、実験記録
- `src/onestring_physics/onestring_pipeline.py`: dataclass fields、CSF候補cost接続
- `src/onestring_physics/paper_local_global_solvers.py`: K3D目的追加・history/hard-mode接続
- `src/onestring_physics/paper_extrusion.py`: 同値幾何の一時solveと元topologyへの展開
- `src/onestring_physics/paper_t3d_20260920_patch.py`: 新launcherのdefault version選択
- `src/onestring_physics/paper_ui_20260916_patch.py`: legacy selector多重wrapperでの新version初期値保持
- `src_backup_before_sideface_contact/onestring_physics/onestring_pipeline.py`: 実stage driverのversion-gated接続とruntime
- `tests/test_extrusion_aware.py`: 小規模回帰・機能テスト
- `docs/EXTRUSION_AWARE_20260923.md`: 本報告
