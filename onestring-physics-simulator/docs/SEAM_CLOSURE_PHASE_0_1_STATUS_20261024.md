# Seam密閉：実装指示書に対するゲート報告（2026-10-24）

対象：`fix/optcuts-real-m2d-footprint-20261023`、出発点 `4dce620`
原本：`OneString 実装指示書：Seam密閉の完成まで.pdf`

## ステータス（先に未達を記載）

**Seam密閉は未完成。Phase 0未通過・Phase 1未通過・Phase 2未着手・Phase 3未着手。**
合成テストが何件成功しても「Bunnyで完成」とは宣言しない。

- Phase 0で必要な本物のBunny 947パネルの、`f05219c` vs `4dce620` の同一ソース・同一カメラの比較動画がまだ存在しない。
- 実機MacのOptCutsバイナリ、ユーザーの再現条件、Bunny入力がGitHub Actionsの合成テストにはない。
- 6x6のbarycentric逆写像によるセルごとの弦誤差 `delta_i` と中央値 `delta_hat` は未計測。元曲面面積加重被覆率も未計測（視覚レンダリングを数値被覆の代わりにしない）。
- バグ解消まで、Phase 2の上流Ω境界/Seamペア制約によるGrid適合やPhase 3のハード制約回帰を勝手に適用しない。元のSやGrid origin/pitch/thresholdも変更しない。

## Phase 0：実装済み（ゲートは未通過）

- 新規 `src/onestring_physics/stage_audit_bundle.py`：
  - `ONESTRING_STAGE_AUDIT_DIR` でopt-inした1回のパイプライン実行に専用の `run_id` を割当てる。
  - `artifacts/<run_id>/stages.npz` へ、S/M2D/M3D/K3D-before-T3D/K3D/T3Dの頂点と面、元UV、元Ω境界とSeam行を記録。
  - `metrics.json` へGit SHA、設定、実際のランタイム配線のmodule/qualname、Stageメトリクスを保存。異常終了時も部分バンドル＋エラー理由を残す。
- `legacy_pipeline_base.py`：S、Ω、M2D、M3D、K3D、T3Dの製造処理を**変更せず**オプショナル記録を追加。
- 新規 `scripts/render_stage_audit.py`：
  - **源Sで固定したカメラ/同じbbox/同じ光条件**、バックフェースカリング無効、裏面マゼンタ/表面青、flat shading＋wire。
  - S/M3D/K3D/T3Dのfront/side/back/earsの16枚PNG。
  - K3D/M3D対のsigned areaに基づく赤/黒の退化パネル図、Seam両側の対応点＋差分リボン図。
  - 各ステージ360°、既定300フレーム30fpsのMP4＋GIF。診断入力が欠落している場合は`missing_artifacts`に明記。
  - 測定していない面積加重被覆率・弦誤差を`UNMEASURED`と明示。近似的な色だけで98%ゲートをPASSにしない。
- 新規 `scripts/compare_stage_audits.py`：
  - 旧版/新版の source S vertices/facesと固定カメラが一致することを検証し、2画面静止画と左右並列MP4/GIFを作る。
  - 異なるSやカメラの動画の見かけ上の比較をエラーとして拒否。

**未完のPhase 0ゲート：**
- `f05219c`側へ製造処理を変えずに同じ診断フックを導入し、旧版実行の診断データを用意する必要がある。
- ユーザーのMacで同じBunny・同じparamsで、旧版と`4dce620`版を実行し、2本の比較動画でスパイク/退化を比較する必要がある。
- Phase 0のMP4生成は合成CIテストの対象外。静止画生成/部分NPZ出力はCIでテストした。

## Phase 1 D1〜D4：パッチと合成テスト実装済み（Bunnyゲート未通過）

| ID | 修正 | 実装・検査 |
|---|---|---|
| D1 | ランク診断未実施・SVD失敗時はhard許可ではなくsoftへ | `optcuts_seam_exact_closure.py`, `optcuts_seam_soft_constraints.py`; cap 200→4000、未評価理由記録 |
| D2 | K3DとT3DのSeamペナルティを`w/(m*h²)`で正規化 | `optcuts_test_k3d_augmented_lagrangian_patch.py`にK3Dのhと有効係数、`paper_extrusion.py`に同じ係数を伝播。K3D h欠落時は黙って別単位を使わずエラー |
| D3 | `remap_rows()`の削除行数と率を監査 | `optcuts_seam_soft_constraints.py`、`extrusion_aware.py`、`split_geometry_constraints.py`、`paper_extrusion.py`。5%超の行消失は例外 |
| D4 | T3DのSeam残差の実数比較をテスト | `test_optcuts_seam_nonaffine_rank_soft.py`で上面・底面の最大残差が入力K3Dの1.5倍を超えないことを確認 |

**未完のPhase 1ゲート：**
- 実Bunny run_idの、全四角形のsigned area最小/p5、反転枚数、平面性inf枚数、Square RMS/max、Seam RMS/max、曲面追従、曲面被覆を **M3D→K3D→T3D** で比較する必要がある。
- 元曲面6×6サンプルから求める`delta_i`と`delta_hat`の数値計測は未実装。
- 所定のK3D合格基準 signed area ≥0.30、曲面追従・Square・SeamがM3Dより悪化なしという判定を未実施。

## Phase 2・3：実装しない理由

本指示書は各Phaseの数値ゲートを満たしてから次へ進むことを要求している。
Phase 0/1の実機データ未取得の段階でSeamペア拘束付きΩ境界分割を変更すると、
製造経路を新たに壊しても検出できない。したがって、
`optcuts_area_grid_boundary.py`、`optcuts_grid_sd_refinement.py`のSeamペア
DP・同時境界スライドの変更と、ハードSeamモードへの本格復帰は実施していない。

また、元メッシュを密閉する`tools/remesh_closed_bunny.py`はユーザー承認がないため一切適用していない。

## Mac上の実行手順

```bash
cd /Users/yuyakato/Documents/onestring-postcut-rebff/onestring-split-validation/onestring-physics-simulator
git pull --ff-only origin fix/optcuts-real-m2d-footprint-20261023
export ONESTRING_STAGE_AUDIT_DIR="$PWD/artifacts"
python3 -m streamlit run app_optcuts_20260923_extrusion_aware.py
```

Streamlitで従来通り同一Bunny・同一paramsで「構築」を実行。
ターミナルに表示される`[STAGE-AUDIT] run_id=...`を控える。

```bash
python3 -m pip install matplotlib imageio imageio-ffmpeg
python3 scripts/render_stage_audit.py \
  --bundle "artifacts/<run_id>/stages.npz" \
  --out "artifacts/<run_id>/"
```

旧版にも同一の**診断だけ**のフックを適用し、旧版の同型bundleが取得できたら:

```bash
python3 scripts/compare_stage_audits.py \
  --baseline artifacts/<old_run_id> \
  --candidate artifacts/<new_run_id> \
  --out artifacts/old_new_comparison
```

「1コマンドだけでBunnyの作成から動画比較まで再現する」という要件は現時点では未達
（Streamlit UI操作と旧版診断フックが必要）。

## 合否表（このコミット時点）

| Phase/条件 | 実測 | 判定 |
|---|---|---|
| 0: run_idつきS/M3D/K3D/T3D段階保存 | 合成テスト成功 | **実装あり、実機未確認** |
| 0: 固定視点図/退化図/Seam図 | 合成テスト成功 | **実装あり、実機未確認** |
| 0: 比較動画（旧版 vs 新版） | 旧版bundleなし | **未通過** |
| 1: D1〜D4実装と合成テスト | GitHub CI合格 | **合成テストのみPASS** |
| 1: Bunny signed area ≥.30 | 未計測 | **未判定** |
| 1: Bunny inf=0 | 未計測 | **未判定** |
| 1: Seam/曲面/Square M3Dより悪化なし | 未計測 | **未判定** |
| 1: delta_hat および視覚の崩壊減少 | 未計測 | **未判定** |
| 2: source_cut_different_grid_partition_count=0 | 未着手 | **未判定** |
| 3: ハード/水密/被覆率98%とT3D | 未着手 | **未判定** |

## 関連ファイル

- `src/onestring_physics/stage_audit_bundle.py`
- `src/onestring_physics/legacy_pipeline_base.py`
- `scripts/render_stage_audit.py`
- `scripts/compare_stage_audits.py`
- `tests/test_stage_audit_phase0.py`
- `src/onestring_physics/optcuts_seam_exact_closure.py`
- `src/onestring_physics/optcuts_seam_soft_constraints.py`
- `src/onestring_physics/optcuts_test_k3d_augmented_lagrangian_patch.py`
- `src/onestring_physics/paper_extrusion.py`
- `src/onestring_physics/extrusion_aware.py`
- `src/onestring_physics/split_geometry_constraints.py`
- `tests/test_optcuts_seam_nonaffine_rank_soft.py`

**確定:** D1〜D4のコード変更と合成テストの成功・段階保存/画像生成のスモークテスト。
**未確認:** Bunny実機の形状維持、旧新版動画、曲面被覆、弦誤差、最終密閉。
