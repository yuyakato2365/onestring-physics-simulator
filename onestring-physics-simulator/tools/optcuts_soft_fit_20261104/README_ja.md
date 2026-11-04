# ソフト制約・境界に添わせる経路（2026-11-04）

OptCuts（σ1 = σ2 のソフト制約）で seam を決め、構造格子（seam を 4 枚で閉じる格子、格子の角は立体の縁の曲がりで決める）を平面図の外周に添わせ、K3D → T3D → K2D → T2D まで流すための道具です。各段階の途中経過は診断用ログ（`docs/TRACE_LOG_FORMAT.md`）に残ります。

## 中身

| ファイル | 役割 |
|---|---|
| `optcuts_multistart.py` | seam の始点の探索。OptCuts 自身が最初に調べる境界の頂点（上位 N^0.8 個）を始点の候補にし、topology の変更回数を 1, 2, 4, … と増やしながら良い半分を残す。最後の 1 本は OptCuts が自分で止まるまで回す（`MS_FINAL_TIMEOUT=0`）。 |
| `connect_interior_slits.py` | OptCuts が境界につながらない切れ目（外周のループが 2 本以上）を残したとき、その先端から元の面の境界まで、ほかの seam に触れない最短の辺の道で切り足す。 |
| `resolve_cut_with_optcuts.py` | 切り足した形を OptCuts に入力として渡し、重なりを許さない解き方で平面図を解き直す（外周 1 本の平面図ができる）。 |
| `run_shape.py` | パイプライン本体（S → … → T2D）。保存した OptCuts の平面図（`CHART_PKL`）を使える。K3D は収束まで回し、エネルギー最小の反復を採用する。格子の配置・パネル最適化・K3D・K2D・T2D の各反復を記録する（`ANIM_TRACE=1`）。 |
| `trace_summary.py` | 診断用ログ（`trace.jsonl`）を読みやすい Markdown にまとめる。 |
| `igloo_export_lib.py` | 上の道具が使う小さな関数（歪みと seam の長さの計算など）。 |
| `optcuts_onestring_fork_20261104.patch` | 公式 OptCuts への差分（σ1 = σ2 の項、seam の折れ曲がりと対称性の罰則、枝分かれの禁止、始点の固定、topology の変更回数の上限、各反復の書き出し）。OptCuts 本体はリポジトリに入れず、この差分だけを共有する。 |

`src/onestring_physics/` 側の変更：

* `structured_grid_20261028.py`：seam を 4 枚で閉じる格子（`structured_grid_seam_zipper`）、格子の角を立体の縁の曲がりで決める（`structured_grid_turn_reference="surface"`）、診断用ログ。
* `aligned_grid_20261028.py`：上の格子の妥当性判定。
* `paper_hinge_solver.py`, `hard_collision_solver.py`：T2D の重なり判定を直したもの（ヒンジでつながった隣どうしの接触は重なりとして数えない、許容値はタイルの辺の 0.1%、開き角が −1〜91° を外れた試行は候補から外す）。`hinge_layout_hard_check="strict"` で以前の判定に戻る。
* `run_trace.py`：診断用ログ。

## アプリ（Streamlit）から使う

Omega parameterization mode で「2026-11-04 | OptCuts（σ1/σ2 のソフト制約）→ 構造格子を境界に添わせる …」（内部名 `optcuts_structured_grid_soft_fit_20261104`）を選びます。

* OptCuts は、下の手順でビルドしたフォーク（`third_party/OptCuts_csf_onestring/build2/OptCuts_bin`、または環境変数 `ONESTRING_SOFT_FIT_OPTCUTS_EXECUTABLE`）を、σ1/σ2 の罰則の重み 16・seam の折れ曲がりの罰則 2 で 1 回実行します（始点の探索は行いません。探索するときは下のスクリプトを使います）。フォークが見つからないと警告を出し、σ1/σ2 の項なしで動きます。
* OptCuts が境界につながらない切れ目を残したときは、自動で境界までつなぎ、OptCuts に平面図を解き直させます（`[SOFT-FIT]` のメッセージ）。
* 格子づくりはプリセット `soft_fit_20261104`（2026-10-30 の向きそろえ・形優先・T3D で上面パネルの形を保つ設定＋seam を 4 枚で閉じる＋格子の角を立体の縁の曲がりで決める）です。
* T2D の重なり判定は新しい判定（既定）です。パネル数はアプリの設定どおりです。
* アプリでは K3D の「エネルギー最小の反復を採用する」処理と、各最適化の記録（アニメーション用）は入っていません。これらは下の `run_shape.py` で動きます。

## 準備：OptCuts のフォーク

```bash
cd third_party
cp -r OptCuts_official OptCuts_csf_onestring          # 公式 OptCuts の複製（場所は ONESTRING_OPTCUTS_EXE で変えられる）
cd OptCuts_csf_onestring
patch -p1 < ../../tools/optcuts_soft_fit_20261104/optcuts_onestring_fork_20261104.patch
mkdir -p build2 && cd build2 && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4 OptCuts_bin
```

## 実行（例：バニー）

```bash
T=tools/optcuts_soft_fit_20261104
# 1. seam の始点の探索（ソフト制約。Igloo のソフト制約と同じ設定）
MS_STALL_SECONDS=180 MS_FINAL_TIMEOUT=0 ONESTRING_ANGLE_W=16 ONESTRING_SEAM_BEND_W=2 ONESTRING_SYM_W=1 \
  python3 $T/optcuts_multistart.py SHAPE.obj out/ms_soft
# 2. 最後に「exactly one UV boundary loop ... got 2」で止まったときだけ：切れ目をつないで解き直す
python3 $T/connect_interior_slits.py out/ms_soft SHAPE.obj 16 out/ms_soft_connected
ONESTRING_ANGLE_W=16 ONESTRING_SEAM_BEND_W=2 python3 $T/resolve_cut_with_optcuts.py out/ms_soft_connected SHAPE.obj
# 3. パイプライン（境界に添わせる経路）
NO_SPLIT=1 T2D_UNLIMITED=1 ANIM_TRACE=1 SEAM_ZIPPER=1 TURN_SURFACE=1 \
  CHART_PKL=out/ms_soft_connected/optcuts_chart.pkl python3 $T/run_shape.py SHAPE.obj out/fit_soft
#    （つながなかったときは CHART_PKL=out/ms_soft/optcuts_chart.pkl）
# 4. ログの要約
python3 $T/trace_summary.py out/fit_soft/trace/trace.jsonl > out/fit_soft/trace/summary.md
```

主な環境変数：`PANELS`（パネル数の目安、既定 600）、`THICK_ABS`（タイルの厚さ）、`SEAM_ZIPPER`、`TURN_SURFACE`、`T2D_UNLIMITED`、`ANIM_TRACE`、`ONESTRING_TRACE_DIR`（ログの場所、既定は出力フォルダの `trace/`）。

## 2026-11-04 時点の状況（バニー、ソフト制約）

* OptCuts は歪み 4.068（上限 4.1）、seam 8.32 で止まるが、背中の下に境界につながらない切れ目（長さ 0.59）が 1 本残る。底の縁まで 2 辺切り足して解き直すと、外周 1 本・歪み 4.068・seam 8.38 になる。面積の比の広がりは 2.47 で、2 を超えている。
* 格子づくりで、外形の直交多角形が見つからずに止まる。ログでは、境界の特徴点が 164 個あり、各辺の目標がパネル 0.7〜1.7 枚分しかないため、辺の長さを整数で決める段階が毎回「解なし」になっている。角の向きを決める段階も 8 秒の時間上限で打ち切られている。
* 枝分かれ禁止の設定で内部分割まで禁止すると、OptCuts は歪み 5.2 で次の一手を失う（その変更は取り消してある）。
