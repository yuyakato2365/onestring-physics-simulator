# 2026-10-27 OptCuts → Zhang et al. (2021) Planar Quad M2D モード

Omega parameterization mode: `optcuts_zhang2021_quad_20261027`
表示名: `2026-10-27 | OptCuts → Zhang et al. 2021 Quad M2D (authors' reference code) + OneString seam/3D adapter`

```
S → 公式OptCuts（3Dカットと UV↔3D 三角形対応は変更しない）
  → [OneString] Ω の前処理 h: 境界の単純化（Hausdorff ≤ tol、Seam 節点とその近傍の向きを保持）
               + 全単射 SLIM → Ω₁（tol は候補パラメータ、0 = 前処理なし）
  → [論文] Zhang, Chai, Liu, Fu, CAD 140 (2021) 103084 — 著者の参照実装 Straighten / Segment / QMSim を
           WholePipeline.bat と同じコマンドラインで実行（移植ビルド、アルゴリズム無変更）。
           N_tar は論文の密度入力（l_tar = √(A/N_tar)）としてだけ使い、ε とともに候補パラメータにする
  → [OneString] Seam を hard に閉じられる標本化:
               (a) Q の境界頂点を ∂Ω へ順序保存で対応付け、Seam 節点（先端・分岐・切替点）を頂点に割当て
               (b) 内部点である節点（カット先端・分岐）の 3D 価数 ≥ 3 を局所修復（対角 collapse / 辺回転）
               (c) 各カット枝の両側を同じ 3D 弧長分率で標本化（一致分割、または入れ子分割 = T 頂点）
               (d) 境界辺の分割（双子辺も同時）+ 境界を Q の境界多角形に固定した全単射 SLIM、
                   境界上の点の配置は 3D 弧長比例（両側で同じ物理点が同じパラメータになる）
               (e) M3D のサイズ・形状一様性を目的とする M2D 内部頂点の最適化（反転禁止）
  → Phase 1: 硬い制約（UV 全単射・四角形位相・被覆・Seam hard 閉合可能・M2D/M3D ゲート・ルート連結）
  → Phase 2: 品質スコア E(c)（枚数項なし）で順位付け
  → Phase 3: 1 位から順に K3D（Seam 厳密拘束 + 品質ゲート）→ T3D → K2D → T2D → 最終検証、
             不合格なら次の候補で再実行（OptCuts と候補評価はキャッシュ）
  → Phase 4: （任意）検証を通った候補の最終品質を比較して採用。全滅なら診断付きで明示的に失敗
```

論文の段階と OneString 独自の段階はメトリクス名で分離しています
（`zhang2021_*` = 論文側、`zhang2021_onestring_*` = OneString 拡張）。
Gmsh や切り抜き正方格子への fallback はありません。失敗はすべて明示的なエラーコード
（`ZHANG2021_*`）で停止します。2026-10-26 の Gmsh プロトタイプ
（`optcuts_planar_quad_20261026`）は変更せず比較用に残しています。

## 1. 論文・補足資料・公開コードの調査

| 項目 | 調査結果 |
|---|---|
| 入力 | 自己交差のない 2D 三角形メッシュ M（境界は単純閉曲線、穴あり可）。参照実装は OBJ の `vt` を領域、`v` を（3D でもよい）歪み評価用の形状として読む |
| 目標 | ①パッチ数が少ない ②反転なし ③内角≈π/2 ④辺長≈l_tar ⑤ Q と M の境界の両側 Hausdorff 距離 < ε かつ境界特徴点を保存 |
| 既定値 | ε = 0.01·d_bb、l_tar = sqrt(area(M)/4000)、l_collapse = 0.05·d_bb、α_th = 0.5 |
| 境界特徴 | 境界辺クラスタリング（θ_normal = π/9、l_th = 0.002·l_bound、θ_th = π/9）。特徴点は内角 > 25π/18 または < 11π/18 の境界頂点 |
| 粗いレイアウト | ①全単射・低歪みの境界直線化（scaffold、軸平行でない目標方向）→ ② cross field に沿った分割線で長方形状の領域へ分割（ECV、[π/3, 2π/3]、穴は One-ECV で先に解消）→ ③ 5 種のテンプレートで領域ごとに四角形化 → ④ T-junction 解消・poly-chord 細分 → ⑤ 幅制限付き poly-chord collapse、SD 幾何最適化、split/collapse で l_tar に近づける |
| パネル数の制御 | 目標辺長 l_tar（= N_tar）のみ。論文本文に「非常に大きい l_tar（非常に小さい N_tar）は Hausdorff 制約のため通常満たされない」と明記 |
| 非凸・穴 | 直線化と分割線で対応（凹頂点を ECV として解消）。穴は分割線で先に連結 |
| 公開コード | Fu 研究室ページの Code（rec.ustc.edu.cn share）に Code_new.zip（ソース）と ExeProgram.zip（Windows 実行ファイル）。ライセンス表記なし |
| 統合可否 | Windows/MSVC 専用（Qt GUI、PARDISO 6.0（個人ライセンス必須）、MKL、MOSEK）。そのままでは macOS/Linux で使えないため、**アルゴリズムを書き換えない移植ビルド**を作成（下記 2.）|

論文の処理と参照実装の対応:

| 論文の節 | 参照実装 | OneString のビルド |
|---|---|---|
| 3.2.1 境界直線化 | Straighten.exe | `z21_straighten` |
| 3.2.2 分割、3.3.1 テンプレート、3.3.2 初期メッシュ | Segment.exe | `z21_segment` |
| 3.3.3 最適化（collapse、SD、Hausdorff 制御） | QMSim.exe | `z21_qmsim` |

## 2. 参照実装の移植と検証

`third_party_patches/zhang2021_quad_coarse_layout/README.md` に変更の全カテゴリと理由を記載しています。
要点:

* 著者コードは**リポジトリに含めません**（ライセンス表記がないため）。
  `scripts/setup_zhang2021_quad_official.py` が著者の共有から取得し、SHA-256 を検証してパッチを適用・ビルドします。
* 変更は構文・ビルド・プラットフォーム互換のみ（`\` の include、`long double(x)`、`void main`、
  MSVC の一時オブジェクト参照、PARDISO/MKL → Eigen SparseLU（同一インターフェース）、MOSEK は到達しない経路のスタブ、
  Qt GUI の除去と batch 経路の再現、OBJ 読み込みの CRLF、未初期化ポインタの UB 修正）。
  MSVC の数値意味を保つため `abs(double)`、FMA 無効化、OpenMP の有無を著者の Release 設定に合わせています。
* 検証: 著者の Windows 実行ファイルを Wine で実行（PARDISO は個人ライセンスのため同 API の Eigen 実装 DLL で代替）し、
  5 つのサンプルで比較。著者の実行ファイル自身が入力 1e-12 の摂動で同程度以上に結果が変わる（model_15: 3380→6874 quads）
  ため、ビット一致は原理的に不可能で、移植版は**原版自身の数値感度の範囲内**で一致しています。
* コンパイラ（GCC と Clang/libc++）でも粗い設定の結果が変わることがあります
  （Bunny 由来の単体 OptCuts チャート, ε=0.05: GCC 206 / Clang 144 quads。アプリが渡したチャートでは両者とも 146）。Mac での数値は結果編と異なり得ます。

セットアップ:

```bash
# macOS (Apple Silicon)。eigen は 3 系（eigen@3）が必要。Homebrew の libomp / eigen@3 の場所と
# Apple Clang 17（libc++）向けの互換フラグはスクリプトが自動で付けます（--no-macos-flags で無効化）
brew install cmake eigen@3 cgal libomp
python3 scripts/setup_zhang2021_quad_official.py
# アプリ用の仮想環境は Python 3.10 以上（例: /opt/homebrew/bin/python3.12 -m venv .venv）
# Ubuntu
sudo apt install cmake g++ libeigen3-dev libcgal-dev
python3 scripts/setup_zhang2021_quad_official.py
# 自動ダウンロードできない場合は共有ページから Code_new.zip を取得して
python3 scripts/setup_zhang2021_quad_official.py --archive ~/Downloads/Code_new.zip
python3 -m streamlit run app_optcuts_20260923_extrusion_aware.py
```

（Paper-oriented computation mode を OFF にすると Omega parameterization mode に表示されます。）

## 3. OneString 拡張（論文外）

### ① UV/3D 対応
Zhang の Q の境界は ∂Ω 上にありません（ε 以内で自由に動く）。そこで Ω 自体を、境界が Q の境界多角形に一致するよう
全単射に再パラメータ化します。境界の対応は順序保存の動的計画法（最寄り射影はスリット付近で順序が崩れるため）で決め、
Seam 節点をQの頂点へ割当て、必要な境界点で境界辺（とカットの双子辺）を分割し、内部は SLIM で解きます。
結果として最終チャートの領域 = Q の領域（面積誤差 0）、かつチャートは S 全体への全単射なので、
**M3D は S を隙間なく・はみ出しなく覆います**（切り抜き方式にはない性質）。代償としてチャートの SD 歪みが増えます（報告値）。

### ② パネル品質（M2D / M3D / K3D）— 修正 B
`zhang2021_panel_quality.py`。すべてスケール非依存です（長さは比か、メッシュ自身の幾何平均辺長に対する対数でだけ使う）。

| 指標 | 定義 |
|---|---|
| E_size | 全辺の mean log²(l_e / l_ref)、l_ref = 辺長の幾何平均 |
| 辺長比 r_q | 各パネルの最長辺 / 最短辺（scaled Jacobian では細長い長方形を検出できないため別に測る） |
| E_angle | 全コーナーの mean cos²θ |
| 面積 CV | パネル面積の標準偏差 / 平均 |
| サイズ比 | 最大/最小パネル（√面積）、p95/p5 |
| 最悪パネル | 各基準の最悪パネル番号、パーセンタイル、パネル偏差の smooth max（log-sum-exp） |
| 退化 / 反転・非凸 | 0 であること |

ゲートは段ごとに独立です（`zhang2021_m2d_gate` / `zhang2021_m3d_gate` / `zhang2021_k3d_gate`、`name=value,...`、空欄は既定値）。
既定値はユーザー設定であって「公式」な値ではありません。

| 段 | 既定のゲート |
|---|---|
| M2D（UV） | 妥当性のみ: 退化・反転なし、角 3°〜177°、辺長比 ≤ 25（カット先端付近では、3D で良いパネルにするために UV 側を細い楔形にする必要がある） |
| M3D | 角 30°〜150°、辺長比 ≤ 3、サイズ比 ≤ 4、p95/p5 ≤ 2.5、E_size ≤ 0.15、E_angle ≤ 0.15、面積 CV ≤ 0.75 |
| K3D | M3D と同じ + 非平面性 ≤ 1e-3 × 平均辺長 |

### ③ Seam の hard 閉合 — 修正 C
* Seam 節点（先端・分岐・切替点）は必ず四角形の頂点に割り当てます（割り当てられなければ失敗）。
* **節点の 3D 価数**: カット先端と分岐は S の内部点なので、全 UV コピーの価数の和が 3 以上必要です
  （1 枚だと折り畳み、2 枚だと 180° の角になる）。不足している場合は局所修復を行います
  （`zhang2021_seam_alignment.repair_node_valences`）。
  - 対角 collapse: 節点の四角形の対角を節点へ統合
  - 辺回転: 節点に面した辺を回転し、節点の周りに 1 枚足す

  どちらも境界頂点を変えず、ダブレットを作らないことを確認してから適用します。修復できない場合は候補を不採用にします。
* **標本化**（`align_branches`）: カット枝ごとに、両側の頂点を 3D 弧長分率で順序保存のまま対応付けます（DP）。
  - 個数が等しい場合は一致分割（頂点 ↔ 頂点）です。
  - 等しくない場合は、少ない側を多い側の部分集合に合わせる入れ子分割です。余った頂点は、相手側の辺上の固定パラメータに置く T 頂点（頂点 ↔ 辺パラメータ）になります。
  - 交互配置（枝全体を直線にしないと閉じられない配置）は作りません。
* チャートの境界点は四角形の各辺に沿って 3D 弧長に比例して配置します。両側で同じ物理点が同じパラメータになるので、
  `source_cut_affine_rows` のすべての行が、頂点の等式と T 頂点の等式から導かれます。
* 判定: `discover_exact_seam_topology` は、本モードに限り入れ子分割を `hard` と判定します。この判定は
  `zhang2021_seam_reduction.nested_partition_audit` で全行を明示的に消去できた場合だけです（他モードの判定は不変）。
  soft は採用不可で、M2D の段で `ZHANG2021_SEAM_NOT_HARD_CLOSABLE` を出して停止します。
* 真の開境界（S の穴）は拘束しません（`source_open_grid_edges`）。
* poly-chord 挿入（`zhang2021_seam_matching`）は任意で、既定は OFF です。入れ子分割で閉じられるため、四角形の追加は必須ではありません。

### ④ 3D パネル品質最適化 — 修正 B / E
`zhang2021_panel_optimizer.py`。最終チャートの中で M2D の内部頂点を動かし、厳密リフトした M3D について
次の E を最小化します（L-BFGS）。

E = mean b_q + α·LSE(b)

b_q は角の項 cos²θ、サイズの項 log²(l/L)、パネル内アスペクトの項、角の和の項 (Σθ − 2π)² の和です。
L は開始時の M3D 辺長の幾何平均で、枚数目標ではありません。

* 境界頂点（Seam 標本）は固定するので、Seam の対応は崩れません。
* UV の四角形が単純かつ正向き（反転なし）であることを障壁で保証します。価数修復の直後などで絡んでいる場合は、先に untangle します。
* 位相は変えません。

### ⑤ 下流との接続と K3D
`UnstructuredQuadGrid` で既存の K3D/T3D/K2D/T2D に接続します（`zhang2021_downstream_route`）。

| ルート | K3D | K2D / T2D | Seam |
|---|---|---|---|
| `paper_local_global_0918`（既定） | Eq.(1) local/global（09-18 Grid と同一関数）。`seam_reduction` で x = E z とし、全反復と平面化 polish を z について解く | Eq.(5) + Sec. 4.3 交互連結、論文 Dual Hinge（硬い非貫通） | 共有変数と T 頂点代入による厳密閉合 |
| `optcuts_test2` | AL K3D（hard Seam の射影子）+ 既存の閉合監査 | 運動学的 K2D、従来の Dual Hinge | 同上（射影子） |

* 09-18 ルートの関数（`optimize_paper_local_global_k3d`、`minimum_displacement_planarity_polish`、`_solve_constraints`）には、
  任意引数 `seam_reduction` / `reduction` を追加しました。既定は None で、他モードの挙動は同一です。
* どちらのルートでも、K3D 直後に K3D ゲート（`zhang2021_verification.k3d_gate`）を通します。品質ゲートと Seam の hard 閉合
  （残差 / 平均辺長 ≤ `zhang2021_seam_closure_tolerance`）を確認し、失敗したら `ZHANG2021_K3D_GATE_FAILED` で停止します。
* 09-18 ルートの交互連結には、内部頂点の価数がすべて偶数であることが必要です（Phase 1 の C7 で事前に判定）。
* 初期 Ω CSF Split は既定 OFF のままです（下記）。
* **暫定変更（2026-10-27）: Zhang2021 モードの初期 Ω CSF Split は既定で OFF です。** 旧方式の固定UV/完全格子線に基づく Split を、Zhang の不規則四角形 M2D にそのまま適用しないためです。OptCuts のカット（Seam）、Zhang の Quad Meshing、境界対応・Seam のためのメッシュ辺分割は **無効化しません**。CSF 値は診断用に記録しますが、上限の達成や製造可能性は保証しません。
* 再有効化は `PipelineParameters(zhang2021_initial_csf_split_enabled=True)`（かつ `enable_csf_splits=True`）で明示的に行えます。現在のアプリUIでは `False` に固定しています。
* OptCuts のマルチスタートで付く代替チャートによる差し替えは、本モードでは無効化します（`zhang2021_optcuts_split_count_candidates_dropped`）。

### ⑥ 前処理と候補の段階的選定 — 修正 A / D / E
呼び出し順は `with_parameterization` → `plan_zhang2021_m2d_coordinated` → `plan_zhang2021_m2d`
（`optcuts_zhang2021_quad_20261027.py`）。選定の部品は `zhang2021_candidate_selection.py` にあります。

1. 候補 tol ごとに `precondition_omega` を実行します。
   - Douglas–Peucker（Hausdorff ≤ tol）
   - Seam 節点の両側を `zhang2021_precondition_node_keep_ratio` × 対角の長さだけ残し、節点での境界の向きを保持します。
     旧版は先端の反射角（例: 245°）を鋭い凸角（例: 96°）に潰しており、先端周りに 3 枚以上置けなくなっていました。
   - 全単射 SLIM と SD 比の上限を確認し、超えた場合は理由付きで不採用にします。
2. (tol × ε × 密度 N_tar) のすべての候補で Zhang を実行します（並列、`zhang2021_parallel_workers`、0 = 自動）。
   密度は `zhang2021_target_quads` × `zhang2021_candidate_density_factors`（既定 1,2,4）です。
   `zhang2021_target_edge_length` > 0 なら、N_tar = 面積(S) / l² で決めます。
3. 各候補について `plan_zhang2021_m2d` を実行します（③④ と M2D/M3D ゲート、Seam の preflight）。
4. **Phase 1**（`feasibility`）: 次の制約を一つでも満たさない候補は不採用です。
   - C1: UV 全単射
   - C2: 四角形の位相
   - C3: S の被覆
   - C4: Seam を hard に閉合できる
   - C6: M2D と M3D のゲート
   - C7: ルートの連結条件
5. **Phase 2**（`candidate_score`）: E(c) = w_size·E_size + w_shape·(E_angle + E_aspect) + w_area·CV² + w_dist·(SD 比 − 1)
   + w_fit·(境界 Hausdorff / 対角) + w_worst·smooth max を、M3D の値に w_m2d × M2D の同じ項を加えて計算します。
   **枚数の項はありません**（`count=` を指定するとエラー）。重みは `zhang2021_selection_weights` で指定します。
6. **Phase 3**（`zhang2021_verification`）: 1 位から順にパイプライン全体を実行します。K3D ゲートで止まった場合や、
   最終検証で不合格の場合は次の候補を試します（上限 `zhang2021_verification_attempts`、既定 3）。2 回目以降は OptCuts と候補評価をキャッシュから再利用します。最終検証の内容:
   - T3D: Seam が hard で、隙間が許容内であること。パネル衝突 ≤ `zhang2021_t3d_max_panel_collisions`（既定 0）
   - K2D: パネル数が K3D と一致すること
   - T2D: 2D 衝突 0、3D 立体衝突 0、`hard_collision_status = verified_feasible`、
     ヒンジ接続誤差 / 平均辺長 ≤ `zhang2021_t2d_max_hinge_error_ratio`（既定 0.05）
   - `fabrication_feasible` は既存の 09-18 Grid の結果でも False なので、ゲートには使わず値だけを記録します。
7. **Phase 4**: `zhang2021_compare_verified_candidates` を ON にすると、検証を通った候補をすべて実行し、K3D 品質とヒンジ誤差で比較して採用します。
8. 全滅した場合（Phase 1 で 0 件、または Phase 3 で全試行が不合格）は、`ZHANG2021_NO_FEASIBLE_CANDIDATE` /
   `ZHANG2021_NO_CANDIDATE_PASSED_VERIFICATION` で明示的に失敗します。メッセージには候補ごとの却下理由が入り、診断フォルダも出力します。

**診断出力**（`zhang2021_diagnostics.py`）は `output/zhang2021_diagnostics/<時刻>/` に書き出します。
- `candidates.json` / `candidates.txt`（全候補の状態、スコアの内訳、却下理由）
- `best_rejected_candidate_m3d.png`（違反パネルを赤で表示）
- `best_rejected_candidate_quality.json`
- `verification.json`
- `k3d_bad_panels.png`

成功時も、候補表を `zhang2021_coordinated_candidates`、検証ログを `zhang2021_verification_log` としてメトリクスに残します。

## 4. 比較結果

`docs/optcuts_zhang2021_quad_20261027_results_ja.md` は、**修正前（35 枚誘導・soft Seam を許した版）** の計測結果です。
今回の修正版での確認はドームの通し実行だけです（同ファイル §9）。Bunny での確認はユーザー側で行う前提です。

## 5. テスト

```bash
python -m pytest -q tests/test_zhang2021_quad_20261027.py tests/test_zhang2021_quality_seam_20261027.py
```
参照プログラム未ビルドの環境では、それを必要とするテストのみ skip されます。

## 6. 2026-10-27 修正指導書への対応表

| 要件 | 実装（関数・制約・評価項） |
|---|---|
| A: 35 枚誘導の撤廃 | `DEFAULT_TARGET_QUADS` を廃止して `DEFAULT_DENSITY_N_TAR`（密度入力）に変更。旧 `SELECTION_WEIGHTS["count"]` を削除。`candidate_densities`（複数密度、l_target からの換算）。UI の表記を「density input, NOT a panel-count goal」に変更 |
| B: サイズ・形状の一様性 | `panel_quality`（E_size、辺長比、E_angle、面積 CV、最悪パネル、smooth max）。M2D/M3D/K3D の独立ゲート。`optimize_panel_quality`（M3D の一様化） |
| C: Seam の 3D 閉合（全ルート） | `align_branches`（一致・入れ子分割）、`repair_node_valences`、3D 弧長による境界配置、`nested_partition_audit`、`build_seam_reduction` + `optimize_paper_local_global_k3d(seam_reduction=…)`、test2 は既存 AL の hard 射影子、`k3d_gate`（soft は失敗）、T3D の監査 |
| D: 段階的選定 | `feasibility`（C1〜C4、C6、C7）、`candidate_score`（枚数項なし）、`install_zhang2021_verification_loop`（Phase 3/4） |
| E: 3D 品質と閉合可能性のフィードバック | M3D ゲートと Seam preflight を Phase 1 に、K3D/T2D の結果を Phase 3 の採否に反映。前処理（節点の向きを保持）、価数修復、内部頂点最適化 |
| §9 禁止事項 | S は変更しない（分割点も S の辺上）。OptCuts のカット位相は不変。soft の採用、警告だけでの通過、Grid/Gmsh への fallback はいずれも無し。初期 CSF Split は OFF のまま。他モードの挙動は不変（全テスト比較で新規失敗 0） |
