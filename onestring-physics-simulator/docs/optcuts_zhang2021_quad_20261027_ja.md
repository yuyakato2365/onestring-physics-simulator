# 2026-10-27 OptCuts → Zhang et al. (2021) Planar Quad M2D モード

Omega parameterization mode: `optcuts_zhang2021_quad_20261027`
表示名: `2026-10-27 | OptCuts → Zhang et al. 2021 Quad M2D (authors' reference code) + OneString seam/3D adapter`

```
S → 公式OptCuts（3Dカットと UV↔3D 三角形対応は変更しない）
  → [OneString拡張 ⑤] Ω の前処理 h: 境界の単純化（Hausdorff ≤ tol、Seam 節点固定）+ 全単射 SLIM → Ω₁
  → [論文] Zhang, Chai, Liu, Fu, "Quad meshing with coarse layouts for planar domains",
           CAD 140 (2021) 103084 — 著者の参照実装 Straighten / Segment / QMSim を
           著者の WholePipeline.bat と同じコマンドラインで実行（移植ビルド、アルゴリズム無変更）
  → [OneString拡張] (a) 境界対応: Q の境界頂点を ∂Ω（カット・開境界）上の点へ順序保存で対応付け
                   (b) Seam 節点（カット先端・分岐・カット/開境界の切替点）を Q の頂点に割当て
                   (c) 任意: カット両側の頂点数を poly-chord 挿入でそろえる（hard seam 化）
                   (d) 境界辺の分割（カットの双子辺も同時に分割し3Dメッシュを conforming に保つ）
                   (e) 境界を Q の境界多角形に固定した全単射 SLIM（対称ディリクレ）で Ω を再パラメータ化
  → M2D = 論文の Q（(c) を適用した場合のみ chord が追加される）
  → M3D = 再パラメータ化 Ω 上の厳密な三角形包含 + 重心座標リフト
  → 既存 K3D / T3D / K2D / T2D
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
# macOS (Apple Silicon)
brew install cmake eigen cgal libomp
python3 scripts/setup_zhang2021_quad_official.py --cmake-arg=-DOpenMP_ROOT=$(brew --prefix libomp)
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

### ② M3D 品質
M2D/M3D の最小・最大角、辺長比、面積比、scaled Jacobian、平面性を出力し、最悪面のインデックスも記録します。

### ③ Seam 整合性
* 節点（先端・分岐・切替）は必ず Q の頂点に割当て（先端は必須。辺の内部に先端があると 3D で辺が折れる）。
* カットの両側で Q の頂点数が等しければ 3D の位置をそろえて hard seam（既存の厳密 Seam 閉合が使える）。
* 異なる場合は soft（既存の affine 制約による近似閉合、隙間が残る）。
* 任意の poly-chord 挿入（`zhang2021_seam_matching`）: 全カットの頂点数（スリットは偶奇＋先端割当て）を
  Dijkstra で最小追加 quad 数になるよう合わせる。全カットが整合し、かつ各カット先端の頂点が 3 枚以上の quad に属する場合のみ採用します
  （2 枚以下だと hard 閉合で先端のパネルが 3D で約 180° に開くか潰れる）。そうでなければ論文の Q のまま soft で続行し、理由を
  `zhang2021_onestring_seam_matching_rejection_reason` に記録します。
* 節点の 3D 価数（各 UV コピーの quad 数の和、内部点の理想 ≈ 4、開境界上 ≈ 2）を報告します。
  先端の価数 2 は 3D で約 180° の角、1 は折り畳みを意味します。

### ④ 下流との接続
`UnstructuredQuadGrid`（2026-10-26 と共通）で既存の K3D/T3D/K2D/T2D に接続します。既存ソルバーは書き換えず、
どの既存ルートへ渡すかだけを選べます（`zhang2021_downstream_route`、UI「OneString downstream route for the Zhang M2D」）。

| ルート | K3D | K2D | T2D | Seam |
|---|---|---|---|---|
| `optcuts_test2`（既定） | AL K3D（optcuts_test2） | 運動学的 K2D | 従来の Dual Hinge | 認証済み Seam 閉合（hard/soft） |
| `paper_local_global_0918` | Eq.(1) local/global（09-18 Grid ルートと同一関数） | Eq.(5) local/global + Sec. 4.3 交互コーナー連結 | 論文 Dual Hinge（硬い非貫通） | 閉合しない（09-18 Grid ルートと同じ。残差は `zhang2021_route_seam_gap_*` に計測値として出力） |

* K3D の Seam 閉合は `optcuts_seam_exact_closure` の条件に本モードのフラグを追加しただけで、ソルバーは変更していません。
* `paper_local_global_0918` の交互コーナー連結は四角形の隣接グラフが二部グラフ（= 全内部頂点の価数が偶数）であることを要します。
  Zhang の粗いレイアウトには価数 3/5 の特異頂点があるため、その場合は
  `ZHANG2021_PAPER_LG_LINKAGE_NON_BIPARTITE`（奇数価数の内部頂点数を含む）で明示的に停止し、別ルートへの自動切替はしません。
* **暫定変更（2026-10-27）: Zhang2021 モードの初期 Ω CSF Split は既定で OFF です。** 旧方式の固定UV/完全格子線に基づく Split を、Zhang の不規則四角形 M2D にそのまま適用しないためです。OptCuts が S を円板状チャートへ展開するためのカット（Seam）、Zhang の Quad Meshing、境界対応・Seam のためのメッシュ辺分割は **無効化しません**。M2D の四角形は初期 CSF Split による切断を受けず、通常の M3D/K3D/T3D/K2D/T2D へ渡されます。CSF 値は診断用に記録しますが上限達成・製造可能性を保証しません。
* 再有効化は `PipelineParameters(zhang2021_initial_csf_split_enabled=True)`（かつ `enable_csf_splits=True`）で明示的に行えます。現在のアプリUIでは事故防止のため `False` に固定し、「初期 Ω CSF Split OFF」と表示します。再有効化すると、既存の `csf_split_bound_certified=False` の扱いにより `ZHANG2021_CSF_SPLIT_UNCERTIFIED` が再発し得ます。旧 Split の修復ではなく、初期 Split のバイパスです。
* 再有効化されたときのみ Simple Split ラッパーが1回の初期 CSF Split を実行します。旧方式の Split 後チェックは従来のまま残します（2026-10-26 Gmsh モードは未変更）。
* OptCuts のマルチスタート（閉じた種数0の入力）で付く代替チャートは、Split 段で Split 数により差し替えられることがあります。
  Zhang の M2D と再パラメータ化チャートは選ばれた1つの OptCuts 結果に対して計算されるため、本モードではこの差し替えを無効化し、
  `zhang2021_optcuts_split_count_candidates_dropped` に記録します。

### ⑤ Quad Meshing 前の Ω の前処理と候補探索（2026-10-26 設計 Step 2〜6）
Zhang の Straighten は分割線を決めるための作業用の座標系を作るだけで、Ω を置き換えません。そこで設計どおり、Zhang の**前に** Ω 自体を変形する段を置きました
（`zhang2021_precondition_enabled`、既定 ON）。

呼び出し順（`with_parameterization` → `plan_zhang2021_m2d_coordinated`、`optcuts_zhang2021_quad_20261027.py`）:

1. 候補 tol ごとに `precondition_omega`（`zhang2021_onestring_precondition.py`）
   1. `split_boundary_chords`: 境界頂点どうしを結ぶ内部辺を中点で分割（UV と 3D の同じ辺上。直線化で潰れる三角形を防ぐ）
   2. `build_source_boundary`: ∂Ω を Seam 節点（カット先端・分岐・切替点）で区切る
   3. `simplified_boundary`: 区間ごとに Douglas–Peucker（許容 tol × bbox 対角）。節点は固定の角、tol を超えて曲がる点だけを角として残し、
      他の境界頂点は順序を保って直線区間上に弧長比で再配置（E_line = E_smooth = 0、E_deviation は Hausdorff ≤ tol のハード制約）
   4. 目標多角形の自己交差と両側 Hausdorff 距離を検査
   5. `progressive_boundary_slim`（`zhang2021_onestring_reparam.py`、既存の後合わせと共通化）: 反転なしで境界を目標へ動かし、内部は S の 3D 三角形に対する対称ディリクレ SLIM
   6. SD 平均比 ≤ `zhang2021_precondition_sd_mean_ratio_max`（既定 1.08）、p95 比 ≤ `..._sd_p95_ratio_max`（既定 1.30）でなければ理由付きで不採用
2. 採用された Ω₁ ごと、候補 ε ごとに `plan_zhang2021_m2d`（Zhang → 既存アダプタ → M3D）
3. 各候補を採点（`SELECTION_WEIGHTS`）: |ln(実数/目標)|、奇数価数の内部頂点数、価数不足の Seam 節点数、M3D 最小角、J_min、最終チャートの SD 増加。
   09-18 ルートでは奇数価数の内部頂点がある候補は不適格。全候補と内訳を `zhang2021_coordinated_candidates` に記録し、最良を採用
4. 境界誤差や SD は**元の** OptCuts チャート Ω₀ に対して報告（`plan.source` = Ω₀）

3D 位置は動かさない（分割点も S の辺上）ので、S との全単射対応と OptCuts のカット位相は保たれます。
h と Q の連続的な同時最適化や Quad 頂点の再配置は行っていません（候補探索による協調）。

## 4. 比較結果

`docs/optcuts_zhang2021_quad_20261027_results_ja.md` に、単純な平面領域・Bunny（代替メッシュ）・ドームでの
従来 Grid / Gmsh 10-26 / Zhang の比較（実パネル数、被覆率、境界誤差、M2D/M3D/K3D 品質、SD 歪み、Seam、T2D、実行時間）をまとめています。
要点（前処理なし）: Bunny 由来 Ω では 30/35 枚は得られず（117 枚以上）、Seam は soft、09-18 ルートの交互連結は特異頂点のため構成できません。
要点（⑤ の前処理あり、結果編 §8）: Bunny で 63 枚・奇数価数の内部頂点 0 となり、09-18 ルートで T2D まで成功（Split 既定値）。
Seam の閉合と K3D の最小角は未解決です。ドームでは前処理の有無にかかわらず T2D まで成功します。

## 5. テスト

```bash
python -m pytest -q tests/test_zhang2021_quad_20261027.py
```
参照プログラム未ビルドの環境では、それを必要とするテストのみ skip されます。
