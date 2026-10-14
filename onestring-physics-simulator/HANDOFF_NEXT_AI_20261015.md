# OneString 引き継ぎ: 矩形BFF再計算、Ωの厳密Grid被覆、Split境界一致、T2D Dual Hinge
更新: 2026-10-15。次のAIは**初めにこのファイルとCI結果を確認**し、成功が未確認なものを成功と報告しないこと。

## 目的・ユーザーの最新要求

ユーザーはZaman et al. (2025) "One String to Pull Them All" の再現を目指している。望むフロー:

1. 入力三角形メッシュ **S** から、入力の境界に適合する四隅を選んで公式**矩形境界BFF**により**Ω**を計算。
2. 面積ヤコビアンから領域CSF（本実験では sqrt(max J / min J)）を計算。**CSF ≤ 2ならSplit 0本**。
3. 超えるときΩで高ガウス曲率点の近くを通る**Grid線でSplit**。同じ2D線で3DメッシュSを正確に切る。
4. Sの両領域を独立に公式**矩形境界BFF**で再計算。切断線の物理的3D両端は**両矩形の隣り合う2隅として明示指定**。
5. 新しい各ΩでCSFを評価し、違反領域だけ再び格子線で切り、上限または候補なしまで反復。
6. 両側が実際につながるSplit境界では**Gridパネルの対応が厳密に一致**するような共通格子を生成し、**Ωの外周まで欠けなく覆う**。K3D/T3D/K2D/T2D Top Hinge/T2D Dual Hingeまで元のOneStringソルバーで求める。
7. 画像で2つのT2D領域が離れている。ユーザーは「境界でパネルが一致するように」「矩形ΩならGridで削られる部分が出ないように、ピッタリ合わせて」と要求した。

### 数学上の重要な留意点

**矩形の形状だけでは、同一間隔 Δ の正方形Gridで外周を隙間なく覆える保証はない。**
矩形の幅 W・高さ H に対し W/Δ と H/Δ が同時に整数である必要がある。
**今回の新しい実装は、外周の端数を非正方形の「境界長方形セル」で吸収する**。UVを異方的に後伸縮してBFF等角性を壊さない。ただし**元論文の正方形タイル条件や製造可能性を自動証明するものではない**。次のAIはこれを明記し、正方形制約と境界完全被覆のトレードオフを検証すること。

## リポジトリ・安全なMac環境

GitHub: https://github.com/yuyakato2365/onestring-physics-simulator
実験ブランチ: \`agent/postcut-rebff-mac-20261014\`
**元の研究コード**: \`~/Documents/OneString-large-steps-mesh-conditioning/onestring-physics-simulator\` に未コミット変更がある。**絶対に git reset --hard / git clean -fd しない**。
独立実験worktree: \`~/Documents/onestring-postcut-rebff/onestring-physics-simulator\`（Git detached HEAD）。
更新:
\`\`\`bash
cd ~/Documents/onestring-postcut-rebff/onestring-physics-simulator
git status --short
git fetch origin
git switch --detach origin/agent/postcut-rebff-mac-20261014
source .venv-postcut-rebff/bin/activate
python3 -m streamlit run app.py
\`\`\`
別の元実験画面は \`python3 -m streamlit run app_postcut_rebff.py\`。
元 \`app.py\` の "Paper-oriented computation mode" をON、"Split後の矩形BFF再計算 → T2D Dual Hinge（実験）"をON、必要時は「CSF>2でもT2D試験計算」をON。既存のOptCuts対応モードは保持。

## 主要実装ファイル

- \`src/onestring_physics/reference_bff.py\`: 官方BFF CLI, \`rectangle_corner_positions\` による3D物理境界4隅指定、矩形検証。
- \`tools/bff_rectangle/bff-rect-command-line.cpp\`: 明示した3D矩形隅（\`--corner-xyz\`）、MeshIO内正規化座標の逆変換。
- \`src/onestring_physics/bff_rectangle_selection.py\`: 初期S境界に適合する矩形4隅をBFFの複数実行で選択（境界フィット・縦横比・CSF）。
- \`app_postcut_rebff.py\`: 初期S→Ωから反復計算、報告とUI。
- \`postcut_rebff_iterative.py\`: CSF閾値判定とGrid Split。新しい子領域矩形BFFには**Sの切断端点を矩形の2隅として指定**。
- \`postcut_rebff_seam_registration.py\`: Sの物理Split線の一致点を抽出、再BFF矩形のUV島を相似変換で共有格子へ位置合わせ。T2Dでは後段の剛体SE(2)配置合わせ。**物理的に同じ境界点を偽造しない**。
- **\`postcut_rebff_exact_rectangle_grid.py\`（今回追加）**: 連結する矩形Ωの四辺の座標を、グローバルGridの**必須線**として挿入。端数境界セルを長方形にし、Gridの一部の正方形を単に落とす従来の挙動を排除。分割各側の物理シーム上の**グリッド辺のUV端点を共通に**。面積=矩形面積を検査。境界の端数セルが非正方形になる点に注意。
- **\`postcut_rebff_full_pipeline.py\`**: \`_assemble_patches\` で矩形BFFなら新しい厳密被覆Grid生成器を呼び、ソースS各領域をひとつの対応付きM2D/M3Dへ統合し、既存パイプラインのK3D→T3D→K2D→T2Dを実行。
- \`postcut_rebff_integrated.py\`: 元HeightFieldを正しく3D Sに変換し、アプリのフルパイプラインに接続。
- \`app_runtime_base.py\`: 元アプリUIの実験トグル、CSFと失敗候補・T2D表示、Split Map。
- \`tests/test_postcut_rebff_exact_rectangle_grid.py\`: 真の長方形（端数あり）、2隣接領域、境界Grid辺の端点一致、面積100%、非矩形拒否などの回帰テスト。
- \`tests/test_postcut_rebff_seam_registration.py\`: Split端点指定・矩形BFFの共有シーム・T2D剛体対応のテスト。
- CI \`.github/workflows/postcut-rebff-mac-smoke.yml\`: pytest、BFF native build、Lilium実行。

## 既知の過去の状況（今回の最新結果とは混同しない）

- 以前の**自由境界**のSplit後BFFではLilium（libigl OBJ）でCSF約1.566/1.242、158枚のT2D Dual Hingeまで完走したことがある。ただしこれは**矩形再BFFの達成を意味しない**。
- **矩形再BFF**ではCSF約7109、Split 1本、2領域、追加候補全棄却 \`all_candidate_cuts_failed\` がユーザーMacに表示された。元論文への忠実な再現結果とは扱えない。
- 過去の「Split後に独立に格子化→位置合わせするだけ」のT2Dでは、13個の対応アンカーと、典型的タイル辺長の約76.5%の残差を観測。**T2D境界完全一致ではなかった**。
- 今回は共通の物理Split境界端点を両矩形BFFの隅として明示し、共通UV軸・必須境界Gridで再タイル化。**T2Dソルバー最終結果で一致することは改めて検証が必要**。
- CSFの定義は本実験 \`sqrt(max(A3D/A2D)/min(A3D/A2D))\`。面積ベースの三角形平均的尺度であり、最大特異値や元論文の物理最大伸縮と同一視しない。

## 今回の技術的変更と正しさの境界

- 新Gridは**分割後の矩形Ω**に対し、完全包含の正方形だけを残す \`_build_reference_m2d\` の代わりに使用する。グローバルx/y線の集合に各矩形の上下左右を必ず含め、各パッチで共有格子からすべてのセルを取り出す。境界を削らない。
- 座標一致は \`matching_seam_edges\` が**同じ2D線分の完全一致**を検査し、各パッチの面積 \`area_fraction_covered=1\` が必要。
- T2Dのヒンジソルバーはパネルを動かすため、**M2D境界一致だけではT2D境界一致が自動で保証されない**。必要なら既存のDual Hinge制約にSplitシーム対応を組み込み、残差上限を厳格にテストする。T2D全辺を重ねることが論文の物理的Split意図に沿うかも確認する。
- 当面 \`split_seam_matching_certified: False\` を削除・Trueに変更してはいけない。M2D側とT2D側の検証を別々に報告。

## テストと次のAIが優先すべき作業

1. **現在のGitHub Actions結果を確認**。失敗したらジョブログ末尾を取り、実際のエラーから直す。成功の根拠なしに「完全一致済み」と言わない。
2. \`python3 -m pytest -q tests/test_postcut_rebff_exact_rectangle_grid.py tests/test_postcut_rebff_seam_registration.py\`。
3. 実際のLilium矩形BFFを \`app_postcut_rebff._execute(...post_policy="target_rectangle",max_splits=1,...)\` で生成し、\`_assemble_patches\` を実行して \`postcut_omega_rectangles_area_fraction_covered\` 全部1、\`postcut_seam_exact_uv_edge_pairs\` に少なくとも1辺、未生成領域ゼロかを確認。CIにテスト追加済み。
4. **矩形版Liliumで元のT2D Dual Hingeまで**実行し、最終的なSplit境界辺の最大位置誤差、隣接タイルの幅の不一致、衝突数、ヒンジ誤差を評価する。元論文に向けCSFとパネル形状（正方形性）も同時に確認する。失敗したらその原因で改修。
5. 元のM2D \`_build_reference_m2d\` は依然、自由境界/従来Paper用には**fully inside**で欠ける。そのコードを全面的に変更すると従来実装を壊すため、必要なら**実験モードのみ有効な完全矩形Grid**へ明示的に切り替える。初回Ωプレビューと最終矩形Ωを混同しない。
6. **パネルの「ぴったり一致」には条件あり**。すべて同寸正方形、等角BFF、任意長矩形の100%被覆は一般に同時に達成できない。境界セルが長方形でも受容できるかはユーザーの設計意図を確認する必要があるが、少なくとも現実装は欠落ゼロを優先。
7. 追加で、Gridが境界上の正方形性違反をどれくらい生んだか、K3D最適化が解けるかを数値出力。シームのメッシュ頂点の共有IDとヒンジ有無は**物理的に切断したのかリンク機構としてつながるのか**の問題なので、便宜的に頂点をweldしない。

## GitHub CI
https://github.com/yuyakato2365/onestring-physics-simulator/actions/workflows/postcut-rebff-mac-smoke.yml

## ユーザーに返す際の注意
- 研究の再現が目的。単なる見栄えの良いT2Dを成功扱いしない。
- 未収束CSFや大きな縫い目残差を成功と表現しない。
- 修正箇所と、成功したテスト、失敗したテストを区別する。
- 元のMac研究用フォルダの未コミット変更は保全する。
