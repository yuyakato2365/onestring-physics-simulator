# Macで「Split後にBFF再計算」を試す

**実験専用**: 公式BFFの矩形Ωを作成 → **初回のCSFを計測し2以下ならSplitなしで終了** → 2を超える場合だけGridに沿って初回Split → 各子領域を**矩形境界BFF**で再パラメータ化 → CSFが2を超える領域だけで追加Splitと再BFFを反復し、全領域で2以下になるまで（または停止条件まで）繰り返します。最終UV・3D・CSF・履歴を表示します。

通常のOneStringパイプラインは変更しません。**各パッチの再BFF後のUVを使った中間M2D/M3Dプレビューに加え、各領域を一つのM2D/M3Dに統合し、既存のK3D・T3D・K2D・T2D Dual Hingeまで実計算する機能を追加しました。**論文の再現成功やT2Dでの衝突改善を示すものではありません。

## 1. 既存クローンで新ブランチを取得

```bash
cd ~/Documents/OneString-large-steps-mesh-conditioning/onestring-physics-simulator
git status --short
git fetch origin
git switch --track origin/agent/postcut-rebff-mac-20261014
```

※ `git status` に未コミット変更が出た場合、ブランチ切替前に退避してください。すでにブランチがある場合は `git switch agent/postcut-rebff-mac-20261014`。

## 2. まとめてセットアップ（推奨）

初回はmacOS用のセットアップスクリプトを使うと、Python 3.11用の独立仮想環境、公式BFFの通常CLI・矩形CLI、Lilium OBJまで用意できます。

```bash
bash scripts/setup_postcut_rebff_mac.sh
source .venv-postcut-rebff/bin/activate
python -m streamlit run app_postcut_rebff.py
```

このスクリプトはHomebrewで必要なツールを導入します。既存のソースや通常版OneStringのPython環境を上書きしません。

以下は手動でインストールしたい場合の代替手順です。

## 3. Pythonと公式BFF CLIを手動で用意

```bash
brew install cmake suite-sparse openblas
python3 --version
python3 -m pip install -e .
mkdir -p third_party
git clone --depth 1 --recursive https://github.com/GeometryCollective/boundary-first-flattening.git third_party/boundary-first-flattening
cmake -S third_party/boundary-first-flattening -B third_party/boundary-first-flattening/build \
  -DBFF_BUILD_GUI=OFF -DBFF_BUILD_CLI=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build third_party/boundary-first-flattening/build --target bff-command-line --parallel 4
python3 scripts/setup_bff_rectangle.py --source third_party/boundary-first-flattening
```

すでに公式BFFのソースがある場合は、clone手順を省略し、`--source` に既存のソースフォルダを指定してください。

確認する実行ファイルはこの2つです。

- `third_party/boundary-first-flattening/build/bff-command-line`
- `third_party/bff_rectangle_build/bff-rect-command-line`

実行ファイルが別の場所なら、実験画面のサイドバーへその絶対パスを指定してください。ビルドが失敗しても代替の調和写像には切り替えません。

## 4. Liliumメッシュを取得

```bash
curl -fL https://raw.githubusercontent.com/libigl/libigl-tutorial-data/master/lilium.obj -o ~/Downloads/lilium.obj
```

## 5. Macのブラウザで実際に計算

```bash
python3 -m streamlit run app_postcut_rebff.py
```

開いたページで：

1. OBJのパスが `~/Downloads/lilium.obj` か確認
2. グリッドは `14`、切断位置は **曲率優先の最初のSplit**
3. Split後のBFFは **`target_rectangle`（矩形境界・初期設定）**を選択。これは初回のSplit後だけでなく、その後の全Splitの子領域にも適用
4. **最大Split数（初回を含む、既定8）**を選び、**Split → BFF再計算を収束まで繰り返す** を押す

画面では各領域のUVとCSFを比較できます。**「従来の全段階を計算してT2D Dual Hingeを表示」**ボタンから、元のOneStringのK3D・T3D・K2D・T2D Dual Hingeソルバーを実行します。実行後は **ZIP保存** から新UV付きOBJと新M3D四角形OBJを取得できます。

## 判定と制約

同じ領域のCSF比 `sqrt(max(A_3D/A_2D) / min(A_3D/A_2D))` を切断前後で測ります。現行の四角形グリッドのCSF評価と**評価領域が違う**ので、M2D上の2.960と、全三角形パッチの値をそのまま比較しないこと。

再BFF後の各UV領域は**互いに独立した座標系**で、まだ一枚の2Dレイアウトに自動配置していません。元論文がSplit後に再BFFを実行したという根拠もありません。これは再現仮説を検証するための独立実験です。

## 反復Splitの判定・停止条件

- 最初に初期矩形BFFの三角形面積ヤコビアンに基づくCSFを測定します。CSF≤2なら**Split 0本で終了**。超えた場合だけGridに沿ってSplitを入れます。
- 初回の切断線は既存の曲率優先M2DアルゴリズムのGrid位置を使います。候補がなければ現在のUVの格子線を曲率優先で探索します。手動位置はGridへスナップします。
- Splitのたびに分かれた両領域で公式BFFを再計算し、領域内の面積ヤコビアン J = A3D/A2D から sqrt(max(J)/min(J)) を求めます。
- すべての領域でCSF比が2以下なら成功して終了します。
- CSFが2を超える領域のみ追加で切断します。複数ある場合はCSFの最も大きい領域を優先します。
- 2回目以降もCSF>2の領域だけ、直前の**矩形BFFの新しいUV格子**とガウス曲率に基づいてGridに一致する切断候補を試します。切断した両側で矩形BFFを実行します。
- 子領域のトポロジーとBFF結果を検証してから切断を採用します。候補が不適切なら次の候補を試します。
- 最大Split数に達した場合、切断候補がない場合、どの候補もBFFに失敗する場合は、未達として停止理由を表示します。
- ZIPには最終的な全領域の再BFF OBJ、新M3DタイルOBJ、全Split履歴と停止理由を出力します。

**注意:** 反復が終了しても、各パッチのUV島は独立しています。K3D・T2Dには接続していません。またCSF比の判定は研究用の近似で、元論文の製造条件を完全に証明するものではありません。

## BFF CLIが見つからないエラーの修復（Mac）

実験用worktreeのルートで以下を実行します。元の研究用worktreeは変更しません。

```bash
cd ~/Documents/onestring-postcut-rebff/onestring-physics-simulator
git fetch origin
git switch --detach origin/agent/postcut-rebff-mac-20261014
bash scripts/repair_postcut_bff_mac.sh
source .venv-postcut-rebff/bin/activate
python3 -m streamlit run app_postcut_rebff.py
```

修復スクリプトは標準BFF CLIと矩形BFF CLIをビルドして、両方の実行ファイルの存在と実行権限を確認します。ビルドが失敗した場合はターミナルの最後のエラーメッセージを確認してください。途中でエラーが起きた場合は、元のOptCuts環境には手を加えません。

**既存OptCutsの扱い:** このアプリは公式BFFの新しい仮説を単独で検証する画面であり、`optcuts_backend.py` や旧`app.py`のOptCuts処理を削除しません。従来の研究用worktreeは別ディレクトリに残っています。再BFFのUVはK3D/T2Dへ統合していません。


## 実験結果と本番OptCutsモードの違い

本実験は初期矩形BFF→Split→各領域再BFFを検証します。OptCutsのソースや既存の本番モードは変更しません。下流の最適化には従来のK3D・T3D・K2D・T2D Dual Hinge実装を呼びます。各領域に新しく敷く四角形グリッドは、元のM2Dに近い枚数となるよう共通間隔を探索します。ただしこれは厳密なタイル保存ではなく、元論文の手法の忠実な再現やT2Dの品質を保証しません。実際のソルバーが失敗した場合、成功したかのように表示せずエラーを表示します。

## Split後にも矩形境界BFFを使う場合（現在の既定設定）

- サイドバーの「Split後のBFF境界条件」は **「矩形境界（初期BFFと同じ・推奨）」** が既定値です。
- 初回のBFF、最初のSplit後の両領域、追加Split後の両領域のすべてに矩形BFFを実行します。
- 矩形の四隅の位相は初期BFFと各子領域で共通です。各子領域の境界上で四隅が独立に選ばれます。
- 公式の矩形BFFアダプターが返すUVを検査し、4辺からなる矩形境界であると検証できない場合は失敗扱いとします。自由境界BFFへは自動的にフォールバックしません。
- `comparison.json` には `postcut_bff: target_rectangle`、`all_postcut_patches_rectangle_verified: true` と、各SplitのCSF履歴が記録されます。
- 矩形条件の方が自由境界より歪みが大きくなりやすく、Splitを繰り返してもCSF≤2へ収束する保証はありません。最大Split数、候補なし、BFF失敗で停止した場合は未達と表示します。
- CSFが2を超えたままでも、**明示的に同意する試験用チェックボックス**を使うとK3D→T2D Dual Hingeの計算を試せます。ただし品質・製造可能性を示すものではなく、下流ソルバーが失敗すればエラーを表示します。


### CSFの改善と停止判定

「改善したか」は切断前後の数値で表示しますが、成功と判定する条件は**全領域でCSF≤2**です。例えばCSFが3→2.4へ改善しても続けてSplitし、CSFが2以下ならSplitなしでも終了します。矩形BFFで歪みが増えたり、切断可能なGrid線がなくなった場合には最大Split数や失敗理由で停止します。元のOptCuts本番処理とK3D/T2D実装は変更しません。
