# Macで「Split後にBFF再計算」を試す

**実験専用**: 公式BFFの矩形Ωを作成 → 現在の曲率優先Splitの最初の1本で元3D三角形を切断 → 両側をそれぞれ**公式BFF**で再パラメータ化 → 2D UV / 3D / CSFを可視化し、再計算後のUV付きOBJをエクスポートします。

通常のOneStringパイプラインは変更しません。**再BFF後のUVを使ったM2D/M3D/K3D/T2Dの再構築は未対応**です。論文の再現成功やT2Dでの衝突改善を示すものではありません。

## 1. 既存クローンで新ブランチを取得

```bash
cd ~/Documents/OneString-large-steps-mesh-conditioning/onestring-physics-simulator
git status --short
git fetch origin
git switch --track origin/agent/postcut-rebff-mac-20261014
```

※ `git status` に未コミット変更が出た場合、ブランチ切替前に退避してください。すでにブランチがある場合は `git switch agent/postcut-rebff-mac-20261014`。

## 2. Pythonと公式BFF CLIを用意

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

## 3. Liliumメッシュを取得

```bash
curl -fL https://raw.githubusercontent.com/libigl/libigl-tutorial-data/master/lilium.obj -o ~/Downloads/lilium.obj
```

## 4. Macのブラウザで実際に計算

```bash
python3 -m streamlit run app_postcut_rebff.py
```

開いたページで：

1. OBJのパスが `~/Downloads/lilium.obj` か確認
2. グリッドは `14`、切断位置は **曲率優先の最初のSplit**
3. Split後のBFFは `automatic_reference`（自由境界）
4. **1本Split → BFF再計算を実行** を押す

画面では各領域の実際の3D形状、元のUV、再BFFしたUV、前後のCSFが表示されます。実行後は **ZIP保存** から新しいUVを保持するOBJを取得できます。

## 判定と制約

同じ領域のCSF比 `sqrt(max(A_3D/A_2D) / min(A_3D/A_2D))` を切断前後で測ります。現行の四角形グリッドのCSF評価と**評価領域が違う**ので、M2D上の2.960と、全三角形パッチの値をそのまま比較しないこと。

再BFF後の2つのUV領域は**互いに独立した座標系**で、まだ一枚の2Dレイアウトに自動配置していません。元論文がSplit後に再BFFを実行したという根拠もありません。これは再現仮説を検証するための独立実験です。
