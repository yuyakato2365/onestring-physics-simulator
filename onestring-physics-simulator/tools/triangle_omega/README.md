# Macで三角形のKagomeを計算する

セットアップ済みのMacでは、onestring-physics-simulator ディレクトリで次を実行します。

```sh
.venv/bin/python -m streamlit run app_optcuts_20260916.py
```

Paper-oriented computation mode をオフにし、Omega parameterization mode で「三角形 Kagome | Ω・ヒンジ・印刷」を選びます。

## seamと格子を調べる

「OptCutsのΩに直接Gridを配置（700枚設定）」をオンにします。入力の既定は底面が開いたBunny（10,251頂点・20,372面）です。枚数は目標値で、入力ファイルと目標枚数は変更できます。seamの曲がりの重みは2、曲面の曲率変化の重みは0.5です。

この設定では正規化した入力にOptCutsを1回適用し、得られた単一チャートを格子に使います。追加の等角制約とΩ2伸縮は省きます。初期格子は重なる面積が50%以上のパネルを採用し、seam両岸の辺数と対応を合わせます。格子の接続を保って3Dに逆写像し、溶接後に曲面上で緩和します。比較実験と同じ開口縁の移動上限は代表辺長の15%です。

計算終了をM3Dに設定し、Run OneString pipeline を押します。View stage で入力曲面上のseam、Ω、格子、M2D、逆写像直後のM3D、緩和後のM3Dを確認できます。Ωでは同じ色の縁が一対です。最終M3Dでは溶接後も残ったパネルの共有辺に同じ色を表示します。「M3D（最小角）」に切り替えると細長いパネルを確認できます。

「大画面3Dビューを作成」→「大画面3Dビューを保存」で、同じデータをブラウザで回転・拡大できる単一のHTMLに保存します。新しく計算した格子には仮採用された三角形も保存されます。

## 曲率変化に対応したOptCuts

計算には `.venv311`（OptCuts呼び出し）、`.venv-triangle`（格子・緩和）、`third_party/OptCuts_csf_onestring/build2` のビルド済みOptCutsを使用します。既存のMacビルドで曲率変化に対応する実行ファイルを更新する場合は次を実行します。

```sh
.venv311/bin/python tools/triangle_omega/controlled/build_optcuts.py "$(mktemp -d)"
```
