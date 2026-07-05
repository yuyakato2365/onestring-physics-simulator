# OneString simulator handoff for ChatGPT

## 目的

このZipは、OneString物理シミュレータで「入力形状 S はきれいなのに、Omega / M2D / K3D / T3D で形状が崩れる」問題を、ChatGPTに説明・相談するための最小セットです。

特に対象にしている入力は、Domeのような単純な凸形状ではなく、雪だるま状・二つのドームが首でつながったような、くびれを持つ開いた表面です。

## 観測された症状

- S は見た目として自然な二山形状になっている。
- Omega では本来の境界形状に対応しない、右上の小さな飛び出しブロックが残ることがある。
- M3D は入力表面の一部しか覆わないように見えることがある。
- K3D は二山形状を保てず、いびつなタイルの組み合わせになることがある。
- T3D ではタイルが長く飛び出す、あるいは崩壊に近い形状になることがある。

当初はSTLメッシュが粗いことや、Blenderでの滑らか化不足を疑ったが、細分化・スムーズ化しても根本的には改善しなかった。したがって、主原因は入力メッシュの粗さだけではなく、S から Omega へ落とす段階と、その後の M2D グリッド化・切断線処理にある可能性が高い。

## 重要な仮説

以前の実装では、Omega の境界を単純な四角形に固定する流れが強かった。

そのため、くびれを持つ非矩形の入力表面でも、Omega 上では矩形領域に押し込まれ、M2D の格子選択時に本来の境界外にあるセルや、首まわりの不自然な接続が残ることがあった。この歪みが K3D / T3D の崩れに伝播していると考えている。

現在の修正では、Omega boundary を単純な四角に固定しないようにし、入力表面の外周形状を保つ投影型パラメータ化を追加している。

## 現在の処理フロー

概念的には次の流れです。

```text
S target surface
  -> Omega planar parameter domain
  -> M2D gridded planar tiles
  -> M3D inverse-mapped tiles on S
  -> K3D kinematic tile layout
  -> K2D / T2D / T3D deployment stages
```

今回の問題では、K3D 以降だけでなく、Omega と M2D の段階ですでに形状の対応が崩れている可能性が高い。

## 実装済みの主な修正

### 1. Omega boundary を矩形固定しない

`src/onestring_physics/onestring_pipeline.py` に `_shape_preserving_projected_uv` と `_build_surface_parameterization` のラッパーを追加した。

この実装では、3D表面頂点をPCA基底へ投影し、2D座標として正規化する。境界を強制的に正方形・長方形に貼り付けるのではなく、元の表面の外形が Omega 上に残ることを優先する。

期待:

- 雪だるま状のくびれや非矩形境界が Omega に反映される。
- M2D で境界外のタイルが残りにくくなる。
- K3D に渡るタイル集合が入力表面に近づく。

限界:

- これは厳密なBFF/LSCM/ARAP系のパラメータ化ではなく、投影ベースの近似。
- オーバーハングや複雑な曲面では、PCA投影だけでは重なりや局所歪みを完全には防げない。

### 2. CSF split は近傍タイルを消さず、既存グリッド線上で頂点を複製する

伸び率が大きい領域を検出し、分割線を M2D の既存グリッド線へスナップする。以前のように分割線近傍のタイルを丸ごと削除するのではなく、切断線をまたぐ接続だけを切る方針に寄せている。

### 3. Split Map 可視化

`src/onestring_physics/visualization.py` の `figure_split_mapping` で、S 上の高CSF点、Omega上に写像された分割サンプル、検出ピークを表示する。

これにより、S のどこを切ろうとしているのか、それが Omega のどこに写っているのかを確認できる。

### 4. Peak-guided split / peak grid alignment

雪だるま形状では、ピークが4枚のパネル頂点の共有点に来ることが期待される。そのため、局所ピークを検出し、M2Dグリッドの頂点配置がピークを避けるのではなく通るように補助している。

### 5. K3D quality guard

K3D最適化後に、辺長比・面積崩壊・スケールドリフト・頂点変位の外れ値を計測し、明らかに崩壊している場合はK3D結果を採用せず M3D にフォールバックする。

これは根本修正ではなく、崩壊した結果を後段に流さないための安全策。

## ChatGPTに相談したいこと

1. `Omega` をPCA投影で外形保持する現在の近似は、この用途に対して妥当か。
2. くびれを持つ開いた表面では、BFF/LSCM/ARAPなど、より厳密なパラメータ化へ置き換えるべきか。
3. `M2D` のタイル選択は、矩形内グリッドではなく、Omega の実境界ポリゴンに対するクリッピングにすべきか。
4. Split line は高CSF領域だけでなく、首・谷・くびれを優先して通すべきか。
5. 対称形状に対して非対称なタイル配置へ収束する主因は、Omega の矩形固定・格子スナップ・順序依存処理のどれが最も疑わしいか。
6. K3Dで崩れるのは、K3D最適化そのものよりも、M2D/M3Dから渡ってくるタイル集合・接続関係がすでに不適切なせいではないか。

## 関連ファイル

- `README.md`
  - 論文の流れと実装の流れの日本語説明。
  - LaTeX数式でCSF、力学更新、アニメーション差分などを記載。
- `src/onestring_physics/onestring_pipeline.py`
  - 主要なパイプライン処理。
  - Omega boundary保持、CSF split、peak-guided split、K3D quality guard など。
- `src/onestring_physics/visualization.py`
  - Streamlit/Plotlyの可視化。
  - Split Map表示。
- `tests/test_onestring_pipeline.py`
  - CSF split、タイル削除回避、対称補助、ピーク整列、非矩形Omega boundary保持のテスト。
- `app.py`
  - Streamlit起動エントリ。

## 起動方法

PowerShellで以下を実行する。

```powershell
cd C:\Users\yjiat\Documents\OneString\onestring-physics-simulator
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

テスト:

```powershell
cd C:\Users\yjiat\Documents\OneString\onestring-physics-simulator
.\.venv\Scripts\python.exe -m pytest tests/test_onestring_pipeline.py -q
```

直近の確認では `tests/test_onestring_pipeline.py` は 19 件すべて通過している。

## 注意

このZipはChatGPT相談用の最小セットであり、`.venv`、キャッシュ、生成物、画像添付ファイルは含めていない。実行には元のリポジトリ環境と依存関係が必要。
