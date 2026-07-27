# OneStringへの公式CEPSバックエンド統合

## 何を組み込んだか

`omega_parameterization_mode="ceps"` を選ぶと、PythonでCEPSらしい近似を再実装するのではなく、Mark Gillespieらが公開している公式C++実装の `parameterize` コマンドを呼び出す。

OneString側は次を行う。

1. 入力曲面 `S` が連結な三角形diskで、境界loopが1本であることを確認する。
2. 3D境界のPCA投影から巡回順序を保つ4頂点を矩形cornerとして選ぶ。
3. cornerでは境界外角 `π/2`、その他の頂点では `0` を指定したcurvature fileを作る。
4. 公式CEPSを `--noFreeBoundary` 付きで実行する。
5. CEPSが生成したcommon-refinement meshとordinary UVをOBJから読み込む。
6. 元のcorner頂点から矩形方向を復元してUVを軸へ揃え、OneStringの `Omega` として渡す。
7. 後段では従来どおり `Omega` 上にquad gridをoverlay/cropし、`M3D`以降を構築する。

CEPS論文では、境界付き曲面は2枚を境界で貼り合わせることで閉曲面問題へ変換する。境界外角を `κ_i*` とすると、二重化した曲面の対応頂点には角度欠損 `2κ_i*` が与えられる。公式コードがこの二重化と制約変換を担当する。

## BFFとの関係

このCEPSモードはBFFの内部処理を少し変更したものではなく、`S -> Omega`を別方式で解く比較・拡張モードである。

- `bff`: Cherrier式、境界scale/curvature、BestFitCurve、harmonic extension。
- `ceps`: intrinsic Delaunay化、頂点scale factorのNewton最適化、Ptolemy flip、common refinement。

OneString原論文が初期parameterizationとして記述するのはBFFであり、CEPSへの置換はこのリポジトリ独自の研究拡張である。

## 「公式CEPS」と「完全に同じ」の範囲

uniformization、Ptolemy flip、normal-coordinate correspondence、common refinementは公式CEPS C++実装が行う。そのため、固定三角形分割上のPython近似を `CEPS` と名付ける実装ではない。

ただしOneStringの後段は2次元の通常のbarycentric inverse mapを要求するため、現時点ではCEPSの `--outputLinearTextureFilename`、つまりordinary linear UV exportを読み込む。CEPSが推奨するhomogeneous/projective texture interpolationをOneStringの逆写像へ直接実装したものではない。この違いはmetricsの次の値で明示する。

```text
ceps_texture_interpolation = ordinary_linear_uv_export
ceps_projective_interpolation_used = False
```

## Windowsで公式CEPSを準備する

Visual Studioの「Desktop development with C++」とCMake、Gitが利用できるPowerShellで実行する。長いパスによるビルド問題を避けるため、既定では `C:\CEPS` にcloneする。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_ceps_windows.ps1
```

完了すると `ONESTRING_CEPS_EXECUTABLE` がユーザー環境変数へ設定される。同じPowerShellでは直ちに利用できる。新しいPowerShellやStreamlitでも利用できる。

手動で指定する場合:

```powershell
$env:ONESTRING_CEPS_EXECUTABLE = "C:\CEPS\build\bin\Release\parameterize.exe"
```

## 単体確認

```powershell
python scripts\verify_ceps_integration.py
```

成功時:

```text
PASS: official CEPS CLI, common refinement, and OneString UV import are active.
```

## アプリ

```powershell
python -m streamlit run app.py
```

`Omega parameterization mode` で `ceps` を選ぶ。`Omega boundary mode` は `paper_default` のままにする。

確認すべきmetrics:

```text
ceps_backend_used = official_ceps_cli
ceps_reference_backend = True
ceps_common_refinement_used = True
ceps_prescribed_boundary_curvature = True
ceps_ptolemy_flips = performed inside official CEPS backend
uv_triangle_flip_count = 0
uv_degenerate_triangle_count = 0
```

## 現在の制約

- 入力は連結なmanifold triangle disk、境界loopは1本。
- 閉じたbunnyなどには、先にcutを設計してdiskへする必要がある。
- 4隅の外角は制御するが、矩形aspect ratioを独立に厳密指定するわけではない。
- CEPSはOneStringのCSF上限、quad配置、split、tile、hinge、string pathを自動決定しない。
- official executableが見つからない場合、BFFやLSCMへ黙ってfallbackせず明示的に停止する。
