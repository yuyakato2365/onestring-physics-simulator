# Igloo の比較実験の入力（2026-11-03〜04、クラウドで使ったもの）

| ファイル | 中身 |
|---|---|
| `igloo.obj` | 元の Igloo（1562 頂点）。クラウドの実験はすべてこれから始めた。 |
| `igloo_symremesh.obj` | 上を `tools/optcuts_soft_fit_20261104/symremesh.py` で対称リメッシュしたもの（1361 頂点） |
| `igloo_symremesh_hard_sym_chart.pkl` | `igloo_symremesh.obj` に、ハード制約（σ1 = σ2、`ONESTRING_ANGLE_W=4096 ONESTRING_ANGLE_BOUND_SD=1`）・seam の折れ曲がりの罰則 2・対称性の罰則 0.5・枝分かれ禁止で seam の始点を探索した OptCuts の結果（始点 9、対称面の上の seam 1 本、長さ 0.68） |

リメッシュ比較のページ（「ハード制約・境界に添わせる：元のメッシュとリメッシュ後」）の右列はこの平面図から作った。
