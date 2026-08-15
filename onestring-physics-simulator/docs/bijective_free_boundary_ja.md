# Bijective Free-Boundary Parameterization

`omega_parameterization_mode="bijective_free_boundary"` は、既存の `bff` と
`ceps` を変更せずに追加された比較実験用モードです。入力は1つの境界を持つ
connected disk triangle meshに限定されます。seam/cut、位相変更、SCAF、
OneString固有grid-loss、lambdaの直接最適化は行いません。

## 処理

1. 入力が1つのconnected diskであることをEuler標数と境界ループで検証する。
2. 3D境界弧長に比例して境界を円へ置き、正の一様重みで内部を解くTutte
   embeddingを有効な初期写像として作る。
3. 各3D三角形を局所2D座標へ剛体展開し、面積重み付きSymmetric
   Dirichlet energyを計算する。
4. Smith & Schaefer 2015-inspiredな局所支持境界バリアを加える。
5. L-BFGS方向に対し、各三角形の符号付き面積が0になる最初の正根と、移動する
   境界edge/vertexが接触する最初の正根より手前にstepを制限する。
6. flip、boundary self-intersection、non-adjacent triangle overlapがない候補だけを
   受理する。

## 論文との差

これは著者実装の完全再現ではありません。論文の空間hashは使わず、境界候補を
O(m^2)で列挙します。最適化は小規模な独自L-BFGSで、受理前に既存の
`count_internal_triangle_overlaps()`を使った全体検証も追加しています。

## lambda

`lambda` は `reference_bff.triangle_jacobian_diagnostics()` と同じく、
UV-to-surface Jacobianの最大特異値です。目的関数はlambdaの均一化ではなく、
一般的なSymmetric Dirichlet distortionです。

比較は次で実行できます。

```powershell
python scripts/compare_bff_bijective_free_boundary.py
```

表示されるlambda差は評価対象であり、新方式が必ず改善することは保証しません。
