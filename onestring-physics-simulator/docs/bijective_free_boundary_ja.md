# Bijective Free-Boundary Parameterization

`omega_parameterization_mode="bijective_free_boundary"` は、既存の `bff` と
`ceps` を変更せずに追加された比較実験用モードです。入力は1つの境界を持つ
connected disk triangle meshに限定されます。seam/cut、位相変更、SCAF、
OneString固有grid-loss、lambdaの直接最適化は行いません。

## 処理

1. 入力が1つのconnected diskであることをEuler標数と境界ループで検証する。
2. 3D境界弧長に比例して境界を等面積円へ置き、正のmean-value weightで内部を解く
   Floater embeddingを有効な初期写像として作る。円の面積を3D表面積へ合わせることで、
   seamが短いBunnyのような入力を初期状態で過度に圧縮しない。
3. 各3D三角形を局所2D座標へ剛体展開し、面積重み付きSymmetric
   Dirichlet energyを計算する。
4. Smith & Schaefer 2015-inspiredな局所支持境界バリアを加える。
5. 全UV頂点を同じL-BFGS問題で更新する。初期逆Hessianには正定値の頂点別対角スケールを
   用い、微小三角形の巨大勾配だけがstepを支配して自由境界を凍結することを防ぐ。
6. L-BFGS方向に対し、各三角形が設定済み正面積marginへ達する最初の正根と、移動する
   境界edge/vertexが接触する最初の正根より手前にstepを制限する。
7. flip、boundary self-intersection、non-adjacent triangle overlapがない候補だけを
   受理する。

## 論文との差

これは著者実装のバイナリ互換な再現ではなく、論文アルゴリズムの独立実装です。
目的関数、自由境界を含む全頂点L-BFGS、境界バリア、first-singularity line searchを
実装しています。数値条件改善として正定値の対角初期逆Hessianを使用します。
境界候補は引き続きO(m^2)で列挙し、最適化には小規模な独自L-BFGSを使います。
non-adjacent triangle overlapの
候補抽出にはuniform-grid spatial hashを使いますが、候補に対するedge intersection、
triangle containment、adjacent triangle除外の厳密条件は旧総当たり版と同じです。
エネルギー・勾配、安全ステップ、重なり候補の狭域判定はNumPyの一括配列演算で処理し、
Pythonループの負荷を抑えています。

line searchではfinite/positive-area/boundary self-intersection、energy、Armijo条件を
順に検査します。円板では全三角形の正向きと単純な境界が大域単射性を与えるため、
高価なglobal overlap全走査はFloater初期値と最終UVで実施します。

## 実行時間と進捗

Streamlitの `Bijective free-boundary settings` では、確認用に最大iteration数と
line-search step数を変更できます。既定値はそれぞれ1000と20です。途中終了でも最終UVは
flip、degeneracy、boundary self-intersection、internal overlapを再検査し、有効な場合だけ
返します。8%から16%の間にはFloater初期化、validity check、各iteration、line search、
最終検査の進捗が表示されます。

性能内訳は `surface_parameterization.metrics` の `overlap_check_*`、
`energy_gradient_*`、`safe_step_*`、`optimization_iteration_log` で確認できます。
再現可能な3規模benchmarkは次で実行します。

```powershell
python scripts/benchmark_bijective_free_boundary.py --large-iterations 1000
```

Streamlitでは計算結果をsession stateに保持します。`View stage`を切り替えても再計算せず、
保存済みの段階を表示するだけです。設定変更後も既存結果を維持し、
`Run OneString pipeline`を押したときだけ新しい設定で再計算します。

`View stage = Omega` は初期円を灰色破線、最終的な自由境界を緑実線で表示します。
表示は再構築した円ではなく、最適化後の `uv_vertices_2d[boundary_loop]` そのものです。

## lambda

`lambda` は `reference_bff.triangle_jacobian_diagnostics()` と同じく、
UV-to-surface Jacobianの最大特異値です。目的関数はlambdaの均一化ではなく、
一般的なSymmetric Dirichlet distortionです。

比較は次で実行できます。

```powershell
python scripts/compare_bff_bijective_free_boundary.py
```

表示されるlambda差は評価対象であり、新方式が必ず改善することは保証しません。
