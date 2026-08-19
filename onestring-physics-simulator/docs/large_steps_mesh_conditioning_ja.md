# Large Steps 入力メッシュ conditioning

`omega_parameterization_mode="bijective_free_boundary"` で sampled input mesh を使う場合、
現在のパイプラインは S→Omega の前に自動で次を実行します。

```text
S_input
  -> Large Steps mesh conditioning
  -> S_conditioned
  -> bijective free-boundary parameterization
  -> Omega
```

## 目的

入力に細長い三角形が含まれると、S→Omega の validity-preserving line search で
triangle-degeneracy step が極端に小さくなり、境界更新がほとんど進まないことがあります。
この stage は connectivity を変更せず、頂点配置だけを整えてその問題を軽減します。

## Large Steps

Nicolet et al., *Large Steps in Inverse Rendering of Geometry* と同じ differential
parameterization

\[
u=(I+\lambda L)v
\]

を使います。`L` は uniform/combinatorial Laplacian です。
Open disk mesh の boundary vertices は固定し、interior variables だけについて
Large-Steps metric を適用します。

最適化では triangle quality と edge-length uniformity を改善しつつ、元表面の法線方向への
移動と全体の位置変化を抑えます。候補頂点は元の local surface patch へ投影されるため、
外部の Mitsuba preprocessing を手作業で行わず、同じパイプライン内で入力メッシュを
conditioning できます。

## CUDA 実装

`device="auto"` が既定値です。PyTorch から CUDA が利用可能なら自動的に CUDA を選びます。
CUDA 使用時は次を GPU 上で実行します。

- `M = I + lambda L` の reduced sparse system
- `M^{-1}` の適用（Jacobi-preconditioned conjugate gradient / 前処理付き共役勾配法）
- triangle-quality / edge-uniformity / surface-preservation energy と gradient
- Large-Steps search direction
- local 2-ring surface projection
- orientation / flip 検査
- 途中の triangle-quality diagnostics

line search ごとに線形方程式を解き直すことはせず、1 iteration につき Large-Steps gradient と
search direction のための sparse solve を行い、その 3D direction の scale だけを line search で変更します。
これにより CPU↔GPU 転送と repeated solve を減らしています。

実行後の metrics では、`large_steps_compute_device`, `large_steps_cuda_used`,
`large_steps_device_name`, `large_steps_cuda_peak_memory_mb` から実際に CUDA が使われたか確認できます。

## 保証・制約

- face connectivity は変更しない。
- 3D boundary vertices は固定する。
- 各 step で元 face normal に対する orientation ratio を検査し、反転候補を reject する。
- conditioning 後の頂点は元表面の local 2-ring triangle patch へ投影する。
- remeshing / edge flip / edge split / edge collapse は行わない。
- したがって connectivity 自体が悪いケースは、この stage だけでは直せない。

## Streamlit

`Omega parameterization mode = bijective_free_boundary` を選ぶと
`Large Steps mesh conditioning` 設定が表示されます。

- enable / disable
- lambda
- conditioning iterations
- learning rate

conditioning 中は専用 progress bar と直近ログを表示します。既定では 5 iteration ごとに、
次の情報を更新します。

- `iteration / max_iterations`
- objective energy
- minimum triangle angle
- triangle-quality p05
- edge-length CV
- accepted / rejected step count
- line-search scale
- gradient / direction の CG iteration count
- elapsed time
- 実際に使っている CUDA device 名

計算後は `View stage -> Conditioned S` で conditioning 後の 3D mesh を確認できます。
タイトルには最小角度と triangle-quality p05 の before/after が表示されます。

## 主な metrics

`surface_parameterization.metrics` に以下を保存します。

- `large_steps_compute_device`
- `large_steps_cuda_used`
- `large_steps_device_name`
- `large_steps_cuda_peak_memory_mb`
- `large_steps_before_minimum_angle_degrees`
- `large_steps_after_minimum_angle_degrees`
- `large_steps_before_triangle_quality_p05`
- `large_steps_after_triangle_quality_p05`
- `large_steps_before_edge_length_cv`
- `large_steps_after_edge_length_cv`
- `large_steps_surface_deviation_max`
- `large_steps_min_orientation_ratio`
- `large_steps_iteration_count`
- `large_steps_rejected_step_count`
- `large_steps_max_gradient_cg_iterations`
- `large_steps_max_direction_cg_iterations`
- `large_steps_max_cg_relative_residual`
- `large_steps_runtime_seconds`
