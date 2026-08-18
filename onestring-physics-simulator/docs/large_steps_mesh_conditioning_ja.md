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
最初の triangle-degeneracy step が極端に小さくなり、境界更新がほとんど進まないことがあります。
この stage は connectivity を変更せず、頂点配置だけを整えてその問題を軽減します。

## Large Steps

Nicolet et al., *Large Steps in Inverse Rendering of Geometry* と同じ differential
parameterization

\[
u=(I+\lambda L)v
\]

を使います。`L` は uniform/combinatorial Laplacian です。
Open disk mesh の boundary vertices は固定し、interior variables だけについて

\[
v_I=M_{II}^{-1}(u_I-M_{IB}v_B)
\]

で 3D 頂点を復元します。

最適化では triangle quality と edge-length uniformity を改善しつつ、元表面の法線方向への
移動と全体の位置変化を抑えます。候補頂点は元の local surface patch へ投影されるため、
外部の Mitsuba preprocessing を手作業で行わず、同じパイプライン内で入力メッシュを
conditioning できます。

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

計算後は `View stage -> Conditioned S` で conditioning 後の 3D mesh を確認できます。
タイトルには最小角度と triangle-quality p05 の before/after が表示されます。

## 主な metrics

`surface_parameterization.metrics` に以下を保存します。

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
- `large_steps_runtime_seconds`
