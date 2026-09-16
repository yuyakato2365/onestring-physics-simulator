# Boundary-sliding LSCM 実装

## 位置づけ

`boundary_sliding_lscm` は、目標矩形境界を持つ LSCM 近似です。Boundary First Flattening (BFF) ではありません。境界頂点を3D累積弧長で矩形へ固定せず、4つの corner anchor 以外を対応する矩形辺上で順序を保って滑らせます。

実行時には `src/onestring_physics/onestring_pipeline.py` が wrapper として読み込まれ、`src_backup_before_sideface_contact/onestring_physics/onestring_pipeline.py` の `build_onestring_design` が参照する `_build_surface_parameterization` を現行 wrapper の実装へ差し替えます。Streamlit は `app.py` から `app_backup_before_mitered_t3d.py` を実行します。

## 数式と離散化

各 surface triangle を局所2次元座標へ展開し、面積重み付き Cauchy-Riemann 残差を行列 `A` に組み立てます。全 UV 未知数を

```text
x = [u_0, ..., u_(n-1), v_0, ..., v_(n-1)]^T
```

とすると、共形エネルギーは

```text
E_lscm = ||A x||^2
grad E_lscm = 2 A^T A x
```

です。`_build_lscm_residual_matrix` は free-boundary LSCM と新方式で共有され、数式を別実装にしていません。

目標矩形を `Gamma(s)` とし、境界頂点 `i` は

```text
uv_i = Gamma(s_i)
```

を満たします。4つの corner anchor は固定し、それ以外は各矩形辺の局所パラメータ `t in (0, 1)` を未知数として扱います。内部頂点は境界 UV を固定した reduced LSCM

```text
min_(x_I) ||A_I x_I + A_B x_B||^2
```

を LSQR で解きます。

境界更新では `2 A^T A x` の各境界頂点成分を矩形辺の接線 `d Gamma / dt` に射影します。3D境界辺長 `l_i` と矩形上の辺長 `l'_i` には弱い正則化

```text
E_length = min_alpha sum_i (l'_i - alpha l_i)^2
```

を加えます。各辺上の隣接パラメータ差には

```text
E_spacing = sum_j (delta t_(j+1) - delta t_j)^2
```

を加え、全体を

```text
E = E_lscm + lambda_length E_length + lambda_spacing E_spacing
```

として減少させます。3D弧長は `E_length` の弱い項だけに使われ、境界対応を固定しません。

## 反復手順

1. `_validate_single_disk_surface` が連結成分数、全境界 loop、non-manifold edge、孤立頂点、Euler characteristic を検査します。単一 manifold disk でなければ例外にします。
2. `_lscm_free_boundary_uv` で2点 pin の自由境界 LSCM を初期解として求めます。
3. 自由境界を PCA 軸へ合わせ、4方向の極値、局所 turning、循環順序を使う小さな候補探索で corner anchor を選びます。
4. corner 間の初期対応を自由 LSCM 境界から投影し、PAV isotonic projection で頂点対応を入れ替えずに最小間隔と順序を満たします。
5. 境界を固定して reduced LSCM を解き、接線方向勾配で境界パラメータを更新します。
6. 各候補について UV signed area と境界自己交差を検査します。flip、ほぼゼロ面積、自己交差を作る更新は backtracking line search で拒否します。安全な step が見つからない場合は成功扱いにせず例外にし、`bff` や PCA へ fallback しません。

## 実装箇所

- `PipelineParameters`: target aspect、反復数、step、正則化重み、flip epsilon、line-search 上限。
- `_mesh_boundary_loops`, `_validate_single_disk_surface`: Step A。
- `_build_lscm_residual_matrix`, `_lscm_free_boundary_uv`: Step B と共有 LSCM 行列。
- `_RectangleBoundaryTarget`: `Gamma(s)`、接線、perimeter、corner。
- `_select_rectangle_corner_vertices`: Step C。
- `_project_strictly_increasing`: 順序制約を保つ isotonic projection。
- `_solve_lscm_with_boundary`, `_boundary_sliding_lscm_uv`: Step D-F。
- `visualization.figure_domain`: 目標矩形、最終境界、対応順、corner、flip、角度歪み。
- `visualization.figure_m3d_overlay`: lookup failure、outside Omega、surface triangle 使用分布。

## 既存方式との差

| mode | 境界 | 内部 solve | 主な性質 |
|---|---|---|---|
| `bff` | 3D累積弧長で矩形へ固定 | cotangent harmonic | 境界対応が固定され、局所形状を圧縮し得る |
| `lscm_paper_like` | 2点 pin 以外は自由 | free-boundary LSCM | 共形性は高いが、不規則な Omega 境界になり得る |
| `boundary_sliding_lscm` | 矩形上を順序付きで滑る。4 corner のみ固定 | constrained LSCM の反復 | 矩形境界と共形 energy の折衷 |

## Reference BFF との差

この実装は Cherrier formula、境界曲率処理、Poincare-Steklov operator、reference BFF library solver を実装していません。将来それらを実装する場合は別 mode `bff_reference` とし、`boundary_sliding_lscm` とは統合しません。
