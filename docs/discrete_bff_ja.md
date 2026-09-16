# 離散 Boundary First Flattening 実装

## 結論

従来の `omega_parameterization_mode="bff"` は、境界頂点を3D弧長に従って矩形上へ固定し、内部の2座標を cotangent harmonic Dirichlet solve で求めていた。この方法は **矩形境界 harmonic parameterization** であり、Boundary First Flattening (BFF) の中核である Cherrier formula、Poincaré–Steklov operator、best-fit boundary closure を実装していなかった。

本実装では `bff` 経路から LSCM を完全に除外し、次の離散BFFへ置き換えた。

1. 三角形の内角から内部 Gaussian curvature `K` と境界 geodesic curvature `k` を計算する。
2. cotangent zero-Neumann Laplacian `A` を構築する。
3. 矩形の4隅に対応する境界頂点を、3D境界のbest-fit planeへのPCA投影から巡回順序を保って選ぶ。
4. 矩形の目標 exterior angles `k_tilde` を作り、合計が `2π` になるよう数値誤差を閉じる。
5. 離散 Cherrier formula に対応する Neumann-to-Dirichlet solve

   `A u = -K - [0; k - k_tilde]`

   により、境界のcompatible log scale factor `u` を求める。
6. 各境界辺の目標長を

   `l_star_ij = exp((u_i + u_j)/2) l_ij`

   で計算する。
7. 目標角度を固定したまま、BFF論文 Eq. (20) の重み付き最小二乗補正で境界が閉じるように辺長を `l_tilde` へ修正する。
8. 得られたBFF-compatible closed boundaryを境界条件として、内部をcotangent harmonic extensionする。

`lscm_paper_like` と `boundary_sliding_lscm` は比較・診断用の別モードとして残るが、`bff` 経路からは呼ばれない。

## 「矩形にできない」場合にBFFが行うこと

外角の合計を `2π` にするだけでは、積分した平面曲線は一般に閉じない。原論文の `BestFitCurve` は、目標外角、すなわち各辺の向きを固定し、閉包制約

`sum(l_tilde_ij T_ij) = 0`

を満たす範囲で、目標辺長 `l_star` からの重み付き変更を最小化する。したがって、BFFは「矩形化に失敗したらLSCMへ戻す」のではない。**角、特に矩形の90度角を維持し、閉じるために必要な最小限だけ辺長を変更する**。

論文は、この補正後の辺長が理論上は負になり得ると明記している一方、実験では観測しなかったとしており、その場合のフォールバック手順は規定していない。本実装では次の方針を採用した。

- 通常時は論文 Eq. (20) の閉形式解をそのまま使う。
- 補正後辺長が非正になる場合だけ、同じ目的関数と閉包制約に `l_tilde > 0` を加えた凸二次計画を解く。
- それでも正値かつ閉じた境界を得られなければ、LSCMや固定矩形harmonicへ黙って切り替えず、明示的に失敗させる。

これは論文にない堅牢化であり、`bff_positive_length_qp_used` などのmetricsで判別できる。

## 矩形のaspect ratioについて

BFFで外角を指定すると矩形の直角性は制御できるが、連続的なaspect ratioを完全に独立指定できるわけではない。離散境界では、どの4頂点を角に割り当てるかとcompatible scale factorによって達成aspectが決まる。この実装は、PCA投影上の方向極値と目標辺比を使って角頂点を選ぶ。

既存設定名 `boundary_target_aspect_mode="lscm_initial"` は互換性のため残したが、BFF経路ではLSCMを実行せず、3D境界のPCA投影aspectとして解釈する。metricsの以下を確認すること。

- `boundary_target_aspect_source`
- `boundary_achieved_unoriented_aspect_ratio`
- `boundary_requested_unoriented_aspect_ratio`
- `boundary_aspect_relative_error`

## 主な検証metrics

- `bff_cherrier_formula_implemented = True`
- `bff_best_fit_curve_implemented = True`
- `bff_uses_lscm = False`
- `bff_gauss_bonnet_error`
- `bff_neumann_compatibility_sum_before_projection`
- `bff_best_fit_closure_error`
- `bff_best_fit_max_relative_length_adjustment`
- `bff_best_fit_negative_length_count_unconstrained`
- `bff_positive_length_qp_used`
- `uv_triangle_flip_count`
- `uv_degenerate_triangle_count`

UV反転や退化が生じた場合も、自動で別方式へ置換しない。結果の方式を偽らないためである。

## 参照

- Rohan Sawhney and Keenan Crane, *Boundary First Flattening*, ACM Transactions on Graphics, 2018.
- GeometryCollective, official `boundary-first-flattening` C++ implementation, `src/project/Bff.cpp`.
