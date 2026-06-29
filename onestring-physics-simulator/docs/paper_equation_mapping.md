# 論文の数式とコードの対応表

このドキュメントは、`onestringpull_authors_version_compressed.pdf` に出てくる主要な式・処理段階が、このリポジトリ内のどのコードに対応しているかを整理したものです。

注意: 現在の実装は、論文の ShapeOp / libigl 実装をそのまま移植したものではありません。論文のパイプラインを追えるようにした「paper-faithful approximation」です。したがって、各式の目的は保ちつつ、一部は NumPy / SciPy / PyTorch による軽量な近似実装になっています。

## Figure 5 の全体パイプライン

論文の流れ:

$$
S \rightarrow \Omega \rightarrow M_{2D} \rightarrow M_{3D}
\rightarrow K_{3D} \rightarrow T_{3D}
$$

および

$$
M_{2D} \rightarrow K_{2D} \rightarrow T_{2D}
$$

対応コード:

- `src/onestring_physics/onestring_pipeline.py`
  - `build_onestring_design(...)`
  - `_build_surface_mesh(...)`
  - `_build_surface_parameterization(...)`
  - `_flatten_to_domain(...)`
  - `_build_m2d(...)`
  - `_lift_m2d_to_m3d(...)`
  - `_optimize_k3d(...)`
  - `_extrude_tiles(...)`
  - `_optimize_k2d(...)`
  - `_make_t2d_from_transforms(...)`
  - `_optimize_dual_hinges(...)`

UI 側の入口:

- `app.py`
  - `Run OneString pipeline`
  - `View stage`
  - 各ステージの Plotly viewer

## Eq. 1: assembled configuration energy

論文の式:

$$
E_{\mathrm{Assembled}}(x)
= \omega_1 E_{\mathrm{Planar}}(x)
+ \omega_2 E_{\mathrm{Square}}(x)
+ \omega_3 E_{\mathrm{Surface}}(x)
$$

意味:

- $E_{\mathrm{Planar}}$: quad 面が平面からどれだけ外れているか
- $E_{\mathrm{Square}}$: quad がどれだけ正方形らしさを保っているか
- $E_{\mathrm{Surface}}$: 目標曲面 $S$ からどれだけ離れているか

対応コード:

- `src/onestring_physics/onestring_pipeline.py`
  - `_optimize_k3d(...)`
  - `_planarity_residuals(...)`
  - `_square_residuals(...)`
  - `_surface_fit_error(...)`
  - `_optimize_k3d_torch(...)`
  - `_optimize_k3d_mesh_torch(...)`

実装メモ:

`_optimize_k3d(...)` が、planarity / square / surface closeness の residual を組み合わせて `K3D` を作っています。UI の `w_planar`, `w_square`, `w_surface` が、それぞれ $\omega_1$, $\omega_2$, $\omega_3$ に対応します。

## Eq. 2: planarity constraint

論文の式の意図:

各 quad の4頂点を、最も近い平面へ射影したときの差を最小化します。

読みやすく書くと:

$$
E_{\mathrm{Planar}}
= \sum_{q \in Q(M_{3D})}
\left\|
V_q - P_{\mathrm{plane}}(V_q)
\right\|_F^2
$$

ここで:

- $q$: quad
- $V_q$: quad $q$ の4頂点をまとめた行列
- $P_{\mathrm{plane}}$: best-fit plane への射影
- $\|\cdot\|_F$: Frobenius norm

対応コード:

- `src/onestring_physics/onestring_pipeline.py`
  - `_planarity_residuals(...)`
  - `_quad_planarity_error(...)`
  - `_tile_face_planarity(...)`
  - `_tile_face_planarity_by_group(...)`

実装メモ:

現在のコードは、ShapeOp の projection stack ではなく、SVD / best-fit-plane 的な residual で近似しています。

## Eq. 3 / Eq. 4: square / edge projection

論文の式の意図:

edge を目標長へ近づけ、quad を正方形らしくします。

edge length 側を読みやすく書くと:

$$
E_{\mathrm{Length}}
= \sum_{(i,j) \in E(M_{3D})}
\left\|
(v_j - v_i) - P_E(v_i, v_j)
\right\|_2^2
$$

射影 $P_E$ は、元の edge 方向を保ったまま目標長 $l_{\mathrm{target}}$ に揃える操作です。

$$
P_E(v_i, v_j)
= l_{\mathrm{target}}(e_{ij})
\frac{v_j - v_i}{\|v_j - v_i\|}
$$

論文ではさらに、quad を closest square へ寄せる shape term も足して:

$$
E_{\mathrm{Square}}
= E_{\mathrm{Length}} + E_{\mathrm{Shape}}
$$

対応コード:

- `src/onestring_physics/onestring_pipeline.py`
  - `_square_residuals(...)`
  - `_square_error(...)`
  - `_edge_length_variance(...)`
  - `_optimize_k3d(...)`

実装メモ:

現在の実装では、edge length と対角線・形状 residual によって正方形らしさを近似しています。論文の closest-square projection そのものを完全実装しているわけではありません。

## Eq. 5: flat configuration energy

論文の目的:

`M2D` を平面上で最適化して、対応する `K3D` の edge length に合う `K2D` を作ります。同時に、製造可能な gap や衝突しない配置も考慮します。

コード内で使っている対応式:

$$
E_{\mathrm{Flat}}(x)
= \omega_1 E_{\mathrm{Edge}}(x)
+ \omega_2 E_{\mathrm{Collision}}(x)
+ \omega_3 E_{\mathrm{Fab}}(x)
$$

ここで:

- $E_{\mathrm{Edge}}$: `K3D` と `K2D` の対応 edge length の不一致
- $E_{\mathrm{Collision}}$: 平面配置での tile overlap
- $E_{\mathrm{Fab}}$: 元の `M2D` から動きすぎないための fabrication / anchor term

対応コード:

- `src/onestring_physics/onestring_pipeline.py`
  - `_optimize_k2d(...)`
  - `_optimize_k2d_torch(...)`
  - `_projective_edge_match_2d(...)`
  - `_strict_k2d_edge_length_solve(...)`
  - `_edge_matching_errors(...)`

実装メモ:

中規模以上の grid では、`K2D` 内で重い collision relaxation を無理に走らせず、edge matching を優先しています。collision-aware な配置は、後段の Dual Hinge layout に寄せています。

## Eq. 6: hinge layout energy

論文の式:

$$
E_{\mathrm{Hinge}}(x)
= \omega_1 E_{\mathrm{Rigid}}(x)
+ \omega_2 E_{\mathrm{Collision}}(x)
+ \omega_3 E_{\mathrm{Conn}}(x)
$$

意味:

- $E_{\mathrm{Rigid}}$: tile の剛体性を保つ
- $E_{\mathrm{Collision}}$: tile 同士の衝突を避ける
- $E_{\mathrm{Conn}}$: hinge vertex pair が一致するようにする

$E_{\mathrm{Conn}}$ を読みやすく書くと:

$$
E_{\mathrm{Conn}}
= \sum_{h \in H(T_{2D})}
\left\|
x_h - P_{\mathrm{Conn}}(x_h)
\right\|_2^2
$$

ここで $P_{\mathrm{Conn}}$ は、対応する hinge pair の相手側位置、または pair の midpoint への射影です。

対応コード:

- `src/onestring_physics/onestring_pipeline.py`
  - `_paper_local_global_se2_layout(...)`
  - `_optimize_t2d_top_hinge_footprint_layout(...)`
  - `_optimize_dual_hinges(...)`
  - `_optimize_rigid_assembly_hinge_layout_2d(...)`
  - `_hinge_constraint_tuples_from_specs(...)`
  - `_count_2d_footprint_collisions_from_pairs(...)`
  - `_sat_polygon_mtv(...)`

実装メモ:

現在の実装では、各 tile を1つの rigid SE(2) pose として扱います。つまり $E_{\mathrm{Rigid}}$ は soft penalty ではなく、pose fit によって強く保たれます。$E_{\mathrm{Conn}}$ は hinge midpoint constraint、$E_{\mathrm{Collision}}$ は SAT / AABB candidate による bounded projection です。

## Lift point selection / GPE

論文の考え方:

gap ごとの gravitational potential energy (GPE) を見て、Morse-Smale segmentation と peak coupling によって、必要最小限の lift point を選びます。

GPE は概念的には次の形です:

$$
g_i
= \sum_{t_j \in T(g_i)}
\frac{1}{4} m_j g (z_j - z_{\min})
$$

ここで:

- $g_i$: gap $i$ の GPE
- $T(g_i)$: gap $i$ の周囲 tile 集合
- $m_j$: tile $j$ の質量
- $z_j$: tile $j$ の重心高さ
- $z_{\min}$: 構造全体の最低高さ

対応コード:

- `src/onestring_physics/onestring_pipeline.py`
  - `_build_gap_graph(...)`
  - `_select_lift_points(...)`
  - `_build_string_path(...)`
  - `_turn_angle_total(...)`

実装メモ:

現在のコードは、完全な Morse-Smale / coupling-DAG ではありません。gap graph に GPE 風スコアを持たせ、`lift_tau` による threshold で高エネルギー peak を選ぶ近似です。

## Channel energy / string path friction

論文の考え方:

Capstan equation に基づき、string channel の摩擦を累積 turn angle で評価します。

概念式:

$$
E_{\mathrm{Channel}}
= T_1 \left(e^{\mu_c \theta_{\mathrm{Total}}} - 1\right)
$$

実装で安定して使うログ形式:

$$
\log E_{\mathrm{Channel}}
\approx \mu_c \theta_{\mathrm{Total}}
$$

対応コード:

- `src/onestring_physics/onestring_pipeline.py`
  - `_build_string_path(...)`
  - `_turn_angle_total(...)`
  - `safe_capstan_friction(...)`

実装メモ:

現在の routing は、gap graph 上の軽量な walk です。metrics として `turn_angle_total`, `theta_total`, `log_channel_cost`, `estimated_channel_friction` を記録します。

## Simulation energy

論文の式:

$$
E_{\mathrm{Simulation}}(x)
= \omega_1 E_{\mathrm{Rigid}}(x)
+ \omega_2 E_{\mathrm{Collision}}(x)
+ \omega_3 E_{\mathrm{Actuation}}(x)
$$

actuation は snap と lift に分かれます。

$$
E_{\mathrm{Actuation}}
= E_{\mathrm{Snap}} + E_{\mathrm{Lift}}
$$

snap の概念:

$$
E_{\mathrm{Snap}}
= \sum_{(a,b) \in R}
\left\|
m_a(x) - m_b(x)
\right\|_2^2
$$

ここで $m_a$, $m_b$ は string path 上の隣接 side-face midpoint です。

lift の概念:

$$
E_{\mathrm{Lift}}
= \sum_{\ell \in L}
\left\|
c_\ell(x) - p_\ell^{\mathrm{target}}
\right\|_2^2
$$

ここで $c_\ell$ は lift gap 周辺 tile の中心、$p_\ell^{\mathrm{target}}$ は prescribed lift target です。

対応コード:

- `src/onestring_physics/onestring_pipeline.py`
  - `simulate_onestring_deployment(...)`
  - `_simulate_onestring_deployment_torch(...)`
  - `_project_rigid_tiles(...)`
  - `_project_aabb_collisions(...)`
  - `_project_snap_constraints(...)`
  - `_project_lift_constraints(...)`
  - `_project_hinge_constraints(...)`
  - `_project_target_pose_fit(...)`
  - `_project_target_contact_guard(...)`

実装メモ:

string は rope particle としてはシミュレーションしていません。論文の説明に合わせて、string の効果を positional constraints として入れています。

- snap: string path 上の side-face midpoint pair を閉じる
- lift: 選ばれた lift gap を 3D target 側へ動かす
- rigid: tile shape を Kabsch projection で保つ
- collision: 現在は軽量 AABB projection

厳密な green-green SAT volume collision は、動画生成や UI 操作を極端に重くしたため、現在の通常経路からは外しています。

## UI / metrics との対応

UI 側:

- `app.py`
  - sidebar の weight / solver controls
  - pipeline progress display
  - CPU / CUDA backend report
  - smooth browser playback

描画:

- `src/onestring_physics/visualization.py`
  - stage figures
  - tile assembly
  - comparison view

アニメーション:

- `src/onestring_physics/animation.py`
  - assembly animation helpers

app の metrics で確認できる主な項目:

- `surface_fit_error`
- `planarity_error_after`
- `square_error_after`
- `edge_matching_error_after`
- `hinge_connection_error`
- `collision_count`
- `snap_error`
- `lift_error`
- `final_deployment_error_to_T3D`

## 実装上の大きな近似まとめ

- BFF は完全実装ではなく、harmonic UV parameterization で近似しています。
- ShapeOp / libigl の projection stack は、NumPy / SciPy / PyTorch の residual / local-global projection で近似しています。
- Morse-Smale lift point selection は、GPE threshold peak selection で近似しています。
- string routing は、完全な最適化ではなく gap graph 上の軽量 route です。
- actuation simulation は rope particle ではなく、snap / lift positional constraints です。
- collision は通常経路では軽量 AABB projection です。
