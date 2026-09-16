# 現行OneString実装の数式処理・離散化・アルゴリズム対応表

この文書は、提供された元論文PDF
`C:/Users/yjiat/Downloads/onestringpull_authors_version_compressed.pdf`
を参照し、現在の `onestring-physics-simulator` が何を数式として扱い、
どのアルゴリズムで近似し、コード上のどこで実装しているかを対応づけた
調査メモである。

参照論文:

- Akib Zaman, Jacqueline Aslarus, Jiaji Li, Stefanie Mueller, Mina Konakovic-Lukovic,
  "One String to Pull Them All: Fast Assembly of Curved Structures from Flat Auxetic Linkages",
  ACM Transactions on Graphics, 44(6), 2025.
- DOI: `https://doi.org/10.1145/3763357`

注意:

- 「論文の数式」は、PDF本文の式番号に対応する内容を記述する。
- 「実装で使っている数式」は、現在のコードがそのまま使っている場合は
  「論文どおり」と書き、差分がある場合は差分を明記する。
- コードは主に
  `onestring-physics-simulator/src/onestring_physics/onestring_pipeline.py`
  を参照する。このファイルは元実装を `_original` として読み込み、その上に
  BFF、分割、T3D、T2D、GPU/進捗計測などの拡張を被せる構成になっている。
- 旧実装の基礎関数は
  `onestring-physics-simulator/src_backup_before_mitered_t3d/onestring_physics/onestring_pipeline.py`
  に残っており、現行ファイルが `_original` 経由で利用する処理も多い。

## 0. 全体パイプライン

### 論文の処理順

論文の図4、図5は、次の流れで処理を整理している。

```text
S -> Omega -> M2D -> M3D -> K3D -> T3D
              \-> K2D -> T2D
              \-> lift points -> string path -> physical deployment
```

ここで:

- `S`: 入力ターゲット曲面
- `Omega`: `S` を平面化した領域
- `M2D`: `Omega` 上に置かれた初期quad mesh
- `M3D`: `M2D` を逆写像で曲面へ持ち上げた初期3D quad mesh
- `K3D`: 組立状態を表す最適化済みquad mesh
- `T3D`: 厚みを持つ組立状態のfrustum tile群
- `K2D`: 平面状態を表す最適化済みquad mesh
- `T2D`: 厚みとヒンジを持つ平面状態のtile linkage

### 実装の対応

実装でもデータ構造上は同じ段階を持つ。

```python
@dataclass
class OneStringDesignState:
    target: TargetSurface
    surface: SurfaceMesh
    parameterization: SurfaceParameterization
    domain: PlanarDomain
    m2d: QuadMesh
    m3d: QuadMesh
    k3d: QuadMesh
    t3d: TileAssembly
    k2d: QuadMesh
    t2d: TileAssembly
```

主な入口:

- `src/onestring_physics/onestring_pipeline.py:882` `PipelineParameters`
- `src/onestring_physics/onestring_pipeline.py:5998` `simulate_onestring_deployment`

実装上の重要な差分は、論文の各段階を完全なShapeOp/libigl実装として再現して
いるのではなく、NumPy/SciPy/Torchによる近似・フォールバック・メトリクス計測
を組み合わせている点である。

## 1. 入力曲面 `S` と離散曲面

### 論文の数式・説明

論文はユーザ入力の自由曲面 `S` を出発点にし、以後の処理を離散三角形メッシュ
またはその上のquad近似として扱う。連続曲面上の「近い点」「法線」「曲率」は、
離散メッシュの頂点、三角形、辺、近傍関係に置き換えられる。

### 離散化

連続曲面 `S` は三角形メッシュ

```math
S_h = (V_S, F_S)
```

として表す。頂点 `v_i in R^3`、面 `f = (i,j,k)` を持ち、最近傍射影
`P_S(x)` は実装上「サンプル頂点または三角形上の近い点」への近似になる。

### 実装で使っている数式

論文どおり、以後のエネルギーは頂点位置の有限集合に対して評価する。ただし、
ユーザ指定のプリセット曲面では、解析的な高さ場や生成メッシュを使う。

高さ場型の入力では概念的に

```math
S(u,v) = (u, v, h(u,v))
```

をサンプルして三角形メッシュに変換する。

### アルゴリズム

1. 入力形状またはプリセットから頂点・面を生成する。
2. 後続の最近傍、面法線、曲率、UV対応で使えるように、同じ `SurfaceMesh` に集約する。
3. 曲面上の演算はすべてこの離散メッシュ上の局所演算として扱う。

### コード対応

旧実装側:

- `src_backup_before_mitered_t3d/onestring_physics/onestring_pipeline.py:956`
  `_build_surface_mesh`

現行拡張側:

- `src/onestring_physics/onestring_pipeline.py:1724`
  `_build_surface_parameterization`

短い引用:

```python
surface_vertices_3d=np.asarray(surface.vertices, dtype=float),
uv_vertices_2d=uv,
faces=np.asarray(surface.faces, dtype=int),
```

## 2. `S -> Omega`: 境界優先平面化と共形写像

### 論文の数式・説明

論文は Boundary First Flattening (BFF) を使い、ターゲット曲面 `S` から平面領域
`Omega subset R^2` への共形写像

```math
c : S -> Omega
```

を作る。可能な場合、境界を矩形になるように指定する。

### 離散化

連続写像 `c` は、各曲面頂点 `v_i` にUV座標 `u_i in R^2` を割り当てる離散写像

```math
c_h(v_i) = u_i
```

として持つ。三角形内部の点は、三角形の重心座標で線形補間する。

離散共形性は、辺長や角度の歪みを小さくするUV配置問題になる。典型的には
cotangent Laplacian

```math
L_{ij} = -1/2 (cot alpha_{ij} + cot beta_{ij})
```

を用いた調和写像、またはLSCM型の線形最小二乗問題として解く。

### 実装で使っている数式

論文はBFFを明示している。現行実装は、外部BFFライブラリそのものではなく、
BFF風の「境界を先に矩形へ置く」実装を追加し、失敗時にLSCM/調和系の平面化へ
フォールバックする。

差分:

- 論文: BFFによる共形写像 `c`
- 実装: `bff` モードでは境界矩形化 + cotan harmonic 型の内部解法
- 実装: `lscm` モードでは自由境界LSCM近似

### アルゴリズム

1. 境界ループを抽出する。
2. 境界長に沿って矩形境界へ割り付ける。
3. 内部頂点をcotangent Laplacianの線形系で解く。
4. 必要に応じてLSCMまたは既存実装へフォールバックする。
5. `SurfaceParameterization` として `surface_vertices_3d`、`uv_vertices_2d`、`faces` を保存する。

### コード対応

- `src/onestring_physics/onestring_pipeline.py:1431`
  `_bff_boundary_first_uv`
- `src/onestring_physics/onestring_pipeline.py:1724`
  `_build_surface_parameterization`

短い引用:

```python
if method == "bff":
    uv, metrics = _bff_boundary_first_uv(surface, params)
```

```python
return _original.SurfaceParameterization(
    method=method,
    surface_vertices_3d=np.asarray(surface.vertices, dtype=float),
    uv_vertices_2d=uv,
    faces=np.asarray(surface.faces, dtype=int),
)
```

## 3. `Omega -> M2D`: 平面領域上のquad grid生成とcrop

### 論文の数式・説明

論文は `Omega` 上へ正方格子を重ね、`Omega` の境界に完全に含まれないquadを
取り除いて `M2D` を作る。この `M2D` が平面状態のtop face候補になる。

### 離散化

平面領域を格子幅 `s` の格子点

```math
g_{ij} = (x_0 + i s, y_0 + j s)
```

でサンプルし、各セル

```math
q_{ij} = (g_{ij}, g_{i+1,j}, g_{i+1,j+1}, g_{i,j+1})
```

をquad候補にする。crop判定は、連続領域包含を多角形包含テストへ落とす。

### 実装で使っている数式

論文は「完全に含まれるquad」を残す。実装では、パラメータにより中心点cropや
厳密cropを選べる。中心点cropの場合、境界をまたぐquadが残る可能性があり、
これは論文からの差分である。

### アルゴリズム

1. `Omega` のAABBから格子点列を作る。
2. 各格子セルについて領域内判定を行う。
3. 残ったセルを `M2D.faces` として登録する。
4. 参照しない格子点を落とし、頂点配列を詰め直す。

### コード対応

- `src/onestring_physics/onestring_pipeline.py:2058`
  `_flatten_to_domain`
- `src/onestring_physics/onestring_pipeline.py:2484`
  `_build_m2d`

短い引用:

```python
return _original.PlanarDomain(
    boundary=np.asarray(boundary, dtype=float),
    uv_vertices=uv,
)
```

```python
mesh = _original.QuadMesh(vertices=np.asarray(vertices), faces=np.asarray(faces), stage="M2D")
```

## 4. CSF判定とmesh splitting

### 論文の数式・説明

論文は、共形写像の局所伸縮を conformal scale factor (CSF) として評価する。
quad auxetic pattern が局所的に対応できる伸縮には限界があり、CSFが2を超える
場合にmesh splitを導入する。分割は高いGauss曲率の近傍を通り、格子方向に沿って
入れると説明される。

概念的には、微分 `dc` の特異値を `sigma_1, sigma_2` とすると、共形写像では

```math
sigma_1 approx sigma_2 approx lambda
```

であり、局所伸縮は `lambda` で測る。論文は最大局所伸縮が閾値2を超える領域を
split対象にする。

### 離散化

各quadまたは三角形で、3D辺長と2D辺長の比を評価する。

```math
rho_e = ||v_i - v_j||_2 / max(||u_i - u_j||_2, epsilon)
```

セルのCSF proxyを、周辺辺の最大比または統計量として扱う。

### 実装で使っている数式

論文の厳密なCSFではなく、実装では辺伸縮比ベースのproxyを用いる。したがって:

- 論文: 共形写像の局所スケール `lambda` を評価
- 実装: 対応する2D/3D辺長比の最大・平均からCSF風メトリクスを評価

これは実装差分であり、完全な離散微分の特異値評価ではない。

### アルゴリズム

1. `M2D` と `M3D` の対応辺を集める。
2. 各辺の伸縮比を計算する。
3. 閾値超過箇所を検出する。
4. split候補線を入れ、接続を切る。
5. split後の局所伸縮メトリクスを再評価する。

### コード対応

- `src/onestring_physics/onestring_pipeline.py:882`
  `PipelineParameters` のCSF/split関連パラメータ
- `src/onestring_physics/onestring_pipeline.py:2484`
  `_build_m2d`
- `src/onestring_physics/onestring_pipeline.py:2668`
  `_lift_m2d_to_m3d`

短い引用:

```python
csf_split_threshold: float = _original.PipelineParameters.csf_split_threshold
csf_estimator: Literal["edge_stretch", "jacobian_svd"] = "edge_stretch"
```

## 5. `M2D -> M3D`: 逆共形写像による持ち上げ

### 論文の数式・説明

論文は、`M2D` の頂点を逆写像

```math
c^{-1} : Omega -> S
```

で3D曲面へ戻し、初期3D quad mesh `M3D` を作る。

### 離散化

UV平面上の点 `u` が三角形 `(u_i,u_j,u_k)` に入るとき、重心座標

```math
u = alpha u_i + beta u_j + gamma u_k,
alpha + beta + gamma = 1
```

を求め、3D位置を

```math
x = alpha v_i + beta v_j + gamma v_k
```

で補間する。

### 実装で使っている数式

論文どおり、UVから曲面上への逆写像を使う。ただし実装では離散三角形探索と
重心補間で近似する。

### アルゴリズム

1. 各 `M2D` 頂点のUV座標を取得する。
2. そのUVを含むパラメータ化三角形を探索する。
3. 重心座標を計算する。
4. 対応する3D三角形上で線形補間し、`M3D.vertices` を得る。

### コード対応

- `src/onestring_physics/onestring_pipeline.py:2668`
  `_lift_m2d_to_m3d`
- `src_backup_before_mitered_t3d/onestring_physics/onestring_pipeline.py:1233`
  `inverse_map_uv_to_surface`

短い引用:

```python
lifted = np.asarray([
    _original.inverse_map_uv_to_surface(parameterization, uv)
    for uv in mesh_2d.vertices
], dtype=float)
```

## 6. `M3D -> K3D`: 組立状態quad mesh最適化

### 論文の数式

論文の式(1)は、組立状態の最適化エネルギーを

```math
E_Assembled(v)
  = w_1 E_Planar(v) + w_2 E_Square(v) + w_3 E_Surface(v)
```

と定義する。

式(2)のplanarity項は、各quad頂点行列 `V_q` を最良平面へ射影し、その差を測る。

```math
E_Planar
  = sum_{q in Q(M3D)} || V_q - P_p(V_q) ||_F^2
```

surface項は各頂点をターゲット曲面へ近づける。

```math
E_Surface
  = sum_{i in V(M3D)} || v_i - P_S(v_i) ||_2^2
```

式(3)のedge length項は、各辺ベクトルを目標長の辺ベクトルへ近づける。

```math
E_Length
  = sum_{(i,j) in E(M3D)}
    || (v_j - v_i) - P_E(v_i, v_j) ||_2^2
```

式(4)の射影は、現在の辺方向を保ちながら長さだけを目標長へ合わせる。

```math
P_E(v_i,v_j)
  = L_Target(e_ij) (v_j - v_i) / ||v_j - v_i||
```

さらに、quadを正方形に近づけるshape項

```math
E_Shape = sum_q || V_q - P_Q(V_q) ||^2
```

を使い、

```math
E_Square = E_Length + E_Shape
```

とする。

論文の実験設定では式(1)の重みは

```math
w_1 = 10000, w_2 = 10, w_3 = 0.1
```

である。

### 離散化

最適化変数はすべてのquad頂点位置を並べたベクトル

```math
v = (x_1,y_1,z_1, ..., x_n,y_n,z_n)
```

である。各エネルギーは残差ベクトル `r(v)` の二乗和

```math
E(v) = ||r(v)||_2^2
```

として実装できる。

### 実装で使っている数式

大枠は論文どおり。差分は次のとおり。

- `P_p`: SVD/PCAでquadの最良平面を求める。
- `P_S`: 厳密な三角形最近点ではなく、実装箇所によって頂点最近傍や近似射影を使う。
- `P_Q`: Umeyama型の類似変換でclosest squareを作る。
- 最適化器はShapeOpではなく、SciPy least_squares、Torch、またはprojective-style
  fallbackを使う。

### アルゴリズム

1. `M3D.vertices` を初期値にする。
2. quadごとのplanarity残差を作る。
3. 辺ごとの長さ残差を作る。
4. quadごとのsquare shape残差を作る。
5. 曲面近接残差を作る。
6. 重み付き残差を最小二乗として解く。
7. 得られた頂点を `K3D` として保存する。

### コード対応

- `src/onestring_physics/onestring_pipeline.py:3183`
  `_optimize_k3d`
- `src_backup_before_mitered_t3d/onestring_physics/onestring_pipeline.py:4355`
  `_planarity_residuals`
- `src_backup_before_mitered_t3d/onestring_physics/onestring_pipeline.py:4373`
  `_surface_fit_error`
- `src_backup_before_mitered_t3d/onestring_physics/onestring_pipeline.py:4388`
  `_square_residuals`

短い引用:

```python
residuals.extend(math.sqrt(params.w_planar) * _planarity_residuals(vertices, faces))
residuals.extend(math.sqrt(params.w_square) * _square_residuals(vertices, faces))
residuals.extend(math.sqrt(params.w_surface) * _surface_residuals(vertices, target))
```

## 7. `K3D -> T3D`: 厚み付けとfrustum tile化

### 論文の数式・説明

論文は `K3D` のquadを厚み付きfrustum tileへ変換する。可能な場合は、quad gridを
主曲率方向に合わせ、各頂点を法線方向へoffsetしてplanar extrusionを行う。
その後、式(2)と同じplanarity制約をtop、bottom、contact facesへ適用して、
面が平面になるように最適化する。

概念的には、top頂点 `p_i` と面法線 `n` からbottom候補を

```math
b_i = p_i - t n
```

で作る。ただし論文ではこの単純offsetだけで終わらせず、contact faceの
planarityも含めて再最適化する。

### 離散化

各quad tileは8頂点

```math
(p_1,p_2,p_3,p_4,b_1,b_2,b_3,b_4)
```

で表され、6つのquad faceを持つ。

```text
top:    (p1,p2,p3,p4)
bottom: (b1,b2,b3,b4)
sides:  (p1,p2,b2,b1), ...
```

### 実装で使っている数式

ここは論文との差分が大きい。現行実装は単純な法線offsetではなく、共有辺の
contact plane/miter planeを使ってbottom vertexを求める実験的なT3D処理である。

各tileで:

- top面法線 `n`
- bottom plane `n dot x = n dot p - t`
- side/contact plane `s_e dot x = s_e dot p_e`

を作り、bottom vertexはbottom planeと隣接する2つのside planeの交点として
解く。

```math
A x = d
```

ここで `A` は `n` と2つのside plane normalを並べた行列である。

さらにガードとして、bottom頂点が本来の厚み方向から大きく外れた場合、
tile全体を法線方向へ平行移動した単純offsetに戻す。

### アルゴリズム

1. `K3D` の各quadからtop tileを作る。
2. 各tile法線を計算し、向きを揃える。
3. 各辺の内向きside normalを計算する。
4. 共有辺では2つのtileのside normalからmiter/contact planeを作る。
5. 各bottom頂点を「bottom plane + 2 side planes」の交点として解く。
6. 厚みの符号やジャンプ量を検査し、破綻時は法線平行移動へ戻す。
7. 8頂点tileと変換行列を `TileAssembly` として返す。

### コード対応

- `src/onestring_physics/onestring_pipeline.py:3456`
  `_extrude_tiles`

短い引用:

```python
"""Extrude K3D tiles using shared-edge miter/contact planes."""
```

```python
normals, normal_orientation_metrics = _orient_tile_normals_consistently(raw_normals, mesh.faces)
side_normals[tile_a][edge_a] = miter
side_normals[tile_b][edge_b] = -miter
```

実装上の注意:

- 論文の「offset後に式(2)でtop/bottom/contact facesを再最適化」とは同一ではない。
- 現行のT3D押し出し方向問題を追う場合、まずこのmiter/contact-plane解法と
  normal orientation guardの相互作用を見る必要がある。

## 8. `M2D/K3D -> K2D`: flat configuration最適化

### 論文の数式

論文の式(5)は、平面状態の最適化エネルギーを

```math
E_Flat(v)
  = w_1 E_Edge(v) + w_2 E_Collision(v) + w_3 E_Fab(v)
```

とする。

- `E_Edge`: 3D組立状態の辺長に平面状態の辺長を合わせる。
- `E_Collision`: 平面配置でtile同士が重ならないようにする。
- `E_Fab`: 製造制約を満たすようにする。

論文の実験設定では

```math
w_1 = 1, w_2 = 1, w_3 = 0.001
```

である。

### 離散化

平面頂点は

```math
u = (x_1,y_1, ..., x_n,y_n)
```

で表す。3Dの目標辺長を

```math
l^3_{ij} = ||p_i - p_j||_2
```

平面の辺長を

```math
l^2_{ij} = ||u_i - u_j||_2
```

として、

```math
E_Edge = sum_{(i,j)} (l^2_{ij} - l^3_{ij})^2
```

を基本残差にする。

### 実装で使っている数式

edge matchingは論文どおりの目的に近い。collision/fabricationは論文補足の
完全実装ではなく、SAT/AABB、最小距離、製造幅などの近似制約として扱う。

### アルゴリズム

1. `M2D` を初期値にする。
2. `K3D` の対応辺長を目標値として集める。
3. 平面上で辺長残差を最小化する。
4. quad同士の重なりを検出し、衝突ペナルティを入れる。
5. 必要に応じてTorch/SciPy/projective fallbackで解く。
6. 得られた頂点を `K2D` とする。

### コード対応

- `src/onestring_physics/onestring_pipeline.py:5233`
  `_optimize_k2d`
- `src_backup_before_mitered_t3d/onestring_physics/onestring_pipeline.py:4435`
  `_projective_edge_match_2d`
- `src_backup_before_mitered_t3d/onestring_physics/onestring_pipeline.py:2514`
  `_sat_polygon_mtv`

短い引用:

```python
target_lengths = _edge_target_lengths(mesh_3d.vertices, mesh_3d.faces)
```

```python
vertices_2d = _projective_edge_match_2d(vertices_2d, faces, target_lengths, params)
```

## 9. `K2D/T3D -> T2D`: 平面側厚み付けと変換行列

### 論文の数式・説明

論文では、3D側のextrusionで得たtop-to-bottomの剛体変換を平面側top faceに適用し、
flat configurationのbottom faceとside facesを生成する。

各tileの変換を

```math
b_i = R p_i + t
```

と見れば、3D側で求めた `(R,t)` を対応する2D側tileへ適用する処理である。

### 離散化

各quad tileごとに4x4同次変換行列

```math
T =
[ R t ]
[ 0 1 ]
```

を持つ。2D側のtop vertexは `z=0` に埋め込み、同次座標でbottomを作る。

### 実装で使っている数式

論文と同じ発想で、T3Dのtile変換行列をT2D側へ適用する。ただし、現行T3Dの
変換行列自体は前節のmiter/contact-plane extrusionの結果から推定される。

### アルゴリズム

1. `K2D` の各quadを `z=0` のtop faceにする。
2. `T3D.transform_matrices` を対応tileに適用する。
3. bottom faceとside faceを生成する。
4. `TileAssembly(stage="T2D")` として返す。

### コード対応

- `src/onestring_physics/onestring_pipeline.py:4096`
  `_make_t2d_from_transforms`

短い引用:

```python
transforms = np.asarray(t3d.transform_matrices, dtype=float)
```

## 10. hinge placement: top hinge / bottom hinge / dual hinge

### 論文の数式・説明

論文は、隣接frustum tileの接続をvertex hingeとして配置する。topに置くかbottomに
置くかは、局所曲率と開く方向のdihedral angleに依存する。

説明上の判断は:

- 曲率が負で、かつdihedral angleが90度を超える場合はbottom hingeが必要。
- それ以外はtop hingeが基本。

さらに論文の式(6)では、hinge位置最適化を

```math
E_Hinge(v)
  = w_1 E_Rigid(v) + w_2 E_Collision(v) + w_3 E_Conn(v)
```

とする。

接続制約は、hinge関連頂点 `x_u` を接続可能集合へ射影する形で

```math
E_Conn
  = sum_{u in H(T2D)} || x_u - P_Conn(x_u) ||_2^2
```

と表される。

論文の実験設定では

```math
w_1 = 100, w_2 = 100, w_3 = 1
```

である。

### 離散化

hinge候補は、隣接tileの対応vertexまたはedge近傍に置かれる有限個の点として
扱う。collisionは平面polygon同士の交差判定へ離散化される。

### 実装で使っている数式

論文と同じ目的を持つが、式(6)の完全なShapeOp最適化ではない。実装は:

- top hingeを基本にする。
- 必要に応じてdual hinge配置を試す。
- rigid placement、SAT衝突判定、接続距離ペナルティで補正する。

### アルゴリズム

1. 隣接tile/gapを走査する。
2. top hinge候補を作る。
3. flat状態で重なりや接続破綻を検査する。
4. 必要ならdual hinge配置へ切り替える。
5. 接続距離・衝突量をメトリクスとして返す。

### コード対応

- `src/onestring_physics/onestring_pipeline.py:4405`
  `_optimize_dual_hinges`
- `src_backup_before_mitered_t3d/onestring_physics/onestring_pipeline.py:2706`
  `_rigidly_place_t3d_tile_in_flat_layout`

短い引用:

```python
return _original._optimize_dual_hinges(grid, mesh_faces, t2d, t3d, params, progress_callback)
```

## 11. GapGraph: gap、lift point候補、境界制約

### 論文の数式・説明

論文は、boundary tilesを先に制約することで全体の形状が決まるという観察を使い、
stringが通るべきgap/lift pointをグラフとして扱う。gapは通常4 tileに囲まれた
四角形の空隙であり、境界やsplit近傍では2または3 tileの場合もある。

### 離散化

gapをノード、隣接関係をエッジとするグラフ

```math
G_gap = (V_gap, E_gap)
```

を作る。各ノードには:

- 2D中心
- 3D中心
- 境界フラグ
- 周辺tile
- 高さ・エネルギーproxy

を持たせる。

### 実装で使っている数式

論文の概念どおりgap graphを作る。ただし、gap検出は現在のquad topologyと
T2D/T3Dの幾何からの近似であり、論文の全ケースを厳密に分類するものではない。

### アルゴリズム

1. quad meshのセル隣接を走査する。
2. 4 tileまたは境界tile群からgap候補を作る。
3. gap中心・隣接gap・boundary属性を計算する。
4. string pathとlift point selectionの入力にする。

### コード対応

- `src/onestring_physics/onestring_pipeline.py:4430`
  `_build_gap_graph`

短い引用:

```python
gap_graph = _build_gap_graph(mesh_faces, t2d, t3d)
```

## 12. lift point selection: GPE、Morse-Smale、Kruskal

### 論文の数式・説明

論文は、lift pointを最小数にするために、gapごとの重力ポテンシャルエネルギー
(GPE)を評価する。PDF本文では、gap `i` のGPEは周辺tileの重心高さを用いる形で
書かれている。

概念的には:

```math
g_i = (1/4) sum_{t in T(g_i)} m_t g (z_t - z_min)
```

ここで `T(g_i)` はgap `i` の周辺tile集合、`m_t` はtile質量、`z_t` はtile重心高、
`z_min` は基準高さである。

論文はこのscalar fieldに対して離散Morse-Smale segmentationを行い、basinを得る。
さらにbasin間の相互作用を重み付きグラフで表し、Kruskal法によるmaximum spanning
treeで結合関係を解析する。閾値 `tau` は初期値 `tau_0 = 0.8` から始め、シミュレーション
が失敗すれば0.1ずつ増やす。

### 離散化

各gapをscalar fieldの頂点とし、

```math
g : V_gap -> R
```

を定義する。隣接gap間にエッジを張り、局所最大、basin、saddle的接続を離散的に扱う。
basinグラフのエッジ重みは、論文では

```math
w(i,j) = min(g_i, g_j)
```

のような障壁値として説明される。

### 実装で使っている数式

現行実装は論文の完全なMorse-Smale複体実装ではなく、gapの高さ/エネルギーproxy、
局所最大、閾値選択を使う近似である。したがって:

- 論文: GPE scalar field + Morse-Smale segmentation + coupling analysis + MST
- 実装: gap graph上のスカラー指標からlift候補を選ぶ軽量近似

### アルゴリズム

1. 各gapの2D/3D中心、周辺tile、境界情報を集める。
2. 高さまたはGPE proxyを計算する。
3. 閾値 `tau` を使って重要gapを選ぶ。
4. boundary constraintと接続性を考慮してlift point集合を返す。
5. deployment simulationで失敗すれば、閾値や候補数を調整する。

### コード対応

- `src/onestring_physics/onestring_pipeline.py:4565`
  `_select_lift_points`

短い引用:

```python
def _select_lift_points(gap_graph, tau: float):
```

## 13. string path: Capstan frictionと閉路探索

### 論文の数式

論文はCapstan equationを使い、stringが曲がるほど摩擦による抵抗が増えるとする。
入口張力 `T_1`、摩擦係数 `mu_c`、総曲げ角 `theta_Total` に対し、channel frictionは

```math
E_Channel = T_1 (exp(mu_c theta_Total) - 1)
```

で表される。

このため経路探索の目的は

```math
min_R theta_Total
```

であり、総曲げ角を最小化する閉じたstring routeを探す。

論文では、gap graph `G=(V,E)` 上の各ノードに向きラベルを持たせる。

```text
x_i = 0   vertical
x_i = 1   horizontal
x_i = -1  virtual boundary entrance
x_i = -2  split boundary
```

split entryを禁止し、境界入口を含む閉路として探索する。

### 離散化

連続的なstring channelは、gap graph上のwalkとして離散化される。

経路 `R = (r_0, r_1, ..., r_k)` に対し、各折れ曲がり角を

```math
theta_j = arccos(
  ((p_j-p_{j-1}) dot (p_{j+1}-p_j))
  / (||p_j-p_{j-1}|| ||p_{j+1}-p_j||)
)
```

とし、

```math
theta_Total = sum_j theta_j
```

を最小化する。

### 実装で使っている数式

Capstan式そのものはメトリクスとして持つ。経路探索は論文の完全な制約付き閉路探索
ではなく、gap/lift pointを結ぶ低角度・低コスト経路の近似である。

### アルゴリズム

1. lift pointとboundary入口候補を決める。
2. gap graph上で候補経路を作る。
3. 曲げ角合計を評価する。
4. 摩擦係数 `mu_c` でCapstan frictionを評価する。
5. 最もコストの低いrouteを `StringPath` として返す。

### コード対応

- `src/onestring_physics/onestring_pipeline.py:4702`
  `_build_string_path`
- `src_backup_before_mitered_t3d/onestring_physics/onestring_pipeline.py:4776`
  `_turn_angle_total`
- `src_backup_before_mitered_t3d/onestring_physics/onestring_pipeline.py:4810`
  `safe_capstan_friction`

短い引用:

```python
theta_total = _turn_angle_total(points)
channel_friction = safe_capstan_friction(mu_c, theta_total)
```

## 14. deployment simulation: 物理シミュレーション

### 論文の数式

論文はシミュレーションエネルギーを

```math
E_Simulation(v)
  = w_1 E_Rigid(v) + w_2 E_Collision(v) + w_3 E_Actuation(v)
```

と書く。`E_Rigid` と `E_Collision` はhinge最適化と同様の剛体性・衝突回避で、
`E_Actuation` はstringによるsnap/lift制約を表す。実装にはProjective Dynamicsと
ShapeOp/libiglが使われると説明される。

### 離散化

tile verticesを時間ステップ `k` ごとの状態

```math
v^k
```

として持ち、各ステップで制約射影を反復する。actuationは、lift pointやstring
channel上の点を目標位置へ近づける拘束として入る。

### 実装で使っている数式

現行実装は論文のProjective Dynamics/ShapeOpそのものではなく、PBD/PD風の軽量な
制約投影とメトリクス計算でdeploymentを近似する。CPU/GPU backendは選べるが、
一部postprocessはCPU側で行われる。

### アルゴリズム

1. `T2D` を初期状態にする。
2. string path/lift pointsからactuation目標を作る。
3. 剛体性、衝突、actuation制約を反復投影する。
4. 各フレームの誤差、残差、backend情報を記録する。
5. 最終状態とレポートを `DeploymentResult` として返す。

### コード対応

- `src/onestring_physics/onestring_pipeline.py:5998`
  `simulate_onestring_deployment`
- `src_backup_before_mitered_t3d/onestring_physics/onestring_pipeline.py:3153`
  deployment projection helpers
- `src_backup_before_mitered_t3d/onestring_physics/onestring_pipeline.py:3395`
  `_optimize_k3d_torch`
- `src_backup_before_mitered_t3d/onestring_physics/onestring_pipeline.py:3496`
  `_optimize_k2d_torch`

短い引用:

```python
result = _original.simulate_onestring_deployment(state, params, progress_callback)
```

## 15. 現行実装と論文の主要差分まとめ

| 段階 | 論文 | 現行実装 |
| --- | --- | --- |
| 平面化 | BFF | BFF風の境界矩形化 + cotan harmonic、LSCM fallback |
| CSF | 共形写像の局所伸縮 | 辺伸縮比proxyが中心 |
| split | CSF>2、高曲率線に沿う分割 | heuristic/localized split |
| K3D | ShapeOp系のprojection energy | SciPy/Torch/projective fallback |
| T3D | 法線offset + face planarity再最適化 | miter/contact plane交点 + guard |
| K2D | edge/collision/fabrication energy | edge matching + SAT/AABB等の近似 |
| hinge | 式(6)のrigid/collision/connection | rigid placement/dual hinge近似 |
| lift | GPE + Morse-Smale + Kruskal MST | gap scalar proxyによる軽量選択 |
| string | 制約付き閉路、Capstan最小化 | 曲げ角/Capstanメトリクス付き近似経路 |
| simulation | Projective Dynamics + ShapeOp/libigl | PBD/PD風の軽量投影 + backend計測 |

## 16. T3D押し出し方向ずれを調べるときの読解ポイント

T3Dのずれは、論文の「normal offset + face planarity」よりも、現行実装固有の
miter/contact-plane extrusionを見るべき問題である。

特に確認すべき不変条件:

```math
(p_i - b_i) dot n_tile > 0
```

つまりbottom vertexがtile normalに対して常に下側へ移動しているか。

さらに、隣接tileのmiter planeでbottom vertexを解くため、次のような場合に方向が
不安定になりやすい。

- 隣接tile normalがほぼ平行または反平行
- side plane normalの差 `q_a - q_b` が小さい
- split boundaryで「隣接していないが位置が一致する辺」をcontact pair扱いする
- surface reference normalの向き補正が、開いた曲面と閉じた曲面で異なる意味を持つ
- bottom planeと2つのside planeの線形系 `A x = d` が悪条件になる

対応箇所:

- `src/onestring_physics/onestring_pipeline.py:3456`
  `_extrude_tiles`
- `_orient_tile_normals_consistently`
- `_orient_tile_normals_to_outward_reference`
- `_edge_inward_normal`
- `_solve_bottom_vertex`

現時点の実装は、論文のT3D手順をそのまま実装しているのではなく、
「接触面を揃える」ことを優先した別アルゴリズムである。したがってT3Dの
押し出し方向問題を根本修正するなら、まずこの段階を論文に近い
「tile normal offset + top/bottom/contact face planarity最小化」へ戻すか、
現在のmiter解法に対して上記不変条件を制約として明示的に入れる必要がある。
