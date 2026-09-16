# Large Steps 入力メッシュ conditioning

## 1. 何をしているのか

`Large Steps in Inverse Rendering of Geometry` の中心アイデアは、頂点位置 `v` を直接最適化する代わりに

\[
u=(I+\lambda L)v
\]

という differential coordinates（微分座標）`u` を最適化変数として使うことです。

ここで

- `v \in R^{n\times 3}`: 3D vertex positions（3D頂点位置）
- `L`: mesh Laplacian（メッシュラプラシアン）
- `\lambda>0`: smoothing scale（平滑化スケール）
- `M=I+\lambda L`

です。

元の頂点位置は

\[
v=M^{-1}u=(I+\lambda L)^{-1}u
\]

として復元します。

重要: Large Steps は remeshing（リメッシュ）ではありません。edge split / collapse / flip は行わず、connectivity は固定です。また Large Steps 自体は「三角形品質」を定義しません。何を良い形とするかは別の目的関数 `E(v)` が決めます。

---

## 2. なぜ "Large Steps" なのか

頂点位置を直接最急降下すると

\[
v_{k+1}=v_k-\eta\nabla_v E
\]

です。

一方 `u=Mv` を変数にすると、連鎖律より概念的には

\[
\nabla_u E=M^{-T}\nabla_v E
\]

となります。`L` が対称なら `M^{-T}=M^{-1}` なので、gradient に

\[
(I+\lambda L)^{-1}
\]

という spatial low-pass filter（空間的ローパスフィルタ）をかけるのと似た作用になります。

Laplacian の固有分解

\[
L=\Phi\,\mathrm{diag}(\mu_1,\ldots,\mu_n)\Phi^T
\]

を考えると

\[
(I+\lambda L)^{-1}
=\Phi\,\mathrm{diag}\left(\frac{1}{1+\lambda\mu_i}\right)\Phi^T.
\]

高周波 mode（局所的でギザギザした変形）は `\mu_i` が大きいため

\[
\frac{1}{1+\lambda\mu_i}
\]

で強く減衰し、低周波 mode（広い範囲が一緒に動く滑らかな変形）は比較的残ります。

したがって、1頂点だけを局所的に動かすより、近傍を含む広い領域を協調して動かす search direction（探索方向）が得られやすくなります。これが "Large Steps" の本質です。

`\lambda` を大きくすると、より広域に滑らかな変形になります。ただし大きすぎると局所修正能力が落ちます。

---

## 3. このプロジェクトでの目的

S -> Omega の bijective optimizer（全単射最適化）では、triangle signed area（符号付き三角形面積）が0に近づくと safe-step が非常に小さくなります。

\[
A_t(\alpha)=A_t(v+\alpha d)
\]

に対し、最初に

\[
A_t(\alpha)=\varepsilon
\]

となる `\alpha` より手前でstepを止めるためです。

skinny triangle（細長い三角形）が多いと、非常に小さい `\alpha` しか許されず、S -> Omega の境界更新がほとんど進まなくなることがあります。

そこで元の意図は

```text
S_input
  -> connectivityを変えずに三角形配置を改善
  -> S_conditioned
  -> bijective free-boundary
  -> Omega
```

でした。

---

## 4. 現在の内製 conditioning の目的関数

現在の実装では、各triangleの品質

\[
q_t=\frac{4\sqrt{3}A_t}{l_1^2+l_2^2+l_3^2}
\]

を使います。正三角形で `q_t=1`、細長くなるほど0に近づきます。

quality term（品質項）は

\[
E_{quality}=\frac{1}{|F|}\sum_t(1-q_t)^2
\]

です。

さらに edge uniformity（辺長一様化）

\[
E_{edge}=\frac{1}{|E|}\sum_e
\left(\log\frac{l_e}{h}\right)^2
\]

を加えます。`h` は初期edge lengthのmedianです。

形状を元surfaceから大きく変えないため

\[
E_{normal}=\frac{1}{n}\sum_i
\big((v_i-v_i^0)\cdot n_i^0\big)^2
\]

と

\[
E_{pos}=\frac{1}{n}\sum_i\|v_i-v_i^0\|^2
\]

も加えます。

全体は

\[
E=
 w_qE_{quality}
 +w_eE_{edge}
 +w_nE_{normal}
 +w_pE_{pos}.
\]

候補頂点は元meshのlocal triangle patchへprojectionし、boundary verticesは固定しています。

---

## 5. 現在わかっている問題

この目的関数は「平均品質」を良くするため、worst triangle（最悪三角形）を直接守っていません。

例えば

```text
q05: 0.459 -> 0.463
minimum angle: 1.24 deg -> 0.63 deg
```

のように、下位5%の平均傾向は少し改善しても、最悪三角形だけさらに悪化することがあります。

つまり問題は Large Steps の座標変換そのものではなく、Large Steps 上に載せている conditioning objective が「S -> Omega のsafe-stepを守る」という目的に十分一致していない点です。

改善案は、例えば lower-tail barrier（低品質側障壁）

\[
E_{bad}=\sum_t\max(q_{target}-q_t,0)^2
\]

や minimum-angle barrier（最小角障壁）を使い、さらに

\[
q_{min}^{new}\ge q_{min}^{old}
\]

をhard acceptance rule（採用条件）にすることです。

---

## 6. Large Steps と remeshing の違い

### Large Steps

- connectivity固定
- vertex positionsだけ変える
- `u=(I+\lambda L)v` というoptimization parameterization
- topology / edge graphは変わらない

### isotropic remeshing（等方リメッシュ）

- edge split
- edge collapse
- edge flip
- vertex relocation

などでconnectivity自体を変えられます。

そのため、元connectivityに非常に悪いskinny triangleが組み込まれている場合は、Botsch-Kobbelt remeshingの方が直接的です。

現在の比較実験では

```text
closed Bunny
 -> Botsch-Kobbelt remesh
 -> Blenderで底面を削除
 -> open disk Bunny
 -> OneString (Large Steps OFF)
```

を推奨しています。

---

## 7. GPU backend

### Alienware / NVIDIA

S -> Omega は full CUDA-resident backend を使います。

```text
S -> Omega FULL CUDA
CUDA Omega i/N
GPU-resident
```

UV、gradient、L-BFGS、safe-step、line search、boundary validity、harmonic responseをCUDA上に保持し、CPUへ戻すのは初期化と最終監査が中心です。

### MacBook Pro M3 Pro / Apple Silicon

S -> Omega は full MPS-resident backend を使います。

```text
S -> Omega FULL MPS
MPS Omega i/N
Metal-resident
```

PyTorch sparse matrix依存を避けるため、harmonic boundary responseはmatrix-free PCGです。

combinatorial Laplacianについて

\[
(Lx)_i=d_ix_i-\sum_{j\in N(i)}x_j
\]

をtriangle edge listから直接評価します。

このため `L_{II}` を疎行列として明示的に保存せず、MPS上の `index_add` でmatvecを構成できます。

### auto selection

```text
CUDA available -> CUDA
else MPS available -> MPS
else -> CPU
```

環境変数:

```text
ONESTRING_BIJECTIVE_DEVICE=auto|cuda|mps|cpu
```

---

## 8. 保証・制約

内製Large Steps conditioningについて:

- face connectivityは変更しない
- 3D boundary verticesは固定
- orientation ratioが閾値以下のcandidateはreject
- candidateを元surface local patchへ投影
- remeshing / edge flip / edge split / edge collapseはしない

S -> Omegaについて:

- accepted stateはpositive signed areaを維持
- boundary self-intersectionをreject
- boundary first-singularity safe-stepを使用
- 最終UVはCPUでglobal overlap audit

Large Steps preprocessingをOFFにしても、CUDA/MPSのS -> Omega backendや後段のM2D/K3D/K2D処理はそのまま利用できます。
