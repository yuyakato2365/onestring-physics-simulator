# onestring-physics-simulator

## WindowsノートPCへの導入

このリポジトリは、Pythonだけで動く設計・表示機能と、別途C++ビルドが
必要なAutodesk ABD物理シミュレーションを含みます。初回はCPU版ABDを
ビルドし、NVIDIA GPU搭載機では必要に応じてCUDA版を追加してください。

```powershell
git clone https://github.com/yuyakato2365/onestring-physics-simulator.git
cd .\onestring-physics-simulator\onestring-physics-simulator
powershell -ExecutionPolicy Bypass -File .\scripts\install_python_environment.ps1 -WithDevDependencies
.\.venv\Scripts\python.exe -m streamlit run app.py
```

`streamlit`コマンドが見つからない場合も、上記のように仮想環境のPythonから
`-m streamlit`で起動してください。ABDを含む完全な導入、CPU/GPUビルド、
検証、別PCへの更新手順は
[WindowsノートPC導入ガイド](docs/windows_laptop_setup_ja.md)を参照してください。

## Version 0.4.0: integrated ABD, discrete BFF, and official CEPS support

Version 0.4.0 combines the variable-topology T3D and Autodesk ABD work from
the 0.3.0 line with the latest discrete BFF and official CEPS integration.
The package metadata and runtime `__version__` use the same semantic version.

## Version 0.3.0: variable-topology T3D and Autodesk ABD bridge

Version 0.3.0 adds two independent changes:

- T3D can use authoritative variable-topology convex solids with classified
  cap/wedge/pyramid/local-thickness recovery. The old eight-vertex tile is kept
  only as a T2D/deployment compatibility proxy.
- If all valid-solid recovery tiers fail, the new selectable version shows only
  the affected panel as a gray one-sided emergency prism and records that it is
  not manufacturing-authoritative. Invalid K3D top faces still fail explicitly.
- Actuation exposes `legacy` and `abd` physics backends. `abd` invokes an
  external Autodesk `affine-body-dynamics` executable; it is not a Python
  reimplementation and never falls back to the legacy SAT projection.

### Autodesk ABD prerequisites (Windows)

The vendored ABD extension can be built as CPU-only or with CUDA CCD. Both
variants also use a TBB-parallel BiCGSTAB Newton linear solve (with the original
Eigen direct solver as a numerical fallback). Build the CPU variant with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_autodesk_abd.ps1
```

For an NVIDIA GPU, install a CUDA Toolkit supported by the local compiler and
build for the GPU architecture. An RTX 4060 Ti uses architecture `89`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_autodesk_abd.ps1 `
  -EnableCuda -CudaArchitectures 89 `
  -BuildDir third_party\affine-body-dynamics\build-gpu
```

The app searches `build-gpu\Release\abd_sim.exe` first, then
`build-parallel\Release\abd_sim.exe`, followed by the legacy `build` folder.
`ONESTRING_ABD_EXECUTABLE` or the executable-path setting still overrides this
order. CUDA accelerates the IPC broad phase/CCD; Hessian assembly, constraints,
line search, and parts of Newton remain CPU work, so GPU utilization is
intermittent rather than continuously high.

Then verify the unmodified official `cube_drop` scene before using OneString:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\verify_autodesk_abd.ps1
```

Set the verified executable:

```powershell
$env:ONESTRING_ABD_EXECUTABLE = "C:\path\to\affine-body-dynamics\build\Release\abd_sim.exe"
python -m streamlit run app.py
```

The stock Autodesk executable supports headless IPC/CCD/friction/Newton contact
and pin joints, but does not expose the unilateral OneString guide-length
constraint required by this project. A full OneString ABD run therefore also
requires an Autodesk-derived extension advertising `--onestring-manifest`.
Without that capability, the `abd` backend stops with an explicit error. It does
not silently run stock ABD without the string and does not switch to `legacy`.

The bridge writes:

- `scene.json`: official Autodesk ABD scene (rest meshes, density, initial
  poses, IPC contact, friction, gravity, stiffness, and pin joints);
- `onestring_manifest.json`: guide points, unilateral pull schedule, prescribed
  shake trajectory, and required per-frame logs;
- `sim.json` / `sim.glb`: official headless outputs;
- `onestring_abd_frames.npz`: Streamlit-ready affine-body frames and logs.

See [ABD backend details](docs/abd_backend.md) and
[T3D recovery details](docs/t3d_variable_topology_recovery.md).

## Version 2026-07-12: one-sided T3D extrusion

The selectable `2026-07-12-one-sided-t3d` version keeps every K3D quad as the
T3D top face and creates the opposite face only on the negative-normal side at
distance `t`. Shared edges continue to use the miter/contact-plane construction
so adjacent non-coplanar panels meet instead of being treated as unrelated
normal-translation prisms.

A Python research prototype inspired by **One String to Pull Them All: Fast Assembly of Curved Structures from Flat Auxetic Linkages**.

## Version 0.2.0: paper-reference initialization

Version 0.2.0 adds `paper_reference_bff`, a fail-fast S-to-M3D path backed by
the official GeometryCollective `bff-command-line` executable. It includes
strict triangle-disk validation, exact per-triangle Jacobian/SVD diagnostics,
an explicit regular square grid, fully-contained-cell cropping, strict
containing-triangle barycentric lifting, split diagnostics, and JSON output.
It never calls LSCM, PCA, harmonic mapping, or nearest geometry as a BFF
fallback. See [BFF backend](docs/bff_backend.md),
[conformal scale factor](docs/conformal_scale_factor.md), and
[traceability](docs/paper_traceability.md).

The optional `boundary_sliding_lscm` mode is LSCM with a prescribed rectangular target boundary and order-preserving sliding boundary correspondence. It is explicitly not Boundary First Flattening: it does not implement the Cherrier formula or a Poincare-Steklov operator. See [docs/boundary_sliding_lscm_ja.md](docs/boundary_sliding_lscm_ja.md).

For a reproducible Windows/CUDA setup, including the wrapper-to-base runtime path that must be present in a clone, follow [docs/home_pc_codex_handoff_ja.md](docs/home_pc_codex_handoff_ja.md) and run `python scripts/verify_home_environment.py --require-cuda` before comparing performance.

This simulator is a paper-audited OneString research prototype, not a complete paper implementation. Bare `PipelineParameters()` now selects the local discrete `bff` backend, which implements the Cherrier/NtD and best-fit boundary path for a single-disk mesh. `rectangular_harmonic_legacy` remains available explicitly for compatibility, `paper_reference_bff` invokes the official BFF CLI, and `ceps` invokes the optional official CEPS CLI. Unsupported topology and unavailable official backends fail explicitly; they do not silently substitute LSCM.

The default Streamlit workflow is:

```text
S -> Omega -> M2D / M3D -> K2D / K3D -> T2D / T3D
-> hinge optimization
-> lift point selection
-> boundary-first string path generation
-> snap + lift + rigid + hinge + collision actuation
```

The deployed physical error is evaluated against the designed assembled tile configuration `T3D`, not against the raw target surface `S`.

## 論文の流れと、この実装の流れ

このリポジトリは、OneString 論文の処理順と物理的な意図を観察するための **paper-audited prototype** です。`paper_reference_bff` は公式BFF CLIを要求し、利用できなければ明示的に失敗します。旧 `bff` 名は `rectangular_harmonic_legacy` のdeprecated aliasであり、BFFとは表示しません。K3D以降は既存近似のままなので、0.2.0全体を完全なOneString再現とは呼びません。

### 論文側の大きな流れ

論文の全体像は、目標曲面 `S` から、平面製造可能な auxetic linkage と、それを紐で引いたときの組み立て挙動を作る流れです。

```text
S
-> Omega
-> M2D
-> M3D
-> K3D
-> T3D

M2D
-> K2D
-> T2D
-> lift point / string path
-> string-driven actuation simulation
```

各段階の意味は次の通りです。

- `S`: 入力の目標曲面。
- `Omega`: `S` を平面に写したパラメータ領域。論文では BFF/LSCM 系の曲面パラメータ化を使う。
- `M2D`: `Omega` 上に敷いた初期 quad mesh。
- `M3D`: `M2D` を逆写像 `c^{-1}: Omega -> S` で曲面上に戻した mesh。
- `K3D`: planarity、square、surface fit を満たすように最適化した 3D mesh。
- `T3D`: `K3D` を厚みのある tile assembly にした、設計上の組み立て後形状。
- `K2D`: `K3D` の edge length に合うように作る平面 mesh。
- `T2D`: 製造時の flat tile linkage。hinge、gap、string channel を持つ。

論文の assembled configuration は、おおまかに次のエネルギーを最小化します。

$$
E_{\mathrm{Assembled}}(x)
= \omega_1 E_{\mathrm{Planar}}(x)
+ \omega_2 E_{\mathrm{Square}}(x)
+ \omega_3 E_{\mathrm{Surface}}(x)
$$

ここで `E_Planar` は quad が平面であること、`E_Square` は quad が正方形的な形を保つこと、`E_Surface` は目標曲面 `S` に沿うことを表します。読みやすく書けば、planarity は次のような「各 quad を best-fit plane に投影したときのずれ」です。

$$
E_{\mathrm{Planar}}
= \sum_{q}
\left\|
V_q - P_{\mathrm{plane}}(V_q)
\right\|_F^2
$$

平面側の `K2D` は、`K3D` の edge length を保ちながら、製造可能な gap と衝突しない配置を探します。

$$
E_{\mathrm{Flat}}(x)
= \omega_1 E_{\mathrm{Edge}}(x)
+ \omega_2 E_{\mathrm{Collision}}(x)
+ \omega_3 E_{\mathrm{Fab}}(x)
$$

その後、`T2D` の hinge layout では、tile が剛体であること、tile 同士が衝突しないこと、hinge connection が閉じることを同時に扱います。

$$
E_{\mathrm{Hinge}}(x)
= \omega_1 E_{\mathrm{Rigid}}(x)
+ \omega_2 E_{\mathrm{Collision}}(x)
+ \omega_3 E_{\mathrm{Conn}}(x)
$$

最後に、lift point と string path を決め、紐を引いたときにどの gap が閉じ、どの場所が持ち上がるかをシミュレーションします。概念的には actuation energy は snap と lift に分かれます。

$$
E_{\mathrm{Simulation}}(x)
= \omega_1 E_{\mathrm{Rigid}}(x)
+ \omega_2 E_{\mathrm{Collision}}(x)
+ \omega_3 E_{\mathrm{Actuation}}(x)
$$

$$
E_{\mathrm{Actuation}}
= E_{\mathrm{Snap}} + E_{\mathrm{Lift}}
$$

snap は、string path 上の向かい合う side-face midpoint を閉じる制約です。

$$
E_{\mathrm{Snap}}
= \sum_{(a,b) \in R}
\left\|
m_a(x) - m_b(x)
\right\|_2^2
$$

lift は、選ばれた lift gap 周辺を prescribed 3D target に近づける制約です。

$$
E_{\mathrm{Lift}}
= \sum_{\ell \in L}
\left\|
c_\ell(x) - p_\ell^{\mathrm{target}}
\right\|_2^2
$$

### この実装側の大きな流れ

この実装の実際の入口は `build_onestring_design(...)` と `simulate_onestring_deployment(...)` です。アプリの通常経路では、以下の順番で `OneStringDesignState` を作ります。

```text
target surface S
-> shape-preserving projected Omega
-> oversized Omega grid + crop -> M2D
-> UV triangle lookup / barycentric inverse map -> M3D
-> K3D optimization
-> T3D extrusion
-> K2D edge-length optimization
-> independent K2D tile layout
-> T2D top hinge
-> T2D dual hinge
-> gap graph
-> GPE threshold lift points
-> boundary-first string path
-> Projective-Dynamics-style snap/lift deployment
```

実装では、`S -> Omega` は BFF/LSCM の完全実装ではなく、入力曲面 `S` の境界形状を2D投影で保つ lightweight parameterization です。以前のように境界を単純な四角へ固定せず、雪だるま型やくびれを持つ入力では `Omega` もその外形に近い境界を持ちます。`M2D -> M3D` は、`M2D` の UV 点を `Omega` 上の三角形に探し、barycentric coordinates で `S` 上の点へ戻します。

$$
p_{3D}
= \lambda_1 s_1 + \lambda_2 s_2 + \lambda_3 s_3,
\quad
\lambda_1+\lambda_2+\lambda_3=1
$$

`K3D` では、論文の `E_Assembled` と同じ目的を保ちつつ、実装上は planarity / square / surface residual を NumPy / SciPy / PyTorch で解きます。大きく破綻して平らにつぶれる結果は downstream に流さず、guard で拒否します。

`K2D` では、`K3D` と edge length が合うことをまず優先します。中規模以上の grid では、重い collision relaxation を `K2D` 内で無理に回さず、collision-aware な配置を後段の independent tile layout / dual hinge layout に寄せています。

`T2D` では、shared mesh としての `K2D` をそのまま製造形状とは見なしません。`K2D` の各 face を独立 tile として扱い、pairwise hinge と gap を持つ flat linkage に変換します。さらに dual hinge layout で、各 tile を rigid SE(2) pose として動かしながら、

- `E_Rigid`: tile shape を保つ
- `E_Conn`: hinge vertex pair を近づける
- `E_Collision`: tile overlap を避ける

を local/global projection 的に解きます。

### Actuation / animation の違い

**一番大事な違い:** 論文は「実物の flat linkage を一本の紐で引くと、どのように組み上がるか」を扱っています。この実装のアニメーションは、その現象をそのまま厳密に再現するのではなく、「紐で起きるはずの効果」をいくつかの制約に置き換えて、タイルがどう動くかを近似的に見せています。

もう少し具体的にいうと、論文のアニメーション的な考え方は次のようなものです。

```text
実物の紐を引く
-> 紐が channel を通る
-> channel / gap に力が伝わる
-> 必要な場所が閉じる
-> lift point が持ち上がる
-> 剛体タイルがヒンジまわりに動いて 3D 形状になる
```

この実装では、紐そのものは粒子やワイヤとしてシミュレーションしていません。代わりに、次のように置き換えています。

```text
string path を先に計算する
-> その path 上の gap に snap constraint をかける
-> lift point に lift constraint をかける
-> hinge constraint で接続を保つ
-> rigid projection で各 tile の形を保つ
-> collision projection で大きな重なりを避ける
-> その途中経過を frame として保存して再生する
```

つまり、画面で見えている動きは **「紐の物理そのもの」ではなく、「紐が作るはずの効果を制約として入れたタイル運動」** です。ここが一番の違いです。

アニメーションを理解するときは、次の3段階に分けると分かりやすいです。

1. `T2D dual hinge` が出発点です。
   これは製造前の平らなタイル配置です。アニメーションはここから始まります。

2. `string path` が「どの gap を動かすか」を決めます。
   ただし、紐のたるみ、張力波、接触摩擦、実際の曲げは直接解いていません。path 上の gap に「閉じる方向の制約」を与えます。

3. solver が snap / lift / hinge / rigid / collision を何度も投影します。
   その結果、タイルが少しずつ `T3D` に近づきます。この途中結果が `DeploymentResult.frames` で、アプリはそれを再生しています。

比較表でいうと、こうです。

| 観点 | 論文・実物で起きること | この実装でやっていること |
| --- | --- | --- |
| 紐 | 実際に channel を通って引かれる | 紐そのものは解かず、先に `string_path` を決める |
| 張力 | 紐の張力が gap や lift point に伝わる | snap / lift の位置制約として代用する |
| gap が閉じる動き | 紐に引かれて channel/gap が閉じる | side-face midpoint pair を目標距離へ近づける |
| 持ち上がり | lift point が構造を起こす | lift gap 周辺 tile center を 3D target へ近づける |
| タイル | 剛体パネルとしてヒンジまわりに動く | Kabsch/SVD projection で各 tile を剛体形状に戻す |
| ヒンジ | 隣のタイルと接続されたまま回転する | hinge vertex pair を近づける constraint をかける |
| 衝突 | 実物同士が接触して押し合う | 軽量な AABB / bounded projection で重なりを減らす |
| アニメーション | 物理現象の観察・検証 | 近似 solver の保存 frame を再生 |

したがって、このアニメーションは次のどちらでもありません。

- 単なる見た目の補間: `T2D` と `T3D` を直線的に混ぜているだけではありません。
- 完全な実物物理: 紐の張力、摩擦、接触、慣性を厳密に解いているわけでもありません。

いちばん正確な言い方は、**「OneString の紐駆動で起きる主要な効果を、snap / lift / hinge / rigid / collision 制約として近似し、その solver の途中経過を再生している」** です。

各 simulation step では、進行度 `alpha` を 0 から 1 に上げながら、以下の投影を反復します。

```text
velocity / damping update
-> lift projection
-> snap projection
-> hinge projection
-> rigid tile projection
-> collision projection
-> target pose/contact guard
-> optional final rigid projection
```

snap projection は、現在の side-face midpoint `p_a, p_b` を見て、string path 上の gap を目標 separation に近づけます。

$$
p_{\mathrm{mid}} = \frac{p_a + p_b}{2}
$$

$$
d_{\mathrm{desired}}(\alpha)
= (1-\alpha)d_{\mathrm{rest}} + \alpha d_{\mathrm{target}}
$$

$$
p_a^\* = p_{\mathrm{mid}} + \frac{1}{2}d_{\mathrm{desired}},
\quad
p_b^\* = p_{\mathrm{mid}} - \frac{1}{2}d_{\mathrm{desired}}
$$

lift projection は、lift gap の周辺 tile center を、2D lift point から 3D lift target へ移動する target に近づけます。

$$
p_{\ell}(\alpha)
= (1-\alpha)p_{\ell}^{2D} + \alpha p_{\ell}^{3D}
$$

rigid projection は、各 tile について rest shape から current shape への best rigid transform を Kabsch/SVD で求め、変形した tile を剛体形状へ戻します。このため、snap/lift が強くても tile 自体がゴムのように伸びるのではなく、剛体 panel として動くように寄せています。

### 論文と実装の主な違い

| 項目 | 論文の考え方 | この実装 |
| --- | --- | --- |
| `S -> Omega` | BFF/LSCM 系の曲面パラメータ化 | boundary-shape-preserving projected UV parameterization |
| `M2D -> M3D` | `c^{-1}: Omega -> S` による曲面への lift | UV triangle lookup + barycentric interpolation |
| `K3D` | ShapeOp/libigl 的な projection stack | NumPy/SciPy/PyTorch residual solve + flattening guard |
| `K2D` | `K3D` edge length と fabrication/collision を同時考慮 | edge matching 優先。重い collision は後段へ寄せる |
| `T2D` | 製造可能な hinge linkage | independent tile layout + top hinge + dual hinge layout |
| lift point | Morse-Smale segmentation / peak coupling | GPE-like score + `lift_tau` threshold |
| string routing | channel energy と friction を考慮した routing | gap graph 上の boundary-first route + Capstan-style turn cost |
| actuation | string-driven deployment | snap/lift positional constraints |
| collision | より厳密な tile collision | 通常経路では軽量 AABB / bounded projection |
| animation | 物理 deployment の可視化 | PD-style constraint simulation frames の再生 |

結論として、この実装は「論文と同じ物理現象を目指した軽量な近似シミュレータ」です。論文の数値結果や solver 実装を完全再現するものではありませんが、`T2D` の flat linkage が string path に沿った snap/lift 制約で `T3D` に近づく、という OneString の機構的な流れを確認するための実装です。

## Quick Start

```powershell
cd onestring-physics-simulator
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
$env:ONESTRING_BFF_EXECUTABLE = "C:\path\to\boundary-first-flattening\binaries\windows-v1.6\bff-command-line.exe"
python scripts\verify_reference_bff.py
streamlit run app.py
```

For reproducing the known-good work-PC environment on another Windows PC,
start with `CHATGPT_HANDOFF.md` and `docs/home_pc_codex_handoff_ja.md`.
The CUDA PyTorch snapshot used on the work PC is captured in
`requirements-local-cu128-lock.txt`; `requirements.txt` alone does not pin a
CUDA-enabled PyTorch build.

If you do not want editable install:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Default Demo

The default demo uses:

- target: dome
- grid: 3x3
- selectable local `bff`, `rectangular_harmonic_legacy`, `lscm_free_boundary`, official `paper_reference_bff`, and official `ceps` modes
- inverse parameterization lift `c^-1` from `M2D` in `Omega` back to `M3D` on `S`
- 3D optimization for `K3D` with planarity, square, and surface objectives
- 2D edge matching for `K2D`
- frustum extrusion for `T3D`, and `T2D` whose top vertices are generated from the optimized `K2D`
- dual-hinge-ready hinge data
- GPE-based lift point selection
- boundary-first gap-graph string routing
- quasi-static snap/lift deployment

## What Is Implemented

- Built-in height-field targets: flat, dome, saddle, wave, gaussian bump
- OBJ/STL/PLY loading through `trimesh`
- `OneStringDesignState` with formal intermediate representations
- Fig. 5-style pipeline viewer
- `SurfaceParameterization` representation for `c : S -> Omega` and `c^-1 : Omega -> S`
- M3D generation by UV triangle lookup and barycentric interpolation on the parameterized surface mesh
- K3D full-3D vertex optimization with a flattening guard
- `S`, `Omega`, `M2D`, `M3D`, `K3D`, `K2D`, `T3D`, `T2D top hinge`, and `T2D dual hinge` views
- Assembled 3D optimization metrics:
  - M3D construction method
  - parameterization method
  - M3D surface distance mean / max
  - UV triangle lookup failures
  - height-field shortcut flag
  - planarity error before / after
  - surface fit error before / after
  - square error before / after
  - edge length variance before / after
- Flat 2D edge matching metrics:
  - edge matching error before / after
  - max absolute `K2D.z` to prove the layout is planar
  - RMS and max displacement from `M2D` to `K2D`
  - independent-tile overlap count and minimum clearance
  - gap angle range
  - fabrication clearance proxy
- `T2D` correctness metrics:
  - top vertices matching `K2D`
  - top-vertex RMS displacement from `M2D`
  - 8 vertices per tile
  - side-face count
- Gap graph visualization with paper-style node labels:
  - `0` vertical gap
  - `1` horizontal gap
  - `-1` virtual boundary entrance
  - `-2` split boundary gap placeholder
- GPE-based lift point selection
- Boundary-first string path with turn angle and simplified channel friction
- Snap/lift actuation with metrics:
  - target surface fit error to `S`
  - final deployment error to `T3D`
  - snap error
  - lift error
  - rigid error
  - hinge error
  - collision count
  - turn angle total
  - estimated channel friction
  - kinetic energy
  - stable state
- High-Fidelity Physical Mode controls:
  - hinge rotational stiffness
  - hinge damping
  - tile mass and inertia proxy
  - gravity
  - contact friction
  - string channel friction
  - quasi-static pulling speed
  - solver substeps
  - damping ratio
- Complexity panel for grid-size scaling estimates
- Batched Plotly rendering for quad meshes and tile assemblies
- Optional PyTorch CUDA backend for tensorized K3D optimization when CUDA is available

## Removed From Default Actuation

The audited default mode does not use these as the default actuator:

- central tendon
- debug goal attraction
- direct target-surface attraction
- rope-particle Verlet simulation as the primary deployment mechanism

Legacy rope/tendon code remains in the package for compatibility with earlier tests and examples, but the app defaults to the OneString-style constraint path.

## Current Approximations

- The default M3D construction uses the stored `Omega` map: M2D vertices live in `Omega`, then each UV point is mapped back to `S` using UV triangle lookup and barycentric interpolation on the corresponding surface triangle. This depends on the reported `flattening_backend` and is tracked with quality metrics rather than treated as exact paper equivalence.
- Direct height-field lifting `[u, v, z=f(u,v)]` is available only through the explicit `analytic_scaled_heightfield_debug` M3D construction mode and is tracked as a debug shortcut.
- Version 0.2.0 exposes `rectangular_harmonic_legacy`, `lscm_free_boundary`, and `paper_reference_bff`. The deprecated `bff` alias selects the legacy rectangular harmonic method and emits the mandatory non-BFF warning. `paper_reference_bff` uses only the official CLI and raises `ReferenceBFFUnavailableError` if it cannot run.
- M2D overlays a grid on `Omega` and then clips whole quads against the Omega boundary polygon. Free-boundary Omega domains may be non-rectangular, so the overlay is rebuilt at a slightly higher density when needed. `rectangular_debug` and `rect_harmonic` remain explicit non-BFF alternatives.
- CSF estimation uses local 3D/UV edge-stretch ratios. Regions whose normalized stretch exceeds `2.0` generate a coarse Omega split line. The implementation detects simple reflection symmetry in both `S` and `Omega`; when symmetry is detected, M2D crop results and CSF split lines are mirrored across the detected axis before the split is applied. The split is snapped to an existing M2D grid line and duplicates the vertices on one side of the line, so the topology is cut without deleting neighboring quads. This is still a lightweight approximation of the paper's split strategy, not a full BFF/CSF segmentation implementation.
- When a dominant surface peak is detected, the Omega overlay grid is shifted so the peak's UV position lands on an M2D grid vertex. This makes the corresponding K3D peak occur at a shared corner where four panels can meet; CSF split lines are allowed to pass through that vertex.
- `K3D` optimization uses a compact least-squares height-field approximation rather than the full projection stack from the paper.
- `K3D` optimization rejects invalid flattened results and falls back to `M3D` rather than passing a collapsed assembled state downstream.
- `K2D` edge matching uses a simplified optimizer/relaxation model. The stored mesh remains planar with `z = 0`, and the app renders an independent per-tile top-face layout with visible gaps instead of a continuous terrain-like quad surface.
- `T2D` is a fabrication-layout approximation generated from the independent `K2D` tile top vertices and projected frustum offsets, with side faces and hinge markers exposed for inspection.
- Frustum extrusion uses per-tile normal offsets plus face-planarity reporting.
- Dual hinge placement uses a local dihedral proxy rather than the full global fabrication optimization.
- Morse-Smale lift point selection is approximated with GPE peaks and threshold clustering.
- Collision handling is AABB-based with projection penalties.
- String channel friction is a simplified Capstan-style estimate from cumulative turn angle.
- High-fidelity mode is an exposed extension mode, not a validated physical contact simulator.

## Performance And GPU Notes

The app avoids generating every stage figure on each Streamlit rerun. Use the `View stage` selector to render only one stage at a time. Animation is generated on demand from the final deployment view.

GPU acceleration is used for tensorized optimization when PyTorch CUDA is available and the compute backend is set to `auto` or `cuda`. UI rendering, Plotly visualization, file I/O, and graph routing remain CPU-side. If `cuda` is explicitly requested but CUDA is unavailable in the Streamlit Python environment, the app raises a visible error instead of silently falling back to CPU.

Current GPU coverage:

- K3D optimization: optional PyTorch CUDA path for analytic height fields
- K2D optimization: optional PyTorch CUDA path, otherwise SciPy/projective NumPy path
- Deployment simulation: optional PyTorch CUDA constraint path, otherwise CPU constraint projection path

The `Complexity / Backend` view separates topology growth from backend status so slowdowns are easier to attribute. It reports `sys.executable`, PyTorch version, PyTorch CUDA version, torch import path, CUDA device count/current device, GPU name, capability, and memory counters. It also includes:

- `Run GPU self-test`, which allocates a CUDA tensor of shape `(4096, 4096)`, runs `x @ x.T`, synchronizes, and reports elapsed time plus peak memory.
- `nvidia-smi` probing, which checks whether the NVIDIA driver can see the GPU independently of PyTorch.

If `nvidia-smi` sees a GPU but `torch_available` or `cuda_available` is false, install a CUDA-enabled PyTorch build into the same `.venv` used by Streamlit, following the current official PyTorch selector for your driver/CUDA combination.

The string routing metrics use `log_channel_cost = mu_c * theta_total` as the stable routing cost. The display-only Capstan friction estimate uses a guarded `expm1` calculation and returns `inf` instead of raising `OverflowError` when the exponent is too large.

## Command Line Smoke Test

```powershell
python -m pytest
```

You can also run the OneString pipeline in Python:

```python
from onestring_physics.input_shape import create_builtin_shape
from onestring_physics.onestring_pipeline import (
    DeploymentParameters,
    PipelineParameters,
    build_onestring_design,
    export_t2d_stl,
    simulate_onestring_deployment,
)

target = create_builtin_shape("dome", {"amplitude": 0.75, "radius": 2.2})
state = build_onestring_design(
    target,
    PipelineParameters(
        nx=3,
        omega_parameterization_mode="bff",
        omega_boundary_mode="paper_default",
    ),
)
stl_bytes, export_metrics = export_t2d_stl(state, "onestring_t2d_dual_hinge.stl", panel_size=0.1)
state.simulation_result = simulate_onestring_deployment(
    state,
    DeploymentParameters(steps=32, solver_iterations=12),
)
print(state.simulation_result.metrics)
print(export_metrics)
```

## Repository Layout

```text
onestring-physics-simulator/
  app.py
  src/onestring_physics/
    onestring_pipeline.py
    visualization.py
    animation.py
    ...
  examples/
  tests/
  docs/
```

See `PAPER_COMPLIANCE_AUDIT.md`, `docs/physics_model.md`, `docs/limitations.md`, and `docs/roadmap.md` for notes. The Streamlit app and `onestring_pipeline.py` are now the canonical entry points for the audited prototype.
