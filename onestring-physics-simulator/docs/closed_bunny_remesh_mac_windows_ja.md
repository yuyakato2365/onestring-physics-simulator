# Closed Bunny remeshing workflow: MacBook Pro M3 Pro / Alienware

## 目的

この手順は次の実験用です。

```text
closed Bunny
  -> Botsch-Kobbelt isotropic remeshing
  -> closed remeshed Bunny
  -> Blender で底面 faces を削除
  -> single-boundary open disk Bunny
  -> OneString input
  -> Large Steps conditioning OFF
  -> bijective free-boundary -> Omega
```

重要: Mitsuba 3 の `mi.ad.LargeSteps` 自体はリメッシャではありません。Mitsuba 3 の公式 shape-optimization tutorial でも、remeshing は GPyToolbox の `remesh_botsch` (Botsch-Kobbelt) を使います。この実験では LargeSteps inverse rendering は使わず、まず入力 triangulation の品質だけを切り分けます。

リポジトリの共通スクリプト:

```text
tools/remesh_closed_bunny.py
```

は Mac / Windows の両方で同じです。

---

## A. MacBook Pro M3 Pro (Apple Silicon)

### 1. Python environment

Terminal:

```bash
cd ~/Documents
python3 -m venv MitsubaBunny/.venv
source ~/Documents/MitsubaBunny/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "mitsuba>=3.9" gpytoolbox trimesh numpy
```

Mitsuba 3.9 以降は Apple Silicon GPU 用の Metal variants を提供します。

確認:

```bash
python - <<'PY'
import mitsuba as mi
print('Mitsuba:', mi.__version__)
print('Variants:', mi.variants())
print('metal_ad_rgb:', 'metal_ad_rgb' in mi.variants())
PY
```

期待:

```text
metal_ad_rgb: True
```

Mitsuba の differentiable rendering / LargeSteps を将来 Mac で使う場合は:

```python
import mitsuba as mi
mi.set_variant('metal_ad_rgb', 'llvm_ad_rgb')
```

とし、Metal が使えない環境では LLVM CPU へ fallback できます。

### 2. Remesh

例:

```bash
cd /path/to/onestring-physics-simulator/onestring-physics-simulator
python tools/remesh_closed_bunny.py \
  ~/Documents/Bunny/bunny_closed.obj \
  ~/Documents/Bunny/bunny_closed_remeshed.obj \
  --iterations 10 \
  --edge-scale 1.0
```

まず `--edge-scale 1.0` を使います。

- `1.0`: 元の median edge length を維持しながら三角形配置を均質化
- `0.75`: 少し高密度
- `0.5`: 約半分の target edge length。かなり高密度になるので、最初から使わない

スクリプトは before / after について以下を表示します。

- minimum angle
- angle p05
- triangle quality q01 / q05
- edge-length CV
- vertex / face count
- watertightness

### 3. Blender

`bunny_closed_remeshed.obj` を開く。

1. Edit Mode
2. Face Select
3. 底面にしたい faces を選択
4. `X -> Faces`
5. `Select -> Select All by Trait -> Non-Manifold`
6. 底の boundary だけが選択されることを確認
7. `A -> Shift+N` で Recalculate Outside
8. OBJ として `bunny_remeshed_open.obj` を export

頂点を直接 Delete Vertices するより、Faces を削除する方が意図しない大きな穴を作りにくい。

### 4. OneString on Mac

現在の `bijective_free_boundary` は CUDA が無い場合 CPU solver へ fallback するため、Apple Silicon でも実行可能です。Large Steps preprocessing は今回 OFF にします。

```text
Omega parameterization mode = bijective_free_boundary
Large Steps mesh conditioning = OFF
```

CUDA 専用の full-resident Omega loop は Mac では使わず CPU fallback になります。Mitsuba 3 の Metal backend と PyTorch/CUDA backend は別物なので混同しないこと。

---

## B. Alienware / Windows / NVIDIA RTX

### 1. Python environment

PowerShell:

```powershell
cd C:\Users\yjiat\Documents
py -3.11 -m venv MitsubaBunny\.venv
& "C:\Users\yjiat\Documents\MitsubaBunny\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install "mitsuba>=3.9" gpytoolbox trimesh numpy
```

確認:

```powershell
python -c "import mitsuba as mi; print('Mitsuba:', mi.__version__); print(mi.variants()); print('cuda_ad_rgb:', 'cuda_ad_rgb' in mi.variants())"
```

期待:

```text
cuda_ad_rgb: True
```

Mitsuba の differentiable rendering / LargeSteps を将来使う場合:

```python
import mitsuba as mi
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
```

### 2. Remesh

```powershell
cd C:\Users\yjiat\Documents\OneString-large-steps-mesh-conditioning\onestring-physics-simulator

python tools\remesh_closed_bunny.py `
  "C:\Users\yjiat\Documents\Bunny\bunny_closed.obj" `
  "C:\Users\yjiat\Documents\Bunny\bunny_closed_remeshed.obj" `
  --iterations 10 `
  --edge-scale 1.0
```

Blender 以降は Mac と同じ。

### 3. OneString on Alienware

```text
Omega parameterization mode = bijective_free_boundary
Large Steps mesh conditioning = OFF
```

NVIDIA CUDA PyTorch が利用可能なら S -> Omega は full GPU-resident CUDA path を使う。

---

## 比較実験で保存する値

同一 closed Bunny から Mac / Windows で前処理した結果について以下を保存する。

```text
Before / remeshed closed / Blender-cut open
- vertex count
- face count
- minimum angle
- angle p05
- q01
- q05
- edge CV
```

OneString 側:

```text
- Floater initialization mode
- S -> Omega runtime
- minimum signed Omega triangle area
- near-degenerate count
- accepted-state flip count
- safe-step minimum / median
- final energy
- K2D result
```

同じ GPyToolbox version と同じ `--iterations`, `--edge-scale` を使えば、Mac と Windows の差よりも入力 triangulation 改善の効果を比較しやすい。
