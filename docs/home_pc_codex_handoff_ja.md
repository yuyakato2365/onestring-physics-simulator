# 家PCのCodex向け OneString完全再現手順

## 正とする状態

この文書を含む `origin/main` の最新commitを、この作業PCから公開した基準状態とします。
家PCでは古いlocal branchをmergeせず、まず新しいcloneで再現確認してください。

対象repository:

```text
https://github.com/yuyakato2365/onestring-physics-simulator.git
```

## 実行時に本当に読み込まれるファイル

現状は互換wrapper構成です。ファイル名だけを見て推測せず、次の経路を維持してください。

```text
app.py
  -> app_backup_before_mitered_t3d.py

src/onestring_physics/onestring_pipeline.py
  -> src_backup_before_sideface_contact/onestring_physics/onestring_pipeline.py
  -> wrapper側がS->Omega、T3D、K2Dなどの差し替え関数を登録
```

以前のGitHub状態では2番目のbase pipelineが `.gitignore` により欠落していました。
今回から次のファイルは名前に `backup` が含まれていても実行必須ファイルとして追跡します。

- `app_backup_before_mitered_t3d.py`
- `src_backup_before_sideface_contact/onestring_physics/onestring_pipeline.py`

家PCのCodexは、これらを不要なbackupとして削除したり、wrapperを再帰的に読み込む形へ
置き換えたりしないでください。

## このPCの基準環境

2026-07-10確認値:

```text
Windows: Windows 11 10.0.26200
Python: 3.12.13 64-bit
GPU: NVIDIA GeForce RTX 4080 Laptop GPU
NVIDIA driver: 581.95
nvidia-smi CUDA Version: 13.0
PyTorch: 2.11.0+cu128
PyTorch CUDA runtime: 12.8
torch.cuda.is_available(): True
NumPy: 2.5.0
SciPy: 1.18.0
Streamlit: 1.58.0
Plotly: 6.8.0
Trimesh: 4.12.2
```

完全なPython package固定値は `requirements-local-cu128-lock.txt` にあります。
`requirements.txt` だけではCUDA版PyTorchや間接依存が固定されないため、速度再現には使わないでください。

## 新規cloneからの再構築

PowerShell:

```powershell
git clone https://github.com/yuyakato2365/onestring-physics-simulator.git
cd onestring-physics-simulator
git checkout main

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-local-cu128-lock.txt
python -m pip install -e . --no-deps
```

PyTorch CUDA 12.8 wheelを先に導入する必要がある場合:

```powershell
python -m pip install -r requirements-gpu-cu128.txt
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]" --no-deps
```

## 必須の自動確認

```powershell
python scripts\verify_home_environment.py --require-cuda
python -m pip check
python -m pytest tests\test_onestring_pipeline.py -q
python -m py_compile app.py app_backup_before_mitered_t3d.py src\onestring_physics\onestring_pipeline.py src\onestring_physics\visualization.py
git status --short
```

検証scriptの出力で、少なくとも次を確認します。

- `runtime_pipeline_wrapper` がclone内の `src/onestring_physics/onestring_pipeline.py`
- `runtime_pipeline_base` がclone内の追跡済み `src_backup_before_sideface_contact/.../onestring_pipeline.py`
- `runtime_override_source` がwrapper側
- `torch_version` が `2.11.0+cu128`
- `torch_cuda_runtime` が `12.8`
- `cuda_available` が `true`
- `required_files_missing` が空

## 起動

必ず有効化した `.venv` のPythonから起動します。

```powershell
python -m streamlit run app.py
```

`streamlit run app.py` だけを使うと、PATH上の別Pythonを拾う場合があります。

## 遅い場合の比較順

1. `scripts/verify_home_environment.py --require-cuda` のJSONをこのPCと比較する。
2. StreamlitのCompute Backendで `requested_backend` と `actual_backend` を確認する。
3. `S -> Omega -> M2D` の頂点・面数と `m2d_general_omega_overlay_rebuilt` を比較する。
4. Dual Hinge前のhinge spec数、候補pair数、各stage時間を比較する。
5. 同じtarget、grid size、surface subdivisions、parameterization modeで比較する。

CUDAが見えていても、すべてのstageがGPUだけで動くわけではありません。CPU geometry処理や
postprocess時間も比較し、CUDA availabilityだけで高速経路と判断しないでください。

## GitHubに載せないもの

次は再構築可能、機種依存、または重複生成物なので不要です。

- `.venv/`
- `__pycache__/`, `.pytest_cache/`
- `streamlit-*.log`
- `src/onestring_physics.zip`
- 一時patch zipと、実行経路に含まれない古いbackup tree

Git cloneの完全性はGit objectで保証されます。`.venv` やCUDA DLLをcommitするのではなく、
固定lockと検証scriptから同じ環境を再構築してください。
