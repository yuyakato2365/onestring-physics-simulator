# 家PCのCodex向け OneString 復旧・再現手順

## 結論

GitHubの `main` は、家PC側でこのPCの基準状態より後のコミットに進んでいました。
しかし、その系列はローカルより大幅に遅くなったと報告されているため、現在は
この作業PCの状態を正とします。

基準にするコミット:

```text
03a92394deda9063cebf3adf2f38e011c2ed6983
Stabilize OneString Omega and hinge pipeline
```

家PC側Codexは、後続のremote commitを安易にmergeしないでください。必要なら
別branchで比較し、`app.py` と `src/onestring_physics/onestring_pipeline.py` の差分を
性能計測付きで一つずつ戻します。

## このPCで確認した環境

2026-07-09に確認した作業PCの環境:

```text
Windows: Windows 11 10.0.26200
Python: 3.12.13
GPU: NVIDIA GeForce RTX 4080 Laptop GPU
NVIDIA driver: 581.95
nvidia-smi CUDA Version: 13.0
PyTorch: 2.11.0+cu128
PyTorch CUDA runtime: 12.8
torch.cuda.is_available(): True
```

この環境差が速度差の最重要候補です。`requirements.txt` だけではCUDA版PyTorchが
固定されません。

## 家PCでの推奨セットアップ

PowerShell:

```powershell
cd C:\path\to\onestring-physics-simulator
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-local-cu128-lock.txt
pip install -e .
```

lockで失敗する場合:

```powershell
pip install -r requirements-gpu-cu128.txt
pip install -r requirements.txt
pip install -e ".[dev]"
```

CUDA確認:

```powershell
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Streamlit確認:

```powershell
streamlit run app.py
```

Streamlit画面内でも `sys.executable`、PyTorch version、CUDA available、GPU名を確認し、
PowerShellで使った `.venv` と同じPythonを使っているか見てください。

## 検証コマンド

```powershell
python -m pytest tests -q
python -m py_compile app.py src\onestring_physics\onestring_pipeline.py tests\test_onestring_pipeline.py
git diff --check
```

`git diff --check` のCRLF警告だけなら、内容破損とは限りません。

## 遅い時の切り分け

まず環境を疑います。

- Streamlitが別のPythonを使っていないか。
- CPU版Torchが入っていないか。
- `torch.cuda.is_available()` がStreamlit内でfalseになっていないか。
- GPUが見えていても、該当ステージのaccepted resultがCPU refinementに戻っていないか。

環境が同じなら、次にトポロジーを疑います。

- `S -> Omega -> M2D` で面数が増えすぎていないか。
- `m2d_general_omega_overlay_rebuilt` がtrueになっていないか。
- `omega_boundary_forced_rectangle` が期待通りtrueか。
- Dual Hinge前のhinge spec数・hinge graph数が急増していないか。
- `fast_t2d` や後続remote commitの重いUI/optimizer変更が混ざっていないか。

## Git運用

この引き継ぎ後のGitHub `main` は、このPCから復旧した状態を正とします。
家PC側で作業する場合は、まず:

```powershell
git fetch origin
git checkout main
git reset --hard origin/main
git clean -fd
```

その後、必要な実験は必ず別branchで行ってください。

## GitHubに載せないもの

`.venv` は数GBあり、CUDA DLLを含むためGitHubには載せません。
代わりに次のファイルを信頼してください。

- `requirements-local-cu128-lock.txt`
- `requirements-gpu-cu128.txt`
- `requirements.txt`
- `pyproject.toml`
- `CHATGPT_HANDOFF.md`
- `docs/current_algorithm_overview_ja.md`

ローカルにある `*.backup_before_*`、`src_backup_before_*`、`chatgpt_handoff_*`、
`streamlit-*.log` は調査用の残骸または過去snapshotであり、基準コードではありません。
