# WindowsでSplit / Panel validation appを使う

この文書は `agent/large-steps-mesh-conditioning` ブランチの `app_split_panels.py` をWindowsで起動する手順です。

この専用UIでは、CSF Split後のpanel分離、seam gap配置、Omega最適化過程、K2D最適化過程を確認できます。Splitの挙動は、指定された既存row/column grid lineでtopologyを切断し、panelを再packせず元配置を保ったままseam gapだけを開く方針です。

## 1. ブランチを取得する

PowerShellで以下を実行します。

```powershell
git clone https://github.com/yuyakato2365/onestring-physics-simulator.git
cd .\onestring-physics-simulator
git switch agent/large-steps-mesh-conditioning
cd .\onestring-physics-simulator
```

すでにclone済みなら、repository rootで次を実行します。

```powershell
git fetch origin
git switch agent/large-steps-mesh-conditioning
git pull --ff-only origin agent/large-steps-mesh-conditioning
cd .\onestring-physics-simulator
```

## 2. Python環境を用意する

既存のWindows用installerを使えます。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_python_environment.ps1 -WithDevDependencies
```

`.venv\Scripts\python.exe` が作成されていることを確認してください。

## 3. 推奨: Windows専用launcherで起動する

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_split_panels.ps1
```

launcherは次を自動で行います。

- `.venv\Scripts\python.exe` を優先して使用する
- PyTorchの `torch.cuda.is_available()` を確認する
- CUDAが使える場合は `ONESTRING_BIJECTIVE_DEVICE=cuda` を設定する
- CUDAが使えない場合は `ONESTRING_BIJECTIVE_DEVICE=cpu` を設定する
- `app_split_panels.py` をStreamlitで起動する
- 既定portとして `8502` を使用する

起動後は通常、以下を開きます。

```text
http://localhost:8502
```

## 4. deviceやportを明示したい場合

CUDAを強制する場合:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_split_panels.ps1 -Device cuda
```

CPUを強制する場合:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_split_panels.ps1 -Device cpu
```

portを変更する場合:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_split_panels.ps1 -Port 8503
```

## 5. launcherを使わず直接起動する場合

NVIDIA CUDA環境では:

```powershell
$env:ONESTRING_BIJECTIVE_DEVICE = "cuda"
.\.venv\Scripts\python.exe -m streamlit run .\app_split_panels.py --server.port 8502
```

CPU環境では:

```powershell
$env:ONESTRING_BIJECTIVE_DEVICE = "cpu"
.\.venv\Scripts\python.exe -m streamlit run .\app_split_panels.py --server.port 8502
```

macOSで使用する `ONESTRING_BIJECTIVE_DEVICE=mps` はApple Silicon向けなので、Windowsでは使用しません。

## 6. CUDAが使われない場合

まず同じvirtual environmentで確認します。

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

`torch.cuda.is_available()` が `False` の場合、CUDA対応PyTorchがその `.venv` に入っていない可能性があります。`nvidia-smi` がGPUを認識していても、CPU版PyTorchではCUDAは使えません。

launcherの `-Device auto` はこの判定を見て自動的にCPUへfallbackします。

## 7. Splitとanimationの確認

パイプラインを一度実行した後、次を確認できます。

- Omega optimizationのaccepted-state animation
- Split / Panel geometry
- K2D optimizationのactual-iteration animation
- `View stage` からOmega/K2D animationの再表示

Splitが有効な例では、ターミナルに概ね以下のようなログが出ます。

```text
[SPLIT-DEBUG] ... START ...
[SPLIT-DEBUG] ... RESULT split_applied=True ... components=4
```

Split診断ログはWindowsでもproject-relativeな次の場所へ保存されます。

```text
logs\split_debug.jsonl
```

## 8. Windows対応上の実装方針

`app_split_panels.py` と関連patchは `pathlib.Path` を使ってproject内のpathを解決しており、`/Users/...` のようなmacOS固有absolute pathを前提にしません。

Windows側で異なる主な点はcompute deviceです。

```text
macOS Apple Silicon: mps
Windows + NVIDIA:    cuda
Windows without CUDA: cpu
```

Split topology、panel gap、Omega/K2D process animationのロジック自体はOS共通です。
