# WindowsノートPC導入ガイド

この文書は、別のWindowsノートPCへリポジトリをクローンし、Streamlitアプリと
OneString対応Autodesk Affine Body Dynamics（ABD）を再構築するための手順です。
ビルド済み`abd_sim.exe`やPython仮想環境はGitHubへ保存しないため、PCごとに
一度だけ構築します。

## 1. 推奨環境

| 項目 | 必須・推奨内容 |
| --- | --- |
| OS | Windows 10/11 64-bit。Windows 11を推奨 |
| CPU | x64 CPU。ABD用に8論理コア以上を推奨 |
| RAM | 最低16 GB、複雑な形状では32 GB以上を推奨 |
| 空き容量 | CPU版で20 GB程度、CUDA版も構築する場合は35 GB以上を推奨 |
| Python | CPython 3.11 64-bitを推奨。プロジェクト要件は3.11以上 |
| C++ | Visual Studio 2022 Build ToolsとC++ビルドツール |
| GPU | 任意。CUDA版ABDにはCUDA対応NVIDIA GPUとCUDA Toolkitが必要 |

ABDの初回CMake構成では依存ソースを取得してコンパイルするため、インターネット
接続が必要です。会社や大学のプロキシ配下ではGit/CMakeのHTTPS通信も許可して
ください。

## 2. リポジトリを取得する

PowerShellを開き、作業したいディレクトリへ移動して実行します。

```powershell
git config --global core.longpaths true
git clone https://github.com/yuyakato2365/onestring-physics-simulator.git
cd .\onestring-physics-simulator\onestring-physics-simulator
```

リポジトリ直下に同名のアプリディレクトリが1段あるため、2行目の`cd`が必要です。
開発ブランチを直接取得する場合は、`git clone -b <branch-name> <URL>`を使用します。

## 3. Python環境を構築する

Python 3.11 x64をインストールし、Windows Python Launcherの`py`が使える状態で
次を実行します。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_python_environment.ps1 -WithDevDependencies
```

このスクリプトはプロジェクト内に`.venv`を作り、`pyproject.toml`からNumPy、
SciPy、Plotly、Streamlit、Trimeshとテスト用依存を導入します。システムPythonへ
直接インストールしません。

確認と起動:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app.py
```

ブラウザが自動的に開かない場合は、PowerShellに表示された
`http://localhost:8501`を開きます。`streamlit`がコマンドとして認識されない場合も、
必ず`.venv\Scripts\python.exe -m streamlit`を使用してください。

## 4. ABD CPUマルチコア版を構築する

まずCPU版で環境全体が正しく動くことを確認します。管理者権限を使えるPowerShellで
次の導入スクリプトを実行します。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_abd_environment.ps1
```

導入対象はGit、CMake、Visual Studio 2022 Build Toolsです。Visual Studioには
`Microsoft.VisualStudio.Workload.VCTools`と推奨コンポーネントが入ります。
完了したらPowerShellを閉じて開き直し、再びアプリディレクトリへ移動します。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_autodesk_abd.ps1 `
  -BuildDir third_party\affine-body-dynamics\build-parallel
```

このビルドはCUDAを必要としません。Newton線形系ではTBB並列BiCGSTABを使い、
数値的に失敗した場合はEigen直接法へフォールバックします。生成物は次です。

```text
third_party\affine-body-dynamics\build-parallel\Release\abd_sim.exe
```

OneString拡張と短い実行を検証します。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\verify_autodesk_abd.ps1 `
  -Executable third_party\affine-body-dynamics\build-parallel\Release\abd_sim.exe `
  -Steps 8
```

アプリは`build-gpu`、`build-parallel`、`build`の順で実行ファイルを探します。
明示的に固定したい場合は、そのPowerShellセッションで次を設定してから起動します。

```powershell
$env:ONESTRING_ABD_EXECUTABLE = (Resolve-Path `
  ".\third_party\affine-body-dynamics\build-parallel\Release\abd_sim.exe").Path
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## 5. NVIDIA GPU版ABDを追加する（任意）

GPU版はIPCの広域衝突候補生成とCCDの一部をCUDAで高速化します。制約、Hessian、
線形ソルバ、line searchなどはCPU処理も残るため、タスクマネージャーのGPU使用率が
常時100%になる実装ではありません。

CUDA Toolkitを追加する場合:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_abd_environment.ps1 `
  -InstallCuda -SkipVisualStudio -SkipGit -SkipCMake
```

PowerShellを開き直し、`nvcc --version`が成功することを確認します。GPUのCompute
Capabilityに合う数値を`-CudaArchitectures`へ渡します。例としてRTX 40シリーズの
Ada GPUは`89`です。

```powershell
nvcc --version
powershell -ExecutionPolicy Bypass -File .\scripts\build_autodesk_abd.ps1 `
  -EnableCuda `
  -CudaArchitectures 89 `
  -BuildDir third_party\affine-body-dynamics\build-gpu

powershell -ExecutionPolicy Bypass -File .\scripts\verify_autodesk_abd.ps1 `
  -Executable third_party\affine-body-dynamics\build-gpu\Release\abd_sim.exe `
  -Steps 8
```

GPUアーキテクチャが不明な場合は、NVIDIAのGPU別Compute Capability一覧で確認して
ください。異なる世代の値を指定すると、ビルドまたは実行時に失敗します。

## 6. PyTorch CUDAはABD CUDAとは別機能

Python側のK3D/K2Dテンソル最適化にもGPUを使う場合だけ、`.venv`へCUDA対応
PyTorchを追加します。ABDのC++/CUDAビルドだけを使う場合、PyTorchは不要です。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu-cu128.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`requirements-gpu-cu128.txt`はCUDA 12.8ランタイム付きPyTorchの固定セットです。
ドライバーが対応しないPCでは、PyTorch公式のインストール選択に従って互換ビルドを
使用してください。ローカルCUDA ToolkitとPyTorch同梱ランタイムの版が完全一致する
必要はありません。

## 7. ABDの机上シミュレーション設定

Assembly Animationで物理バックエンドに`abd`を選択し、次を確認します。

- `ABD gravity Z`: `-9.81`
- `support panels on fixed desk plane`: ON
- `desk initial clearance`: `0.005`
- 結果だけ確認する初回テスト: 8～12 steps、比較的小さいグリッド

机は固定水平支持面として、各パネル底面4頂点へ滑らかな片側法線反力を与えます。
重力は有効なままです。現在の机モデルには接線摩擦を入れていないため、パネルは
水平方向へ滑れます。パネル同士のIPC接触・摩擦とは別の支持モデルです。

## 8. 別PCで更新する

ソースだけ更新し、仮想環境とビルド成果物はそのPCに保持します。

```powershell
git pull
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

`third_party/affine-body-dynamics`以下のC++ソースやCMake設定が更新された場合は、
CPU/GPUの`build_autodesk_abd.ps1`を再実行してください。問題が疑われる場合は
`-Clean`も指定できますが、依存関係を含めて再構築するため時間がかかります。

## 9. よくある問題

### `streamlit`が認識されない

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### `cmake`、`git`、`nvcc`が認識されない

インストール直後のPowerShellを閉じ、新しいPowerShellを開いてください。それでも
見つからなければ、インストール先が`PATH`へ登録されているか確認します。

### ABDが600秒でタイムアウトする

最初はグリッド、steps、接触数を減らします。GPU版でも全工程がGPU化されるわけでは
ありません。`output\abd_run\abd_stdout.log`と`abd_stderr.log`で停止箇所を確認します。

### ABD結果が下へ落ち続ける

最新の`abd_sim.exe`を再ビルドしたか、`support panels on fixed desk plane`がONかを
確認します。古い実行ファイルは机支持マニフェストを解釈できません。

### 完全に初期化したい

`.venv`と対象のABDビルドディレクトリを削除し、Python導入とABDビルドをやり直します。
Git管理されるソースファイルは削除しないでください。
