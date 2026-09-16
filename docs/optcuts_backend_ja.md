# OptCuts 外部 backend（試験統合）

この branch では、SIGGRAPH Asia 2018 の **OptCuts: Joint Optimization of Surface Cuts and Parameterization**
著者公開実装を、既存の OneString pipeline を壊さない外部 backend として試験的に接続する。

## 方針

既存の BFF / CEPS / `bijective_free_boundary` / M2D Split は変更しない。

```text
S (triangle surface)
  -> official OptCuts executable
  -> cut topology + UV
  -> Omega
  -> existing M2D / M3D / K3D / K2D / T2D pipeline
```

OptCuts が見つからない、失敗する、または出力 topology を現在の bridge が扱えない場合は
**明示的に失敗する**。BFF 等へ silent fallback しない。

## 1. 公式 OptCuts を取得・build

macOS / Linux / Windows 共通の補助 script:

```bash
python scripts/setup_optcuts.py
```

これは未取得なら `third_party/OptCuts` に公式 repository
`https://github.com/liminchen/OptCuts.git` を clone し、CMake build を実行する。
既存 checkout がある場合は自動 `git pull` しない。

build だけ省略する場合:

```bash
python scripts/setup_optcuts.py --no-build
```

公式コードは古い libigl / Eigen / TBB 等を含む研究コードなので、現在のOS/compilerでは
upstream側のbuild修正が必要になる可能性がある。特にWindowsについては公式READMEにも
手動environment設定やEigen backendの速度問題への注意がある。

## 2. 実行

既存の安定した Split validation app に OptCuts selector だけを追加した入口:

```bash
python -m streamlit run app_optcuts.py --server.port 8504
```

sidebar の `Omega parameterization mode` で `optcuts` を選ぶ。

同じ場所に以下の設定が表示される。

- OptCuts executable
- Symmetric Dirichlet distortion bound（4より大きい値）
- initial lambda
- bijectivity
- initial cut option
- timeout

通常は executable 欄を空にしておけば、以下を自動探索する。

```text
third_party/OptCuts/build/OptCuts_bin
third_party/OptCuts/build/OptCuts_bin.exe
third_party/OptCuts/build/Release/OptCuts_bin.exe
```

または環境変数で指定できる。

macOS / Linux:

```bash
export ONESTRING_OPTCUTS_EXECUTABLE="/path/to/OptCuts/build/OptCuts_bin"
```

Windows PowerShell:

```powershell
$env:ONESTRING_OPTCUTS_EXECUTABLE = "C:\path\to\OptCuts\build\Release\OptCuts_bin.exe"
```

## 3. bridge が呼ぶ公式CLI

OptCuts公式READMEにある headless mode (`100`) を使う。

概念的には:

```text
OptCuts_bin 100 input.obj 0.999 1 0 4.1 1 0 <unique-tag>
```

出力の `finalResult_mesh.obj` を読み込み、OBJの

- `v`: 3D surface vertices
- `vt`: optimized UV vertices
- `f v/vt`: face-wise 3D/UV correspondence

を保持したまま `SurfaceParameterization` に変換する。

Seamにより `v` index と `vt` index が異なっていても、`surface_faces` と `uv_faces` を
別々に保持するため、barycentric inverse mappingで対応可能。

## 4. 現在の意図的な制限

この最初の bridge は、OptCuts後のUV topologyが **1個のsimple boundary loop** を持つ場合だけ受理する。
複数UV boundary loopが出た場合は明示エラーにする。

また、OptCutsが作るsurface seamをOneStringの規則M2D grid seamへ直接snapする処理はまだ入れていない。
つまり今回の段階はまず、

> OptCutsで中心潰れ / Symmetric Dirichlet distortionが本当に改善するか

を同じ入力Sで確認するためのもの。

既存の `simple_split_panel_patch.py` のSplitロジックは変更していない。

## 5. 診断値

backendは少なくとも以下を記録する。

- per-triangle `sigma1`, `sigma2`
- per-triangle Symmetric Dirichlet energy
- mean / max Symmetric Dirichlet
- UV flip count
- UV degenerate triangle count
- OptCuts executable SHA-256
- CLI parameters
- runtime
- stdout/stderr tail

これにより、BFF等と「中心の両特異値が同時に小さくなっているか」を直接比較できる。
