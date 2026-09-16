# 現在の OneString アルゴリズム概要

この文書は、家PC側のCodexが現在の実装を誤読しないための短い地図です。
論文の完全実装ではなく、現在のリポジトリに存在する研究プロトタイプの説明です。

## パイプライン

現在の主な流れは次の通りです。

```text
S target surface
 -> Omega planar domain
 -> M2D regular quad overlay
 -> M3D inverse map to S
 -> K3D assembled 3D quad optimization
 -> T3D thick panels
 -> K2D flattened edge-length layout
 -> T2D top / dual hinge fabrication layout
 -> Gap graph / LiftPoint / StringPath
 -> deployment simulation / preview animation
```

## 重要な前提

- `paper_reference_bff` は公式GeometryCollective BFF CLIだけを使用し、利用不能なら
  `ReferenceBFFUnavailableError` で停止します。LSCM等へのfallbackはありません。
- 旧 `bff` は `rectangular_harmonic_legacy` のdeprecated aliasです。BFFとは表示しません。
- `lscm_free_boundary` は自由境界LSCM診断用です。デフォルトの速い状態として
  乱用しないでください。
- `Omega` を自由境界や極端な非矩形領域にすると、`M2D` のoverlayやhinge候補が
  爆発し、Dual Hinge前段で極端に遅くなる可能性があります。
- `K3D`、`K2D`、deployment の一部はPyTorch/CUDAを使えますが、Plotly描画、
  file I/O、graph routing、Streamlit UIはCPU側です。
- `actual_backend="cuda"` のような表示は、最終的に採用された計算がCUDA経路で
  作られた場合だけ信用してください。

## 遅くなった時に最初に見る場所

`src/onestring_physics/onestring_pipeline.py` で次を確認します。

- `PipelineParameters`
- `_build_surface_parameterization(...)`
- `_rebuild_domain_overlay_for_general_omega(...)`
- `_flatten_to_domain(...)`
- `_build_m2d(...)`
- `_make_t2d_from_transforms(...)`
- `_optimize_dual_hinges(...)`
- file末尾の `_original...` monkey patch wiring

特に次の値を記録してください。

- `len(mesh_2d_initial.faces)`
- `len(mesh_2d_optimized.faces)`
- hinge spec count
- dual hinge count
- `m2d_general_omega_overlay_rebuilt`
- `m2d_general_omega_effective_nx`
- `m2d_general_omega_effective_ny`
- `omega_boundary_forced_rectangle`
- `omega_boundary_shape`

面数やhinge数が急増している場合、optimizer自体よりも
`S -> Omega -> M2D` の位相・overlay・split処理が原因である可能性が高いです。

## 実装上の注意

- このPCの基準コミットは `03a92394deda9063cebf3adf2f38e011c2ed6983` です。
- その後に家PCからGitHubへ入ったremote commit群は、性能劣化の疑いがあるため
  デフォルトでは信用しません。
- 変更する場合は、まずこのPCの `.venv` と同じCUDA PyTorch環境を再現し、
  `python -m pytest tests -q` とStreamlit内のCUDA表示を確認してください。
- `.venv` 自体はGitHubへ載せません。代わりに
  `requirements-local-cu128-lock.txt` と `requirements-gpu-cu128.txt` を使います。
