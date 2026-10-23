# Default-on K3D SmoothMax (2026-10-24)

## Scope

In the active `app_optcuts_20260923_extrusion_aware.py` -> `app_optcuts_core_20260916.py` path, the visible Omega option **OptCuts Ω + Area-Threshold Grid Boundary** is numerically routed via `omega_parameterization_mode="optcuts_test"`. The OptCuts-test Augmented Lagrangian planarization stage already implemented a smooth-maximum worst-panel square energy, but it was **incorrectly gated** behind the unrelated optional `ONESTRING_K3D_SHAPE_PRESERVING_PLANARITY` mode.

The objective now unconditionally includes the existing global square term whenever `params.w_square > 0`, regardless of optional shape-preservation:

```
E_square = (w_square / 2) * (
    mean_i(d_i^2) + alpha * LSE_beta(d_1,...,d_N)^2
)
LSE_beta = log(mean_i(exp(beta*d_i))) / beta
d_i^2 = mean([four successive edge-length difference residuals^2,
               diagonal length-difference residual^2])
```

The alpha and beta defaults remain `4` and `12`; `beta=0` retains its differentiable continuous limit `mean(d)`. `alpha=0` disables *only* the additional worst-panel term; `w_square=0` disables the full square term, preserving the intentional UI option.

The optional `shape_mode` still determines **only** whether the distinct source-reference metric preservation term is present. It does not gate the global square term. Original source mesh, Ω/Grid/M2D, seam topology, AL quad-planarity constraints and tolerances are unchanged.

## Preserving the objective across later K3D stages

The subsequent exact source-Seam K3D closure historically moved vertices to minimize planarity with exact paired-vertex equality, without accounting for square energy. This can undo the earlier AL square improvement. It now performs a *second direct* analytic SmoothMax optimization over equality-group-reduced K3D unknowns after obtaining an exact closed planarized baseline:

- Every pair of matching source-Seam samples uses one optimization unknown; no approximate welding.
- Global SmoothMax term is differentiated analytically via `_square_energy_and_gradient`.
- An explicit normalized quad-planarity residual and rest-position anchor penalize infeasible movement.
- A trial is **accepted only when** its true SmoothMax energy decreases, the seam gap remains < 1e-12, its quad-plane distance does not exceed the existing closed baseline (apart from a 0.5% numerical margin), and all coordinates are finite. Otherwise the baseline exact-Seam result is retained.
- Nonconforming two-sided Grid seam subdivisions remain a hard error; this change does not pretend to fix them.

The Paper T3D Eq.(2) extrusion still has a separate face-planarity solve and can move/promote its top vertices to the final K3D. This stage is **not claimed to be SmoothMax-constrained**. To avoid hiding regressions, a read-only end-to-end check measures the *same K3D SmoothMax energy* before and after T3D top promotion, storing `t3d_final_k3d_smoothmax_energy_before`, `_after`, `_change`, and logging `[K3D-SMOOTHMAX-T3D-AUDIT]`.

## Runtime

```bash
git pull --ff-only origin fix/optcuts-real-m2d-footprint-20261023
python3 -m streamlit run app_optcuts_20260923_extrusion_aware.py
```

Expected traces **once M3D -> K3D is reached**:

```
[K3D-SMOOTHMAX] stage=AL enabled=true w_square=... alpha=4 beta=12 shape_preserving=false
[OPTCUTS-TEST-K3D-AL] ... square_rms=... ...
[SEAM-K3D] ... smoothmax_accepted=... smoothmax_before=... smoothmax_final=...
[K3D-SMOOTHMAX-T3D-AUDIT] before=... after=... delta=...
```

In the K3D metrics, verify `k3d_smoothmax_enabled=True`, `k3d_global_square_regularization_enabled=True` and `k3d_global_square_weight_effective == UI w_square`.

The Bunny is not locally available in CI with the official native OptCuts binary. A native Bunny run is still required to show numerical feasibility, actual improvement and whether exact seam pairing is conforming.
