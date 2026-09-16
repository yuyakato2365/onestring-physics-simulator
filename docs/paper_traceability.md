# Paper traceability for version 0.2.0

`Exact` below means exact relative to the stated discrete operation or official
backend, not a claim that the entire OneString system has been reproduced.

| Pipeline stage | Paper section/equation | Implemented function | Exact / Approximate / Unspecified |
|---|---|---|---|
| Input triangle disk validation | BFF Sec. 1.1 and Appendix B input | `validate_reference_mesh` | Exact validation; no repair |
| S -> Omega | OneString Sec. 4.1; BFF Appendix B Algorithm 1 | `run_official_bff`, `_build_surface_parameterization` | Exact official BFF CLI call; OneString boundary policy is Unspecified |
| BFF automatic boundary data | BFF official CLI `BFF::flatten(u, true)` | `run_official_bff` | Exact official CLI default (`boundary_scale_zero`) |
| Triangle differential and singular values | OneString Sec. 4.1/4.5 physical scale discussion | `triangle_jacobian_diagnostics` | Exact discrete Jacobian/SVD |
| Lambda mapping direction | OneString Sec. 4.5 | `normalize_uv_and_compute_csf` | Approximate interpretation: flat-to-surface maximum singular value |
| Lambda global normalization | OneString Sec. 4.5, `1 <= lambda <= sigma` | `normalize_uv_and_compute_csf` | Unspecified in paper; `hypothesis_a` is labelled |
| Omega -> M2D regular grid | OneString Sec. 4.1 | `_reference_flatten_to_domain` | Exact regular square construction for explicit spacing/rotation/origin |
| M2D crop | OneString Sec. 4.1, quads fully contained in Omega | `_build_reference_m2d` | Exact polygon containment policy for quad vertices and boundary crossings |
| M2D -> M3D | OneString Sec. 4.1, inverse `c^-1` | `strict_inverse_map_uv_to_surface`, `_lift_m2d_to_m3d` | Exact containing-triangle barycentric interpolation |
| Split diagnostics | OneString Sec. 4.5 | `_reference_flatten_to_domain` | Paper rule recorded; reparameterization Unspecified |
| M3D -> K3D | OneString Sec. 4.2 | `_optimize_k3d` | Approximate existing implementation |
| K3D -> T3D | OneString Sec. 4.2/4.4 | `_extrude_tiles` | Approximate existing implementation |
| K3D -> K2D | OneString Sec. 4.3 | `_optimize_k2d` | Approximate existing implementation |
| K2D -> T2D | OneString Sec. 4.3/4.4, Eq. 6 | `_make_flat_tile_layout`, `_make_t2d_from_transforms`, `_optimize_dual_hinges` | Approximate existing implementation |

Because Approximate and Unspecified entries remain, version 0.2.0 is **not** a
complete OneString reproduction.  Its supported claim is:

```text
S_TO_M3D: paper-reference candidate
K3D_AND_LATER: existing approximate implementation
```

