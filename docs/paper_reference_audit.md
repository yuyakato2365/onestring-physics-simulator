# Paper-reference pipeline audit

Audit date: 2026-07-11 (JST)

This document records the implementation that existed before the
`paper_reference_bff` work started.  It is intentionally an audit, not a claim
of paper equivalence.

## Primary sources checked

- Zaman et al., *One String to Pull Them All: Fast Assembly of Curved
  Structures from Flat Auxetic Linkages*, ACM TOG 44(6), 2025,
  DOI [10.1145/3763357](https://doi.org/10.1145/3763357), especially Sections
  4.1-4.5.
- The authors' [supplementary information](https://onestringtopullthemall.github.io/static/pdfs/OneStringPull-supplement.pdf).
- Sawhney and Crane, *Boundary First Flattening*, 2017,
  DOI [10.1145/3132705](https://doi.org/10.1145/3132705), especially Sections
  3-6 and Appendix B.
- The authors' [official BFF implementation](https://github.com/GeometryCollective/boundary-first-flattening),
  including its `bff-command-line` automatic flattening path.

## Findings that constrain the reference mode

The OneString paper states that BFF constructs `c: S -> Omega`, that the
boundary is made rectangular *where possible*, that only quads fully contained
in Omega are retained, and that `c^-1` lifts M2D to M3D.  It also states that a
quad linkage has an upper bounded conformal scale factor of 2 and describes
hierarchical complete splits along a grid direction through the point of
highest Gaussian curvature.

The paper does **not** identify the exact BFF API call, the default/automatic
boundary-data vector, a deterministic rectangle-corner selection rule, the
global similarity normalization used before comparing lambda with 2, or a
complete post-split reparameterization algorithm.  These are recorded as
`UNSPECIFIED_IN_PAPER`; the reference implementation must not silently invent
them.

The official BFF automatic path is minimal-area-distortion flattening.  BFF is
not merely a fixed boundary followed by two cotangent harmonic solves: its
boundary data are related by the Cherrier formula and the Dirichlet-to-Neumann
or Neumann-to-Dirichlet map, the adjusted boundary lengths are closed, the
boundary curve is integrated, and the interior is extended through harmonic
extension plus harmonic conjugation.

## Current implementation audit

| Pipeline stage | Current function(s) | Input | Output | Formula / algorithm currently used | Heuristics and fallbacks | Agreement with paper | Difference from paper/reference |
|---|---|---|---|---|---|---|---|
| S -> Omega | `_build_surface_parameterization`; `_bff_boundary_first_uv`; `_lscm_free_boundary_uv`; `_boundary_sliding_lscm_uv` | Triangulated `SurfaceMesh`, target height field, grid, `PipelineParameters` | `SurfaceParameterization` with paired 3D/UV triangles | Legacy `bff`: boundary vertices are distributed by 3D boundary arclength on a rectangle, then `L_II u_I=-L_IB u_B` and the same for `v`. LSCM minimizes triangle Cauchy-Riemann residuals with two pinned boundary vertices. | The old `bff` path falls back from singular `spsolve` to `lsqr`; other experimental PCA/height-field paths exist. | Produces a map and paired triangles used by later stages; the paper says to use BFF and permits rectangular Omega where possible. | The old `bff` path contains no Cherrier formula, Poincare-Steklov operator, boundary closure solve from BFF, or harmonic conjugate. It is rectangular cotangent harmonic parameterization, not BFF. LSCM/PCA are not valid substitutes for the paper-reference mode. |
| Omega -> M2D | `_flatten_to_domain`; `_rebuild_domain_overlay_for_general_omega`; `_align_domain_grid_to_uv_points`; `_build_m2d`; `_clip_m2d_faces_to_omega_boundary`; `_symmetrize_m2d_faces` | UV parameterization, requested grid, crop/split settings | Cropped regular quad `QuadMesh` in UV | Axis-aligned regular grid over the UV bounding box; polygon-inclusion tests select cells. | Automatic peak alignment, optional doubled density, mirror/symmetry completion, hidden grid shifts, and either center or strict-vertex crop policy. | Uses a regular grid and crops it to Omega. | The paper says quads not fully contained are cropped; the default center policy is looser. Automatic peak alignment and symmetry repair are not paper defaults. Rotation, origin, spacing, and resolution are not all explicit independent reference inputs. |
| M2D -> M3D | `_lift_m2d_to_m3d`; `inverse_map_uv_to_surface`; `_inverse_map_uv_to_surface_regular` | M2D vertices and paired UV/3D triangle meshes | Lifted quad `QuadMesh` on S | Find a UV triangle, compute barycentric coordinates, apply them to the corresponding 3D triangle. | If no containing triangle is found, current code selects a near triangle, clips barycentric coordinates, or returns a nearest surface vertex. An analytic height-field shortcut also exists. | The primary containing-triangle barycentric path matches the paper's inverse-map intent. | Silent nearest/clipped fallback violates reference requirements and can move an outside point onto S. Failure causes are not separated into outside-Omega, numerical tolerance, and clipping errors. |
| Conformal scale factor | `_parameterization_stretch_csf` | Paired surface/UV vertices and triangle connectivity | One scalar per UV vertex | Edge ratios `||e_3D||/||e_UV||`; per-vertex 90th percentile; divided by the global median; clamped to at least 1. | Median normalization and percentile aggregation are heuristic. | Attempts to locate locally large stretch before splitting. | It is not the differential/Jacobian conformal scale factor. It does not record per-triangle singular values, anisotropy, determinant/area scale, flips, or UV degeneracy. The normalization against the paper's bound is unsupported. |
| Mesh splitting | `_csf_split_lines`; `_peak_guided_csf_split_lines`; `_localized_csf_split_segments`; `_split_m2d_along_existing_grid_line`; `_build_m2d` | Heuristic vertex CSF, UV mesh, grid, threshold | Duplicated vertices and disconnected M2D components | Pick row/column lines from high proxy-CSF bands or surface-height peaks; optionally localize them; duplicate existing grid-line vertices. | Peak guidance, symmetry mirroring, band thresholds, line snapping, residual masking, maximum split counts, and localized rather than complete splits. `max_csf_after_split` can be estimated without reparameterizing. | Splits follow grid directions and preserve quads. | The paper specifies complete hierarchical bisection through highest Gaussian curvature after M2D/M3D exist, repeated until each part satisfies the bound. Current code uses proxy CSF, often local segments, and no verified post-split BFF reparameterization. |
| M3D -> K3D | `_optimize_k3d`; original `_optimize_k3d`; Torch/SciPy/projective helpers | M3D quad mesh, surface parameterization, weights | Optimized `K3D` quad mesh | Approximate residual `sqrt(w_planar) E_planar + sqrt(w_square) E_square + sqrt(w_surface) E_surface` plus an extra XY anchor. | CUDA, SciPy, and fast NumPy paths are selected by availability/size. Quality guards can reject output and replace it with M3D. | Uses the three energy families and documented default weights. | Projection operators and solver sequence are approximate. Extra anchor, backend substitution, fast path, and M3D fallback are not a strict paper-reference solve. This stage remains explicitly approximate in version 0.2.0. |
| K3D -> T3D | `_extrude_tiles`; `_solve_bottom_vertex`; shared-edge miter helpers | K3D top quads, thickness | Eight-vertex `TileAssembly` | Offset bottom plane plus independently constructed shared-edge miter/contact planes. | Singular bottom-vertex solves have local fallbacks; optional intersection trimming exists. | Produces thick quadrilateral-frustum tiles with planar/contact-oriented faces. | The current miter construction is an independent implementation and has not been traced to the paper's exact extrusion and contact-face projection operators. It remains approximate. |
| K3D -> K2D | `_optimize_k2d`; projective/SciPy/Torch edge solvers | M2D initialization, K3D target edge lengths | Flat shared-vertex `K2D` mesh | Match each K2D edge length to the corresponding K3D edge, weakly anchor to M2D, then optionally relax collisions. | Backend varies with size and GPU; stricter solver may replace a fast result; collision relaxation is conditional. | Targets corresponding edge lengths and a planar non-overlapping layout, consistent with the paper's flat-configuration goal. | Current solver/backend switching and conditional relaxations are approximate and do not reproduce one fixed paper solver. |
| K2D -> T2D | `_make_flat_tile_layout`; `_make_t2d_from_transforms`; `_optimize_dual_hinges` | K2D, K3D, T3D transforms, hinge topology | Independent flat tiles, top/dual-hinge T2D | Duplicate quad vertices into rigid tiles, optimize one SE(2) pose per tile, apply T3D top-to-bottom transforms, then optimize `E_Hinge`. | Expansion, collision sweeps, time budgets, candidate-pair limits, anchor weights, and layout guards are implementation choices. | Separates rigid tiles, avoids overlap, and aligns hinge vertices as described in Sections 4.3-4.4. | Several projection operators, initial placement details, stopping rules, and collision handling are not fully specified in the paper and are approximate here. |

## Required mode rename resulting from the audit

- `rectangular_harmonic_legacy`: the existing rectangular boundary plus
  cotangent harmonic solve.
- `lscm_free_boundary`: the existing two-pin free-boundary LSCM.
- `paper_reference_bff`: official BFF command-line backend only; no LSCM,
  PCA, harmonic, Tutte, or nearest-point fallback.
- `bff`: deprecated alias of `rectangular_harmonic_legacy`, always accompanied
  by: `This is not Boundary First Flattening. It is rectangular-boundary
  cotangent harmonic parameterization.`

## Reference-mode stop conditions

The reference mode must fail rather than substitute another algorithm when:

- the official BFF executable cannot be found or returns a non-zero status;
- the input is not a finite, consistently oriented, manifold triangle disk
  with exactly one boundary loop and no degenerate faces;
- BFF output cannot be matched one-to-one to input triangle corners/vertices;
- UV output contains a degenerate triangle, a flip, or a reference fallback;
- a requested split requires the unspecified post-split reparameterization.

