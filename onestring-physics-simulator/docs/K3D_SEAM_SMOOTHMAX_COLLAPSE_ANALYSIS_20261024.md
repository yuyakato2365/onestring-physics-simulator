# K3D source-Seam + SmoothMax: collapsed quads and undefined normals
Date: 2026-10-24

## Verified diagnosis of the user error

The user observed:

```
SEAM_K3D_AFFINE_SMOOTHMAX_PLANARITY_NOT_CONVERGED:
  source_cut_gap=2.22044605e-16
  plane_distance=inf
  plane_tol=8.83871033e-08
```

The `_quad_plane_distances()` implementation explicitly returns
`inf` when the chosen quad's base-triangle normal norm is at most
`1e-12`. This is a **degenerate quad**, not an ordinary finite
planarity residual. The seam affine equations were satisfied to
floating-point precision; this log does NOT implicate seam
classification and does not support turning off Seam constraints.

The SmoothMax square loss penalizes *differences* between neighboring
edge lengths and between diagonals. Every quad vertex collapsing to
the same point also has equal edge lengths (all zero), so the square
loss by itself has a false global minimum at complete collapse.
Under the hard affine source-Seam constraints this can produce severe
spikes or triangle-like panels, especially after independent planar
consensus projections.

## Source-preserving anti-collapse fix

- The standard K3D SmoothMax loss remains enabled.
- When source-Seam hard constraints are active, turn on
  dimension-preserving edge-and-diagonal reference-metric energy
  independently of optional `shape_mode`:
  `shape_weight=max(user_shape_weight,8,3*w_square)`.
- Add an oriented source-triangle signed-double-area barrier on
  each original K3D quad triangle, with reference-aligned normals,
  `area_weight=max(50,10*w_square)`; keep the source quad face IDs.
  This penalizes both area collapse and inversion rather than merely
  making all edge lengths equal.
- Backtrack every planar-consensus step that would reduce minimum
  oriented area below its previous valid threshold. Since all
  candidate and prior vertices already satisfy the same linear
  Seam equations, the backtracking interpolation stays in their
  affine feasible subspace.
- Before declaring a projected SmoothMax result feasible, independently
  audit its original-M3D-relative two-triangle signed area, not
  merely the planarity distance.
- If original post-AL K3D contains collapsed/inverted panels or the
  minimum reference-relative signed area is below 8%, restart the
  joint solve **from original noncollapsed M3D**, NOT the corrupted K3D,
  whose near-zero reference edges would destroy the scale and make
  the collapsed layout self-validating.
- Reject final results with less than 2% unsigned triangle area,
  less than 1% signed triangle area, or inverted triangles, and
  report affected quad IDs in
  `SEAM_K3D_COLLAPSED_QUADS_IN_JOINT_OPTIMIZATION`.
- If quads are nondegenerate and the Seam equations hold but the
  plane tolerance is still unmet, report
  `SEAM_K3D_AFFINE_SMOOTHMAX_PLANARITY_NOT_CONVERGED`
  with finite residuals, restart source, and signed area diagnostics.

## What this does and doesn't guarantee

The code guards against **accepting** collapsed quads and adds
stabilizing area/shape penalties. It does not mathematically prove
that all 947 quad panels can simultaneously fit the specific source
surface shape, exact source-Seam affine equations, and an extremely
tight planar tolerance at the chosen topology. These constraints may
compete, requiring a different seam/cell pattern, targeted splitting,
a different local optimizer, or a higher-dimensional global
constraint solver. Hard-Seam and nondegenerate-quad checks must never
be silently relaxed merely to avoid an error.

Passing GitHub synthetic regressions validates equations and failure
detection. Native OptCuts/Bunny 947-quads end-to-end feasibility
remains unverified until a Mac run, and any resulting T3D geometric
constraints must still be audited independently.
