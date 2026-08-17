# Conformal scale factor diagnostics

## Per-triangle differential

For each surface triangle, version 0.2.0 constructs an orthonormal local 2D
coordinate system.  With the first vertex as the origin:

```text
D_S  = [p1_local - p0_local, p2_local - p0_local]
D_UV = [uv1 - uv0,             uv2 - uv0]
J_(S->UV) = D_UV inverse(D_S)
J_(UV->S) = D_S  inverse(D_UV)
```

SVD is evaluated directly for both mapping directions.  The JSON output stores
the two singular values, anisotropy `sigma_max / sigma_min`, absolute
determinant (area scale), UV degeneracy, orientation-relative triangle flips,
and logarithmic scale for every triangle.  No edge percentile is involved.

## Direction used for lambda

The linkage begins flat and expands toward the target surface.  Version 0.2.0
therefore records the candidate auxetic expansion as:

```text
lambda_raw(t) = sigma_max(J_(UV->S)(t))
```

Both forward and inverse singular values remain available in the programmatic
diagnostic object.  This mapping-direction choice follows the physical
flat-to-assembled interpretation, while the OneString paper's Section 4.5 does
not provide an explicit discrete Jacobian equation.

## Similarity normalization

BFF has a global similarity-scale degree of freedom.  The OneString paper states
`1 <= lambda <= sigma` and `sigma=2` for the quadrilateral linkage, but does not
state how its BFF result was globally scaled before this comparison.

The default is therefore explicitly labelled:

```text
hypothesis_a / min_to_one_hypothesis_a:
scale UV so min_t sigma_max(J_(UV->S)(t)) = 1
```

This is not called an exact paper setting.  `none_unspecified` is also available
to preserve the official CLI's raw similarity scale.  The old median edge-ratio
normalization is not reused in reference mode.

## Split decision

The exact per-triangle normalized lambda is compared with 2.  If the bound is
exceeded, diagnostics record the vertex with the greatest discrete Gaussian
angle defect and the paper's complete grid-direction split rule.  The main
paper does not completely define post-split BFF reparameterization, so the
strict default stops with `UNSPECIFIED_SPLIT_REPARAMETERIZATION` instead of
applying the legacy localized/high-band split heuristic.  Set
`reference_stop_on_required_split=False` only to inspect diagnostics without
claiming a completed strict reference run.

