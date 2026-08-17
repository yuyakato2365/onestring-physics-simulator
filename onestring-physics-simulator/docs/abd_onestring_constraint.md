# OneString constraint extension for Autodesk ABD

This upgrade completes the missing C++ half of the Python ABD bridge.

## What was missing

The branch `agent/t3d-abd-backend-0.3.0` already exported:

- closed per-tile OBJ meshes;
- an Autodesk ABD scene JSON;
- pin-joint hinge constraints;
- a OneString manifest;
- result parsing back into Streamlit frames.

However, the vendored ABD executable did not accept `--onestring-manifest`, did
not add a string force to the Newton objective, and did not emit
`onestring_stats`.  The original verification script also referenced the
excluded upstream `cube_drop` sample.

## Added model

For a body guide with local material point `r_i`, body translation `p_b`, and
affine transform `A_b`, the world guide position is

```text
g_i(q) = p_b + A_b r_i
```

The smoothed total path length is

```text
L(q) = sum_i sqrt(||g_(i+1)-g_i||^2 + epsilon^2)
```

The unilateral constraint is

```text
L(q) <= L_command(t)
```

and the first implementation uses the one-sided potential

```text
E_string(q,t) = 0.5 * k * max(0, L(q)-L_command(t))^2
```

When the string is slack, the energy, gradient, and Hessian are exactly zero,
so the model never produces a compressive string force.

The Hessian defaults to a Gauss-Newton rank-one approximation.  Setting
`use_exact_hessian=true` in the manifest adds the positive-semidefinite length
Hessian as well.

## Support anchor

Manifest v2 explicitly adds a fixed world anchor named `support` before the
body guide points.  This is essential: without a world support point every tile
is dynamic, so gravity would translate the entire assembly instead of
suspending it.

For compatibility, the C++ parser still accepts v1 manifests and synthesizes a
support anchor at the initial position of the first guide.

## Output

`sim.json` now includes:

```json
{
  "onestring_stats": {
    "string_length": [],
    "command_length": [],
    "constraint_violation": [],
    "active": []
  }
}
```

Each array has one value per animation state, including the initial state.
