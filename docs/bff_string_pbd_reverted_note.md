# OneString BFF / String PBD Implementation Snapshot Before Revert

This note preserves the temporary implementation that was tested and then
reverted at the user's request.

- A temporary implementation added `PipelineParameters.parameterization_method`
  with `harmonic`, `bff`, and `conformal` options.
- The BFF/conformal path used a lightweight cotangent-weight conformal fallback
  labeled `method_used="lscm_fallback"`, with metadata for
  `method_requested`, `method_used`, `fallback_reason`,
  `conformal_scale_stats`, and `boundary_condition`.
- `DeploymentParameters` was extended with target-pose-fit off by default,
  velocity-based gravity (`dt`, `gravity`, `damping`, `enable_gravity`), and
  string-physics controls (`enable_string_physics`,
  `string_constraint_weight`, `string_constraint_iterations`, `pull_fraction`,
  `pull_schedule`, `string_slack`).
- CPU deployment was changed to a PBD-style loop: velocity/gravity integration,
  pull schedule, string length constraint along `StringPath` gap centers,
  lift/snap/hinge/rigid/collision/contact guard, and optional target pose fit
  only when explicitly enabled.
- Metrics added included `simulation_model="string_length_constraints_pbd"`,
  target-pose-fit enabled state, string length values/error, gravity enabled,
  target error, hinge error, and collision count.
- Tests added checked BFF metadata and default string-PBD deployment metrics.
- Verification before revert: `py_compile`, `git diff --check`, and
  `pytest tests -q` passed with `22 passed`.
- The implementation was reverted to restore the previous simulator behavior,
  while this note keeps the design snapshot available for future work.
