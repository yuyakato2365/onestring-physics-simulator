# BFF rectangular target-domain implementation notes

Implemented requested change: keep the rectangular Omega constraint while making the default `bff` path boundary-first rather than falling back to free-boundary LSCM.

## Main code change

`src/onestring_physics/onestring_pipeline.py`

- Added `_bff_boundary_polygon(...)` to estimate a BFF-style natural boundary candidate from 3D boundary edge lengths and boundary turning angles.
- Added `_rectangularize_boundary_by_arclength(...)` to prescribe existing boundary vertices onto a centered rectangle by 3D boundary arclength.
- Added `_bff_boundary_first_uv(...)` to solve interior UV coordinates with the cotangent Laplacian under that prescribed rectangular boundary.
- Changed `omega_parameterization_mode="bff"` so it now uses this rectangular target-domain solve by default.
- Kept `lscm_paper_like` as the explicit free-boundary diagnostic mode.
- Because `omega_boundary_forced_rectangle=True`, the general free-boundary Omega overlay rebuild is skipped in the default path, avoiding the high-density/non-rectangular M2D overlay that was suspected of causing Dual Hinge slowdown.

## Metrics expected in default mode

- `parameterization_method = "bff"`
- `parameterization_exactness_label = "bff_rectangular_boundary_local"`
- `flattening_backend = "local_bff_rectangular_boundary_cotan_harmonic"`
- `bff_implemented = True`
- `bff_reference_backend_available = False`
- `omega_boundary_forced_rectangle = True`
- `omega_boundary_shape = "rectangular"`
- `omega_boundary_constraint_model = "bff_prescribed_rectangular_boundary_by_3d_arclength"`
- `bff_boundary_rectangular_correction_applied = True`
- `harmonic_solve_performed = True`

## Verification performed here

- `python -m py_compile src/onestring_physics/onestring_pipeline.py app_backup_before_mitered_t3d.py tests/test_onestring_pipeline.py`

Full pytest could not be run in this handoff folder because the ZIP contains only selected files; `onestring_physics.input_shape` and other package modules are not present in the extracted handoff package.
