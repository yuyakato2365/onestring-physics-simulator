# Paper equation to code mapping

This document maps the main equations and algorithmic stages from
`onestringpull_authors_version_compressed.pdf` to the current simulator code.
The implementation is a paper-faithful approximation, not a direct port of the
authors' ShapeOp/libigl implementation.

## Figure 5 pipeline

Paper stage:
`S -> Omega -> M2D -> M3D -> K3D/T3D` and `M2D -> K2D -> T2D`.

Code:
- `src/onestring_physics/onestring_pipeline.py`
  - `build_onestring_design(...)`
  - `_build_surface_mesh(...)`
  - `_build_surface_parameterization(...)`
  - `_flatten_to_domain(...)`
  - `_build_m2d(...)`
  - `_lift_m2d_to_m3d(...)`
  - `_optimize_k3d(...)`
  - `_extrude_tiles(...)`
  - `_optimize_k2d(...)`
  - `_make_t2d_from_transforms(...)`
  - `_optimize_dual_hinges(...)`

The Streamlit control surface for this sequence is in `app.py`, especially the
`Run OneString pipeline` action and the stage viewer options.

## Eq. 1: assembled configuration energy

Paper:

```text
E_Assembled(x) = w1 E_Planar(x) + w2 E_Square(x) + w3 E_Surface(x)
```

Code:
- `src/onestring_physics/onestring_pipeline.py`
  - `_optimize_k3d(...)`
  - `_planarity_residuals(...)`
  - `_square_residuals(...)`
  - `_surface_fit_error(...)`
  - `_optimize_k3d_torch(...)`
  - `_optimize_k3d_mesh_torch(...)`

Implementation note:
`_optimize_k3d(...)` builds the residual vector from planarity, square-like
shape, and surface closeness terms. The user-facing weights are `w_planar`,
`w_square`, and `w_surface`; the default values match the paper notes shown in
the app.

## Eq. 2: planarity projection

Paper:
Planarity is the squared deviation of each quad from its best-fit plane.

Code:
- `src/onestring_physics/onestring_pipeline.py`
  - `_planarity_residuals(...)`
  - `_quad_planarity_error(...)`
  - `_tile_face_planarity(...)`
  - `_tile_face_planarity_by_group(...)`

Implementation note:
The code uses a compact SVD/best-fit-plane residual rather than the exact
ShapeOp projection stack.

## Eq. 3 and Eq. 4: square / edge projection

Paper:
Edges are projected toward target lengths, and quads are encouraged toward
square-like shapes.

Code:
- `src/onestring_physics/onestring_pipeline.py`
  - `_square_residuals(...)`
  - `_square_error(...)`
  - `_edge_length_variance(...)`
  - `_optimize_k3d(...)`

Implementation note:
The simulator approximates the paper's square energy through edge-length and
diagonal residuals. This is enough to expose square distortion metrics in the
app, but it is not the full authors' projection operator.

## Eq. 5: flat configuration energy

Paper:
`K2D` is optimized from `M2D` so that flat edge lengths match corresponding
`K3D` edge lengths, while maintaining fabrication clearance.

Code:
- `src/onestring_physics/onestring_pipeline.py`
  - `_optimize_k2d(...)`
  - `_optimize_k2d_torch(...)`
  - `_projective_edge_match_2d(...)`
  - `_strict_k2d_edge_length_solve(...)`
  - `_edge_matching_errors(...)`

Implementation note:
`_optimize_k2d(...)` uses `E_Flat = w1*EEdge + w2*ECollision + w3*EFab` in
the metrics. For medium and large grids, collision relaxation is intentionally
deferred to the dual-hinge layout stage so edge matching is not destroyed by a
slow all-pairs collision pass.

## Eq. 6: hinge layout energy

Paper:

```text
E_Hinge(x) = w1 E_Rigid(x) + w2 E_Collision(x) + w3 E_Conn(x)
```

Code:
- `src/onestring_physics/onestring_pipeline.py`
  - `_paper_local_global_se2_layout(...)`
  - `_optimize_t2d_top_hinge_footprint_layout(...)`
  - `_optimize_dual_hinges(...)`
  - `_optimize_rigid_assembly_hinge_layout_2d(...)`
  - `_hinge_constraint_tuples_from_specs(...)`
  - `_count_2d_footprint_collisions_from_pairs(...)`
  - `_sat_polygon_mtv(...)`

Implementation note:
The current implementation enforces tile rigidity by optimizing one rigid
SE(2) pose per tile. `E_Conn` is represented by hinge vertex midpoint
constraints, and `E_Collision` uses bounded SAT/AABB candidate projections.

## Lift point selection and GPE

Paper:
The paper selects a minimum set of lift points from gap gravitational potential
energy (GPE), Morse-Smale basins, and peak coupling.

Code:
- `src/onestring_physics/onestring_pipeline.py`
  - `_build_gap_graph(...)`
  - `_select_lift_points(...)`
  - `_build_string_path(...)`
  - `_turn_angle_total(...)`

Implementation note:
The simulator stores GPE-like gap scores and selects high-energy peaks with a
threshold (`lift_tau`). This approximates the Morse-Smale / coupling-DAG method
rather than implementing the full topological segmentation.

## Channel energy / string path friction

Paper:
The path objective is based on Capstan-style channel friction and cumulative
turn angle.

Code:
- `src/onestring_physics/onestring_pipeline.py`
  - `_build_string_path(...)`
  - `_turn_angle_total(...)`
  - `safe_capstan_friction(...)`

Implementation note:
The route is a lightweight graph walk over gaps. The metrics record
`turn_angle_total`, `theta_total`, `log_channel_cost`, and
`estimated_channel_friction`.

## Simulation energy

Paper:

```text
E_Simulation(x) =
    w1 E_Rigid(x) + w2 E_Collision(x) + w3 E_Actuation(x)
```

with actuation split into snap and lift constraints.

Code:
- `src/onestring_physics/onestring_pipeline.py`
  - `simulate_onestring_deployment(...)`
  - `_simulate_onestring_deployment_torch(...)`
  - `_project_rigid_tiles(...)`
  - `_project_aabb_collisions(...)`
  - `_project_snap_constraints(...)`
  - `_project_lift_constraints(...)`
  - `_project_hinge_constraints(...)`
  - `_project_target_pose_fit(...)`
  - `_project_target_contact_guard(...)`

Implementation note:
The string is not simulated as rope particles. It is encoded as positional
constraints:
- snap closes paired side-face midpoints along the selected string path;
- lift moves selected lift gaps toward prescribed 3D targets;
- rigid projection preserves tile shape;
- collision is currently a lightweight AABB projection, not the strict
  green-green SAT volume collision path that was removed because it made
  animation generation unusably slow.

## UI and metrics

Code:
- `app.py`
  - sidebar weights and solver controls
  - pipeline progress display
  - backend reporting for CPU/CUDA
  - smooth browser playback for simulation frames
- `src/onestring_physics/visualization.py`
  - stage figures and overlays
- `src/onestring_physics/animation.py`
  - assembly and tile animation helpers

The metrics table in the app exposes the current approximation status:
surface fit, planarity, square error, K2D edge matching, hinge connection,
collision count, snap error, lift error, and deployment error to `T3D`.
