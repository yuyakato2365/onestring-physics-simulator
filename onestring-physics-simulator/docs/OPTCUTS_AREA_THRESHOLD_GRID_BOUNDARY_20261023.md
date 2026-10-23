# Experimental OptCuts: 50%-area Grid footprint -> Omega boundary -> harmonic reparameterization

Branch: `fix/optcuts-real-m2d-footprint-20261023` (corrected real-M2D implementation).

## User's idea (implemented)

1. Generate original, successful official OptCuts UV chart **without** the
   previously unsuccessful post-OptCuts Grid-aware seam-vertex nudging.
   Preserve OptCuts's seam topology and the existing CSF-tail refinement.
2. **Actually generate the production Ω→M2D lattice first** using the
   unmodified OptCuts UV via the established `_flatten_to_domain` stage.
   Take its **actual origin, step, phase, resolution, and extra margins from
   `domain.uv_vertices`**. Do NOT independently estimate an origin/step
   from the OptCuts UV bounding box. **Never refit this lattice after moving
   the Ω boundary.**
3. Compute each cell's **exact** covered area by clipping the original UV
   triangles to the cell rectangle and summing their intersection areas:

   `r_q = area(original_Omega intersect cell_q) / area(cell_q)`.

   Retain cells `r_q >= threshold`, default `threshold = 0.50`;
   the UI slider permits `0.10 ... 0.95` in `0.05` increments.
4. Construct the polyomino union boundary of those selected rectangular
   tiles. Require a *single simple contour*: no detached islands, pinched
   vertices, holes or branching boundary segments. Refuse invalid output.
5. Preserve the cyclic order of original OptCuts UV boundary vertices and
   associate distinct boundary vertices with all corners of the target
   rectilinear polygon, interpolating other vertices along each grid edge.
   **If the source Ω has fewer boundary vertices than the target polygon has
   corners, automatically subdivide its boundary triangles in UV AND the
   corresponding source 3D surface**, retaining face-by-face correspondence.
6. Enforce these correspondence targets at the boundary and recompute
   interior UV using a sparse **fixed-boundary harmonic Dirichlet solve**
   with mild regularization to the original low-distortion OptCuts chart.
   Evaluate stages at boundary-displacement fractions 25/50/75/100%;
   all must be nonflipped and nonoverlapping, with the full final target
   realized. Refuse if distortion increases excessively.
7. Recompute derived Omega/CSF metadata for the moved UV, then restore
   **the exact original production lattice** including margins.
   Construct **the full, uncropped M2D overlay**. Apply the saved 50% mask
   directly, **BEFORE any full-cell/center boundary cropping**. Do not
   use the old strict crop first, because discarded boundary-crossing
   panels cannot be recovered by filtering an already cropped mesh.
   Continue through the existing CSF Split stage afterwards when enabled.
   Check the number of saved mask cells against the M2D cell count.
8. Send the resulting UV into the original paper-aligned M3D/K3D/K2D/T2D
   processing through the proven `optcuts_test2` carrier. The later
   solver formulas remain unchanged.

## Selection in Streamlit

- Paper-oriented computation mode: **OFF**
- Omega parameterization mode:
  **2026-10-23 | OptCuts Ω + Area-Threshold Grid Boundary → Reparameterization**
- Expand **Area-Threshold Grid Boundary（新Ω生成モード）**.
- Adjust **GridセルのΩ内面積割合・採用閾値** from its default **0.50**.
- In **View stage > Omega** review number of accepted tiles, number of
  grid outline corners, Symmetric Dirichlet before/after. The actual
  M2D should contain the same selected cell set.

The original 2026-09-18 `optcuts_test2`, the previous
`optcuts_grid_aware` experiment, `optcuts`, `optcuts_grid`,
Paper mode and all K3D/K2D solvers remain unchanged.

## Mathematical model / limitations

We are solving a **new experimental boundary-constrained parameterization**,
not invoking the authors' official BFF implementation and not modifying
the internal C++ OptCuts cut/sew optimizer.

- Candidate cell area test is exact for a nonoverlapping UV triangle mesh.
- A polyomino may be disconnected or have holes. Unlike a smoothing
  heuristic, this implementation refuses topological mismatches rather
  than silently throwing panels away.
- If the grid outline contains *more corners than the original UV boundary
  has vertices*, boundary triangle edges are now subdivided in parallel in
  the UV and source 3D meshes. Inverse mapping must use these updated paired
  faces. Extremely sharp outlines may still fail orientation tests.
- A nonconvex staircase is **not guaranteed** to admit an orientation-
  preserving harmonic chart with fixed boundary correspondence. The
  implementation fails with explicit
  `GRID_BOUNDARY_REPARAMETERIZATION_FOLDOVER` if necessary.
- UV similarity/SD and overlap checks do **not** establish downstream
  K3D/T2D physical fabrication feasibility.
- The phase and cell size used in this experiment are those of the
  **regular Omega overlay**, which can be non-square when the two
  original UV extents differ. A true equal-square lattice is a separate
  improvement.
- If original OptCuts has more than one UV boundary loop, this prototype
  explicitly refuses instead of silently fitting the first loop.
- The provisional official OptCuts start is selected before this fitting.
  Its *unfitted* secondary multistart UV chart is disabled as a potential
  downstream replacement **only in this experiment**. The 2026-09-18
  mode keeps its original multistart behavior.

## Validation

The CI test collection includes:
`tests/test_optcuts_area_grid_boundary.py` (exact polygon clipping,
50%-cells, topology guards, corner mapping, fully wired Omega->M2D
integration without native OptCuts),
`tests/test_streamlit_paper_startup.py` (independent Omega UI mode, the
default 50% slider, changing to 65%, and returning to baseline), and the
existing regression suite.

**Native OptCuts + real Bunny/Lilium full-pipeline execution has not
been performed in CI.** A pass in the synthetic tests should not be
described as a successful reproduction of the paper's physical pipeline.

Main implementation:
`src/onestring_physics/optcuts_area_grid_boundary.py`.
Integration:
`src/onestring_physics/optcuts_test_boundary_reparameterization_patch.py`,
`app_optcuts_core_20260916.py`, `app_runtime_base.py`.


## Audit correction: why the preceding branch was NOT the user's requested method

The original experimental branch was defective despite passing synthetic tests:

- It independently invented a lattice based on `min(uv)` and
  `(max(uv)-min(uv))/(2*nx,2*ny)`. In the real production pipeline,
  lattice origin/margins and possible alignment are established by
  `_flatten_to_domain`. Therefore the alleged retained panels were not
  necessarily the real M2D panels.
- It called `base_m2d`, which may **crop away boundary-intersected cells**,
  before applying the 50% mask. A mask cannot revive cells already removed.
- It attempted to fit the boundary **before** constructing the actual grid.
  This reversed the user's intended dependency.
- It raised `NOT_ENOUGH_VERTICES` instead of refining boundary triangles
  in UV and corresponding 3D geometry.
- It could bypass the usual source M2D Split pipeline.

The corrected pipeline explicitly does the opposite, in the required order:
`Original OptCuts S→Ω → ACTUAL M2D overlay (unfiltered lattice only)
→ exact per-cell Ω area coverage at 50%
→ selected M2D tile boundary
→ ordered UV/3D boundary subdivision/correspondence
→ fixed-boundary interior harmonic reparameterization
→ preserve original lattice
→ construct full M2D overlay → apply the SAVED area mask
→ existing Split/M3D/K3D/K2D`.

The Omega UI shows the original boundary, selected real M2D cells, target
cell-union contour, and fitted UV contour in one overlay plot. It reports
accepted counts and full-stage mean Symmetric Dirichlet to help diagnose
actual outcomes. The test
`test_area_mode_m2d_does_not_call_prior_strict_crop` ensures that a
pre-crop is never invoked in this mode.

**Known limitations:** the fixed-boundary harmonic mapping can fail on
strongly non-convex staircases or on a seam topology with multiple charts;
this is an explicit failure, not successful fabrication. Native OptCuts
on Bunny/Lilium and a full final K2D/T2D run have not been independently
validated here.
