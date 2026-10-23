# Experimental OptCuts: 50%-area Grid footprint -> Omega boundary -> harmonic reparameterization

Branch: `feature/optcuts-area-grid-boundary-20261023`.

## User's idea (implemented)

1. Generate original, successful official OptCuts UV chart **without** the
   previously unsuccessful post-OptCuts Grid-aware seam-vertex nudging.
   Preserve OptCuts's seam topology and the existing CSF-tail refinement.
2. Overlay the same fixed axis-aligned lattice used for fabrication:
   `step_x = (max_u-min_u)/(2*grid.nx)`,
   `step_y = (max_v-min_v)/(2*grid.ny)`. The production extra margin is
   retained. **Do not refit this lattice after the boundary is moved.**
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
6. Enforce these correspondence targets at the boundary and recompute
   interior UV using a sparse **fixed-boundary harmonic Dirichlet solve**
   with mild regularization to the original low-distortion OptCuts chart.
   Evaluate stages at boundary-displacement fractions 25/50/75/100%;
   all must be nonflipped and nonoverlapping, with the full final target
   realized. Refuse if distortion increases excessively.
7. Reuse the exact *original fixed grid* and exactly those selected cells
   when building M2D. Never run strict full-cell clipping again against the
   old Omega. No UV/M2D refeedback, grid regeneration or silent fallback.
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
  has vertices*, the target cannot be represented without subdividing
  boundary triangles. The current implementation raises
  `GRID_BOUNDARY_NOT_ENOUGH_VERTICES`; automatic subdivision is **not**
  implemented yet.
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
