# New OmegaParameterizationMode: `optcuts_seamless_lattice` (experimental)

Date: 2026-10-25
Base branch: `fix/optcuts-real-m2d-footprint-20261023` @ `71d3058`
Feature branch: `feature/optcuts-seamless-lattice-omega-20261025`

## Purpose

Test the October 25 external research report, *OneString：Ωの作り方とK3Dパネル形状・T2D衝突*, without modifying working Omega modes or their defaults.

The report establishes on synthetic dome/tube examples that snapping an OptCuts chart boundary to the exact staircase outline of retained Grid cells imposes 3D boundary corner angles of ~170–180° at convex staircase corners and ~60° at concave corners. It hypothesizes that imposing **lattice-compatible transitions between two source-cut UV copies**, instead of conforming the source boundary to the grid staircase, can avoid the source-induced degeneration.

A full seamless quad atlas requires global transition consistency, cone planning, **and a seam-aware quad-cell M2D topology**. The feature implemented here is an independently selectable UV-stage *pilot*, NOT a full production-grade quad atlas. Do not confuse transition residual 0 with assembled mesh closure.

## UI and isolation

Choose from Streamlit **Omega parameterization mode**:

`2026-10-25 | EXPERIMENTAL Seamless Lattice Ω (90°/integer transitions, no boundary staircase)`

The frontend chooses the proven `optcuts_test` carrier and sets `PipelineParameters.optcuts_seamless_lattice_enabled=True` for **this session only**. It never enables `optcuts_area_grid_boundary_enabled`. All other modes are unchanged.

## Implemented UV method

1. Run original OptCuts with its original physical seam/cut topology. Bypass the legacy optcuts_test Taubin boundary smoothing in this mode only.
2. Pair both UV copies **by original surface edge and physical vertex IDs**, using `_surface_seam_records`, `_physical_chains`, `_uv_copy_chains_for_physical_chain`. Do NOT use UV nearest-neighbor matching.
3. Search over four grid automorphisms:
   `u_B = R^k u_A + h*n`, `k in {0,1,2,3}`, `n in Z²`, `h=tile_size`.
   **2026-10-25 bug fix:** independently rounding the nearest `(k,n)` for every cut chain can introduce an impossible nonzero translation around a source seam junction (`SEAMLESS_INCONSISTENT_TRANSLATIONAL_HOLONOMY`). Now `optcuts_seamless_joint_transitions.py` enumerates alternatives per physical chain and uses a bounded joint beam search with a C4/integer-offset union-find. Every partial candidate passes an EXACT global cycle/translated cone check before any UV optimization; original source UV correspondence is validated edge by edge. No seam equation is silently dropped. Each chain gets one rotation/translation. It is NEVER straightened into an artificial parallel horizontal/vertical line.
4. Build a global graph of transform equations. Propagate root-to-vertex affine maps. Reject inconsistent translational holonomy at source junctions. The discrete search **tries coordinated changes to chain transforms first** instead of aborting upon an independently chosen inconsistent round-off. A cone with compatible lattice-grid fixed point is permitted; partial/incompatible/off-grid cones are reported explicitly and rejected. **General cone placement (MIQ/QuadCover), arbitrary change in official cut topology and exhaustive global search are not implemented**. Default beam width 24 and 24 local candidates per source chain are research-budget limits, not proof of infeasibility. When no compatible assignment is found, the exception `SEAMLESS_JOINT_TRANSITION_INFEASIBLE` includes source chain, frontier, rejection counts and example conflicting integer steps.
5. Minimize a sparse quadratic combining source OptCuts UV anchors and original triangle-edge vectors over independent root UV positions. Seam transition equations are **exact by variable elimination**, not strong soft penalty. All original OptCuts surface triangles and the original Grid spacing remain unchanged.
6. Validate the original surface/UV face orientation and source-relative signed triangle areas. A flip/collapse raises `SEAMLESS_UV_FLIP_OR_COLLAPSE`. UV folded candidates NEVER fall back to the current area-grid staircase.
7. Update `SurfaceParameterization.uv_vertices_2d` and `omega_boundary` in the real production builder. Record rotation, integer shift, source-chain IDs, UV residual, cone count, worst signed area and RMS UV motion as metrics.

## Actual downstream status

The existing `optcuts_test` M2D strict whole-cell crop remains the downstream builder. The pilot adds a **read-only audit** of the 2D source-UV area covered by the resulting actual M2D cell footprint, calculated from the existing `_intersection_ratios()`. Stored metrics:

- `seamless_lattice_m2d_source_uv_area_coverage`
- `seamless_lattice_m2d_source_uv_uncovered_area_fraction`
- `seamless_lattice_m2d_source_uv_coverage_pass_98pct`
- `seamless_lattice_m2d_3d_area_coverage_verified=False`

This is a UV-domain area coverage measure, **not** the original S surface area-weighted coverage gate; in particular the latter is not declared successful. There is NO invented M2D cross-seam cell topology, no source remeshing and no guaranteed K3D exact closure. The K3D SmoothMax and current T2D are left unchanged. The experimental result can be inspected as an ablation, but should not be used to claim a valid final OneString design.

## Why this is explicitly partial

A transition `u_B=R^k u_A+h*n` only proves that both UV copies share a periodic lattice coordinate system. Correct M2D construction still requires enumerating cells on the **quotient atlas** around a seam, lifting vertices through the appropriate face chart (potentially several copies), and proving no covered source triangle is lost or counted twice. A single-chart whole-cell crop does not achieve that. Similarly, the cone network may need new singularities and cuts on Bunny to be solvable.

A next implementation milestone is therefore **seam-aware quad atlas M2D**:
- build per-cell chart/physical-triangle ownership across all paired Seam copies;
- resolve transition transport across seams and cone junctions;
- maintain consistent original 3D correspondence for every grid vertex;
- count faces and edges after physical seam gluing and prove geometric coverage, no T-junctions and no duplicate faces;
- only then enable source-equivalent `equality_groups` for K3D and compare to the prior soft-Seam K3D results.

Do **not** reactivate the old hard `source_cut_affine_rows` for mismatched partitions, because its nullspace can force a fully collapsed seam. The new mode currently leaves that route untouched, instead of claiming the required cut/quad atlas is solved.

## Reproduce

```bash
git fetch origin feature/optcuts-seamless-lattice-omega-20261025
git switch feature/optcuts-seamless-lattice-omega-20261025
python3 -m streamlit run app_optcuts_20260923_extrusion_aware.py
```

Select the new mode, use the same Bunny, tile size, grid, K3D, T2D settings as baseline. Look for:

```
[OMEGA-SEAMLESS-LATTICE] chains=... UV_cut_residual=... staircase_snap=OFF
[OMEGA-SEAMLESS-M2D-AUDIT] source_uv_area_coverage=... 3D_coverage=UNMEASURED
```

If a cut-tip cone violates the lattice constraints or the constrained UV chart folds, expect an explicit `SEAMLESS_...` exception. This is correct **research evidence**, not a reason to silently change a working mode or change source topology.

## Validation gates (none claimed for real Bunny yet)

1. Source cut chains each obey the 90°/integer translation transform to <=1e-7 grid units.
2. No UV triangle flips, self-overlaps or catastrophic symmetric-Dirichlet increase.
3. M2D source-UV and S area-weighted coverage >=98%, no fabricated seam strips.
4. Same physical grid cells contribute exact 3D seams; K3D angle range and Seam gap after optional hard source vertex identification measured.
5. T2D collision pairs, hinge RMS and negative opening angles compared with baseline, with Collision-driven Split ON (default settings).

Only gate 1 and triangle signed-area of gate 2 are currently enforced. Self-overlap, strict 3D coverage and hard closure remain **unverified**.

The October 25 tube script `omega/exp3_seam_vs_seamless.py` is a special known periodic arrangement; it is not evidence that arbitrary OptCuts charts or Bunny automatically admit a compatible lattice-atlas mapping.

## 2026-10-25 Bunny failure and follow-up

Reported failure before this fix:
```
SEAMLESS_INCONSISTENT_TRANSLATIONAL_HOLONOMY: vertex=8129 mismatch=0.6324555
```
This means that the independently picked source-seam lattice transforms did not compose to the identity around one UV graph cycle. It **does not by itself prove** the Bunny cut topology mathematically impossible: a different integer translation / 90-degree rotation assignment may satisfy the graph.

The new jointly constrained discrete search and explicit UV edge-copy validation are implemented in the **separate experimental mode only**. Synthetic tests include a graph in which both locally best independent transforms conflict but the global solver chooses a second-best transform and closes the graph exactly. The existing area-grid, BFF, OptCuts, SmoothMax and T2D modes are unchanged.

**Not yet verified:** the user's same Bunny mesh and original full Streamlit parameter set; 3D seam closure, K3D degenerate panel counts and collision behavior. If the next run raises `SEAMLESS_JOINT_TRANSITION_INFEASIBLE`, share its `frontier`, `first_conflicts` and `rejected` diagnostics. This may indicate a real cone-placement/cut-topology obstruction **or** the bounded search missing valid alternatives. Do not force this by weakening the exactness threshold or erasing Seam equations.
