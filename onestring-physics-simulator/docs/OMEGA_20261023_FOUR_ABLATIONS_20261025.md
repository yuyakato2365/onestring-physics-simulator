# Four isolated October 23 OptCuts Ω experiments (2026-10-25)

**Branch**: `feature/optcuts-omega-phase-tail-remesh-20261025`.

This branch starts from `fix/optcuts-real-m2d-footprint-20261023`.
It does **not** merge the separate 2026-10-25 Claude Cone-first / Seamless
Lattice approach, does not modify the authors' OptCuts algorithm, and does
not change the K3D / K2D objectives. Original Omega modes remain selected
and executed exactly as before.

## New Omega parameterization choices in Streamlit

| Visible mode | Underlying flags | Meaning |
| --- | --- | --- |
| `optcuts_area_grid_phase_search` | phase_search | ① Candidate grid orientation + half-cell translation search |
| `optcuts_area_grid_tail_sd` | tail_sd | ③ Tail-aware 3D-to-UV Symmetric Dirichlet optimization |
| `optcuts_area_grid_adaptive_remesh` | adaptive_remesh | ⑤ Paired source triangle subdivision adjacent to target grid turns |
| `optcuts_area_grid_combined` | all three | ① + ③ + ⑤ |
| `optcuts_area_grid_boundary` | none | Original unmodified October 23 baseline |

All variants use `omega_parameterization_mode="optcuts_test"` internally
and `optcuts_area_grid_boundary_enabled=True`, so downstream
10/23 M2D/M3D/K3D/K2D behavior and quality guards are shared.

### ① Grid search

Existing 10/23 search compares original official OptCuts start cuts and
rigid UV orientation candidates at 0, 22.5, 45, 67.5 degrees. The new
phase-search mode additionally tries production-grid origin offsets
`(0,0), (.5,0), (0,.5), (.5,.5)` **in units of the original cell spacing**.
It extracts cell spacing and grid origin from the *actual* candidate
production M2D overlay, then shifts the copied candidate overlay before
any selected-cell or UV boundary optimization. Does not alter mesh/cut,
cell pitch, overlay dimensions, or boundary coordinates in the source
chart. The mask and accepted origin are frozen after candidate choice.

Search is deliberately bounded (normally up to 32 candidate lattices for
two OptCuts starts and four rotations). This is only coarse half-cell
phase exploration, not arbitrary continuous phase optimization. If a
shifted candidate is invalid, it is rejected explicitly.

### ③ Tail-aware UV distortion

Original October 23 flow minimizes harmonic position and only invokes
3D-to-UV SD refinement when mean SD exceeds the hard acceptance limit.
Tail mode always refines a valid full-grid UV solution and adds a smooth
penalty to the top 5% of **3D-face-area-weighted** per-face SD distortions.
The threshold is fixed from the incoming UV; the objective uses softplus,
with an analytic gradient shared with the standard SD optimizer.

An improved tail candidate may be adopted only if:
- It has strictly improved smooth tail score.
- Mean SD stays below the existing hard acceptance threshold.
- Mean SD increases by no more than 2% from that candidate's start.
- UV signed area ratio remains ≥ 0.005 and the independent overlap test
  rejects all global collisions.
- Exact grid corners are retained. Boundary slide is confined to the
  original grid edges with ordered noncrossing bounds.

The original control path (tail disabled) is unchanged; it still optimizes
only when mean SD fails its original threshold.

### ⑤ Adaptive paired boundary-corner triangulation

After original triangle/fully-boundary-face corner correspondence
feasibility preflight, choose UV faces incident to original boundary
vertices assigned to the target grid's 90/270-degree corners.
Split up to 600 such faces with one **barycenter per triangle**:

`[a,b,c] → [a,b,m], [b,c,m], [c,a,m]`.

This is performed in **both UV and 3D**, with identical face ordering
and barycentric geometry interpolation. It adds only unconstrained
interior vertices (never splits any existing mesh edge), so there are
no hanging nodes, the original 3D piecewise-linear surface is
geometrically unchanged, and UV↔3D triangle correspondence remains
one-to-one. It is an adaptive *local vertex-density* refinement
targeted to candidate grid corners, not an arbitrary remesh of the
source asset and not a replacement for UV optimization.

The original full-boundary-ear preflight runs before subdivision:
splitting cannot silently make an infeasible corner correspondence pass.
Final signed-area and global overlap audits still apply.

## Scope, warnings and validation

- **These are four ablation modes, not verified improvements on Bunny.**
  Synthetic geometry/gradient/selection tests can pass even when actual
  Bunny boundary or K3D outcomes do not improve.
- The 50% original area mask rule and disk-topology constraint are
  retained. No grid cell is added or removed to artificially straighten
  a staircase, so a rectilinear staircase generally remains.
- Optional adaptive remeshing increases the number of UV and 3D source
  triangulation faces. On dense source meshes this can increase run time.
- The branch does not circumvent or silence
  `OptCutsUnavailableError`; official OptCuts binary is still required.
- Actual final diagnostics to compare: corner count, reflex count,
  final mean SD, area-weighted p95 / p99 of per-triangle SD, SD near
  original grid corners, 3D shape fidelity of M3D and K3D, and later
  collision count. For fair comparison use identical mesh, nominal grid
  size, 50% cell threshold and extrusion/solver settings.

## Running independently, without modifying existing 10/23 worktree

From the **outer repository worktree**:

```bash
git fetch origin
git worktree add -b local/omega-four-ablation-test \
  ../onestring-omega-four-ablation-test \
  origin/feature/optcuts-omega-phase-tail-remesh-20261025
cd ../onestring-omega-four-ablation-test/onestring-physics-simulator
python3 -m pip install -e ".[dev]"
python3 -m pytest -q tests/test_optcuts_omega_phase_tail_remesh.py
python3 -m streamlit run app_optcuts_20260923_extrusion_aware.py
```

Choose among the four `EXPERIMENTAL 10/23 OptCuts Ω` options in
**Omega parameterization mode**. All four intentionally need Official
OptCuts; build it with `python3 scripts/setup_optcuts_official.py` if no
verified binary exists in a sibling Git worktree.
