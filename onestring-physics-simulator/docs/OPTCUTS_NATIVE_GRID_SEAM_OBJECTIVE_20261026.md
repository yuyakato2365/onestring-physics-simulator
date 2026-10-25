# Native Grid-Seam OptCuts experiment (2026-10-26)

Branch: feature/optcuts-grid-seam-native-20261026. This branch extends the 10/23 area-grid Omega mode and retains the previous four ablation modes. The original Official OptCuts binary/source are left unchanged.

## Native C++ objective

Builds an isolated `third_party/OptCuts_grid_seam_onestring` tree from upstream `liminchen/OptCuts` using `scripts/setup_optcuts_grid_seam.py`. It patches verified anchors in `src/TriMesh.cpp` so *candidate boundary/interior splits* incur a new seam-direction cost and *candidate merges* are rewarded for removing diagonal edges.

edge_axis_cost = physical_edge_length / virtualRadius * sin^2(2*theta) = physical_edge_length / virtualRadius * 4*du^2*dv^2/(du^2+dv^2)^2.

**Current V3 behavior:** the native scoring is a *sign-preserving grid-aware
candidate-ranking heuristic*, not a separate global energy. The original
candidate decrease S is retained for S <= 0. For S > 0, Grid alignment
adjustment is bounded to min(lambda_t * beta * edge_axis_cost, 0.25*S),
subtracted on splits and added on merges. This preserves the sign of the
original move and all lambda-controller inputs. Previous V1/V2 formulas
and their hard move exclusions are no longer used. Full details follow
in the V3 repair section below.

The authors' UV quality and inversion checks are retained. This is a *local direction surrogate for staircase cost*, NOT an exact grid cell-mask penalty inside C++; true 50% area-selected mask corners, reflex counts and perimeter are still evaluated against the original production M2D grid in Python.

## New Omega selector

Select `EXPERIMENTAL 10/23 OptCuts Ω | Native C++ seam penalty + actual-grid boundary fitting`, then adjust `Grid seam penalty strength` (try 0, 0.05, 0.10, 0.25). The UI preserves the normal optcuts_test downstream geometry stack, but sets an isolated native backend flag. The new mode requires the patched binary, and never falls back to Official OptCuts.

After the exact-grid boundary fitting (including the original flip/overlap/mean SD guards), up to three valid candidates are compared using the existing true-grid contour score plus a UV-to-3D local square grid *proxy*: squared cosine of the two inverse mapping directions + squared logarithmic side-length ratio. It is NOT the actual M3D or K3D quad-angle metric.

## Mac setup

From the repository's outer worktree:

```bash
git fetch origin
git worktree add -b local/native-grid-seam \
  ../onestring-native-grid-seam \
  origin/feature/optcuts-grid-seam-native-20261026
cd ../onestring-native-grid-seam/onestring-physics-simulator
python3 -m pip install -e '.[dev]'
python3 scripts/setup_optcuts_grid_seam.py --jobs 4
python3 -m pytest -q tests/test_optcuts_grid_seam_native.py
python3 -m streamlit run app_optcuts_20260923_extrusion_aware.py
```

Compilation requires the same C++ toolchain, network and dependencies as the existing Official OptCuts setup. The isolated setup first builds clean upstream source, patches it, then rebuilds the C++ executable. Original OptCuts remains untouched.

## Research validation

Compare original 10/23, the four prior ①/③/⑤ ablations, and beta=0/0.05/0.1/0.25 under the same Bunny, actual grid spacing, 50% coverage, and downstream K3D settings. Inspect changes in the actual physical Cut edge set, grid corner/reflex counts, UV signed-area minima, mean/p95/p99 SD, number of selected panels, and actual M3D/K3D angle deviation (median/p95).

Validation status: Python regressions verify source patch transformation, binary resolution, parameter routing, and synthetic grid-square proxy. A full compiled native C++ run and real Bunny improvement remain UNVERIFIED and must not be claimed from those tests.

## V3 repair (2026-10-26): repeated -DBL_MAX and 600s timeout

Mac Bunny failure: the V2 native binary repeatedly printed

    evaluate vertex splits, 7792 candidate verts
    max E_dec = -1.79769e+308 < thres 0

The V2 hard gate on origEwDec <= 1e-12 incorrectly replaced legitimate
negative/zero original OptCuts split/merge candidate scores with -DBL_MAX.
The authors' dual lambda controller uses those signed energyChanges values.
Erasing candidates can prevent its switching logic from making progress.

V3 uses grid orientation ONLY to rerank already-positive original moves.
Let S be the original weighted OptCuts local energy decrease, G>=0 be
the grid alignment edge cost, lambda_t=1-lambda_init the old seam weight,
and beta the new relative grid strength:

    if S <= 0: new_score = S
    if S > 0: delta = min(lambda_t*beta*G, 0.25*S)
    if S > 0 and split: new_score = S-delta
    if S > 0 and merge: new_score = S+delta

Thus every nonpositive original score and every sentinel passes through
unchanged, positive scores remain positive, and beta=0 is identical to
the original local selection score. Original energyChanges used by the
lambda controller are kept unchanged. This is a conservative proposal
ranking heuristic, NOT a unified global Grid+UV objective.

V3 also adds explicit diagnostic failure instead of unlimited loops:
- Experimental TriMesh.cpp: three consecutive all-invalid candidate slates
  at identical lambda and source vertex count => clear error.
- Experimental main.cpp: four lambda searching loops are bounded and detect
  unchanged lambda; they report GRID-CUT-NATIVE-LAMBDA-STALLED.

This never fabricates a valid Cut/UV result or relaxes the distortion,
bijectivity, or collision constraints. The clean Official OptCuts remains
unchanged. The dedicated V3 resolver requires marker
ONESTRING_NATIVE_GRID_SEAM_V3_SAFE_RANK in BOTH C++ source files and
an executable newer than both.

### Update and rebuild on Mac

    cd /Users/yuyakato/Documents/onestring-postcut-rebff/onestring-native-grid-seam/onestring-physics-simulator
    git status
    git fetch origin
    git merge --ff-only origin/feature/optcuts-grid-seam-native-20261026
    python3 scripts/setup_optcuts_grid_seam.py --jobs 4
    python3 -m pytest -q tests/test_optcuts_grid_seam_native.py
    python3 -m streamlit run app_optcuts_20260923_extrusion_aware.py

Stop Streamlit before rebuilding. The build setup resets/repatches only
its own isolated C++ clone under third_party/OptCuts_grid_seam_onestring.

### Verification status

GitHub Actions currently passes synthetic and compiled-C++ helper tests,
plus prior geometry tests. The entire patched external OptCuts C++ build,
and Bunny runtime/quality, are NOT yet validated on the user's Mac.
Compare beta=0 against beta=0.1 with identical Bunny and Grid settings.
Do not increase the timeout or hide geometry invalidity to mask failures.

## 2026-10-26: Repeated boundary-fit continuation after successful native C++

The new native C++ run did finish in 61.4 seconds. The uploaded log shows
`GRID-SEAM-SEARCH` ranking four rotations in 10.7 seconds, then
`GRID-BOUNDARY-FIT` spending over 254 seconds on 8 of 12 boundary
correspondences for the first rotation. The first seven attempted
correspondences failed at 0.5%-6.5% of the boundary interpolation.
Their diagnostic bundles repeatedly identify triangle 2058 and
`boundary_crossings=2` on the *intermediate intended boundary*.

Thus the bottleneck is NOT C++ OptCuts but post-C++ boundary fitting:
the old method linearly interpolates the OptCuts boundary into the
grid-cell polygon, and for some choices this intermediate boundary
self-intersects. No interior UV solver can make that intermediate
boundary globally bijective.

Native-only repair:
- Pre-screen *all* original ear-safe boundary correspondences for actual
  witnessed proper crossings on a fixed sequence of intermediate
  boundary fractions. Do not spend the 12 expensive solver attempts on
  witnesses of an impossible straight-line intermediate state.
- If a narrow crossing interval escapes the sampled preflight, test each
  actual adaptive continuation step BEFORE calling harmonic/untangling.
  Abort the blocked linear path immediately with an explicit error.
- Re-rank the remaining ear-safe correspondences by existing displacement
  cost. The 50% cell mask, grid size, original Seam, 3D geometry, hard
  signed-area, SD, and overlap checks remain identical.
- Skip only the **specific straight-line correspondence/path** with a
  witnessed crossing. The new error is not a proof that the final target
  polygon lacks some other nonlinear bijective homotopy.

Important: this repair can avoid hundreds of seconds of futile work,
but does NOT create a new viable nonlinear boundary map. If every
existing correspondence is blocked, the correct next research task is
to design an injectivity-preserving nonlinear boundary isotopy or
to change the underlying Cut/grid mask before fitting. Never fake success
with a folded UV, mutate the 3D target, or raise the timeout as a fix.

Only Python source files changed in this repair; the V3 native C++
executable does not require rebuilding. Stop Streamlit, fetch/merge
the branch and restart with the same experimental Omega mode.
