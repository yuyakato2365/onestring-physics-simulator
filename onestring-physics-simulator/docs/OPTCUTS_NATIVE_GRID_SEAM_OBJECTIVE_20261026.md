# Native Grid-Seam OptCuts experiment (2026-10-26)

Branch: feature/optcuts-grid-seam-native-20261026. This branch extends the 10/23 area-grid Omega mode and retains the previous four ablation modes. The original Official OptCuts binary/source are left unchanged.

## Native C++ objective

Builds an isolated `third_party/OptCuts_grid_seam_onestring` tree from upstream `liminchen/OptCuts` using `scripts/setup_optcuts_grid_seam.py`. It patches verified anchors in `src/TriMesh.cpp` so *candidate boundary/interior splits* incur a new seam-direction cost and *candidate merges* are rewarded for removing diagonal edges.

edge_axis_cost = physical_edge_length / virtualRadius * sin^2(2*theta) = physical_edge_length / virtualRadius * 4*du^2*dv^2/(du^2+dv^2)^2.

Each new split edge subtracts beta*edge_axis_cost from the original split score, and each removed merge edge adds beta*edge_axis_cost to the original merge score. Beta is additive and independent of the existing OptCuts seam-length/SD mixing lambda. Defaults to beta=0.10; beta=0 produces the original split/merge decision scores.

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