# Native Grid-Seam OptCuts experiment (2026-10-26)

Branch: feature/optcuts-grid-seam-native-20261026. This branch extends the 10/23 area-grid Omega mode and retains the previous four ablation modes. The original Official OptCuts binary/source are left unchanged.

## Native C++ objective

Builds an isolated `third_party/OptCuts_grid_seam_onestring` tree from upstream `liminchen/OptCuts` using `scripts/setup_optcuts_grid_seam.py`. It patches verified anchors in `src/TriMesh.cpp` so *candidate boundary/interior splits* incur a new seam-direction cost and *candidate merges* are rewarded for removing diagonal edges.

edge_axis_cost = physical_edge_length / virtualRadius * sin^2(2*theta) = physical_edge_length / virtualRadius * 4*du^2*dv^2/(du^2+dv^2)^2.

**Revised after a real 600-second Bunny timeout:** each proposed split now subtracts lambda_t*beta*edge_axis_cost from the ORIGINAL OptCuts score; each merge adds lambda_t*beta*edge_axis_cost. Here lambda_t=1-lambda_init, normally 0.001 when lambda_init=0.999. We do NOT add beta independently: independent beta=0.10 can be 100x stronger than the 0.001 seam-weight scale. For beta>0, only split/merge candidates with strictly positive ORIGINAL OptCuts objective decrease are eligible. The unmodified global OptCuts energy is therefore not knowingly increased by a new grid reward. Beta=0 keeps the original split/merge scores unchanged. This is a candidate-scoring heuristic, not a globally optimized new energy; it still requires geometric and actual Bunny convergence validation.

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

## Timeout diagnosis and V2 upgrade

After an actual 600-second native OptCuts timeout on Bunny, the initial V1 objective was found to overscale the grid penalty and potentially accept topological operations that did not improve OptCuts' original energy. **V1 builds are now rejected**: the resolver requires `ONESTRING_NATIVE_GRID_SEAM_V2_MONOTONE` in the patched C++ source with a fresh executable. The isolated builder resets the *experimental* C++ source to the upstream commit before applying the patch, so the following commands are sufficient; no destructive reclone is necessary:

```bash
git fetch origin
git pull --ff-only
python3 scripts/setup_optcuts_grid_seam.py --jobs 4
python3 -m pytest -q tests/test_optcuts_grid_seam_native.py
python3 -m streamlit run app_optcuts_20260923_extrusion_aware.py
```

The dedicated native runner now emits a PID and a periodic 20-second heartbeat, preserving the partial C++ stdout/stderr for timeout reports. An unchanged 10/23 mode still uses the original Official OptCuts subprocess. In a rigorous experiment, run beta=0 first (same input/settings) to measure the original C++ runtime before beta=0.10. The 600s limit is NOT increased or suppressed. If beta=0 also exceeds 600s, the modified grid score is not sufficient to explain the entire slowdown: capture C++ tail logs and reassess original OptCuts cost and solver convergence.
