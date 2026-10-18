# OneString Split validation (experimental, isolated)

This research-only CLI compares **explicitly named** metrics without changing
`paper_mesh_splitting.py`, `paper_fidelity_mode.py`, the Streamlit UI, K3D
or T2D. Based on base commit `00f935a7e172291156268f8630cd48e4521aae95`.

## What it computes

- S: the input source triangle disk mesh. Input SHA is recorded (Lilium is
  `4f62ff741b41bd629ccbeaeb0bb2e44ac060cdff88f090129911bede790e8ab3`).
- Initial Ω: official rectangular BFF, either its default arclength corner
  policy or a preregistered 4-lobe turning-angle selection. The lobe selection
  is a **hypothesis**, not the authors' choice.
- Grid: equal square cells, either initial-rectangle origin and max-side
  pitch or centered min-side pitch. Only fully contained cells remain.
- Complete Grid cuts: retain the original tile identifiers and initial UV
  coordinates, and split only the connectivity into two groups.
- Diagnostic BFF: clip original curved triangles along the candidate
  **parent Ω Grid line**; **only for the metric**, independently flatten each
  child with the existing official BFF CLI using boundary scale factors zero.
  Do not tile in the child free-BFF coordinates.
- Metric `free_bff_weighted_tile_area_v1`: for each child BFF triangle
  `J = area_3D / area_free_BFF`; for each original Ω square tile, compute
  the **mean of J weighted by overlap area in the original Ω**; for a child
  `sigma = max(averaged_tile_J) / min(averaged_tile_J)`.
  This ratio is dimensionless AREA RATIO, not linear CSF. It is neither
  the max/min *inside* each tile, nor a published author discretization.
  `fixed_uv_weighted_tile_area_v1` uses the *same* tile averaging on
  the initial rectangular BFF restriction instead.
- Stop candidate recursion when all groups have sigma <= 2, or cuts fail
  or the explicit budget is exhausted. Candidate selection is the nearest
  sampled intrinsic Gaussian curvature at interior Grid vertices after
  3-ring neighbor smoothing (not author-proven), with best of row/column
  based on child sigma. All rejected attempts are recorded.

**IMPORTANT:** B0 is the fixed-chart **metric proxy with matching experimental
tile aggregation**, not a byte-for-byte invocation of current Paper mode.
For the actual baseline, separately execute
`scripts/run_paper_full_pipeline.py` on the original `00f935a` branch.
Do not falsely equate the results. B4 and B6 change the chart with the
same averaging method; B7 replaces child free BFF with child rectangle BFF.

## Requirements and Mac command

**Important: this GitHub repo has an inner Python project directory**
`onestring-physics-simulator/`. Git worktree/clone root contains
`.github/` and that directory; commands must run from **inside**
the inner project directory containing `pyproject.toml`.

For an existing worktree placed at
`~/Documents/onestring-postcut-rebff/onestring-split-validation`:

```bash
cd "$HOME/Documents/onestring-postcut-rebff/onestring-split-validation/onestring-physics-simulator"
ls pyproject.toml scripts/validate_lilium_split.py tests/test_split_validation_independent.py
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q tests/test_split_validation_independent.py

export ONESTRING_BFF_EXECUTABLE="$(find "$HOME/Documents" -type f -name bff-command-line -print -quit)"
export ONESTRING_BFF_RECTANGLE_EXECUTABLE="$(find "$HOME/Documents" -type f -name bff-rect-command-line -print -quit)"
test -x "$ONESTRING_BFF_EXECUTABLE" && test -x "$ONESTRING_BFF_RECTANGLE_EXECUTABLE" && echo "BFF executables found"

curl -fL -o /tmp/lilium.obj https://raw.githubusercontent.com/libigl/libigl-tutorial-data/master/lilium.obj
python scripts/validate_lilium_split.py /tmp/lilium.obj \
  --require-lilium-sha --grid 14 --cases B0,B6 \
  --output output/lilium_split_validation
```

If `python3.11` is unavailable, install a Python version >=3.11 first,
for example `brew install python@3.11` when Homebrew is installed.
Use a **virtual environment** rather than the Apple CommandLineTools
system Python (old Python/pip builds cannot reliably editable-install this
`pyproject.toml` package). If either BFF executable test fails, build
or locate official BFF before running the experiment. The two `find`
commands are a **convenience**, not a proof of which BFF build was selected;
verify the printed paths if more than one BFF installation exists.
Never copy rendered Markdown links or `\\~` into the terminal; use
plain URLs and `"$HOME/Documents"`. Interactive zsh may reject pasted
`#` comment lines, so the copyable block intentionally has no comments.

For a native BFF build recipe see the repository-root
`.github/workflows/lilium-split-validation-experiment.yml`
and the inner project's `scripts/setup_bff_rectangle.py`.

Optional high-cost evaluations:

```bash
python scripts/validate_lilium_split.py /absolute/path/to/lilium.obj \
  --require-lilium-sha --cases B0,B1,B2,B3,B4,B5,B6,B7 \
  --sweep --sensitivity --output output/lilium_split_validation_full
```

A run writes `environment.json`, `run_config.json`,
`run_results.json`, `ablation.csv`, `cut_candidates.csv`,
`sensitivity.csv`, `figures/B*_cut_map.png`, and `index.html`.
The results are recorded without automatically rewriting thresholds
to get the expected 1-Split result.

## Evidence boundaries

- The author's Lilium reportedly has **193** tiles, whereas the attached
  hypothesis predicts **196**. A match on Split count is not geometric or
  mechanism reproduction.
- Figure 18 of the supplemental PDF establishes a maximum geometric
  area expansion of 2 for the depicted square linkage. It does **not**
  establish this program's `max(tile-mean-J)/min(tile-mean-J) <= 2`
  as a necessary-and-sufficient fabrication criterion.
- A child free-BFF chart is NOT used for manufacturing; numerical acceptance
  does not prove that the original-grid K2D/T2D linkage can open and avoid
  collisions. Its positive result is Level A (criterion-only), not Level C.
- The original code by Zaman et al. is not supplied with these materials.
  `lilium_reflatten_split_v2.patch` was referenced in the handoff but
  **not present** in the provided files or visible baseline branch.
  This package is an independent reconstruction, not an application
  of that missing patch.
- In particular, a parent's Jacobian span may exceed the product of its
  two children spans **even without reflattening** if their global scaling
  differs. Therefore the handoff's proposed product inequality must never
  be cited as proof of reflattening.

## Pending verification

CI has established that Lilium clipped child meshes are accepted by the
official BFF CLI for the Lilium input. Inputs, numbers, and images below are
intentionally **not** pre-filled or asserted before the workflow completes.
Comparing the hypothesis with the authors' published 193-panel layout and
running a real K2D/T2D feasibility check require separate experiments.


## Verified 2026-10-19 findings

See [RESULTS_20261019.md](RESULTS_20261019.md) for complete measured B0-B7
and real unchanged Paper mode baseline comparison. The B6 area-ratio
reconstruction yields 196 panels, **one cut (154+42)**, child ratios
1.9691/1.2505. Across 20 Grid/smoothing settings, one cut occurred 12 times,
not 18. The unchanged original Paper mode yields 48 cuts (grid interpolated)
or 45 cuts (overlap extrema) from 156 panels and still violates its CSF bound.
These are different discretizations and quantities; no author-code identity
or K2D/T2D manufacturability conclusion follows.
