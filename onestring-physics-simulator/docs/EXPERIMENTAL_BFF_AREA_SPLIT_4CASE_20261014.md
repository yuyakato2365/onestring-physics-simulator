# Experimental BFF corner selection × Fig.18 area bound: exact Lilium 4-case test

2026-10-14. This is **NOT the original OneString author code**.
The original paper does not publish Lilium BFF corner IDs, grid origin
or CSF sampling details. Do not infer one Lilium Split from "one string".

## Controls implemented (all OFF by default)

- `PipelineParameters.bff_rectangle_selection_mode`: `fixed_phase`
  (unchanged historical Paper) or
  `min_whole_source_area_span_4phase` (experimental).
  The selection criterion is ONLY the uncut, **entire original 3D
  surface** Jacobian area-span `max(J)/min(J)`, with
  `J=dA_S/dA_Omega`. The preset phases are
  `[0,1/16,1/8,3/16]` of the boundary perimeter.
  Every phase uses prescribed four 90-degree rectangle exterior angles
  with the unmodified official BFF library. No filename, Split count,
  tile count or tuned target output influences selection.
- `paper_split_area_limit_experiment`: `False` (unchanged default,
  `sqrt(Jmax/Jmin)<=2`) or `True` (experimental Fig.18 **area**
  ratio <=2, equivalently `sqrt(Jmax/Jmin)<=sqrt(2)`).
- `paper_csf_sampling_method`: default `overlap_extrema` or the
  separately labeled `interpolated_grid_vertices` diagnostic.

The new check is based on a **fixed UV chart and a potential
piecewise uniform scale**. It is NOT a fabricated-linkage
certificate, and no per-piece reflattening is performed. The 
standalone *post-cut re-BFF* experiment remains independent.

## Official BFF same-mesh 2×2 result

Input SHA256:
`4f62ff741b41bd629ccbeaeb0bb2e44ac060cdff88f090129911bede790e8ab3`.
All 4 cases: Lilium libigl OBJ, official BFF, M2D 14 grid,
interpolated M2D-vertex CSF sampling, 156 initial M2D quads,
no forced Split count.

| Corner selector | Cut limit | Split events | Linear CSF before -> after | Final area span | Termination | Approx 3D covered area |
|---|---|---:|---|---:|---|---:|
| fixed arclength phase 0 | linear 2 | 47 | 9.774 -> 2.051 | 4.207 | no admissible cut | 75.96% |
| fixed arclength phase 0 | **area 2** | 64 | 9.774 -> 2.051 | 4.207 | budget exhausted | 75.96% |
| auto min source-area span | linear 2 | **7** | 2.960 -> 2.125 | 4.515 | no admissible cut | 79.44% |
| auto min source-area span | **area 2** | **23** | 2.960 -> 2.125 | 4.515 | no admissible cut | 79.44% |

All four FAIL both the paper's strict area-span-2 interpretation
and the fixed-chart full-grid Split convergence to its configured
threshold. **Never report 7 or 23 as reproduced paper success.**
In particular, more cuts can be added with no reduction in the
maximum remaining CSF.

Source-rectangle phase evaluations from SAME input and official BFF:

| Phase (boundary fraction) | Whole-source area Jacobian span |
|---:|---:|
| 0 | 271.556 |
| **0.0625** | **17.155** |
| 0.125 | 26.662 |
| 0.1875 | 349.466 |

Thus `0.0625` is selected using a **pre-split geometric criterion**
rather than a Lilium split-count target.

### Why can't an additional grid split meet the bound?

The per-quad **irreducible local distortion** under the current
interpolated-grid CSF field is:

| Phase | At linear limit 2 | At Fig18 area limit 2 (linear sqrt(2)) |
|---|---:|---:|
| 0 | 1 / 156 | **34 / 156** |
| 0.0625 | 1 / 156 | **6 / 156** |

These quads individually exceed the applicable bound BEFORE any cuts.
Deleting M2D edges cannot remove intra-quad distortion when the
underlying UV and discretization are unchanged. This is a
**fixed-chart numerical impossibility**, not a proof that the
original authors' parameterization or later optimization cannot
build this shape.

### Evidence

- [All four official BFF pipelines, no count fitting](https://github.com/yuyakato2365/onestring-physics-simulator/actions/runs/38521453591) – success.
- [Paper and OFF-mode regression, including new BFF selection and
  area-unit tests](https://github.com/yuyakato2365/onestring-physics-simulator/actions/runs/38521615291) – success.

The original **overlap-extrema** default method has 49 cuts at phase 0,
not 47: the table above deliberately fixes the separately labeled
vertex-interpolated sampler for a controlled experiment.

## Run same experimental conditions on Mac

```bash
cd ~/Documents/OneString-large-steps-mesh-conditioning/onestring-paper-rectangle/onestring-physics-simulator
git fetch origin
git switch --detach origin/agent/paper-split-area-verified-20261014
source .venv/bin/activate
python scripts/setup_bff_rectangle.py

# Compare under the SAME interpolated-field convention
python scripts/verify_lilium_paper_split.py ~/Downloads/lilium.obj \
  --grid 14 --sampling-method interpolated_grid_vertices \
  --rectangle-corner-selection min_whole_source_area_span_4phase \
  --report output/lilium_corner_auto_linear.json

python scripts/verify_lilium_paper_split.py ~/Downloads/lilium.obj \
  --grid 14 --sampling-method interpolated_grid_vertices \
  --rectangle-corner-selection min_whole_source_area_span_4phase \
  --area-limit-two \
  --report output/lilium_corner_auto_area.json
```

If `setup_bff_rectangle.py` cannot find the official BFF source,
supply `--source /path/to/boundary-first-flattening` (exact upstream C++
source). **Rebuild** the adapter before selecting non-zero rectangle
corner phases on a Mac with an older binary.

The Streamlit Paper-reference BFF settings expander has the three
separate research switches with default OFF. Change them then use
the normal Run button to recompute the pipeline.

## Next faithful step

For the 6 locally overstretched quads after geometry-driven corner
selection under Fig18 area limit, determine if author uses a
distinct conformal bound interpretation, map, grid placement or
a post-optimization mechanical feasibility criterion.
Redesigning local geometry or using a new conformal chart must be
explicitly separated from already tested general-purpose Split
selection. Do NOT set a Lilium-only cutoff, phase or forced one-Split
budget.
