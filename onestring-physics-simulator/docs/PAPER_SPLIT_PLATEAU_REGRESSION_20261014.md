# Paper hierarchical Split plateau fix — actual code + regression test

This is implementation work, not a claim that the authors' numeric
implementation has been fully reproduced.

## Concrete defect removed from Paper mode

Previously `paper_mesh_splitting._split_mesh_single` required **each**
cut to immediately lower the worst child `sqrt(max(J)/min(J))`.
This constraint was **not in OneString §4.5**. As an actual failing
counterexample, four two-row columns with constant per-quad
`J=[1,9,1,9]` are attainable by three full grid cuts, with final
span 1.0, but the old code stopped with 0 cuts, span 3.0.

**Paper mode now accepts a zero-improvement (plateau) full cut** when
it divides one connected quad region into exactly two nonempty
components. The existing curvature peak priorities and grid-direction
complete cuts remain unchanged. A formerly cut-boundary curvature
peak can guide the nearest interior gridline in a remaining component.
There are strictly fewer faces in each new component, so recursion
cannot loop forever in a fixed finite quad mesh.

Legacy non-Paper code is unchanged. The former restrictive policy
remains testable using
`paper_split_allow_neutral_cuts=False`. The default in Paper mode is
now `True`.

## Reproducible results

- [60 regression tests passed, including exactly 3 plateau-aware
  cuts on the synthetic counterexample](https://github.com/yuyakato2365/onestring-physics-simulator/actions/runs/38532560812)
- [Actual Lilium official BFF 2×2 comparison under new policy](https://github.com/yuyakato2365/onestring-physics-simulator/actions/runs/38532495517) (success)

| Corner selection | CSF stopping model | Old event count | New event count | New final linear span | New final area span |
|---|---|---:|---:|---:|---:|
| fixed phase 0 | linear span ≤2 | 47 | 48 | 2.0510 | 4.2067 |
| fixed phase 0 | area span ≤2 | 64 | 64 | 2.0510 | 4.2067 |
| selected 4-phase rectangle | linear span ≤2 | 7 | 8 | 2.1249 | 4.5152 |
| selected 4-phase rectangle | area span ≤2 | 23 | 40 | 2.1249 | 4.5152 |

**Lilium is still NOT reproduced or certified.** Both the original
and the modified algorithm hit the same maximal individual-quad
CSF lower bound on a fixed M2D and fixed BFF chart. More cuts can
resolve a *greedy policy defect* but cannot change the scale field
inside a quad. Do not treat higher split counts as successful
or alter the input-specific cutoff.

The next end-to-end implementation bottleneck is to construct a
different physically valid flat/curved linkage geometry that satisfies
the area/length bounds while retaining the target surface coverage.
Neither changing max-split budget nor editing the tie breaker can
achieve that for the existing fixed UV mapping.

## Reproduce on Mac

```bash
git fetch origin
git switch --detach origin/agent/paper-split-area-verified-20261014
source .venv/bin/activate
python scripts/verify_lilium_paper_split.py ~/Downloads/lilium.obj \
  --grid 14 --sampling-method interpolated_grid_vertices \
  --rectangle-corner-selection min_whole_source_area_span_4phase \
  --area-limit-two
```
