# OneString Lilium: input-matched paper Split reproducibility audit (2026-10-14)

## Success criterion (NOT achieved)

Run the **same generic** Split program on the **same author Lilium input
and algorithmic settings** and reproduce the published number, location,
direction and geometry of the cuts. Never force one cut based on the
filename or tune acceptance to the desired cut count.

Paper sources:

- Zaman et al. 2025 [authors' paper](https://onestringtopullthemall.github.io/static/pdfs/onestringpull_authors_version_compressed.pdf)
  Sec.4.1: BFF S->Omega, rectangle boundary *where possible*, regular
  quad grid, crop fully interior quads; Split if CSF>2.
- Sec.4.5 and Fig.6: complete grid-direction bisections through high
  Gaussian curvature locations, hierarchical until each part meets
  bound 2; authors **do not publish the discrete field sampling,
  BFF rectangle corner vertex IDs, tiling parameters or exact cut
  acceptance algorithm** in these sections.
- Fig.15: assembled physical Lilium has **193 tiles**, not the 156
  produced by the current 14-grid rectangle hypothesis.
- [Supplement Fig.18](https://onestringtopullthemall.github.io/static/pdfs/OneStringPull-supplement.pdf)
  demonstrates maximal **area expansion factor 2** for square quad
  auxetics. A conformal *length* factor lambda and an *area* ratio
  lambda^2 cannot be treated as identical without a stated convention.

Input SHA-256:
`4f62ff741b41bd629ccbeaeb0bb2e44ac060cdff88f090129911bede790e8ab3`.
The same file bytes were used by macOS and all GitHub Actions below;
they are the published *libigl tutorial* Lilium OBJ, but the **authors'
preprocessing of their Lilium input remains unknown**.

## Unforced actual-mesh/official-BFF results

All tests run the *same* general-purpose hierarchical Split procedure.
For example, no --expect-splits was used in CI. All tests use the
**same uploaded mesh SHA**. Parameter changes only affect UV boundary,
regular grid density or CSF *discretization hypothesis*.

| Policy | M2D grid request | CSF sampling | Quad count | Cuts | max linear CSF before -> after |
| --- | ---: | --- | ---: | ---: | --- |
| Prescribed rectangle, arclength-quartile corners | 14 | overlap triangle extrema | 156 | 49 | 12.4663 -> 2.2757 |
| Prescribed rectangle, same corners | 14 | interpolated grid vertices | 156 | 47 | 9.7739 -> 2.0510 |
| Standard BFF free boundary, same input | 14 | interpolated grid vertices | 86 | 0 | 1.5384 -> 1.5384 |
| Prescribed rectangle, same corners | 15 | interpolated grid vertices | 182 | 48 | 10.0754 -> 2.0904 |
| Prescribed rectangle, same corners | 16 | interpolated grid vertices | 210 | 57 | 8.9156 -> 2.0363 |
| **Published Lilium** | **unknown** | **unknown** | **193 physical tiles** | **one visible complete slit** | **not published numerically for Lilium** |

Caution: `M2D quad count` and `physical tile count` need final
topology comparison; assuming identity is a test hypothesis, not a
proven equivalence.

### Data sources

- [Official BFF with existing and vertex-interpolated CSF](https://github.com/yuyakato2365/onestring-physics-simulator/actions/runs/38514889984)
- [Official rectangular vs official free-boundary BFF](https://github.com/yuyakato2365/onestring-physics-simulator/actions/runs/38515401140)
- [Grid 15 and 16 under exact OBJ](https://github.com/yuyakato2365/onestring-physics-simulator/actions/runs/38515547138)

### One-cut audit

For 14-grid, prescribed rectangle, overlap-extrema: 23 complete
candidate cuts, best worst-child CSF 11.122, no candidate <=2.
With grid-vertex interpolation: 23 complete candidates, best
worst-child CSF 9.188, no candidate <=2.

This **only proves fixed-UV one-cut infeasibility under the audited
discrete field**. It does not prove that the authors' BFF map, grid
layout or post-cut geometric optimization could not achieve one cut.

## Actual new code

- `src/onestring_physics/paper_csf_discrete.py`: generic,
  *opt-in* UV-face-area-weighted log-Jacobian recovery followed by
  barycentric sampling at M2D grid vertices; no sliver-coverage cutoff.
- `src/onestring_physics/paper_mesh_splitting.py`: distinguish
  `overlap_extrema` (current default) vs
  `interpolated_grid_vertices` research sampler.
- `tests/test_paper_csf_discrete.py`: unit tests, in CI.
- `scripts/verify_lilium_paper_split.py`: sampling method and
  official BFF boundary-policy switches; neither changes Split budget,
  trigger, curvature rule, cut acceptance or test expected result.
- `.github/workflows/lilium-official-bff-regression.yml` and
  `.github/workflows/lilium-grid-density-comparison.yml`: execute
  real native BFF, SHA-gated exact mesh and unforced Split, save
  reports as artifacts.

**The optional interpolated field has NOT replaced the current default**:
a paper-authored finite-element CSF sampling specification has not been
verified. Using a 5% triangle-overlap cutoff, custom 1-cut budget or
filename-specific branching is deliberately forbidden.

## Key finding and priorities

1. **BFF corner and boundary conditions** have a dramatic measured
   effect: with all other choices held fixed, a prescribed rectangle
   produces 47 splits versus **zero** with the official standard BFF
   boundary. The existing rectangle implementation chooses its four
   physical corner vertices by boundary arclength quartiles **without
   evidence that the authors used these vertices**.
2. **The grid is not matched**: observed 156, 182, and 210 M2D quads
   at grid counts 14,15,16 versus the physical 193-tile reference.
   Grid orientation/origin and cropped-topology comparison are needed;
   neither should be optimized just to minimize Split count.
3. **The Split criterion remains a candidate, not a physical
   certificate**: current code only evaluates `sqrt(max(J)/min(J))`
   in disconnected pieces; it does not recompute the conformal map
   or actually rescale the flat pieces and then certify the linkage
   deformation bound. The manuscript does not disclose enough
   detail in Sec.4.5 to identify the authors' numerical procedure.

Next practical reproduction step: obtain the Lilium-specific original
BFF boundary corner IDs/conditions, M2D grid placement, and exact
postcut CSF evaluation method from author artifacts or supplemental
data if they exist. Compare the resulting \Omega and M2D visually
and quantitatively **before** attempting to modify the curvature
Split acceptance code. A second path is to implement and compare
explicit per-piece UV scaling/reparameterization on the generic solver,
but label that an independently reconstructed method until authors'
details are available.

## Local commands

```bash
git fetch origin
git switch --detach origin/agent/paper-split-area-verified-20261014

python scripts/verify_lilium_paper_split.py ~/Downloads/lilium.obj \
  --grid 14 --sampling-method overlap_extrema \
  --report output/lilium_original_sampling.json

python scripts/verify_lilium_paper_split.py ~/Downloads/lilium.obj \
  --grid 14 --sampling-method interpolated_grid_vertices \
  --report output/lilium_vertex_sampling.json
```

For `--bff-boundary-policy automatic_reference`, also provide
`--bff-executable` pointing to the **official**
`bff-command-line`, not the rectangular adapter.
