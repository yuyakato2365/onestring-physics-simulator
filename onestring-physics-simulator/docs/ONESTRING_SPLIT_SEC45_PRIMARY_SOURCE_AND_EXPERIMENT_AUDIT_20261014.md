# OneString §4.5: primary-source, references, and numerical reproduction audit

Date: 2026-10-14. Worktree:
`agent/paper-split-area-verified-20261014`.

**Status: NOT REPRODUCED and NOT VERIFIED against the authors' original
Lilium numerical Split output.** The shape named Lilium in the published
robot demonstration has **193 physical tiles and ONE CONTINUOUS STRING**.
The paper does **not** state that the Lilium design contains exactly ONE
Split. Figure 6 is a generic multi-stage splitting illustration and
does not identify Lilium. Do not use "1 Split in Lilium" as a published
ground-truth acceptance criterion without an original author figure,
dataset, code, or explicit confirmation. Different input assets and
cropped final tile topology may differ.

## Primary sources inspected directly

1. [Zaman et al. 2025 author paper](https://onestringtopullthemall.github.io/static/pdfs/onestringpull_authors_version_compressed.pdf),
   full PDF text via `pdftotext -raw`, §4.1, §4.5, Figures 5,6,10,15,
   discussion §6, bibliography.
2. [OneString supplement](https://onestringtopullthemall.github.io/static/pdfs/OneStringPull-supplement.pdf),
   Figure 18. This depicts quad-linkage **AREA** expansion by a factor
   of two, not an unambiguous factor-two bound on **LINEAR** stretch.
3. §4.5's exact four relevant citations:
   - Mina Konaković et al. 2016, *Beyond Developable: Computational
     Design and Fabrication with Auxetic Materials*, TOG 35(4).
     Bounded conformal mapping of triangular auxetics; cuts/cone
     singularities and nonlinear geometric optimization.
   - Mina Konaković-Luković et al. 2018, *Rapid Deployment of Curved
     Surfaces via Programmable Auxetics*, TOG 37(4).
     Bounded local isotropic scale and conformal cone singularities,
     then physical auxetic mechanism geometry.
   - Caigui Jiang et al. 2022, *Shape-Morphing Mechanical Metamaterials*,
     Computer-Aided Design 143:103146. Quad-kirigami mechanical mapping
     and special topology; **not** an implementable specification of
     OneString's Split optimizer.
   - Siheng Chen, Gary P. T. Choi, L. Mahadevan 2020,
     *Deterministic and Stochastic Control of Kirigami Topology*,
     PNAS 117(9):4511–4517. The **minimum links/topology/rigidity**
     argument, **NOT** a CSF-reduction solver.

## What the OneString authors actually specify

- §4.1: Flatten disk-like surface `S` by BFF, `c:S->Omega`.
  Only **where possible** force a rectangle; overlay a regular quad
  grid, retaining **fully enclosed** quads. For CSF >2 introduce
  Splits in M2D, then inverse-map M2D vertices via `c^-1` to M3D.
- §4.5: Require a bounded conformal scale factor `1<=lambda<=sigma`,
  give quad auxetic threshold `sigma=2`, invoke Figure 18's area
  bound, and hierarchically insert full grid-direction bisectors
  through high-Gaussian-curvature locations until disconnected
  components meet the limit. Avoid combinatorial singularities.
- Authors **do not specify** the Lilium original input mesh byte
  identity; BFF boundary vertices/pinning; rectangle corner indices;
  grid origin, margin, dimensions or angle; continuous-to-discrete
  CSF rule; candidate selection/tie breaking; or whether an
  independent post-Split BFF solve is performed.
- **No explicit post-Split BFF solve appears in §4.5 or the Figure 5
  pipeline**. The figure routes M2D Split to inverse lift; a blanket
  claim that the author re-flattens is unsubstantiated.
- §4.5 wording relates maximal **length** scale `lambda` to an
  **area** expansion limit. A conformal map has `J=lambda^2`.
  If the physical area range is 2, isotropic length range is
  `sqrt(2)`, not 2. Which quantity the authors use internally
  remains unpublished/ambiguous in these passages.
- §6 limitation: complicated Stanford Bunny requires numerous splits
  and optimal Split locations/orientations are beyond their current
  algorithm. There is no guarantee of one/few Splits for all meshes.

## Our numerical measurements: SAME libigl Lilium OBJ

SHA256: `4f62ff741b41bd629ccbeaeb0bb2e44ac060cdff88f090129911bede790e8ab3`.
File: 3389 surface vertices, 6590 triangular faces, one disk boundary.
The input OBJ is byte-identical between macOS and CI, but **is not
verified byte-identical to the authors' actual Lilium surface input**.

The **same general-purpose Split code** ran for all rows. No
name-dependent branch or split-count cap.

| BFF boundary | Grid N | Discrete CSF | Retained quads | Cuts | max sqrt(Jmax/Jmin), before -> after |
|:--|--:|:--|--:|--:|:--|
| forced rectangle, boundary arclength quartiles | 14 | all triangle/quad positive overlaps | 156 | 49 | 12.466 -> 2.276 |
| same rectangle | 14 | UV-area-weighted source-vertex logJ interpolated to M2D vertices | 156 | 47 | 9.774 -> 2.051 |
| standard free-boundary official BFF | 14 | M2D vertex interpolant | 86 | 0 | 1.538 -> 1.538 |
| forced rectangle | 15 | M2D vertex interpolant | 182 | 48 | 10.075 -> 2.090 |
| forced rectangle | 16 | M2D vertex interpolant | 210 | 57 | 8.916 -> 2.036 |
| author Lilium | UNKNOWN | UNKNOWN | 193 **physical** tiles | UNKNOWN | UNKNOWN |

The older independently computed cotangent-harmonic square-Dirichlet
14-grid **proxy** had 193 quads and in one linear-ratio variant could
adopt a single cut, but that is NOT official BFF and not demonstrably
the authors' configuration. Numerical coincidences do not establish
author fidelity.

### Test for misleading loss of source-surface coverage

We added independent coverage calculations, since fewer retained
tiles can yield a misleading improvement. Results for N=14:

| BFF | Omega-area crop coverage | Approximate lifted 3D quad-area vs source area |
|:--|--:|--:|
| forced rectangle | 81.19% | 75.96% |
| standard free boundary | 74.99% | 74.75% |

Surface coverage is of **similar magnitude**. Reduced split count
cannot be explained solely by a grossly reduced *total area*.
These fractions **do not verify matched spatial footprint**; M3D
lifted quad area is a triangulation-based proxy.

### Exhaustive one-cut audit (fixed UV only)

- Overlap extrema: 23 admissible complete initial-grid cuts; best
  single cut lowers max linear scale-span 12.466 -> 11.122, none <=2.
- Interpolated grid-vertex sampler: 23 complete cuts; best
  9.774 -> 9.188, none <=2.
- Full run with old overlap extrema: **49 separate cut events**,
  but **23 distinct (axis,position) combinations**; 26 cuts break
  exactly one shared pair of tile edges. These are separate local
  grid-piece events, not 49 distinct whole-initial-grid lines.
- After 49 events worst local linear span is 2.276 (area span 5.179).
  4 quads individually exceed threshold under the overlapping
  triangle extrema. `no_admissible_cut_remaining`, not convergence.

This does not prove author code could not succeed; it demonstrates
the present fixed-chart CSF discretization does not reproduce
the alleged one-cut output.

### A physical units red flag: official free-boundary false pass

Zero-cut standard BFF ends at **linear span 1.538** and
**area span 2.367**. Hence it passes the *current linear threshold 2*
but **not** a literal maximum *area expansion 2*. Merely switching
to free BFF is not a physical feasibility certificate.

### Controlled REAL surface-cut / re-BFF hypothesis

Cut the actual source triangle mesh by one original UV grid line;
duplicate seam boundary vertices; confirm each patch is a disk; run
the unmodified official BFF on the two 3D surface patches. Evaluate
full-patch triangle `sqrt(Jmax/Jmin)` after reparameterization:

| Initial seam (row) | Fixed original BFF: two patches | re-BFF with forced arclength-quartile rectangles | re-BFF with official free boundary |
|---|---|---|---|
| 0.378065792089 | 13.230 / 15.839 | 456.952 / 1281.424 | **1.595 / 1.262** |
| 0.084014620464 | 11.568 / 15.839 | 4853.426 / 28.846 | **1.111 / 1.452** |

One cut **can** reduce the full-source-patch linear scale span below 2
if a **new unconstrained** conformal solve is permitted after cutting.
This directly supports a possible mechanism for a small cut count.
It does NOT prove authors perform re-BFF. The output UV patches must
be tiled again, aligned, independently scaled and linked. In the
free-BFF first row, one patch has area span `1.595^2>2`;
thus it still cannot be certified under the supplement's literal
area expansion bound. The huge values under forced rectangles show
severe boundary/corner sensitivity and may include near-corner
discretization effects; they do not prove rectangular BFF always fails.

## Mathematical falsification of "just rescale each cut patch"

For a fixed component C and a uniform flat scaling s:
`J -> J/s^2`, so
`sqrt(Jmax/Jmin)` is invariant. Rescaling the old UV geometry
without recomputing the nonlinear/nonuniform conformal field
**cannot** lower the metric used by the current implementation.
It can change the absolute scale gauge, not the intrinsic span.

## Concrete code/provenance findings

- `src/onestring_physics/paper_mesh_splitting.py`: Paper mode
  samples `J=dA_S/dA_Omega` per source triangle and takes extremes
  from ANY triangle overlapping a retained M2D quad by >1e-12 quad
  area. Most extremes are selected by <1% overlaps (111/156 min,
  138/156 max). Raising the floor to 5% removes all four local
  floor violations but does **not** improve global scale span below
  7; this floor is **not** paper documented and was not deployed.
- Opt-in `paper_csf_discrete.py` vertex interpolation makes
  only a modest improvement (49 -> 47) and is **not author-proven**;
  original default remains unchanged.
- `paper_fidelity_mode.py` currently **forces** rectangular BFF,
  while §4.1 says rectangle **where possible**. C++ adapter chooses
  four 3D-boundary-arclength quartiles: an original **unverified**
  assumption, not published corner indices.
- Previous BFF provenance JSON confusingly recorded the enclosing
  OneString Git HEAD as the upstream BFF SHA. Corrected by recording
  actual upstream git SHA at independent adapter BUILD TIME, paired
  with executable SHA-256, and refusing to invent SHA for older
  builds without manifest. Existing older Mac binaries need rebuild
  to recover exact source provenance.
- A single split satisfying a `lambda` span is insufficient proof
  of full deployment: explicit flat tile scaling, optimization,
  contact/collision and physical auxetic expansion require
  independent verification.

## Reproducible GitHub Actions evidence

- [Same-mesh old vs vertex CSF, original rectangular BFF](https://github.com/yuyakato2365/onestring-physics-simulator/actions/runs/38514889984)
- [Rectangle vs free BFF](https://github.com/yuyakato2365/onestring-physics-simulator/actions/runs/38515401140)
- [Grid 15/16 comparison](https://github.com/yuyakato2365/onestring-physics-simulator/actions/runs/38515547138)
- [Same-mesh measured UV and 3D surface coverage](https://github.com/yuyakato2365/onestring-physics-simulator/actions/runs/38518176725)
- [Actual one-seam triangle clipping and free vs rectangular re-BFF](https://github.com/yuyakato2365/onestring-physics-simulator/actions/runs/38518407744)
- [Primary paper PDF extracted by GitHub Actions](https://github.com/yuyakato2365/onestring-physics-simulator/actions/runs/38517950096)

## Bottom line and next objective decision

1. **MATCHED**: high-level intended pipeline BFF to M2D/M3D,
   hierarchical full grid cuts, source Gaussian curvature priority,
   generic no-shape-specific code, same private/user OBJ verified
   across Mac and CI.
2. **FAILED TO MATCH**: figure-specific baseline input/UV coordinates,
   author corner constraint/crop, 193 final Lilium physical tiles,
   exact bound meaning, source field discretization, postcut
   deformation and fabricated linkage, author cut locations.
3. **UNVERIFIABLE FROM RELEASED PAPER**: Lilium's count of Split events,
   original mesh bytes and BFF settings, exact discrete Split choices,
   whether each cut triggers a re-BFF solve.

Highest-priority reproducibility work is acquiring the authors'
Lilium input/project/UV data, per-example BFF boundary and grid
conditions, and their numerical Fig6 Split field, *not* forcing
any particular cut count. If unavailable, establish and label a
new physically certified algorithm with known differences rather
than call it faithful replication. Splits cannot be certified from
`sqrt(Jmax/Jmin)` alone in the present software.
