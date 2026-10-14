# Lilium BFF boundary corners and CSF-limit reproducible test (2026-10-14)

**Status: numerical tests complete; original OneString author's implementation NOT reproduced.**
Official input source used here: the byte-identical `lilium.obj` from the user's Mac
and libigl tutorial (SHA256
`4f62ff741b41bd629ccbeaeb0bb2e44ac060cdff88f090129911bede790e8ab3`).
The author's exact Lilium input / parameterization files were **not published in
the project webpage repository**, so input equivalence to the **author** is
NOT established, only equivalence Mac ↔ GitHub CI.

## 1. One-variable phase sensitivity of BFF rectangle corners

Four C++ BFF calls use the **same unmodified upstream BFF library**, original
triangle OBJ, M2D 14×14 grid, general-purpose hierarchical full-grid Split,
and **vertex-interpolated** CSF sampling (a separately labeled research hypothesis).
Only the four 90° exterior-angle BFF corner positions are shifted along
the boundary by the stated arclength fraction.

| Corner arclength phase | Actual initial M2D quads | Approx source 3D area crop fraction | Initial linear CSF span | Generic Split events | Final max linear CSF | Final area span |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0000 (prior default) | 156 | 0.75964 | 9.77386 | 47 | 2.05101 | 4.20666 |
| 0.0625 | 156 | 0.79437 | 2.95993 | **7** | 2.12491 | 4.51524 |
| 0.1250 | 156 | 0.79181 | 3.71341 | **7** | 2.24464 | 5.03840 |
| 0.1875 | 156 | 0.74708 | 10.18812 | 50 | 2.27732 | 5.18617 |

**Every entry is measured, none meets its own end-of-Split linear CSF ≤2.**
The experiment does NOT select the phase with lowest Split count as "author
correct" or set a Lilium-specific override. It proves that the **arbitrary
start position of four arclength-quartile rectangle corners** can make a
large difference, *even for identical tile count*; physical coverage
locations still differ and were not proven pointwise identical.

[Actual native BFF 4-phase matrix (all 4 passed)](https://github.com/yuyakato2365/onestring-physics-simulator/actions/runs/38519667889).

## 2. Source-formula and unit validation

Sec.4.5 says `1 ≤ lambda ≤ sigma`, gives the square auxetic `sigma=2`,
but cites the supplement's **area expansion** ratio 2
([supplement Figure 18](https://onestringtopullthemall.github.io/static/pdfs/OneStringPull-supplement.pdf)).
For a truly conformal map,
`J = lambda^2`. With disconnected component normalization by
uniform scaling, the *minimal* linear span is
`sqrt(Jmax/Jmin)`; corresponding *area* span is `Jmax/Jmin`.
A square-rotation toy linkage has limiting linear pitch `sqrt(2)`
and area pitch `2`. Thus a linear ratio <2 is insufficient for a
literal twofold **area** bound.

**Measured sanity check:** normal free-boundary BFF at grid 14 has
0 Splits and final linear CSF span 1.538, **but area span is 2.367 >2**.
This is NOT a physically certified OneString solution.

Three independent CI tests demonstrate:
(1) linear 1.5 / area 2.25 give opposite pass/fail against 2,
(2) uniform UV rescaling leaves component `sqrt(Jmax/Jmin)` invariant,
(3) ideal rotating-square pitch squares to exactly area factor 2
at maximum deployment.
[Paper-mode unit and algorithm regression](https://github.com/yuyakato2365/onestring-physics-simulator/actions/runs/38519884600)
succeeded.

## 3. All 23 complete initial-grid cuts + actual re-BFF of two real surface patches

Previous fixed BFF audit found **0/23** single initial-grid cut candidates
can attain linear span ≤2 by merely splitting the fixed UV.
To falsify or corroborate the *unpublished* "after-cut re-BFF" hypothesis,
for EVERY generic candidate we:
- clip the actual source triangular surface along its BFF-UV line,
- validate both new triangular meshes as disk topology,
- run **unmodified official standard BFF** independently on each half,
- evaluate **full-source-patch** triangle `sqrt(Jmax/Jmin)` and `Jmax/Jmin`,
- demand both halves satisfy the bound; no shape-specific cut placement.

All 23 candidates completed (46 separate patch solves):

| Test | Results |
|---|---:|
| Both patch length spans ≤2 | **23 / 23** |
| Both patch area spans ≤2 | **0 / 23** |
| Best candidate by maximum area span | `col:0.378065792089` |
| Best maximum length span | 1.436724 |
| Best maximum area span | **2.064176** |

**A proof of one-cut sufficiency under the current true area threshold
WAS NOT obtained.** The best is close but remains >2.
This shows reparameterization can profoundly change scale distortion,
but OneString §4.5 does **not say** it re-runs BFF after each split.
This is a **research hypothesis only**. Re-BFF on entire source patches
also changes their planar quad layouts; the actual M2D flat linkage
constraints were **not** rebuilt/certified.

[All-cut exhaustive official re-BFF test](https://github.com/yuyakato2365/onestring-physics-simulator/actions/runs/38520102068).

## 4. Source fidelity and unresolved author settings

[Original paper](https://onestringtopullthemall.github.io/static/pdfs/onestringpull_authors_version_compressed.pdf)
§4.1 uses BFF and sets rectangular boundary only **where possible**.
§4.5 prescribes high-Gaussian-curvature full grid cuts, hierarchical
until every region's conformal scale bound is met. It does NOT publish
Lilium BFF corner vertex IDs, exact mesh preprocessing, grid
orientation/origin or specific CSF discretization.

Figure 15 states **193 physical Lilium tiles and one continuous string**:
this is NOT a published claim of exactly **one Split** on Lilium.
Figure 6 is a schematic multi-cut example and is not identified
as Lilium. Matching any arbitrary one-cut criterion is therefore not
proof of author code reproducibility.

The author project page GitHub includes paper/supplement/video/static
media but, in inspected directories, **no original Lilium source code,
mesh, UV map or parameterization settings**.

## 5. Changes applied and safe defaults

- Added an optional four-corner phase argument to the **independent**
  rectangle BFF adapter and explicit verification script. **Phase 0**
  is unchanged from the original implementation, as verified by
  identical 47-Split baseline.
- Added **actual rectangle boundary wedge position diagnostics**
  to strict BFF metadata.
- Added automated four-phase native BFF experiment; no hardcoded Lilium
  Split behavior.
- Added exhaustive candidate + re-BFF source-patch falsification.
- Added unit tests for the `J=lambda^2` distinction and uniform
  rescale invariance.
- **Did not** change the default paper CSF sampling, Split acceptance
  logic, or automatically select rectangle corners based on the result.

### Reproduce the corner-phase experiment on Mac

Ensure the BFF rectangle adapter is rebuilt from current source:
`python scripts/setup_bff_rectangle.py` (or specify
`--source /path/to/official/boundary-first-flattening`).

```bash
git fetch origin
git switch --detach origin/agent/paper-split-area-verified-20261014
source .venv/bin/activate
python scripts/setup_bff_rectangle.py
python scripts/verify_lilium_paper_split.py ~/Downloads/lilium.obj \
  --grid 14 --sampling-method interpolated_grid_vertices \
  --rectangle-corner-phase 0.0625 \
  --report output/lilium_corner_phase_0625.json
```

The **remaining goal** is obtain author's Lilium mesh, BFF corner/boundary
and grid settings, pin down lambda vs area-span convention from actual
linkage geometry, and compare full geometry and physical deployment.
This report **does not** claim original paper output match.
