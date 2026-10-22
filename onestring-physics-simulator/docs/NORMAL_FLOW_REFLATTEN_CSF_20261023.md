# Reflatten tile-area CSF in the normal (Paper OFF) flow

This change reuses the **existing Paper-oriented `reflatten_tile_area` engine unchanged** for the normal OneString pipeline. It is an experimental reconstruction and is not verified author code.

## UI
1. Turn **Paper-oriented computation mode** OFF.
2. Choose the usual target mesh, Omega parameterization and grid (for example, the existing `rectangular_harmonic_legacy` mode). Note: harmonic/LSCM initial maps are **not** official BFF, even though reflatten assessment uses official free-boundary BFF.
3. Select **Split CSF判定（Paper / 通常フロー共通）** -> **3Dピース自由境界BFF再展開 → タイル面積比 ≤ 2**.
4. Supply an official free-boundary `bff-command-line` binary via the existing **official bff-command-line path (nonrectangle)** field (or the existing environment variable).
5. Run the existing OneString pipeline. `fixed_uv` remains the default in Paper OFF with non-reference Omega, so old experiments are not silently changed.

## Shared algorithm
For each current 3D source patch:
1. Compute a new official *free-boundary* BFF parameterization of that patch.
2. For each original M2D grid quad, lift its four sampled corners onto the original 3D source and transfer them into the new patch UV using barycentric coordinates.
3. Compute `J_q=A3D_q/A2D_q(reBFF)` and `sigma_P=max_q J_q / min_q J_q`.
4. A patch needs a Split if `sigma_P>2`. Candidate complete grid-row or grid-column cuts pass through a smoothed-Gaussian-curvature peak. Compute official free-boundary BFF again for each candidate child and choose the direction minimizing the worse child `sigma_P`, unless manually overridden.
5. Add all planned cuts to actual M2D topology, preserving existing downstream layout, 3D optimization, and independently accepted collision-driven cuts.

## Safety and interpretation
- The reflatten UVs are **for scoring only** and are **not** used to reshape the final manufacturing M2D tiles. A passing score is not a physical deployability or collision-free certificate.
- Any initial Omega mapping must be finite, nondegenerate and locally unfolded. The normal flow refuses locally flipped/degenerate charts; it never silently repairs them.
- The initial S->Omega method is recorded as `csf_split_initial_omega_method`. With harmonic/LSCM Omega, it is explicitly *not* a literal Paper reproduction.
- `sigma_P` is an **area-span ratio**, not a linear conformal factor `sqrt(sigma_P)`; the threshold 2 retains the previous Paper experimental convention.
- The reflatten calculation requires the native official BFF CLI. No fallback to harmonic mapping is permitted on failed child BFF.
- Imported meshes are checked as single connected disk-topology triangular surfaces when this option is active.

## Regression tests
`python -m pytest -q tests/test_paper_split_reflatten.py`

The tests check dispatch from both Paper ON and Paper OFF, UI selection/default, and collision-driven seam preservation with native BFF mocked. Full end-to-end native-BFF validation on a representative mesh remains necessary.
