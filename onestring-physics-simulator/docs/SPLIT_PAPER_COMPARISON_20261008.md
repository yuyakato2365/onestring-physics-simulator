# Split correction against the published OneString paper

Reference: Zaman et al., *One String to Pull Them All*, ACM TOG 44(6),
Article 234 (2025), supplied `3763357 (1).pdf`, pp. 4–7, especially
§4.1, §4.5 and Figure 6; Supplement Figure 18 for the area bound.
This correction starts from commit `ad7cfb5`.

## Paper requirements and implementation

| Requirement | Implementation after this correction |
|---|---|
| §4.1: crop a regular quad grid, split M2D when required, then lift with the inverse map | Retain the existing M2D cut owner and canonical, pre-display-gap UV coordinates. §4.5 describes corresponding M2D/M3D meshes; this does not require changing the explicit §4.1 ordering. |
| §4.5: hierarchical bisection, complete grid-aligned splits, regular connectivity | A candidate must be one connected seam chain with two boundary endpoints, no intermediate boundary contact, full face partition, and exactly two child components. Quads and stable face IDs are preserved. |
| §4.5: pass through the highest Gaussian curvature point | Rank intrinsic signed source Gaussian curvature by angle defect / dual area. Artificial UV seams no longer suppress valid source interior vertices. Points on existing split boundaries can still seed cuts along the other grid direction. Physical source boundaries have no interior angle-defect estimate and remain excluded. |
| Figure 6: continue until each part satisfies sigma <= 2 | One offending quad no longer blocks its entire component. Recursion accepts only cuts that reduce the worst child CSF. A component at its retained-chart within-quad lower bound stops intact as `grid_resolution_limit`; candidates with no worst-child reduction stop as `no_csf_improvement`. |
| Supplement Figure 18: quad area expansion bound 2 | Keep area-Jacobian evaluation distinct from linear stretch. A collapsed positive-area source triangle's UV image is an invalid inverse map, not zero expansion demand; its Jacobian remains NaN; usable regions continue, but `uncertified_uv_degeneracy` prevents a satisfied status. An entirely unusable field still raises an error. |

## Implementation choices that are not specified by the paper

The paper does not specify a post-cut parameterization solver, boundary
conditions, a numerical CSF normalization procedure, or tie-breaking between
the two grid directions. This implementation retains the chosen piecewise-affine
UV map. It measures source triangle area Jacobian extrema intersecting each
quad, then normalizes each component by its minimum Jacobian. It does not
reparameterize each child and does not claim to reproduce Figure 6's values.
A nonconformal input chart also prevents an area bound alone from establishing
physical realizability; the existing anisotropy diagnostic remains applicable.

Feature/grid coincidence is not generally possible (§4.5 explicitly leaves
feature alignment to the user). The implementation snaps to the nearest
internal grid line and records `grid_snap_distance`; it does not claim an
off-grid peak is crossed exactly. At the highest-curvature sample allowing a
complete cut, directions are ranked by unresolved face count, worst child
sigma, and then face-count balance. Reported CSFs are not rounded; ordering
keys alone use a numerical tolerance so rotating the chart does not change a
tie through roundoff.

The retained-chart within-quad ratio is a limit of this discretization, not a
proof that no parameterization or finer grid could realize the surface. Budget
and resolution stops must not be presented as achieving the paper's bound.

## Limited flow checks

- Actual launcher installation chain to flatten, cropped M2D, Split, and M3D:
  face correspondence, finite lifted coordinates, seam identities and display
  samples remain connected. The UI body was intercepted; this is not an
  interactive UI test.
- Small existing linkage fixture: seam metadata reaches gap labels and a closed
  string route. No new routing algorithm was introduced in this correction.
- Small synthetic split fixture: an offending quad does not block its parent;
  hierarchy partitions every parent, preserves faces, and ends with only
  unresolved singleton quads (9 cuts / 10 components in that fixture).
- The same fixture under chart rotation retains identical seam topology;
  duplicated UV indexing retains intrinsic curvature; a hole-crossing line
  producing multiple slits is refused; a partially collapsed UV field continues without bound certification.

These are control/data-flow checks, not shape quality, physical feasibility,
convergence, or paper-result reproduction experiments. Those remain with the
user as requested.

Runtime correction: the pre-crop all-or-nothing UV rejection was too strict. Degenerate face IDs/counts are retained; Split Map displays a partial-evaluation warning. This changes error handling, not the UV coordinates, and does not repair the parameterization. A synthetic partial/valid-field flow and syntax checks passed; the user's 2964-triangle input has not been rerun.

## Over-splitting correction

The previous singleton-isolation policy could fragment a component even when its worst CSF did not change. Cuts now require relative worst-child improvement greater than 1e-8. Peaks on an existing boundary cannot move that same-axis cut inward to an unrelated grid row. The within-quad lower bound stops a whole component only when its current CSF has already reached that bound. This is a conservative guard for the retained-chart estimator, not an algorithm specified by the paper; it can leave a region unresolved when improvement would require several intermediate cuts. Small control-flow fixtures confirm that floor-limited and non-improving cases retain topology, while an improving bisection still executes. The screenshot input has not been rerun.
