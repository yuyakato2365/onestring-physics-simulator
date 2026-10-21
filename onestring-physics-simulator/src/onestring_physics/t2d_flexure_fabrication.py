"""Experimental one-material thin-lamella hinges for FINAL OneString T2D STL.

Inspired by the PP lamella connections in IsoGami (Han et al., SIGGRAPH 2026).
OneString is a *3D folding* linkage, not IsoGami's planar linkage: this is an
engineering adaptation, NOT a reproduction of IsoGami's undisclosed CAD.

The ONLY authority for a hinge is the supplied final HingeGraph. In particular,
surface='top'/'bottom' and local_vertex indices are never re-inferred from XY
proximity: split boundaries must remain disconnected.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FlexureParameters:
    width_mm: float = 1.2
    skin_mm: float = 0.5
    anchor_inset_mm: float = 2.5
    embed_mm: float = 0.18  # additional in-plane insertion into the rigid tile
    # An optimized rigid-hinge point can retain a small residual gap. The
    # printable strip BRIDGES that gap without modifying T2D or welding seams.
    # Limit is deliberately finite; physically large gaps remain rejected.
    max_joint_gap_mm: float = 1.5
    max_face_z_spread_mm: float = 0.25
    max_miter_ratio: float = 3.0


def _unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < 1e-8:
        raise ValueError("degenerate flexure path: anchors or joint coincide")
    return vector / length


def _strip_mesh(path: np.ndarray, width: float, zlo: float, zhi: float,
                max_miter_ratio: float) -> np.ndarray:
    """Single watertight bent ribbon (3 cross-sections, 2 surfaces)."""
    d0 = _unit(path[1] - path[0])
    d1 = _unit(path[2] - path[1])
    n0 = np.array([-d0[1], d0[0]])
    n1 = np.array([-d1[1], d1[0]])
    middle_normal = n0 + n1
    if np.linalg.norm(middle_normal) <= 1e-7:
        raise ValueError("180-degree reversal at flexure hinge")
    middle_normal = _unit(middle_normal)
    denominator = float(np.dot(middle_normal, n1))
    if denominator <= 1e-6 or 1.0 / denominator > max_miter_ratio:
        raise ValueError("flexure miter too sharp; reduce width or adjust T2D")
    half = 0.5 * width
    offsets = [n0 * half, middle_normal * (half / denominator), n1 * half]
    # Vertex indices: section 0..2, side 0=left/1=right, level 0=low/1=high
    verts = np.empty((3, 2, 2, 3), dtype=float)
    for i in range(3):
        for side, sign in ((0, 1.), (1, -1.)):
            xy = path[i] + sign * offsets[i]
            verts[i, side, 0] = (xy[0], xy[1], zlo)
            verts[i, side, 1] = (xy[0], xy[1], zhi)
    # Fail instead of writing a folded-over ribbon.
    for i in (0, 1):
        quad = np.array([
            verts[i, 0, 1, :2], verts[i, 1, 1, :2],
            verts[i + 1, 1, 1, :2], verts[i + 1, 0, 1, :2],
        ])
        area = 0.5 * float(np.sum(quad[:, 0] * np.roll(quad[:, 1], -1)
                                  - quad[:, 1] * np.roll(quad[:, 0], -1)))
        if area <= 1e-8:
            raise ValueError("self-folded/degenerate flexure ribbon")
    faces: list[np.ndarray] = []

    def add_quad(a, b, c, d):
        faces.extend([np.stack((a, b, c)), np.stack((a, c, d))])

    for i in (0, 1):
        # Upper/lower face: shared middle cross-section, no internal caps.
        add_quad(verts[i, 0, 1], verts[i, 1, 1], verts[i+1, 1, 1], verts[i+1, 0, 1])
        add_quad(verts[i, 1, 0], verts[i, 0, 0], verts[i+1, 0, 0], verts[i+1, 1, 0])
        add_quad(verts[i, 0, 0], verts[i, 0, 1], verts[i+1, 0, 1], verts[i+1, 0, 0])
        add_quad(verts[i, 1, 1], verts[i, 1, 0], verts[i+1, 1, 0], verts[i+1, 1, 1])
    add_quad(verts[0, 1, 1], verts[0, 0, 1], verts[0, 0, 0], verts[0, 1, 0])
    add_quad(verts[2, 0, 1], verts[2, 1, 1], verts[2, 1, 0], verts[2, 0, 0])
    return np.asarray(faces)


def make_flexure_triangles(
    scaled_tiles: np.ndarray,
    hinge_graph,
    parameters: FlexureParameters = FlexureParameters(),
) -> tuple[list[np.ndarray], dict[str, object]]:
    """Make top/bottom face-specific closed lamella solids in millimetres.

    The resulting STL is an assembly of individually closed, *overlapping*
    shells (rigid tile prisms + lamellae). The lamella end pads intersect the
    designated tiles. A slicer Boolean-union/merge and physical testing are
    still required; no joint durability or collision-free folding is asserted.
    """
    v = np.asarray(scaled_tiles, dtype=float)
    if v.ndim != 3 or v.shape[1:] != (8, 3) or not np.isfinite(v).all():
        raise ValueError("T2D flexures require finite (N,8,3) scaled tile prisms")
    if hinge_graph is None or not hasattr(hinge_graph, "hinges"):
        raise ValueError("final authoritative HingeGraph is required")
    p = parameters
    values = (p.width_mm, p.skin_mm, p.anchor_inset_mm, p.embed_mm,
              p.max_joint_gap_mm, p.max_face_z_spread_mm, p.max_miter_ratio)
    if not all(np.isfinite(val) and val > 0 for val in values):
        raise ValueError("all flexure settings must be finite positive values")
    if p.anchor_inset_mm <= p.width_mm / 2:
        raise ValueError("anchor inset must be larger than half the ribbon width")
    if p.max_miter_ratio < 1:
        raise ValueError("max_miter_ratio must be >= 1")

    # Top faces must be physically above the bottom faces for 'top'/'bottom'
    # labels to specify a reliable printing side. Never silently swap them.
    if len(v) and np.any(v[:, :4, 2].mean(axis=1) <= v[:, 4:, 2].mean(axis=1)):
        raise ValueError("T2D face orientation not printable: top is not above bottom")
    triangles: list[np.ndarray] = []
    surfaces = {"top": 0, "bottom": 0}
    gaps: list[float] = []
    seen: set[tuple[int, int, int, int]] = set()
    for index, h in enumerate(hinge_graph.hinges):
        a, b = int(h.tile_a), int(h.tile_b)
        ca, cb = int(h.local_vertex_a), int(h.local_vertex_b)
        surface = str(h.surface).lower()
        if surface not in surfaces:
            raise ValueError(f"hinge {index}: unknown side {surface!r}")
        offset = 0 if surface == "top" else 4
        if not (0 <= a < len(v) and 0 <= b < len(v) and a != b):
            raise ValueError(f"hinge {index}: invalid tile IDs")
        if not (offset <= ca < offset + 4 and offset <= cb < offset + 4):
            raise ValueError(f"hinge {index}: {surface} disagrees with stored local vertices")
        key = (min(a, b), max(a, b), ca if a < b else cb, cb if a < b else ca)
        if key in seen:
            raise ValueError(f"duplicate hinge {index}")
        seen.add(key)
        face_a, face_b = v[a, offset:offset+4], v[b, offset:offset+4]
        spread = max(float(np.ptp(face_a[:, 2])), float(np.ptp(face_b[:, 2])))
        if spread > p.max_face_z_spread_mm:
            raise ValueError(f"hinge {index}: tilted {surface} face (z spread {spread:.3g} mm)")
        joint_a, joint_b = v[a, ca], v[b, cb]
        gap = float(np.linalg.norm(joint_a - joint_b))
        if gap > p.max_joint_gap_mm:
            raise ValueError(
                f"hinge {index}: attachment gap {gap:.3g} mm exceeds the printable "
                f"bridge limit {p.max_joint_gap_mm:.3g} mm. "
                "This is residual T2D hinge misalignment: improve the Dual Hinge "
                "optimization or explicitly increase the fabrication bridge limit "
                "after inspecting the physical geometry."
            )
        if abs(float(joint_a[2]-joint_b[2])) > p.max_face_z_spread_mm:
            raise ValueError(f"hinge {index}: mismatched {surface} heights")
        gap_xy = 0.5 * (joint_a[:2] + joint_b[:2])
        # Attach each ribbon end INSIDE the designated tile face. The central
        # section follows the stored K2D joint (no nearest-neighbour guessing).
        center_a, center_b = face_a[:, :2].mean(axis=0), face_b[:, :2].mean(axis=0)
        interior_a = _unit(center_a-joint_a[:2])
        interior_b = _unit(center_b-joint_b[:2])
        # End overlap is IN PLANE. This avoids putting bottom-side lamellae
        # below the build plate (the former implementation did precisely that).
        step_a = min(p.anchor_inset_mm + p.embed_mm, 0.45 * np.linalg.norm(center_a-joint_a[:2]))
        step_b = min(p.anchor_inset_mm + p.embed_mm, 0.45 * np.linalg.norm(center_b-joint_b[:2]))
        if min(step_a, step_b) < p.width_mm / 2:
            raise ValueError(f"hinge {index}: tile corner is too small for requested ribbon width")
        a_anchor = joint_a[:2] + step_a * interior_a
        b_anchor = joint_b[:2] + step_b * interior_b
        path = np.stack((a_anchor, gap_xy, b_anchor))
        zface = 0.5 * float(joint_a[2] + joint_b[2])
        # Both strips stay WITHIN the common fabrication slab [0, plate_thickness].
        # The material on the panel face is already present over the anchored
        # end, so independent intersecting shells require a slicer Boolean union.
        if surface == "top":
            zlo, zhi = zface - p.skin_mm, zface
        else:
            zlo, zhi = zface, zface + p.skin_mm
        if zlo < -1e-6 or zhi > float(np.min(v[:, :4, 2])) + 1e-6:
            raise ValueError(f"hinge {index}: skin thickness exceeds printable panel slab")
        try:
            mesh = _strip_mesh(path, p.width_mm, zlo, zhi, p.max_miter_ratio)
        except ValueError as exc:
            raise ValueError(f"hinge {index}: {exc}") from exc
        triangles.extend(mesh)
        gaps.append(gap)
        surfaces[surface] += 1

    metrics = {
        "flexure_export_model": "experimental_IsoGami_inspired_lamella_for_OneString_3D_hinges",
        "flexure_hinge_count": len(gaps),
        "flexure_top_count": surfaces["top"],
        "flexure_bottom_count": surfaces["bottom"],
        "flexure_triangle_count": len(triangles),
        "flexure_max_joint_gap_mm": max(gaps, default=0.),
        "flexure_bridge_limit_mm": p.max_joint_gap_mm,
        "flexure_nonzero_gap_count": sum(gap > 1e-5 for gap in gaps),
        "flexure_gap_semantics": "residual_T2D_joint_misalignment_bridged_by_lamella_not_solver_fixed",
        "flexure_width_mm": p.width_mm,
        "flexure_skin_mm": p.skin_mm,
        "flexure_anchor_inset_mm": p.anchor_inset_mm,
        "flexure_embed_mm": p.embed_mm,
        "flexure_embed_semantics": "additional_XY_anchor_inset_not_vertical_extension",
        "flexure_z_layout": "inside_shared_panel_slab_printable_on_bed",
        "flexure_units": "mm",
        "flexure_source": "authoritative final HingeGraph, not new geometric inference",
        "flexure_boolean_union": "not_performed_overlapping_closed_solids_require_slicer_merge",
        "flexure_physical_validation": "not_performed",
        "flexure_collision_and_folding_validation": "not_performed",
    }
    return triangles, metrics


def export_final_t2d_flexure_stl(
    source,
    *,
    hinge_graph=None,
    panel_size_mm: float = 40.0,
    panel_thickness_mm: float = 2.0,
    parameters: FlexureParameters = FlexureParameters(),
) -> tuple[bytes, dict[str, object]]:
    """Independent OPTIONAL fabrication export; leaves optimization and stock STL untouched.

    STL is unitless: coordinates are in **mm** for direct use in Bambu Studio.
    Importers must merge intersecting shells. No boolean, relief cuts, or
    folding/stress validation is performed.
    """
    from . import onestring_pipeline as pipeline

    if not (np.isfinite(panel_size_mm) and panel_size_mm > 0
            and np.isfinite(panel_thickness_mm) and panel_thickness_mm > 0):
        raise ValueError("panel size and thickness must be positive millimetres")
    assembly = pipeline._resolve_t2d_assembly(source, "dual_hinge")
    graph = hinge_graph if hinge_graph is not None else getattr(source, "hinge_graph", None)
    if graph is None:
        raise ValueError("the final T2D Dual Hinge graph is required")
    tiles, _original_triangles, metrics = pipeline._t2d_stl_mesh_and_metrics(
        assembly, panel_size=float(panel_size_mm), thickness=float(panel_thickness_mm)
    )
    # The old STL helper scales each tile's Z extent about its OWN Z centre.
    # Even when the source is conceptually T2D, this leaves different top/bottom
    # face levels and makes top/bottom fabrication hinges impossible to share
    # a common plate. For this opt-in export ONLY, preserve every source XY
    # footprint and the explicit top/bottom vertex IDs, but level all print
    # panels to a common bed and uniform user-selected thickness.
    tiles = tiles.copy()
    original_top_z = tiles[:, :4, 2].copy()
    original_bottom_z = tiles[:, 4:, 2].copy()
    pre_top_spread = float(np.ptp(original_top_z.mean(axis=1))) if len(tiles) else 0.
    pre_bottom_spread = float(np.ptp(original_bottom_z.mean(axis=1))) if len(tiles) else 0.
    pre_tile_face_warp = max(
        float(np.max(np.ptp(original_top_z, axis=1))) if len(tiles) else 0.,
        float(np.max(np.ptp(original_bottom_z, axis=1))) if len(tiles) else 0.,
    )
    if parameters.skin_mm > panel_thickness_mm:
        raise ValueError(
            f"flexure skin {parameters.skin_mm:g} mm exceeds print panel "
            f"thickness {panel_thickness_mm:g} mm"
        )
    tiles[:, :4, 2] = float(panel_thickness_mm)
    tiles[:, 4:, 2] = 0.
    local_tris = pipeline._prism_triangle_indices()
    tile_triangles = [tile[list(tri)] for tile in tiles for tri in local_tris]
    ribbons, flexure_metrics = make_flexure_triangles(tiles, graph, parameters)
    all_triangles = tile_triangles + ribbons
    result_metrics = dict(metrics)
    result_metrics.update(flexure_metrics)
    result_metrics["t2d_face_count"] = len(all_triangles)
    result_metrics["t2d_connected_component_count"] = len(tiles) + len(graph.hinges)
    result_metrics["t2d_fabrication_unit"] = "mm (STL coordinates, not metadata)"
    result_metrics["t2d_export_exactness_label"] = "experimental_flexure_geometry_not_validated"
    result_metrics["flexure_print_z_normalized"] = True
    result_metrics["flexure_preprint_top_level_spread_mm"] = pre_top_spread
    result_metrics["flexure_preprint_bottom_level_spread_mm"] = pre_bottom_spread
    result_metrics["flexure_preprint_max_face_warp_mm"] = pre_tile_face_warp
    result_metrics["flexure_print_z_bounds_mm"] = [0., float(panel_thickness_mm)]
    result_metrics["flexure_print_shape_semantics"] = (
        "XY geometry preserved; per-tile Z flattened to common printable slab; "
        "upstream T2D solution unchanged; not proof of rigid congruence"
    )
    data = pipeline._ascii_stl_bytes("onestring_t2d_dual_flexure_mm", all_triangles)
    return data, result_metrics
