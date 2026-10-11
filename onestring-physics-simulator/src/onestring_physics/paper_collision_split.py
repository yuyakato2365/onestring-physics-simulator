"""Collision-driven, boundary-aware complete grid-cut proposal ranking.

Candidates are ONLY proposals. Committing one requires a fresh M2D->T2D
pipeline build and an explicit collision/hinge improvement audit. We never
fake a Split by removing an Eq.6 penalty without changing source topology.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from .final_split_panel_pass import (
    _complete_cut_candidate, _component_edge_data, _edge_components,
    _internal_grid_values,
)
from .hard_collision_solver import footprint_contacts


def _boundary_distance(centers: np.ndarray, vertices: np.ndarray,
                       faces: np.ndarray, component: np.ndarray) -> np.ndarray:
    """Exact distances from face centroids to their OWN component boundary."""
    _, boundary_edges, _ = _component_edge_data(faces, component)
    if not boundary_edges:
        return np.zeros(len(component), float)
    edges = np.asarray(sorted(boundary_edges), dtype=int)
    a = vertices[edges[:, 0], :2]
    v = vertices[edges[:, 1], :2]-a
    l2 = np.maximum(np.einsum("ij,ij->i", v, v), 1e-30)
    answer = np.zeros(len(component), float)
    for offset in range(0, len(component), 48):
        x = centers[component[offset:offset+48]]
        w = x[:, None, :]-a[None, :, :]
        t = np.clip(np.einsum("nmd,md->nm", w, v)/l2[None, :], 0., 1.)
        delta = w-t[:, :, None]*v[None, :, :]
        answer[offset:offset+48] = np.sqrt(
            np.min(np.einsum("nmd,nmd->nm", delta, delta), axis=1)
        )
    return answer


def _collision_depths(tiles: np.ndarray):
    if len(tiles) < 2:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, float)
    hulls = [ConvexHull(tile[:, :2]).vertices.tolist() for tile in tiles]
    width = max(map(len, hulls))
    faces = np.asarray(
        [[8*i+j for j in hull+[hull[-1]]*(width-len(hull))]
         for i, hull in enumerate(hulls)], dtype=int
    )
    i,j,_,_,depth = footprint_contacts(tiles[:, :, :2], faces)
    return i,j,depth


def rank_collision_split_candidates(state, *, boundary_weight: float = 1.,
                                    max_candidates: int = 12,
                                    maximum_hotspots: int = 16) -> tuple[list[dict], dict]:
    """Rank topology-valid ONE-line cuts using collision severity and depth.

    Boundary distance is measured in the *unsplit canonical M2D chart*, not
    the post-layout XY coordinates, so a crowded layout cannot falsify it.
    Larger boundary_weight favors severe collisions deeper in a component.
    Candidate lines must be COMPLETE boundary-to-boundary connected seams
    splitting exactly one existing face component in two.
    """
    if not np.isfinite(boundary_weight) or boundary_weight < 0:
        raise ValueError("collision split boundary weight must be finite and >= 0")
    mesh = state.mesh_2d_initial
    dual = state.tiles_2d_dual_hinge
    if not dual.metrics.get("section_4_3_linkage", False):
        return [], {"status": "unsupported_T2D_topology", "collision_pairs": None}
    source_faces = np.asarray(mesh.faces, int)
    if source_faces.ndim != 2 or source_faces.shape[1] != 4:
        return [], {"status": "unsupported_source_quads"}
    panels = np.asarray(dual.vertices, float)
    if len(source_faces) != len(panels):
        return [], {"status": "tile_identity_mismatch"}
    raw = getattr(mesh, "_split_panel_source_vertices", None)
    vertices = np.asarray(raw if raw is not None else mesh.vertices, float).copy()
    if not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(panels)):
        return [], {"status": "nonfinite_input"}
    # _split_mesh_single internally rotates by this same matrix before cutting.
    domain = state.conformal_domain
    angle = np.deg2rad(float(getattr(domain, "reference_grid_rotation_degrees", 0.)))
    c,s = float(np.cos(angle)), float(np.sin(angle))
    rot = np.array([[c,-s],[s,c]], float)
    vertices[:, :2] = vertices[:, :2]@rot
    centers = vertices[source_faces, :2].mean(axis=1)
    tile_scale = float(np.median(np.linalg.norm(
        np.roll(panels[:, :4, :2], -1, axis=1)-panels[:, :4, :2],
        axis=2,
    )))
    chart_scale = float(np.median(np.linalg.norm(
        np.roll(vertices[source_faces, :2], -1, axis=1)-
        vertices[source_faces, :2], axis=2,
    )))
    chart_scale = max(chart_scale, 1e-10)
    tile_scale = max(tile_scale, 1e-10)
    ci,cj,depths = _collision_depths(panels)
    # Match Eq.5's SAT cutoff: zero-depth numerical noise is not a structural
    # conflict. Per-panel penalty is the summed normalized squared depth.
    active = depths > tile_scale*1e-8
    ci,cj,depths = ci[active],cj[active],depths[active]
    severity = np.zeros(len(source_faces), float)
    degree = np.zeros(len(source_faces), int)
    np.add.at(severity,ci,(depths/tile_scale)**2)
    np.add.at(severity,cj,(depths/tile_scale)**2)
    np.add.at(degree,ci,1)
    np.add.at(degree,cj,1)
    metrics = dict(
        status="no_collisions" if not len(depths) else "candidates_pending",
        collision_pairs=int(len(depths)),
        sum_normalized_penetration_squared=float(np.sum((depths/tile_scale)**2)),
        maximal_normalized_penetration=float(np.max(depths/tile_scale, initial=0.)),
        affected_tiles=int(np.count_nonzero(degree)),
        boundary_weight=float(boundary_weight),
        tile_scale=float(tile_scale),
        chart_scale=float(chart_scale),
    )
    if not np.any(severity):
        return [], metrics
    ranked = []
    components = _edge_components(source_faces)
    for component in components:
        if len(component) < 2:
            continue
        distances = _boundary_distance(centers, vertices, source_faces, component)
        # Interior conflicts receive a bonus; the bonus is bounded to avoid
        # drowning out a much more severe boundary collision.
        interior = 1.+boundary_weight*np.minimum(distances/(2.*chart_scale),2.)
        priority = severity[component]*interior
        if not np.any(priority > 0):
            continue
        face_distance = dict(zip(component.tolist(), distances.tolist()))
        top = component[np.argsort(-priority,kind="stable")[:maximum_hotspots]]
        top = [int(i) for i in top if severity[i] > 0.]
        internal = {axis: _internal_grid_values(vertices,source_faces,component,coord)
                    for axis,coord in (("col",0),("row",1))}
        attempted = set()
        for hotspot in top:
            for axis,coord in (("col",0),("row",1)):
                values = internal[axis]
                if len(values) == 0:
                    continue
                # Try the two seams nearest the hotspot, rather than trying
                # all O(grid_size) grid lines on every iteration.
                nearby = np.argsort(np.abs(values-centers[hotspot,coord]),kind="stable")[:2]
                for value_idx in nearby:
                    requested=float(values[value_idx])
                    key=(axis,round(requested,11),int(component.min()))
                    if key in attempted:
                        continue
                    attempted.add(key)
                    cand = _complete_cut_candidate(vertices,source_faces,component,axis,requested)
                    if cand is None:
                        continue
                    # Score severity around the actual seam, rather than
                    # treating a remote conflict as evidence for every cut.
                    width=max(chart_scale*1.5,1e-10)
                    distance=np.abs(centers[component,coord]-cand["value"])
                    local_weight=np.exp(-0.5*(distance/width)**2)
                    weighted_severity=float(np.dot(priority,local_weight))
                    seam_length=float(sum(np.linalg.norm(vertices[a,:2]-vertices[b,:2])
                                          for a,b in cand["seam_edges"]))
                    # Prefer fewer/shorter cuts when severity is comparable.
                    cut_score=weighted_severity/(1.+0.12*seam_length/chart_scale)
                    if cut_score <= 0:
                        continue
                    ranked.append(dict(
                        axis=str(axis),value=float(cand["value"]),
                        source_face_id=int(hotspot),
                        component_seed_face=int(component.min()),
                        component_face_count=int(len(component)),
                        hotspot_severity=float(severity[hotspot]),
                        hotspot_collision_pairs=int(degree[hotspot]),
                        boundary_distance=float(face_distance[hotspot]),
                        boundary_distance_tiles=float(face_distance[hotspot]/chart_scale),
                        seam_edge_count=int(len(cand["seam_edges"])),
                        seam_length=float(seam_length),
                        collision_score=float(weighted_severity),
                        score=float(cut_score),
                    ))
    ranked.sort(key=lambda d:(-d["score"],-d["hotspot_severity"],
                              d["seam_length"],d["axis"],d["value"]))
    metrics["valid_candidate_count"]=len(ranked)
    metrics["status"]="candidates_ranked" if ranked else "no_valid_complete_grid_cut"
    return ranked[:max(0,int(max_candidates))], metrics


def accept_if_improved(before: dict, after: dict, *,
                       minimum_relative_improvement: float = 0.02,
                       max_hinge_rms_multiplier: float = 3.0) -> tuple[bool,str]:
    """Conservative accept/reject after a full downstream rebuild.

    The split must meaningfully reduce collision energy or pair count without
    producing unacceptably larger hinge errors. Zero overlap is preferred.
    Scores are computed on the *same final T2D metric definitions*.
    """
    if after.get("status") != "ok":
        return False,"rebuild_missing_valid_T2D"
    before_pairs=int(before["collision_pairs"])
    after_pairs=int(after["collision_pairs"])
    e0=float(before["collision_energy"])
    e1=float(after["collision_energy"])
    h0=float(before["hinge_rms"])
    h1=float(after["hinge_rms"])
    if not all(np.isfinite(x) for x in (e0,e1,h0,h1)):
        return False,"nonfinite_collision_or_hinge_metrics"
    if h1 > max(h0*max_hinge_rms_multiplier,
                float(before.get("tile_scale",1.))*0.1):
        return False,"hinge_connection_regression"
    if after_pairs == 0 and before_pairs > 0:
        return True,"no_remaining_collision_pairs"
    # Lexicographic pair-count priority is restricted to a meaningful
    # energy improvement, to avoid accepting more severe overlaps.
    if before_pairs > 0 and after_pairs < before_pairs and (
        e1 <= e0*(1.-min(.5,minimum_relative_improvement/2.))
    ):
        return True,"collision_pair_count_and_energy_reduced"
    if e0 > 0 and e1 < e0*(1.-minimum_relative_improvement) and after_pairs <= before_pairs:
        return True,"penetration_energy_reduced"
    return False,"insufficient_verified_improvement"


def collision_quality(state) -> dict:
    dual = getattr(state,"tiles_2d_dual_hinge",None)
    if dual is None:
        return {"status":"missing_T2D"}
    metrics=dual.metrics
    tiles=np.asarray(dual.vertices,float)
    if len(tiles)==0:
        return {"status":"missing_T2D"}
    scale=float(np.median(np.linalg.norm(
        np.roll(tiles[:,:4,:2],-1,axis=1)-tiles[:,:4,:2],axis=2,
    )))
    i,j,depth=_collision_depths(tiles)
    depth=depth[depth>max(scale,1e-10)*1e-8]
    return dict(status="ok",collision_pairs=int(len(depth)),
                collision_energy=float(np.sum(depth**2)),
                penetration_max=float(np.max(depth,initial=0.)),
                hinge_rms=float(metrics.get("hinge_rms",float("inf"))),
                tile_scale=float(scale),
                fabrication_feasible=bool(metrics.get("fabrication_feasible",False)))
