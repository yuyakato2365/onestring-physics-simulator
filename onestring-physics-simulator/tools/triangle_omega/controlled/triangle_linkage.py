"""Fabricate a triangular dual-hinge linkage inside its assembled tile envelope."""

import json
import numpy as np
import manifold3d as md
import trimesh
import shapely
from shapely.geometry import Polygon, MultiPoint
from shapely.strtree import STRtree
from scipy.spatial.distance import pdist
from tempfile import TemporaryDirectory
from pathlib import Path
from dual_triangle import choose, place_shell
from mac_hinges import _hinge_fill
from manufacturing import union
from face_membrane import mesh_of


import numpy as np, trimesh
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def weld(mesh, tolerance):
    """Collapse sub-tolerance coincident STL vertices before triangle cleanup."""
    pts = np.asarray(mesh.vertices)
    pairs = cKDTree(pts).query_pairs(tolerance, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), pairs.T), shape=(len(pts), len(pts)))
    _, label = connected_components(graph, directed=False)
    _, first = np.unique(label, return_index=True)
    tri = label[mesh.faces]
    valid = (
        (tri[:, 0] != tri[:, 1]) & (tri[:, 0] != tri[:, 2]) & (tri[:, 1] != tri[:, 2])
    )
    out = trimesh.Trimesh(pts[first], tri[valid], process=False)
    out.update_faces(out.unique_faces())
    out.remove_unreferenced_vertices()
    return out


def manifold(points, triangles):
    mesh = md.Mesh64(
        np.ascontiguousarray(points), np.ascontiguousarray(triangles, dtype=np.uint64)
    )
    mesh.merge()
    g = md.Manifold(mesh)
    if g.status() != md.Error.NoError or g.volume() <= 0:
        raise ValueError("パネルまたは膜の立体が反転しています。")
    return g


def bore(start, end, r):
    axis = end - start
    length = np.linalg.norm(axis)
    z = axis / length
    u = np.cross(z, np.eye(3)[np.argmin(abs(z))])
    u /= np.linalg.norm(u)
    # Circumscribe the required open circle.
    tube = md.Manifold.cylinder(length, r / np.cos(np.pi / 32), circular_segments=32)
    return tube.transform(np.column_stack([u, np.cross(z, u), z, start]))


def audited_layout(out, cfg):
    mm = cfg["mm_per_unit"]
    with np.load(out / "tiles.npz") as t:
        folded = t["T3D"] * mm
        flat = t["flat_solids"] * mm
        faces = t["surface_tris"]
    with np.load(out / "flat.npz") as f:
        hinges = f["hinges"]
        cells = f["regular_cells"]
        up = f["lattice_up"]
    side = choose(folded, hinges)
    a, k, b, l = hinges.T
    initial = np.linalg.norm(flat[a, k + side] - flat[b, l + side], axis=1)
    seed_angle = None
    if len(hinges):
        flat, seed_angle = place_shell(flat, hinges, side, cells, up, (None, 30, -30, 15, -15, 0),
                                       refine=cfg.get("extrusion_mode") == "outer_layer")
    spans = np.linalg.norm(flat[a, k + side] - flat[b, l + side], axis=1)
    error = max(abs(pdist(x) - pdist(y)).max() for x, y in zip(flat, folded))
    # The four corners and the triangulation of the physical mating wall must
    # agree under a single rigid transform per tile.
    gap = 0.0
    common_area = []
    channel_error = 0.0
    import pickle

    with (out / "remesh3d.pkl").open("rb") as stream:
        ids = pickle.load(stream)["T"]
    for i, ki, j, kj in hinges:
        match = ids[i, :, None] == ids[j, None, :]
        if match.sum() != 2 or ids[i, ki] != ids[j, kj]:
            raise ValueError(f"パネル {i}–{j} の側面対応が不正です。")
        li, lj = np.where(match)
        if li[0] == ki:
            li = li[::-1]
            lj = lj[::-1]
        # Also test the shared edge midpoint at both levels: the fixed-skin
        # variant has a separately adjusted midpoint on each side wall.
        mid = lambda pair: 3 + next(
            s for s in range(3) if set([s, (s + 1) % 3]) == set(pair)
        )
        ca = [*li, mid(li)]
        cb = [*lj, mid(lj)]
        ca = np.r_[ca, np.array(ca) + 7]
        cb = np.r_[cb, np.array(cb) + 7]
        A = folded[i, ca]
        B = folded[j, cb]
        gap = max(gap, np.linalg.norm(A - B, axis=1).max())
        channel_error = max(
            channel_error,
            np.linalg.norm(A[[0, 1, 3, 4]].mean(0) - B[[0, 1, 3, 4]].mean(0)),
        )
        walls = faces[i][np.isin(faces[i], ca).all(1)]
        walls_b = faces[j][np.isin(faces[j], cb).all(1)]

        def connectivity(tris, indices):
            return {
                tuple(sorted(indices.tolist().index(int(v)) for v in tri))
                for tri in tris
            }

        if connectivity(walls, ca) != connectivity(walls_b, cb):
            raise ValueError("組立姿勢で側面の対角線が一致しません。")
        pts = folded[i, walls]
        common_area.append(
            np.linalg.norm(
                np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0]), axis=1
            ).sum()
            / 2
        )
    report = dict(
        extrusion_mode=cfg["extrusion_mode"],
        hinge_design="dual_triangle",
        hinges=len(hinges),
        kagome_seed_angle_deg=seed_angle,
        max_gap_before_mm=float(initial.max(initial=0)),
        max_gap_mm=float(spans.max(initial=0)),
        max_rigidity_error_mm=float(error),
        assembled_side_gap_mm=float(gap),
        assembled_channel_center_gap_mm=float(channel_error),
        minimum_mating_area_mm2=min(common_area, default=0.0),
        inner_hinges=int(sum(side == 0)),
        offset_hinges=int(sum(side == 7)),
        converged=False,
    )
    (out / "linkage_check.json").write_text(json.dumps(report, indent=2))
    if (
        spans.max(initial=0) > 2.000001
        or gap > 1e-6
        or error > 1e-6
        or (common_area and min(common_area) <= 1e-6)
    ):
        raise ValueError(
            f"剛体・側面・角ヒンジ条件を満たしません。linkage_check.json 参照。最大角間隔 {spans.max():.3f} mm"
        )
    return flat, folded, faces, hinges, side, report


def make_linkage(out, cfg):
    flat, folded, faces, h, side, report = audited_layout(out, cfg)
    mm = cfg["mm_per_unit"]
    t = cfg["hinge_mm"]
    r = cfg["hole_mm"] / 2
    p = flat.copy()
    reflected = cfg["extrusion_mode"] != "outer_layer"
    if reflected:
        p[:, :, 2] *= -1
        faces = faces[:, :, ::-1]
    p[:, :7, 2] = 0.0
    n = len(p)
    has = np.zeros((n, 2), bool)
    for (a, _, b, _), s in zip(h, side):
        has[[a, b], s // 7] = True
    skin_depths = []
    bodies = []
    pads = []
    skins = {}
    holes = []
    tiles = []
    rim_channels = None
    if cfg.get("string_route") == "perimeter":
        from perimeter_string import perimeter_channels
        rim_channels = perimeter_channels(out, folded, p, r)
    for i, q in enumerate(p):
        low = q[:7]
        high = q[7:]
        height = high[:, 2] - low[:, 2]
        if height.min() < -1e-7:
            raise ValueError(f"パネル {i} の押し出しが反転しています。")
        budget = height * 0.5 - r / np.cos(np.pi / 32) - 0.30
        if np.any(budget <= 0.05):
            raise ValueError(
                f"パネル {i} は薄すぎて指定径の紐穴と外壁を収められません。"
            )
        skin_depths.append(np.minimum(t, budget))
        alpha = (skin_depths[-1] / height)[:, None]
        lower = low + alpha * (high - low) if has[i, 0] else low
        upper = high - alpha * (high - low) if has[i, 1] else high
        tile = manifold(q, faces[i])
        tiles.append(tile)
        rigid = manifold(np.vstack([lower, upper]), faces[i])
        if rim_channels is not None:
            channel = rim_channels[i]
        else:
            center = q[[0, 1, 2, 7, 8, 9]].mean(0)
            arms = []
            for k in range(3):
                j = (k + 1) % 3
                mouth = q[[k, j, k + 7, j + 7]].mean(0)
                d = mouth - center
                d /= np.linalg.norm(d)
                arms.append(bore(center - r * d, mouth + 2 * r * d, r))
            channel = union(arms)
        holes.append(channel)
        bodies.append(rigid - channel)
        for layer, slab in [
            (0, np.vstack([low, lower])),
            (1, np.vstack([upper, high])),
        ]:
            if has[i, layer]:
                skin = manifold(slab, faces[i]) - channel
                skins[i, layer] = skin
                pads.append(skin)
    polygons = [MultiPoint(v[:, :2]).convex_hull for v in p]
    tree = STRtree(polygons)
    bridges = []
    preview_bridges = []
    owners = []
    minimum = np.inf
    for (a, k, b, l), s in zip(h, side):
        qa = p[a, s : s + 3]
        qb = p[b, s : s + 3]
        outline, dist = _hinge_fill(qa[:, :2], k, qb[:, :2], l, 1.04, 2.0)
        if outline is None:
            raise ValueError(f"角間隔 {dist:.2f} mm が四角形版の上限を超えます。")
        poly = shapely.make_valid(Polygon(outline)).buffer(0.10, quad_segs=6)
        za, zb = qa[k, 2], qb[l, 2]
        # The flexible neck spans the two selected top or bottom skins.
        ha = skin_depths[a][k]
        hb = skin_depths[b][l]
        bot, top = (min(za - ha, zb - hb), max(za, zb)) if s else (0.0, min(ha, hb))
        ring = np.array(poly.exterior.coords)[:-1]
        bridge = (
            md.CrossSection([ring], md.FillRule.EvenOdd)
            .extrude(top - bot)
            .translate([0, 0, bot])
        )
        # Remove the PLA and bore volumes, keeping both intended pad attachments.
        full_bridge = bridge
        for j in tree.query(poly):
            if j in (a, b):
                bridge = bridge - bodies[j] - holes[j]
                full_bridge = full_bridge - holes[j]
            elif (bridge ^ tiles[j]).volume() > 1e-6:
                clearance = tiles[j].minkowski_sum(md.Manifold.sphere(0.03, 8))
                bridge = bridge - clearance
                full_bridge = full_bridge - clearance
        preview_bridges.append(full_bridge)
        attach = min((bridge ^ skins[j, s // 7]).volume() for j in (a, b))
        if attach < 1e-7:
            raise ValueError(f"ヒンジ {a}–{b} が指定面の膜に届きません。")
        minimum = min(minimum, attach)
        bridges.append(bridge)
        owners.append(poly)
    bt = STRtree(owners)
    for i, poly in enumerate(owners):
        for j in bt.query(poly):
            if j > i and (bridges[i] ^ bridges[j]).volume() > 1e-6:
                raise ValueError(f"異なるヒンジ {i},{j} が干渉します。")
    pla = union(bodies)
    film = union(pads + bridges)
    if (pla ^ film).volume() > 1e-5:
        raise ValueError("材料領域が重なっています。")
    # Combined exterior uses undivided tile envelopes (same physical boundary)
    # to avoid float32 cracks along internal material interfaces.
    whole = union([tile - hole for tile, hole in zip(tiles, holes)] + preview_bridges)
    if len([m for m in whole.decompose() if m.volume() > 1e-5]) != 1:
        raise ValueError("全体が一体につながっていません。")
    from fixed_skin import basis

    R = basis(folded[:, :3]).transpose(0, 2, 1) @ basis(flat[:, :3])
    translation = folded[:, :3].mean(1) - np.einsum(
        "nij,nj->ni", R, flat[:, :3].mean(1)
    )
    assembled = []
    for i, (tile, hole) in enumerate(zip(tiles, holes)):
        # Move the same drilled tile; a flexible neck has no prescribed rigid pose.
        rot = R[i].copy()
        if reflected:
            rot[:, 2] *= -1
        assembled.append(
            mesh_of((tile - hole).transform(np.column_stack([rot, translation[i]])))
        )
    assembly = trimesh.util.concatenate(assembled)
    dest = out / "prototype"
    dest.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=dest) as tmp:
        paths = []
        for name, g in [("PLA_tiles", pla), ("TPU_membrane", film), ("T2D", whole)]:
            path = Path(tmp) / (name + ".stl")
            for simplification in [0.00002, 0.0005, 0.003]:
                mesh_of(
                    g.set_tolerance(simplification).simplify(simplification)
                ).export(path)
                base = trimesh.load(path, force="mesh")
                for tolerance in [0.0001, 0.0003, 0.001]:
                    clean = weld(base, tolerance)
                    if clean.is_volume:
                        clean.export(path)
                        break
                m = trimesh.load(path, force="mesh")
                if m.is_volume:
                    break
            if not m.is_volume:
                raise ValueError(f"{name} のSTLが閉じていません。")
            if name == "PLA_tiles" and len(m.split()) != n:
                raise ValueError("PLAタイルが不正に連結されています。")
            if name == "T2D" and len(m.split()) != 1:
                raise ValueError("書き出した全体が一体化していません。")
            paths.append(path)
        for path in paths:
            path.replace(dest / path.name)

    def display(solid):
        m = mesh_of(solid)
        if reflected:
            m.apply_transform(np.diag([1.0, 1.0, -1.0, 1.0]))
        return m.vertices / mm, m.faces

    v, f = display(film)
    pv, pf = display(pla)
    np.savez_compressed(
        out / "assembled_drilled.npz",
        vertices=assembly.vertices / mm,
        faces=assembly.faces,
        flat_to_target_rotation=R,
        translation_mm=translation,
    )
    np.savez_compressed(
        out / "closure.npz",
        T2D=flat / mm,
        hinge_vertices=v,
        hinge_faces=f,
        pla_vertices=pv,
        pla_faces=pf,
        hinge_layer=side,
    )
    report.update(
        converged=True,
        components=1,
        unattached_hinge_ends=0,
        unintended_contacts=0,
        joint_intersections=0,
        validation_scope="flat_and_assembled_endpoints",
        string_route=cfg.get("string_route", "radial"),
        bore_diameter_mm=cfg["hole_mm"],
        minimum_skin_mm=float(np.min(skin_depths)),
        maximum_skin_mm=float(np.max(skin_depths)),
        minimum_attachment_mm3=float(minimum) if h.size else None,
    )
    (out / "closure.json").write_text(json.dumps(report, indent=2))
    (out / "linkage_check.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)
