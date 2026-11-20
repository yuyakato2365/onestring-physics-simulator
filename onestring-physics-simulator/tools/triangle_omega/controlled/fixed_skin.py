"""Fit shared inward columns with the assembled outer surface held fixed.
All geometry calculations in this module use millimetres. Stage archives
and the UI use normalized mesh units. STL files use millimetres.
"""

import json
import numpy as np
import manifold3d as md
import trimesh
from shapely.geometry import MultiPoint
from shapely.strtree import STRtree
from separated_layout import separated_layout


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def basis(v):
    u = unit(v[:, 1] - v[:, 0])
    w = unit(np.cross(u, v[:, 2] - v[:, 0]))
    return np.stack([u, np.cross(w, u), w], 1)


def intersections(parts):
    box = np.array([g.bounding_box() for g in parts])
    hit = []
    for i in range(len(parts)):
        for j in (
            np.flatnonzero(
                (box[i, 3:] >= box[i + 1 :, :3]).all(1)
                & (box[i + 1 :, 3:] >= box[i, :3]).all(1)
            )
            + i
            + 1
        ):
            vol = (parts[i] ^ parts[j]).volume()
            if vol > 1e-5:
                hit.append((int(i), int(j), vol))
    return hit


def as_mesh(part):
    mesh = part.to_mesh64()
    return trimesh.Trimesh(mesh.vert_properties[:, :3], mesh.tri_verts, process=False)


def maximum_depth_fraction(room, outflow):
    return np.clip(
        np.divide(
            room,
            np.maximum(outflow, 1e-12),
            out=np.ones_like(room),
            where=outflow > 1e-10,
        ),
        0,
        1,
    )


def build_shell(out, mesh, baseline, layout, cfg, depth):
    mm = cfg["mm_per_unit"]
    X0 = mesh["X"] * mm
    F = mesh["T"]
    raw = baseline * mm
    top = raw[:, :3]
    n = len(top)
    if cfg.get("extrusion_mode") == "outer_layer":
        from outer_layer import check_shared_vertex_clearance
        check_shared_vertex_clearance(F, basis(top)[:, 2], depth * mm,
                                      cfg.get("outer_lateral_mm", 1.), cfg.get("depth_tolerance_mm"))
    thin = layout["rest"].copy()
    thin[:, 3:, :2] = thin[:, :3, :2]
    plane = thin[:, :3] * mm
    independent_shell = (cfg.get("extrusion_mode") == "outer_layer"
                         and cfg.get("depth_tolerance_mm") is not None
                         and cfg.get("outer_spacing_percent", 0.) == 0)
    if len(layout["hinges"]) and not independent_shell:
        pose = separated_layout(
            thin,
            layout["hinges"],
            layout["regular_cells"],
            layout["lattice_up"],
            np.deg2rad(10),
            steps=60,
            progress=lambda i, total: print(
                f"外面固定 · 平面配置の探索 {i}/{total}", flush=True
            ),
        )
        plane = pose[:, :3] * mm
    if cfg.get("extrusion_mode") == "outer_layer":
        centers = plane[:, :, :2].mean(axis=1)
        plane[:, :, :2] += (cfg.get("outer_spacing_percent", 0.0) / 100.0) * (
            centers - centers.mean(0)
        )[:, None, :]
    R = basis(plane).transpose(0, 2, 1) @ basis(top)
    trans = plane.mean(1) - np.einsum("nij,nj->ni", R, top.mean(1))

    def flatten(v):
        return v @ R.transpose(0, 2, 1) + trans[:, None]

    # Midpoints are shared by edge ID, and each face owns its center.
    edges = np.stack([F, np.roll(F, -1, 1)], 2)
    ek, inv = np.unique(np.sort(edges, 2).reshape(-1, 2), axis=0, return_inverse=True)
    id7 = np.c_[F, len(X0) + inv.reshape(-1, 3), len(X0) + len(ek) + np.arange(n)]

    def expanded(t):
        return np.concatenate([t, (t + np.roll(t, -1, 1)) / 2, t.mean(1)[:, None]], 1)

    outer = expanded(top)
    outward = cfg.get("extrusion_mode") == "outer_layer"
    if outward:
        from outer_layer import outward_offsets

        delta = expanded(
            outward_offsets(
                top,
                F,
                basis(top),
                R,
                None if independent_shell else flatten(top),
                depth * mm,
                cfg.get("outer_lateral_mm", 1.0),
                cfg.get("depth_tolerance_mm"),
            )
        )
    else:
        delta = expanded(raw[:, 3:]) - outer
    if independent_shell and len(layout["hinges"]):
        from dual_triangle import choose, place_shell
        shell = np.concatenate([outer, outer + delta], axis=1)
        rest = flatten(shell)
        hinges = layout["hinges"]
        side = choose(shell, hinges)
        best, _ = place_shell(rest, hinges, side, layout["regular_cells"], layout["lattice_up"],
                              (30, -30, 15, -15, 0, None))
        plane = best[:, :3]
        R = basis(plane).transpose(0, 2, 1) @ basis(top)
        trans = plane.mean(1) - np.einsum("nij,nj->ni", R, top.mean(1))
    q = flatten(outer)
    d = delta @ R.transpose(0, 2, 1)
    outline = [
        MultiPoint(v).convex_hull for v in np.concatenate((q, q + d), 1)[:, :, :2]
    ]
    i, j = STRtree(outline).query(outline, predicate="intersects")
    pairs = np.c_[i[i < j], j[i < j]]
    scale = np.ones(id7.max() + 1)
    axes = np.stack(
        [np.cos(np.arange(720) * np.pi / 360), np.sin(np.arange(720) * np.pi / 360)], 1
    )
    # For each potentially overlapping pair, preserve the top triangles and
    # place a separating plane in their gap. Prefer keeping the center deep.
    weights = np.array([1, 1, 1, 2, 2, 2, 6])
    for i, j in ([] if outward else pairs):
        de = np.concatenate(
            [np.roll(z, 1, 0) - z for z in (q[i, :3, :2], q[j, :3, :2])]
        )
        extra = unit(de[:, ::-1] * [-1, 1])
        axis = np.vstack([axes, extra, -extra])
        u = q[i, :, :2] @ axis.T
        v = q[j, :, :2] @ axis.T
        left = u.max(0)
        right = v.min(0)
        ok = right >= left - 1e-8
        if not ok.any():
            raise ValueError(f"平面配置で外面が交差しています（パネル {i}, {j}）。")
        axis = axis[ok]
        left = left[ok]
        right = right[ok]
        u = u[:, ok]
        v = v[:, ok]
        du = d[i, :, :2] @ axis.T
        dv = d[j, :, :2] @ axis.T
        best = None
        for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
            cut = left + (right - left) * frac

            ca = maximum_depth_fraction(cut - u, du)
            cb = maximum_depth_fraction(v - cut, -dv)
            score = weights @ (
                np.minimum(ca, scale[id7[i], None])
                + np.minimum(cb, scale[id7[j], None])
            )
            ix = score.argmax()
            if best is None or score[ix] > best[0]:
                best = score[ix], ca[:, ix], cb[:, ix]
        for face, cap in [(i, best[1]), (j, best[2])]:
            np.minimum.at(scale, id7[face], cap)
    # A shared center/edge column has one 3D endpoint for both incident panels.
    bot = outer + delta * scale[id7, None]
    ring = [0, 3, 1, 4, 2, 5]
    faces = []
    for s, u in enumerate(ring):
        v = ring[(s + 1) % 6]
        faces.extend([[6, u, v], [13, v + 7, u + 7]])

    def panel_triangles(i):
        tris = faces.copy()
        for s, u in enumerate(ring):
            v = ring[(s + 1) % 6]
            if id7[i, u] < id7[i, v]:
                tris.extend([[u, u + 7, v + 7], [u, v + 7, v]])
            else:
                tris.extend([[u, u + 7, v], [u + 7, v + 7, v]])
        return tris

    triangles = np.asarray([panel_triangles(i) for i in range(n)])

    if outward:
        triangles = triangles[:, :, ::-1].copy()

    def solid(i):
        pts = np.vstack([outer[i], bot[i]])
        mesh = md.Mesh64(pts, triangles[i].astype(np.uint64))
        mesh.merge()
        g = md.Manifold(mesh)
        if g.status() != md.Error.NoError:
            raise RuntimeError((i, g.status()))
        return g

    fold = [solid(i) for i in range(n)]
    for attempt in range(24):
        hit = intersections(fold)
        if outward:
            layer_normals = np.cross(bot[:, 1] - bot[:, 0], bot[:, 2] - bot[:, 0])
            inverted = (
                np.einsum("ij,ij->i", layer_normals, basis(top)[:, 2]) <= 0
            ).any()
        else:
            inverted = False
        if not hit and not inverted:
            break
        if outward and cfg.get("depth_tolerance_mm") is not None:
            raise ValueError("指定厚みで組立時に干渉または面の反転が生じました。")
        affected = np.unique([i for a, b, _ in hit for i in (a, b)])
        if outward:
            # A uniform backoff preserves the outer planes and shared seams.
            scale *= 0.82
        else:
            scale[np.unique(id7[affected])] *= 0.82
        bot = outer + delta * scale[id7, None]
        fold = [solid(i) for i in range(n)]
    area_move_mm = 0.0
    if outward and cfg.get("restore_panel_area"):
        from area_relaxation import repair_offset
        shared = X0.copy()
        shared[F] = bot[:, :3]
        move = repair_offset(F, X0, shared) - shared
        initial_bot = bot.copy()
        normals = basis(top)[:, 2]
        tolerance = cfg.get("depth_tolerance_mm")
        for factor in (1., .5, .25, .125, .0625):
            if not np.any(move): break
            candidate = initial_bot + factor * expanded(move[F])
            displacement = candidate - outer
            thickness = np.einsum("nkj,nj->nk", displacement, normals)
            lateral = displacement - thickness[:,:,None]*normals[:,None]
            if np.any(thickness <= 0) or np.max(np.linalg.norm(lateral, axis=2)) > cfg.get("outer_lateral_mm",1.)+1e-6: continue
            if tolerance is not None and np.any((thickness < depth[:,None]*mm-tolerance-1e-6) | (thickness > depth[:,None]*mm+1e-6)): continue
            projected = np.cross(candidate[:,1]-candidate[:,0],candidate[:,2]-candidate[:,0])
            if np.any(np.sum(projected*normals,axis=1)<=0): continue
            bot = candidate
            trial = [solid(i) for i in range(n)]
            if not intersections(trial) and not intersections([g.transform(np.c_[R[i], trans[i]]) for i,g in enumerate(trial)]):
                fold = trial
                area_move_mm = float(factor*np.linalg.norm(move,axis=1).max())
                break
            bot = initial_bot
    if outward:
        layer_normals = np.cross(bot[:, 1] - bot[:, 0], bot[:, 2] - bot[:, 0])
        if np.any(np.einsum("ij,ij->i", layer_normals, basis(top)[:, 2]) <= 0):
            raise ValueError("外側の三角形が反転しています。")
    flatbody = [g.transform(np.c_[R[i], trans[i]]) for i, g in enumerate(fold)]

    flat_hits = intersections(flatbody)
    folded_hits = intersections(fold)
    if flat_hits or folded_hits:
        raise ValueError(
            f"押し出し調整後もパネルが干渉します。T2D {len(flat_hits)} 組、T3D {len(folded_hits)} 組。"
        )
    solids = np.concatenate([outer, bot], axis=1)
    normalized = solids / mm
    normalized[:, :3] = baseline[:, :3]
    actual = -np.einsum("nij,nj->ni", bot - outer, basis(top)[:, 2])
    if outward:
        actual *= -1
    tolerance = cfg.get("depth_tolerance_mm") if outward else None
    if tolerance is not None and np.any(
        (actual < depth[:, None] * mm - tolerance - 1e-6)
        | (actual > depth[:, None] * mm + 1e-6)
    ):
        raise ValueError("出力するパネルの厚みが指定範囲を外れました。")
    np.savez_compressed(
        out / "tiles.npz",
        K3D=baseline[:, :3],
        T3D=normalized,
        surface_tris=triangles,
        depth=actual[:, 6] / mm,
        requested_depth=depth,
        flat_solids=flatten(solids) / mm,
        rotation=R,
        translation_mm=trans,
        **{"layout_" + k: v for k, v in layout.items()},
    )
    dest = out / "prototype"
    dest.mkdir(exist_ok=True)
    trimesh.util.concatenate([as_mesh(g) for g in fold]).export(dest / "T3D.stl")
    report = dict(
        panels=n,
        extrusion_mode=cfg.get("extrusion_mode", "fixed_skin"),
        miter_apex_panels=[],
        area_repair_max_move_mm=area_move_mm,
        top_coordinate_error_mm=float(np.max(abs(top - X0[F]))),
        corner_depth_mm=[float(actual[:, :3].min()), float(actual[:, :3].max())],
        center_depth_mm=np.percentile(actual[:, 6], [0, 50, 100]).tolist(),
        corner_depth_range=(
            np.array([actual[:, :3].min(), actual[:, :3].max()]) / mm
        ).tolist(),
        near_zero_columns=int(np.sum(scale < 0.01)),
        columns=len(scale),
        flat_body_intersections=len(flat_hits),
        assembled_body_intersections=len(folded_hits),
    )
    if outward:
        normal = basis(top)[:, 2]
        lateral = (bot - outer) - actual[:, :, None] * normal[:, None]
        report["maximum_lateral_mm"] = float(np.linalg.norm(lateral, axis=2).max())
        report["near_zero_columns"] = len(
            np.unique(id7[actual < 0.01 * mm * depth[:, None]])
        )
    (out / "tiles_summary.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)


def connect(out, cfg):
    import pickle

    with (out / "remesh3d.pkl").open("rb") as f:
        mesh = pickle.load(f)
    mm = cfg["mm_per_unit"]
    F = mesh["T"]
    X0 = mesh["X"] * mm
    with np.load(out / "flat.npz") as arc:
        flat = {"hinges": arc["hinges"]}
    with np.load(out / "tiles.npz") as shell:
        outer = shell["T3D"][:, :7] * mm
        flat_solids = shell["flat_solids"]
        triangles = shell["surface_tris"]
        R, trans = shell["rotation"], shell["translation_mm"]
    flatbody = []
    for pts, tris in zip(flat_solids * mm, triangles):
        surface = md.Mesh64(pts, tris.astype(np.uint64))
        surface.merge()
        g = md.Manifold(surface)
        if g.status() != md.Error.NoError:
            raise ValueError("T2Dパネルの立体を構成できません。")
        flatbody.append(g)
    a, k, b, l = flat["hinges"].T
    other = (F[a, :, None] == F[b, None, :]).any(2) & (F[a] != F[a, k, None])
    e = F[a, other.argmax(1)]

    # Attach inside the original perimeter, 1.8 mm along the designated shared edge.
    def anchors(i, j):
        u = outer[i, j]
        return (
            u
            + unit(X0[e] - u)
            * np.minimum(1.8, 0.30 * np.linalg.norm(X0[e] - u, axis=1))[:, None]
        )

    P = anchors(a, k)
    P2 = anchors(b, l)
    if not np.allclose(P, P2, rtol=0, atol=1e-8):
        raise ValueError("ヒンジの共有辺が一致しません。")
    pa = np.einsum("nij,nj->ni", R[a], P) + trans[a]
    pb = np.einsum("nij,nj->ni", R[b], P) + trans[b]
    # Bond a compliant bridge of the selected diameter. In the folded target
    # its centerline is collapsed, so verification concerns the flat fabricated shape.
    ball = md.Manifold.sphere(cfg["hinge_mm"] / 2, 16)
    joints = [
        md.Manifold.batch_hull([ball.translate(x), ball.translate(y)])
        for x, y in zip(pa, pb)
    ]
    boxes = np.array([g.bounding_box() for g in flatbody])
    third = 0
    touch = []
    # Open a clearance in the flexible material only around unintended bodies.
    broad = np.array([g.bounding_box() for g in joints])
    for row, g in enumerate(joints):
        bo = broad[row]
        for j in np.flatnonzero(
            (boxes[:, 3:] >= bo[:3] - 0.1).all(1)
            & (boxes[:, :3] <= bo[3:] + 0.1).all(1)
        ):
            if j in (a[row], b[row]):
                continue
            if (g ^ flatbody[j]).volume() > 1e-8:
                joints[row] -= flatbody[j].minkowski_sum(md.Manifold.sphere(0.08, 8))
        for j in np.flatnonzero(
            (broad[:row, 3:] >= bo[:3] - 0.1).all(1)
            & (broad[:row, :3] <= bo[3:] + 0.1).all(1)
        ):
            if (joints[row] ^ joints[j]).volume() > 1e-8:
                joints[row] -= joints[j].minkowski_sum(md.Manifold.sphere(0.08, 8))
        pieces = joints[row].decompose()
        if not pieces:
            raise ValueError(f"ヒンジ {row} を設置する空間がありません。")
        joints[row] = max(pieces, key=lambda x: x.volume())
    for row, g in enumerate(joints):
        box = g.bounding_box()
        for j in np.flatnonzero(
            (boxes[:, 3:] >= box[:3]).all(1) & (boxes[:, :3] <= box[3:]).all(1)
        ):
            vol = (g ^ flatbody[j]).volume()
            if j not in (a[row], b[row]) and vol > 1e-5:
                third += 1
        touch.append([(g ^ flatbody[j]).volume() for j in (a[row], b[row])])
    jbox = np.array([g.bounding_box() for g in joints])
    cross = 0
    for i, g in enumerate(joints):
        for j in (
            np.flatnonzero(
                (jbox[i, 3:] >= jbox[i + 1 :, :3]).all(1)
                & (jbox[i + 1 :, 3:] >= jbox[i, :3]).all(1)
            )
            + i
            + 1
        ):
            cross += (g ^ joints[j]).volume() > 1e-5

    combined = md.Manifold.batch_boolean(flatbody + joints, md.OpType.Add)
    components = sum(piece.volume() > 1e-5 for piece in combined.decompose())
    missing = int((np.array(touch) <= 1e-5).sum())
    ok = components == 1 and not (missing or third or cross)
    report = dict(
        extrusion_mode=cfg.get("extrusion_mode", "fixed_skin"),
        hinges=len(joints),
        span_mm=(
            np.percentile(np.linalg.norm(pa - pb, axis=1), [50, 100]).tolist()
            if len(pa)
            else []
        ),
        components=components,
        unattached_hinge_ends=missing,
        unintended_contacts=third,
        joint_intersections=int(cross),
        converged=bool(ok),
    )
    (out / "closure.json").write_text(json.dumps(report, indent=2))
    if not ok:
        raise ValueError(f"T2Dの接続検査に失敗しました。{report}")
    hinge_mesh = (
        trimesh.util.concatenate([as_mesh(g) for g in joints]) if joints else None
    )
    np.savez_compressed(
        out / "closure.npz",
        T2D=flat_solids,
        hinge_vertices=hinge_mesh.vertices / mm if joints else np.empty((0, 3)),
        hinge_faces=hinge_mesh.faces if joints else np.empty((0, 3), int),
    )
    as_mesh(combined).export(out / "prototype/T2D.stl")
    print(report, flush=True)
