"""Continuous TPU under the broad faces of the rigid tiles, in print coordinates."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import manifold3d as md
import numpy as np
import shapely
from shapely.geometry import Polygon, LineString
from shapely.strtree import STRtree
import trimesh

from mac_hinges import _hinge_fill


def mesh_of(solid):
    mesh = solid.to_mesh64()
    return trimesh.Trimesh(mesh.vert_properties[:, :3], mesh.tri_verts, process=False)


def make_membrane(out, cfg):
    mm, thickness = cfg["mm_per_unit"], cfg["hinge_mm"]
    with np.load(out / "tiles.npz") as saved:
        flat = saved["flat_solids"]
        faces = saved["surface_tris"]
    with np.load(out / "flat.npz") as saved:
        hinges = saved["hinges"]
    points = flat.copy() * mm
    outward = cfg["extrusion_mode"] == "outer_layer"
    if not outward:
        points[:, :, 2] *= -1
        faces = faces[:, :, ::-1]
    points[:, :7, 2] = 0.0
    if points[:, :, 2].min() < -1e-6:
        raise ValueError("パネルが膜の接合面を貫通しています。")
    xy = points[:, :3, :2]
    pads = [Polygon(tri) for tri in xy]
    tree = STRtree(pads)
    bridges = []
    minimum_bond = float("inf")
    for a, ka, b, kb in hinges:
        # Keep a full-face pad on each owner. Extend the curved neck just
        # inside both pads for a finite-width material connection.
        outline, _ = _hinge_fill(xy[a], ka, xy[b], kb, 1.04, np.inf)
        bridge = shapely.make_valid(Polygon(outline)).buffer(0.08, quad_segs=6)
        for attempt in range(2):
            if attempt:
                anchors = []
                for i, k in ((a, ka), (b, kb)):
                    corner = xy[i, k]
                    delta = xy[i].mean(0) - corner
                    anchors.append(
                        corner + delta * min(0.25, 1.0 / np.linalg.norm(delta))
                    )
                bridge = LineString(anchors).buffer(0.70, quad_segs=8)
            for j in tree.query(bridge.buffer(0.08)):
                if j not in (a, b):
                    bridge = bridge.difference(pads[j].buffer(0.08))
            if (
                bridge.geom_type == "Polygon"
                and min(bridge.intersection(pads[j]).area for j in (a, b)) > 1e-4
            ):
                break
        else:
            raise ValueError(f"ヒンジ {a}–{b} の膜を両方のパネルに接続できません。")
        minimum_bond = min(
            minimum_bond, *(bridge.intersection(pads[j]).area for j in (a, b))
        )
        bridges.append(bridge)
    tree_pads = tree
    tree = STRtree(bridges)
    for i, bridge in enumerate(bridges):
        for j in tree.query(bridge, predicate="intersects"):
            if (
                j > i
                and not set(hinges[i, [0, 2]]).intersection(hinges[j, [0, 2]])
                and bridge.intersection(bridges[j]).area > 1e-6
            ):
                raise ValueError(f"異なるヒンジ膜 {i} と {j} が交差します。")
    # Small TPU lip around the PLA footprint, limited by neighbouring pads.
    skirts = []
    for i, pad in enumerate(pads):
        nearby = [
            pad.distance(pads[j]) for j in tree_pads.query(pad.buffer(0.20)) if j != i
        ]
        lip = min(0.05, 0.2 * min(nearby, default=1.0))
        skirts.append(pad.buffer(lip, quad_segs=4))
    film = shapely.union_all(skirts + bridges, grid_size=1e-7)
    if film.geom_type != "Polygon":
        raise ValueError("ヒンジを接続したTPU膜が一体になりません。")
    rings = [np.asarray(r.coords)[:-1] for r in [film.exterior, *film.interiors]]
    membrane = md.CrossSection(rings, md.FillRule.EvenOdd).extrude(thickness)
    parts = []
    for vertices, tris in zip(points, faces):
        mesh = md.Mesh64(
            vertices + np.array([0.0, 0.0, thickness]), tris.astype(np.uint64)
        )
        mesh.merge()
        tile = md.Manifold(mesh)
        if tile.status() != md.Error.NoError or tile.volume() <= 0:
            raise ValueError("PLAパネルの閉じた立体を生成できません。")
        parts.append(tile)
    rigid = md.Manifold.batch_boolean(parts, md.OpType.Add)
    if (rigid ^ membrane).volume() > 1e-5:
        raise ValueError("PLAとTPUの材料体積が重なっています。")
    # Extend the underside into its own pad in the single-material preview.
    # Its external boundary is unchanged; this removes coincident internal faces.
    preview_tiles = []
    for vertices, tris in zip(points, faces):
        vertices = vertices.copy()
        vertices[:7, 2] -= min(0.01, thickness / 4)
        mesh = md.Mesh64(vertices + [0.0, 0.0, thickness], tris.astype(np.uint64))
        mesh.merge()
        preview_tiles.append(md.Manifold(mesh))
    whole = md.Manifold.batch_boolean(
        preview_tiles + [membrane], md.OpType.Add
    ).simplify(1e-5)
    dest = out / "prototype"
    dest.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=dest) as temp:
        tmp = Path(temp)
        for name, solid in [
            ("PLA_tiles", rigid),
            ("TPU_membrane", membrane),
            ("T2D", whole),
        ]:
            mesh_of(solid).export(tmp / (name + ".stl"))
            saved = trimesh.load(tmp / (name + ".stl"), force="mesh")
            regions = saved.split(only_watertight=False)
            expected = len(points) if name == "PLA_tiles" else 1
            if not saved.is_volume or len(regions) != expected:
                raise ValueError(
                    f"{name}.stl の立体検査に失敗しました（{len(regions)} / {expected} 個）。"
                )
        for item in tmp.iterdir():
            item.replace(dest / item.name)
    # Show the membrane on the correct side of the original panel layout.
    display = mesh_of(membrane)
    display.apply_translation([0.0, 0.0, -thickness])
    if not outward:
        display.apply_transform(np.diag([1.0, 1.0, -1.0, 1.0]))
    np.savez_compressed(
        out / "closure.npz",
        T2D=flat,
        hinge_vertices=display.vertices / mm,
        hinge_faces=display.faces,
    )
    ratios = [film.intersection(face).area / face.area for face in pads]
    report = dict(
        extrusion_mode=cfg["extrusion_mode"],
        hinge_design="face_membrane",
        hinges=len(hinges),
        components=1,
        unattached_hinge_ends=0,
        unintended_contacts=0,
        joint_intersections=0,
        converged=True,
        membrane_thickness_mm=thickness,
        min_face_contact_fraction=float(min(ratios)),
        rigid_panels=len(parts),
        pla_z_min_mm=thickness,
        tpu_z_range_mm=[0, thickness],
        min_bridge_pad_overlap_mm2=minimum_bond if bridges else None,
    )
    (out / "closure.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)
