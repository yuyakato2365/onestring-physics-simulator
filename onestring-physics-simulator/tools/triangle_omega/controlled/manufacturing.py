"""Manufacture the posed Kagome solids in mm, in a common plate coordinate frame."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import manifold3d as md
import numpy as np
import trimesh

from mac_hinges import _hinge_fill


def union(parts):
    return md.Manifold.batch_boolean(parts, md.OpType.Add)


def channels(panel, radius):
    # Horizontal bores start at each side centre and terminate at one common junction.
    z = .5*np.min(panel[3:,2])
    middle=panel[:3]+(panel[3:]-panel[:3])*(z/panel[3:,2])[:,None]
    ends=(middle+np.roll(middle,-1,axis=0))/2
    centre=middle.mean(0)
    segments=48
    r=radius/np.cos(np.pi/segments)
    parts=[md.Manifold.sphere(r,segments).translate(centre)]
    for end in ends:
        axis=end-centre
        length=np.linalg.norm(axis)
        u=axis/length
        v=np.array([-u[1],u[0],0.])
        # Extend outside the entrance to open even an oblique side wall.
        cylinder=md.Manifold.cylinder(length+np.ptp(panel,axis=0).max(),r,circular_segments=segments)
        parts.append(cylinder.transform(np.column_stack([v,np.cross(u,v),u,centre])))
    return union(parts)


def manufacture(out):
    cfg=json.loads((out/'settings.json').read_text())
    diagnostic=json.loads((out/'closure.json').read_text())
    if not diagnostic['converged']:
        raise ValueError('ヒンジ位置または食い込みがT2Dの許容差を超えています。closure.json に残差を保存しました。')
    scale=cfg['mm_per_unit']; t=cfg['hinge_mm']; r=cfg['hole_mm']/2
    wall=.5
    with np.load(out/'closure.npz') as archive:
        p=archive['T2D']*scale
    with np.load(out/'flat.npz') as archive:
        joints=archive['hinges'].reshape(-1,4)
    p[:,:,2]*=-1
    p[:,:,2]-=p[:,:3,2].mean()
    with np.load(out/'tiles.npz') as archive:
        tets=archive.get('tets')
    depth=p[:,3:,2].min(1)
    p[:,:3,2]=0.
    separate=cfg['material']=='separate'
    joined=set(joints[:,[0,2]].ravel())
    membrane=np.array([t if i in joined else 0. for i in range(len(p))])
    needed=2*(r+wall)/np.cos(np.pi/48) + (2*membrane if separate else 0.)
    needed=np.maximum(needed,membrane)
    if np.any(depth<=needed):
        k=int(np.argmin(depth/needed))
        raise ValueError(f'パネル {k} の厚み {depth[k]:.2f} mm に穴・外壁・ヒンジを収められません。現在の形状に必要な印刷倍率は {np.max(scale*needed/depth):.2f} mm/単位を超える値です。')
    bodies=[]; membranes=[]; liners=[]; panel_bodies=[]
    corners={i: set() for i in joined}
    for a,ka,b,kb in joints:
        corners[a].add(ka); corners[b].add(kb)
    for i,panel in enumerate(p):
        mid=panel[:3,:2]+(panel[3:,:2]-panel[:3,:2])*(.5*depth[i]/panel[3:,2])[:,None]
        side=np.roll(mid,-1,axis=0)-mid
        ray=mid+.5*side-mid.mean(0)
        sin_angle=np.abs(side[:,0]*ray[:,1]-side[:,1]*ray[:,0])/(np.linalg.norm(ray,axis=1)*np.linalg.norm(side,axis=1))
        if np.any(.5*np.linalg.norm(side,axis=1)*sin_angle <= (r+wall)/np.cos(np.pi/48)):
            raise ValueError(f'パネル {i} の側面に指定径の紐穴と外壁を収められません。')
        tile=(md.Manifold.hull_points(panel) if tets is None else
              union([md.Manifold.hull_points(panel[indices]) for indices in tets[i]]))
        panel_bodies.append(tile)
        bore=channels(panel,r)
        pockets=[]
        for k in corners.get(i,()):
            # Leave clearance above the membrane where two rigid corners would
            # otherwise meet along a zero-area vertical edge in an STL.
            corner=panel[k]
            pocket=md.Manifold.cylinder(panel[3:,2].max()+1.,.16,circular_segments=32).translate([*corner[:2],t])
            pockets.append(pocket)
            tile=tile-pocket
        if separate and i in joined:
            rigid,film=tile.split_by_plane((0.,0.,1.),t)
            membranes.append(film)
            for pocket in pockets:rigid=rigid-pocket
            bodies.append(rigid-bore)
        else:
            if separate:
                bodies.append(tile-bore)
            else:
                sleeve=channels(panel,r+wall)
                bodies.append(tile-sleeve)
                liners.append((sleeve^tile)-bore)
    bridges=[]
    for a,ka,b,kb in joints:
        outline,gap=_hinge_fill(p[a,:3,:2],ka,p[b,:3,:2],kb,1.04,2.)
        if outline is None: raise ValueError(f'ヒンジ {a}–{b} の隙間 {gap:.3f} mm が大きすぎます。')
        # The Mac membrane touches both edges. A small in-plane overlap makes
        # that interface volumetric even when the sides retreat with depth.
        contour=md.CrossSection([outline.tolist()],md.FillRule.EvenOdd).offset(.05,circular_segments=16)
        bridge=contour.extrude(t)
        for i in (a,b):
            attachment=bridge ^ panel_bodies[i]
            if attachment.volume()<1e-7:
                raise ValueError(f'ヒンジ {a}–{b} がパネル {i} に接続できません。')
        bridges.append(bridge)
    if separate:
        pla=union(bodies)
        tpu=union(membranes+bridges)
    else:
        pla=union(liners)
        tpu=union(bodies+bridges)
    if (tpu^pla).volume()>1e-6:
        raise ValueError('PLAとTPUの材料領域が重なっています。')
    # Publish the pair together only after checking the round-tripped STL meshes.
    with TemporaryDirectory(dir=out) as temp:
        dest=Path(temp)/'print';dest.mkdir()
        for name,solid in [('PLA',pla),('TPU',tpu)]:
            if solid.status()!=md.Error.NoError or solid.volume()<=0: raise ValueError(f'{name} のソリッドを生成できません。')
            mesh=solid.simplify(2e-5).to_mesh()
            path=dest/(name+'.stl')
            trimesh.Trimesh(mesh.vert_properties[:,:3],mesh.tri_verts,process=False).export(path)
            reloaded=trimesh.load(path,force='mesh')
            reloaded.vertices=np.round(reloaded.vertices,5)
            reloaded.merge_vertices(digits_vertex=5)
            reloaded.update_faces(reloaded.nondegenerate_faces(height=1e-8))
            reloaded.update_faces(reloaded.unique_faces())
            reloaded.remove_unreferenced_vertices()
            reloaded.export(path)
            saved=trimesh.load(path,force='mesh')
            if name=='TPU' and len(saved.split(only_watertight=False))!=1:
                raise ValueError('TPUのヒンジが一体につながっていません。')
            if not saved.is_volume: raise ValueError(f'書き戻した {name}.stl のメッシュが閉じていません。')
        dest.rename(out/'print')
    (out/'print_summary.json').write_text(json.dumps(dict(panels=len(p),hinges=len(joints),bore_diameter_mm=2*r,
        hinge_thickness_mm=t,minimum_panel_depth_mm=float(depth.min()),material=cfg['material'],mm_per_unit=scale),indent=2))
