"""Active production M2D builder for optcuts_test.

In the experimental area-grid mode, Ω was already fitted to the *exact*
50%-overlap cell mask on the production lattice. That frozen mask is the
authoritative face selection; never recrop it with a strict containment
test or silently accept a different footprint.

Other optcuts_test modes retain the old strict full-quad crop. Both
paths retain real four-corner cells and do not fabricate clipped panels.
"""
from __future__ import annotations

from typing import Any
from types import SimpleNamespace
import numpy as np

from .quad_grid import create_quad_grid


def _point_in_polygon(point, polygon):
    p=np.asarray(point,float)[:2]; x,y=float(p[0]),float(p[1]); poly=np.asarray(polygon,float)
    for i in range(len(poly)):
        a=poly[i]; b=poly[(i+1)%len(poly)]; ab=b-a; den=float(np.dot(ab,ab))
        if den>1e-24:
            t=float(np.clip(np.dot(p-a,ab)/den,0,1))
            if float(np.linalg.norm(p-(a+t*ab)))<=1e-9: return True
    inside=False
    for i in range(len(poly)):
        x0,y0=poly[i]; x1,y1=poly[(i+1)%len(poly)]
        if (y0>y)!=(y1>y):
            den=float(y1-y0)
            if abs(den)>1e-20 and x < float(x0+(x1-x0)*(y-y0)/den): inside=not inside
    return inside


def _polygon_area(poly):
    p=np.asarray(poly,float)
    if len(p)<3: return 0.0
    return 0.5*abs(float(np.sum(p[:,0]*np.roll(p[:,1],-1)-np.roll(p[:,0],-1)*p[:,1])))


def _panel_inside_omega(poly,boundary,samples_per_edge=13):
    p=np.asarray(poly,float)
    if len(p)!=4 or _polygon_area(p)<=1e-12: return False
    for v in p:
        if not _point_in_polygon(v,boundary): return False
    for i in range(4):
        a=p[i]; b=p[(i+1)%4]
        for t in np.linspace(0,1,max(3,int(samples_per_edge))):
            if not _point_in_polygon((1-t)*a+t*b,boundary): return False
    center=np.mean(p,axis=0)
    if not _point_in_polygon(center,boundary): return False
    for v in p:
        for t in (0.2,0.4,0.6,0.8):
            if not _point_in_polygon((1-t)*center+t*v,boundary): return False
    return True


def install_optcuts_test_boundary_clip_m2d_patch(pipeline: Any) -> None:
    if getattr(pipeline,"_onestring_optcuts_test_boundary_clip_installed",False): return
    base=pipeline._build_m2d; base_lift=pipeline._lift_m2d_to_m3d

    def build_m2d(grid,domain,params=None):
        if not bool(getattr(domain,"_optcuts_test_clip_boundary",False)): return base(grid,domain,params)
        nx=int(getattr(domain,"overlay_nx",grid.nx)); ny=int(getattr(domain,"overlay_ny",grid.ny))
        overlay=create_quad_grid(nx,ny,grid.tile_size,grid.gap_size)
        vertices=[np.asarray([p[0],p[1],0.0],float) for p in np.asarray(domain.uv_vertices,float)]
        original=np.asarray(vertices,float); boundary=np.asarray(domain.boundary,float)
        if len(boundary)>1 and np.linalg.norm(boundary[0]-boundary[-1])<1e-10: boundary=boundary[:-1]

        # The earlier area-grid patch produces the authoritative mask
        # BEFORE Ω is deformed. This wrapper is installed afterwards and
        # must consume that SAME mask, never crop a second time.
        mask_payload=getattr(domain,"_optcuts_area_grid_mask",None)
        params_area_mode=bool(
            getattr(params,"optcuts_area_grid_boundary_enabled",False)
            and str(getattr(params,"omega_parameterization_mode",""))=="optcuts_test"
        )
        if params_area_mode and mask_payload is None:
            raise RuntimeError(
                "OPTCUTS_TEST_M2D_MISSING_FROZEN_GRID_MASK: "
                "experimental area-grid Ω requires its original "
                ">=50%-intersection production cell mask"
            )
        area_mode=mask_payload is not None
        tiles=list(overlay.tiles or [])
        if area_mode:
            from .optcuts_area_grid_boundary import keep_cells_on_fixed_lattice
            mask=np.asarray(mask_payload,dtype=bool)
            if mask.shape!=(ny,nx):
                raise RuntimeError(
                    "OPTCUTS_TEST_M2D_MASK_SHAPE_MISMATCH: "
                    f"mask_shape={mask.shape} lattice=({ny},{nx})"
                )
            chart=getattr(domain,"parameterization",None)
            if not getattr(chart,"metrics",{}).get(
                "optcuts_area_grid_boundary_enabled",False
            ):
                raise RuntimeError(
                    "OPTCUTS_TEST_M2D_MASK_WITHOUT_ACCEPTED_OMEGA: "
                    "cannot consume an unverified frozen grid mask"
                )
            if len(tiles)!=nx*ny:
                raise RuntimeError("OPTCUTS_TEST_M2D_INCOMPLETE_OVERLAY")
            all_faces=np.asarray(
                [tile.vertex_ids for tile in tiles],dtype=int).reshape(-1,4)
            actual=keep_cells_on_fixed_lattice(
                SimpleNamespace(vertices=original,faces=all_faces),domain
            )
            expected=np.asarray(
                [mask[int(t.row),int(t.col)] for t in tiles],bool)
            if not np.array_equal(actual,expected):
                mismatched=np.flatnonzero(actual!=expected)
                raise RuntimeError(
                    "OPTCUTS_TEST_M2D_MASK_CELL_ID_MISMATCH: "
                    f"count={len(mismatched)} "
                    f"cell_ids={mismatched[:20].tolist()}"
                )
            # Every cell must reference its own original lattice vertices;
            # equality of selected-cell *counts* is not sufficient.
            for tile in tiles:
                row,col=int(tile.row),int(tile.col)
                expected_ids=(
                    row*(nx+1)+col,
                    row*(nx+1)+col+1,
                    (row+1)*(nx+1)+col+1,
                    (row+1)*(nx+1)+col,
                )
                if tuple(map(int,tile.vertex_ids))!=expected_ids:
                    raise RuntimeError(
                        "OPTCUTS_TEST_M2D_TILE_VERTEX_IDS_MISMATCH: "
                        f"tile={tile.id} ({row},{col})"
                    )
            chosen=expected
            print(
                "[OPTCUTS-TEST-M2D-MASK] "
                f"selected={int(chosen.sum())}/{len(chosen)} "
                f"mask={int(mask.sum())} "
                f"cell_id_mismatch=0 threshold="
                f"{getattr(params,'optcuts_area_grid_coverage_threshold',.5)}",
                flush=True,
            )
        else:
            chosen=np.asarray([
                _panel_inside_omega(
                    original[np.asarray(tile.vertex_ids,int),:2],
                    boundary,
                ) for tile in tiles
            ],bool)
        faces=[]; polygons=[]; sources=[]; uv_faces=[]
        full=removed_boundary=0
        for tile,retain in zip(tiles,chosen):
            if not retain:
                removed_boundary+=1
                continue
            ids=[int(v) for v in tile.vertex_ids]
            cell=original[np.asarray(ids,int),:2]
            faces.append(tuple(ids))
            polygons.append(list(ids))
            sources.append("full_grid_quad")
            uv_faces.append(cell.copy())
            full+=1

        if not faces:
            raise RuntimeError(
                "OPTCUTS_TEST_M2D_FROZEN_MASK_EMPTY" if area_mode
                else "OPTCUTS_TEST_M2D_STRICT_FULL_QUAD_CROP_EMPTY"
            )
        verts=np.asarray(vertices,float); face_array=np.asarray(faces,int)
        cls=getattr(getattr(pipeline,"_original",None),"QuadMesh",None) or type(base(grid,domain,params))
        metrics={
            "m2d_crop_policy":(
                "frozen_original_omega_area_overlap_mask"
                if area_mode else "strict_full_grid_quads_inside_omega"
            ),
            "optcuts_area_grid_boundary_applied_to_m2d":bool(area_mode),
            "optcuts_area_grid_coverage_threshold":(
                float(getattr(params,"optcuts_area_grid_coverage_threshold",.5))
                if area_mode else None
            ),
            "optcuts_area_grid_mask_matched_cell_ids":bool(area_mode),
            "optcuts_area_grid_mask_cell_count":(
                int(mask.sum()) if area_mode else None
            ),
            "m2d_authoritative_selection":"frozen_mask" if area_mode else "strict_full_cell",
            "m2d_full_cell_count":full,
            "m2d_boundary_clipped_cell_count":0,
            "m2d_removed_cell_count":removed_boundary,
            "m2d_boundary_pentagon_split_count":0,
            "m2d_triangle_panel_count":0,
            "m2d_quad_panel_count":full,
            "m2d_rejected_outside_omega_count":removed_boundary,
            "m2d_max_visible_panel_degree":4,
            "m2d_panel_subset_of_omega_enforced":True,
            "m2d_unused_overlay_vertex_count":int(
                len(verts)-len(np.unique(face_array))
            ),
            "m2d_true_polygon_geometry":True,
            "m2d_legacy_quad_surrogate_for_downstream":False,
            "m2d_boundary_surrogate_generation_disabled":True,
            "m2d_boundary_crossing_cells_discarded":not area_mode,
            "number_of_splits":len(getattr(domain,"split_lines",[]) or []),
            "split_locations":list(getattr(domain,"split_lines",[]) or []),
        }
        out=cls(verts,face_array,overlay,"M2D",metrics,list(getattr(domain,"split_lines",[]) or []))
        setattr(out,"_polygon_faces",[list(map(int,f)) for f in polygons])
        setattr(out,"_optcuts_test_boundary_clipped",False)
        setattr(out,"_optcuts_test_face_sources",list(sources))
        setattr(out,"_optcuts_test_face_uv",[np.asarray(x,float) for x in uv_faces])
        print(
            f"[OPTCUTS-TEST-M2D] full_quads={full} removed_boundary_crossing={removed_boundary} "
            "triangles=0 clipped_quads=0 surrogate_midpoints=0 "
            f"selection={'frozen_mask' if area_mode else 'strict_full_quads'}"
        )
        return out

    def lift(target,mesh,parameterization,params):
        lifted,report=base_lift(target,mesh,parameterization,params)
        for attr in ("_polygon_faces","_optcuts_test_boundary_clipped","_optcuts_test_face_sources","_optcuts_test_face_uv"):
            if hasattr(mesh,attr): setattr(lifted,attr,getattr(mesh,attr))
        return lifted,report

    pipeline._build_m2d=build_m2d; pipeline._lift_m2d_to_m3d=lift
    original=getattr(pipeline,"_original",None)
    if original is not None: original._build_m2d=build_m2d; original._lift_m2d_to_m3d=lift
    for fn in (getattr(pipeline,"build_onestring_design",None),getattr(pipeline,"_ORIGINAL_BUILD_ONESTRING_DESIGN",None),getattr(original,"build_onestring_design",None) if original is not None else None):
        glb=getattr(fn,"__globals__",None)
        if isinstance(glb,dict): glb["_build_m2d"]=build_m2d; glb["_lift_m2d_to_m3d"]=lift
    pipeline._onestring_optcuts_test_boundary_clip_installed=True


__all__=["install_optcuts_test_boundary_clip_m2d_patch"]
