"""Native Grid-Seam geometry preflight on the exact 50%-mask M2D lattice."""
from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics.optcuts_grid_seam_preflight import (
    preview_fixed_grid_seam_topology,
)
from onestring_physics.optcuts_seam_exact_closure import (
    SeamTopologyMismatch,discover_exact_seam_topology,
)
from onestring_physics.quad_grid import create_quad_grid


def _source_seam_chart(*, short_physical_edge=False, native=True):
    xyz=np.array([
        [0.,0.,0.],[1.,0.,0.],[2.,0.,0.],
        [0.,1.,0.],[1.,1.,0.],[2.,1.,0.],
    ])
    if short_physical_edge:
        # Source cut is geometrically nonzero but shorter than the
        # historical global-span quantizer (2e-7), not shorter than
        # the original 3D source-edge resolution (2e-9).
        xyz[4]=[1.,5e-8,0.]
    src_faces=np.array([[0,1,4],[0,4,3],[1,2,5],[1,5,4]])
    uv=np.array([
        [0.,0.],[1.,0.],[1.,1.],[0.,1.],
        [2.,0.],[3.,0.],[3.,1.],[2.,1.]
    ])
    uv_faces=np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7]])
    return SimpleNamespace(
        surface_vertices_3d=xyz,
        surface_faces=src_faces,
        uv_vertices_2d=uv,
        uv_faces=uv_faces,
        metrics={
            "optcuts_grid_seam_native_enabled":native,
            "optcuts_area_grid_overlay_nx":3,
            "optcuts_area_grid_overlay_ny":1,
            "optcuts_area_grid_grid_origin":[0.,0.],
            "optcuts_area_grid_grid_step":[1.,1.],
        },
        _optcuts_area_grid_mask=np.array([[True,False,True]],bool),
    )


def test_native_preflight_reproduces_exact_masked_production_quad_mesh():
    chart=_source_seam_chart()
    result=preview_fixed_grid_seam_topology(
        chart,SimpleNamespace(tile_size=1.,gap_size=.08))
    assert result["status"]=="conforming"
    assert result["seam_grid_edges"]==2
    assert result["grid_face_count"]==2
    assert result["exact_mask_unchanged"]
    grid=create_quad_grid(3,1,1.,.08)
    expected=np.asarray([tile.vertex_ids for tile in grid.tiles])[1-1:1]
    assert len(expected)==1
    # IDs 0..7 match the production paper_m2d() full-grid convention.
    assert grid.tiles[0].vertex_ids==(0,1,5,4)
    assert grid.tiles[2].vertex_ids==(2,3,7,6)
    assert chart._optcuts_area_grid_mask.tolist()==[[True,False,True]]


def test_native_preflight_keeps_physically_distinct_close_seam_endpoints():
    chart=_source_seam_chart(short_physical_edge=True,native=True)
    result=preview_fixed_grid_seam_topology(
        chart,SimpleNamespace(tile_size=1.,gap_size=.08))
    assert result["status"]=="conforming"
    assert result["seam_grid_edges"]==2
    # The legacy 10/23 path remains unchanged, including its originally
    # too-coarse 3D physical key resolution.
    chart.metrics["optcuts_grid_seam_native_enabled"]=False
    with pytest.raises(SeamTopologyMismatch,
                       match="SEAM_DEGENERATE_GRID_BOUNDARY_EDGE"):
        preview_fixed_grid_seam_topology(
            chart,SimpleNamespace(tile_size=1.,gap_size=.08))


def test_seam_preflight_invalid_shape_fails_explicitly_and_preserves_mask():
    chart=_source_seam_chart()
    chart._optcuts_area_grid_mask=np.array([[True,False]])
    with pytest.raises(ValueError,match="GRID_SEAM_PREFLIGHT_FROZEN_LATTICE_INVALID"):
        preview_fixed_grid_seam_topology(
            chart,SimpleNamespace(tile_size=1.,gap_size=.08))


def test_seam_preflight_uses_exact_same_faces_after_original_compaction():
    chart=_source_seam_chart()
    grid=create_quad_grid(3,1,1.,.08)
    original_faces=np.asarray([tile.vertex_ids for tile in grid.tiles],int)
    mask=chart._optcuts_area_grid_mask.reshape(-1)
    faces=original_faces[mask]
    used=np.unique(faces.ravel())
    remap=np.full(8,-1,int)
    remap[used]=np.arange(len(used))
    compact_faces=remap[faces]
    coords=np.array([
        [float(i),float(j)] for j in range(2) for i in range(4)
    ])
    compact_coords=coords[used]
    topology=discover_exact_seam_topology(chart,compact_coords,compact_faces)
    report=preview_fixed_grid_seam_topology(
        chart,SimpleNamespace(tile_size=1.,gap_size=.08))
    assert report["seam_grid_edges"]==topology["seam_grid_edges"]
    assert report["boundary_grid_edges"]==topology["boundary_grid_edges"]
