import numpy as np
from types import SimpleNamespace

from onestring_physics.optcuts_planar_quad_uv_reparam_20261026 import (
    boundary_cycle, harmonically_smoothed_uv, surface_uv_sd, coarse_boundary_reparameterize
)
from onestring_physics.optcuts_planar_quad_refine_20261026 import (
    validate_disk_mesh, relax_m3d_quality
)
from onestring_physics.optcuts_planar_quad_20261026 import (
    _triangle_inverse, UnstructuredQuadGrid
)
from onestring_physics.optcuts_planar_quad_downstream_20261026 import (
    install_unstructured_quad_downstream_20261026
)


def uv_disk_source():
    uv = np.asarray([[0, 0], [.5, .015], [1, 0], [1, 1], [0, 1],
                     [.5, .5]], float)
    faces = np.asarray([[5, i, (i + 1) % 5] for i in range(5)], int)
    xyz = np.column_stack((uv, np.zeros(len(uv))))
    return uv, faces, xyz


def test_uv_boundary_and_interior_reparam_preserves_mapping_and_sd():
    uv, faces, xyz = uv_disk_source()
    before = surface_uv_sd(xyz, faces, uv, faces)
    modified, metrics = harmonically_smoothed_uv(xyz, faces, uv, faces)
    assert np.isfinite(modified).all()
    assert metrics["planar_quad_uv_reparam_sd_after"] <= 1.035 * before + 1e-8
    assert len(boundary_cycle(faces)) == 5
    assert np.allclose(modified.shape, uv.shape)
    for sample in (.3, .5, .7):
        _triangle_inverse(modified, faces, xyz, faces,
                          modified[[0, 5]].mean(axis=0, keepdims=True))


def test_poor_m3d_quads_can_be_relaxed_without_changing_boundary_or_source():
    uv = np.array([[0,0],[.5,0],[1,0],[0,.5],[.32,.38],[1,.5],
                   [0,1],[.5,1],[1,1]], float)
    fquad = np.array([[0,1,4,3],[1,2,5,4],[3,4,7,6],[4,5,8,7]],int)
    # A paired independent source UV triangle mesh.
    tris = np.array([[0,1,4],[0,4,3],[1,2,5],[1,5,4],
                     [3,4,7],[3,7,6],[4,5,8],[4,8,7]],int)
    xyz = np.column_stack((uv, np.zeros(len(uv))))
    source = SimpleNamespace(uv_vertices_2d=uv.copy(),uv_faces=tris,
                             surface_vertices_3d=xyz,surface_faces=tris)
    moved, report = relax_m3d_quality(uv, fquad, source, _triangle_inverse)
    validate_disk_mesh(moved, fquad)
    assert np.allclose(moved[[0,1,2,3,5,6,7,8]],uv[[0,1,2,3,5,6,7,8]])
    assert report["planar_quad_refine_3d_energy_after"] <= report["planar_quad_refine_3d_energy_before"] + 1e-8


def test_unstructured_downstream_collision_pairs_are_not_empty():
    from onestring_physics import legacy_pipeline_base as old
    install_unstructured_quad_downstream_20261026()
    uv = np.array([[0,0],[1,0],[2,0],[0,1],[1,1],[2,1]],float)
    faces = np.array([[0,1,4,3],[1,2,5,4]],int)
    g = UnstructuredQuadGrid(uv, faces,1,.08)
    assert old._collision_candidate_pairs(2,g,False) == [(0,1)]
    assert len(old._boundary_tile_edges(g)) == 6
    out=old._initial_auxetic_vertex_hinge_layout(uv[faces],g)
    assert out.shape == (2,4,2)

def test_coarse_boundary_preserves_bijective_source_triangle_correspondence():
    boundary = np.array([[0,0],[.35,.01],[.67,-.01],[1,0],
                         [1,.34],[1.01,.66],[1,1],[.67,.99],
                         [.34,1.01],[0,1],[-.01,.66],[.01,.34]],float)
    uv = np.vstack((boundary,[.5,.5]))
    tris = np.array([[len(boundary),i,(i+1)%len(boundary)]
                     for i in range(len(boundary))],int)
    xyz=np.column_stack((uv,np.zeros(len(uv))))
    moved,m=coarse_boundary_reparameterize(xyz,tris,uv,tris,16)
    assert m["planar_quad_coarse_uv_sd_after"] <= 1.08 * m["planar_quad_coarse_uv_sd_before"] + 1e-8
    if m["planar_quad_coarse_uv_accepted"]:
        assert m["planar_quad_coarse_boundary_points_after"] < len(boundary)
    _triangle_inverse(moved,tris,xyz,tris,moved[[0,2,12]])

def test_actual_onestring_m2d_to_m3d_lift_accepts_unstructured_quads():
    from onestring_physics import legacy_pipeline_base as old
    uv = np.array([[0,0],[.5,0],[1,0],[0,.5],[.5,.5],[1,.5],
                   [0,1],[.5,1],[1,1]],float)
    tri = np.array([[0,1,4],[0,4,3],[1,2,5],[1,5,4],
                    [3,4,7],[3,7,6],[4,5,8],[4,8,7]],int)
    quads = np.array([[0,1,4,3],[1,2,5,4],[3,4,7,6],[4,5,8,7]],int)
    xyz = np.column_stack((uv,np.zeros(len(uv))))
    original = SimpleNamespace(
        uv_vertices_2d=uv, uv_faces=tri, surface_vertices_3d=xyz,
        surface_faces=tri, omega_boundary=np.vstack((uv[[0,1,2,5,8,7,6,3]],uv[0])),
        method="optcuts_test", metrics={}, triangle_acceleration=None)
    quadgrid = UnstructuredQuadGrid(uv, quads, .5, .01)
    m2d = old.QuadMesh(xyz, quads, quadgrid, "M2D", {}, [])
    params = old.PipelineParameters(nx=3,ny=3,tile_size=.5,
                                    m3d_construction_mode="mesh_harmonic")
    m3d,report = old._lift_m2d_to_m3d(None, m2d, original, params)
    assert len(m3d.faces) == 4
    assert np.allclose(m3d.vertices,xyz,atol=1e-6)
    assert m3d.metrics["m3d_outside_omega_count"] == 0
