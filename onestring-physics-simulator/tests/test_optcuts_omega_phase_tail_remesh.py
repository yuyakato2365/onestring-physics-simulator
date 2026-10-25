"""Ablation tests: ① phase ③ tail SD ⑤ topology-preserving local remesh."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix

from onestring_physics.optcuts_grid_aware_seam_selection import (
    _shift_production_lattice,
)
from onestring_physics.optcuts_grid_corner_remesh import (
    subdivide_grid_corner_neighborhood,
)
from onestring_physics.optcuts_grid_sd_refinement import (
    SymmetricDirichletProblem,
)


def _area3d(xyz, faces):
    tri = xyz[faces]
    return .5*np.linalg.norm(np.cross(
        tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)


def test_phase_search_shifts_only_candidate_grid_before_mask():
    xx, yy = np.meshgrid(np.arange(5.), np.arange(5.), indexing="xy")
    base = np.stack((xx, yy), axis=-1).reshape((-1, 2))
    domain = SimpleNamespace(
        overlay_nx=4, overlay_ny=4, uv_vertices=base.copy())
    shifted = _shift_production_lattice(domain, .5, .5)
    np.testing.assert_allclose(shifted.uv_vertices, base + .5)
    np.testing.assert_allclose(domain.uv_vertices, base)
    assert shifted.overlay_nx == domain.overlay_nx
    assert shifted.overlay_ny == domain.overlay_ny


def test_corner_remesh_preserves_surface_exactly_and_adds_uv_freedom():
    xy = np.array([[0., 0.], [2., 0.], [2., 2.],
                   [0., 2.], [1., 1.]])
    xyz = np.column_stack((xy, [0., 0., .2, 0., .1]))
    faces = np.array([[0, 1, 4], [1, 2, 4],
                      [2, 3, 4], [3, 0, 4]], int)
    obj = SimpleNamespace(
        uv_vertices_2d=xy.copy(), surface_vertices_3d=xyz.copy(),
        uv_faces=faces.copy(), surface_faces=faces.copy(),
        triangle_acceleration="stale",
    )
    before_area = _area3d(xyz, faces).sum()
    report = subdivide_grid_corner_neighborhood(
        obj, np.array([0, 1, 2, 3]), xy[:4], xy[:4])
    assert report["subdivided_triangles"] == 4
    assert report["added_interior_vertices"] == 4
    assert len(obj.uv_faces) == 12
    assert len(obj.surface_faces) == 12
    np.testing.assert_allclose(_area3d(
        obj.surface_vertices_3d, obj.surface_faces).sum(), before_area)
    np.testing.assert_allclose(obj.surface_vertices_3d[:len(xyz)], xyz)
    np.testing.assert_allclose(obj.uv_vertices_2d[:len(xy)], xy)
    assert obj.triangle_acceleration is None
    for i, face in enumerate(faces):
        np.testing.assert_allclose(obj.uv_vertices_2d[len(xy)+i],
                                   xy[face].mean(axis=0))
        np.testing.assert_allclose(obj.surface_vertices_3d[len(xyz)+i],
                                   xyz[face].mean(axis=0))
    # Every original edge is present; centroid splits create no T-junctions.
    def edge_set(ff):
        return {tuple(sorted((int(t[i]),int(t[(i+1)%3]))))
                for t in ff for i in range(3)}
    assert edge_set(faces).issubset(edge_set(obj.uv_faces))


def test_tail_sd_analytic_gradient_and_energy():
    xyz = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    source_faces = np.array([[0, 1, 2]])
    uv = np.array([[0., 0.], [1.2, 0.15], [-.15, .8]])
    model = SymmetricDirichletProblem(xyz, source_faces,
                                     source_faces, uv)
    tau = 4.2
    energy, grad = model.energy_gradient(
        uv, tail_threshold=tau, tail_strength=.5)
    assert np.isfinite(energy) and np.isfinite(grad).all()
    np.testing.assert_allclose(energy, model.tail_score(
        uv,tau,strength=.5), atol=1e-9)
    eps = 1e-6
    for i in range(len(uv)):
        for j in range(2):
            pos=uv.copy()
            neg=uv.copy()
            pos[i,j]+=eps
            neg[i,j]-=eps
            plus=model.energy_gradient(pos,tail_threshold=tau)[0]
            minus=model.energy_gradient(neg,tail_threshold=tau)[0]
            np.testing.assert_allclose(grad[i,j],(plus-minus)/(2*eps),
                                       rtol=1e-4,atol=5e-5)


def test_four_modes_are_distinct_and_independently_cached():
    root=Path(__file__).resolve().parents[1]
    launcher=(root/"app_optcuts_core_20260916.py").read_text()
    ui=(root/"app_runtime_base.py").read_text()
    core=(root/"src/onestring_physics/onestring_pipeline.py").read_text()
    for suffix in ("phase_search","tail_sd","adaptive_remesh","combined"):
        assert '"optcuts_area_grid_'+suffix+'"' in launcher
    for flag in (
        "optcuts_area_grid_phase_search_enabled",
        "optcuts_area_grid_tail_sd_enabled",
        "optcuts_area_grid_adaptive_remesh_enabled",
    ):
        assert flag+"=optcuts_area_grid_" in ui
        assert flag+": bool = False" in core
    assert '"phase_search": value in {' in launcher
    assert '"tail_sd": value in {' in launcher
    assert '"adaptive_remesh": value in {' in launcher
