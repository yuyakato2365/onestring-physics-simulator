from __future__ import annotations
from types import SimpleNamespace
import numpy as np


def test_paper_t3d_patch_forces_fixed_frustum(monkeypatch):
    from onestring_physics import onestring_pipeline as pipeline
    from onestring_physics.paper_t3d_20260920_patch import install_paper_t3d_20260920_patch

    old = pipeline._extrude_tiles
    try:
        install_paper_t3d_20260920_patch(pipeline)
        assert pipeline._extrude_tiles is not old
        assert getattr(pipeline, "_onestring_paper_t3d_20260920_installed", False)
    finally:
        pipeline._extrude_tiles = old
        pipeline._onestring_paper_t3d_20260920_installed = False


def test_paper_t3d_shared_vertices_do_not_separate():
    from onestring_physics import onestring_pipeline as pipeline
    from onestring_physics.paper_extrusion import extrude_paper_face_planarity

    # Two non-coplanar quads sharing edge (1, 4).  This is deliberately not a
    # trivial planar sheet: Eq.(2) must move vertices, while the shared copies in
    # the returned per-tile representation must still coincide exactly.
    vertices = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.10],
        [2.0, 1.0, 0.25],
        [2.0, 0.0, 0.05],
    ])
    faces = np.asarray([
        [0, 1, 3, 2],
        [1, 5, 4, 3],
    ])
    mesh = SimpleNamespace(
        vertices=vertices,
        faces=faces,
        stage="K3D",
        metrics={
            "paper_t3d_planarity_iterations": 12,
            "paper_t3d_planarity_weight": 1.0,
            "paper_t3d_rest_weight": 1e-3,
        },
    )

    assembly, _ = extrude_paper_face_planarity(mesh, 0.1, "T3D", pipeline)
    tiles = np.asarray(assembly.vertices)

    # face0 local 1 == face1 local 0 (global vertex 1)
    # face0 local 2 == face1 local 3 (global vertex 3)
    np.testing.assert_allclose(tiles[0, 1], tiles[1, 0], atol=1e-12)
    np.testing.assert_allclose(tiles[0, 2], tiles[1, 3], atol=1e-12)
    np.testing.assert_allclose(tiles[0, 5], tiles[1, 4], atol=1e-12)
    np.testing.assert_allclose(tiles[0, 6], tiles[1, 7], atol=1e-12)

    assert assembly.metrics["paper_t3d_shared_vertex_separation_max"] <= 1e-12
    assert assembly.metrics["paper_t3d_optimization_topology"].startswith("shared top/bottom")


def test_paper_t3d_keeps_eight_vertices_per_tile():
    from onestring_physics import onestring_pipeline as pipeline
    from onestring_physics.paper_extrusion import extrude_paper_face_planarity

    mesh = SimpleNamespace(
        vertices=np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.1],
            [0.0, 1.0, 0.0],
        ]),
        faces=np.asarray([[0, 1, 2, 3]]),
        stage="K3D",
        metrics={"paper_t3d_planarity_iterations": 4},
    )
    assembly, _ = extrude_paper_face_planarity(mesh, 0.1, "T3D", pipeline)
    assert np.asarray(assembly.vertices).shape == (1, 8, 3)
    assert np.asarray(assembly.top_faces).shape == (1, 4)
    assert np.asarray(assembly.bottom_faces).shape == (1, 4)
    assert np.asarray(assembly.side_faces).shape == (4, 4)
