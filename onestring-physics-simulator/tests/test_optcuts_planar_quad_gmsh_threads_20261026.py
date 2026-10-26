"""Exercise Gmsh from background script threads, as Streamlit actually does.

The previous code called gmsh.initialize() with its default interruptible=True,
which sets SIGINT and raises ValueError outside Python's main interpreter thread.
"""
from concurrent.futures import ThreadPoolExecutor
import signal
from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics.optcuts_planar_quad_20261026 import build_planar_quad


def test_gmsh_meshing_from_concurrent_streamlit_style_worker_threads():
    pytest.importorskip("gmsh")
    pytest.importorskip("shapely")
    uv = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    xyz = np.column_stack((uv, np.zeros(4)))
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    source = SimpleNamespace(
        uv_vertices_2d=uv, uv_faces=triangles,
        surface_vertices_3d=xyz, surface_faces=triangles,
        omega_boundary=np.vstack((uv, uv[0])),
    )
    original_sigint = signal.getsignal(signal.SIGINT)

    # Two concurrent sessions must not race Gmsh's global model/finalization.
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(build_planar_quad, source, 16, refine_m3d=False)
            for _ in range(2)
        ]
        results = [future.result(timeout=60) for future in futures]

    for quv, faces, metrics in results:
        assert len(quv) > 0
        assert len(faces) > 0
        assert faces.shape[1] == 4
        assert metrics["planar_quad_omega_undercoverage_fraction"] < .02

    # The worker must not modify the host's installed Ctrl-C handler.
    assert signal.getsignal(signal.SIGINT) == original_sigint


def test_full_quad_gmsh_accepts_short_boundary_curves_with_odd_auto_counts():
    """Every separate CAD edge needs even 1D mesh subdivisions.

    A beveled square contains four very short diagonal curves and four long
    segments. Gmsh's default full-quad boundary mesher can give a short
    curve exactly one segment and raise "1D mesh cannot be divided by 2".
    """
    pytest.importorskip("gmsh")
    pytest.importorskip("shapely")
    uv = np.array([
        [.10, 0.], [.90, 0.], [1., .10], [1., .90],
        [.90, 1.], [.10, 1.], [0., .90], [0., .10],
    ], dtype=float)
    center = np.array([[.5, .5]], dtype=float)
    source_uv = np.vstack((uv, center))
    triangles = np.array(
        [[8, k, (k + 1) % 8] for k in range(8)], dtype=int)
    source_xyz = np.column_stack((source_uv, np.zeros(len(source_uv))))
    source = SimpleNamespace(
        uv_vertices_2d=source_uv, uv_faces=triangles,
        surface_vertices_3d=source_xyz, surface_faces=triangles,
        omega_boundary=np.vstack((uv, uv[0])),
    )
    quv, faces, metrics = build_planar_quad(source, 35, refine_m3d=False)
    assert len(faces) > 0
    assert faces.shape[1] == 4
    assert metrics["planar_quad_gmsh_per_curve_even_subdivisions"] is True
    assert metrics["planar_quad_gmsh_boundary_curve_count"] == 8
    assert metrics["planar_quad_gmsh_boundary_curve_elements"] >= 16
