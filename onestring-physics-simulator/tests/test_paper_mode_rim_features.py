"""Paper mode keeps small sharp rim features in the rectangular BFF target."""
import os

import numpy as np
import pytest

from onestring_physics.paper_split_reflatten import boundary_feature_angles, lobe_tip_corner_positions


def _sheet_with_tooth(n=20, tooth=0.4):
    xs = np.linspace(-1.0, 1.0, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    V = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    if tooth:
        mid = (n // 2) * (n + 1) + 0          # boundary vertex in the middle of the y=-1 side
        V[mid, 1] -= tooth * (2.0 / n)          # a sharp outward spike
    F = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = i*(n+1)+j, (i+1)*(n+1)+j, (i+1)*(n+1)+j+1, i*(n+1)+j+1
            F += [[a, b, c], [a, c, d]]
    return V, np.asarray(F)


def test_tooth_becomes_a_zero_net_feature_and_square_has_none():
    V, F = _sheet_with_tooth()
    corners = lobe_tip_corner_positions(V, F, snap_to_corners=True)
    feats = boundary_feature_angles(V, F, corners)
    assert len(feats) >= 3 and abs(sum(a for *_, a in feats)) < 1e-9
    assert sorted(map(tuple, np.round(np.abs(corners[:, :2]), 9))) == [(1.0, 1.0)] * 4
    V0, F0 = _sheet_with_tooth(tooth=0)
    assert boundary_feature_angles(V0, F0, lobe_tip_corner_positions(V0, F0, snap_to_corners=True)) == []


@pytest.mark.skipif(not (os.environ.get("ONESTRING_BFF_EXECUTABLE") and os.environ.get("ONESTRING_BFF_RECTANGLE_EXECUTABLE")),
                    reason="needs the official BFF executables")
def test_mild_tooth_needs_no_feature_fallback():
    """Features are only a fallback: a tooth BFF can flatten keeps the plain rectangle."""
    from onestring_physics.reference_bff import normalize_uv_and_compute_csf, run_official_bff

    V, F = _sheet_with_tooth()
    corners = lobe_tip_corner_positions(V, F, snap_to_corners=True)
    plain = run_official_bff(V, F, boundary_policy="target_rectangle", rectangle_corner_positions=corners)
    _, diff = normalize_uv_and_compute_csf(V, plain.uv_vertices, F)
    assert diff["uv_degenerate_triangle_count"] == 0 and diff["uv_triangle_flip_count"] == 0
