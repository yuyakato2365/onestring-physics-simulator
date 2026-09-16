from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics.ceps_strict_adapter import install_ceps_strict_adapter


def _dummy_module():
    module = SimpleNamespace()

    def inverse(point, parameterization):
        uv = np.asarray(point, dtype=float)
        outside = bool(uv[0] > 0.5)
        return np.asarray([uv[0], uv[1], 0.0]), (-1 if outside else 0), outside

    def clip(mesh, domain, params=None):
        return np.asarray(mesh.faces, dtype=int).copy(), {
            "m2d_boundary_clip_input_quad_count": int(len(mesh.faces))
        }

    module.inverse_map_uv_to_surface = inverse
    module._clip_m2d_faces_to_omega_boundary = clip
    module._CEPS_STRICT_ADAPTER_INSTALLED = False
    return module


def test_ceps_inverse_map_disables_nearest_fallback() -> None:
    module = _dummy_module()
    install_ceps_strict_adapter(module)
    parameterization = SimpleNamespace(
        method="ceps", metrics={"ceps_backend_used": "official_ceps_cli"}
    )

    with pytest.raises(RuntimeError, match="Nearest-triangle/nearest-vertex fallback is disabled"):
        module.inverse_map_uv_to_surface(np.asarray([0.75, 0.2]), parameterization)


def test_ceps_m2d_clip_requires_actual_uv_triangle_coverage() -> None:
    module = _dummy_module()
    install_ceps_strict_adapter(module)
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.4, 0.4, 0.0],
            [0.0, 0.4, 0.0],
            [0.6, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.4, 0.0],
            [0.6, 0.4, 0.0],
        ],
        dtype=float,
    )
    mesh = SimpleNamespace(
        vertices=vertices,
        faces=np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=int),
    )
    parameterization = SimpleNamespace(
        method="ceps", metrics={"ceps_backend_used": "official_ceps_cli"}
    )
    domain = SimpleNamespace(parameterization=parameterization)

    kept, metrics = module._clip_m2d_faces_to_omega_boundary(mesh, domain, None)

    assert kept.shape == (1, 4)
    assert np.array_equal(kept[0], [0, 1, 2, 3])
    assert metrics["ceps_m2d_quad_rejected_outside_actual_uv_union_count"] == 1
    assert metrics["ceps_nearest_inverse_fallback_allowed"] is False
