"""2026-10-28 structured-grid mode: minimal smoke tests (pure Python)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from test_zhang2021_quad_20261027 import cylinder_chart


def _params(**kw):
    from onestring_physics.onestring_pipeline import PipelineParameters
    p = PipelineParameters()
    p.optcuts_structured_grid_20261028_enabled = True
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def test_mode_uses_09_18_route_and_zhang_hooks():
    from onestring_physics import optcuts_zhang2021_quad_20261027 as z
    p = _params()
    assert z.mode_enabled(p) and z.structured_grid_enabled(p)
    assert z.downstream_route(p) == z.ROUTE_PAPER_LOCAL_GLOBAL
    assert not z.mode_enabled(SimpleNamespace())
    assert p.zhang2021_initial_csf_split_enabled is False


def test_cylinder_structured_grid_is_regular_and_seam_closable():
    from onestring_physics.structured_grid_20261028 import plan_structured_grid
    xyz, f3, uv, f2 = cylinder_chart(n_theta=24, n_z=6, h=1.0)
    plan = plan_structured_grid(xyz, f3, uv, f2, _params(structured_grid_auto_panels=60))
    m = plan.metrics
    assert m["structured_grid_interior_valence_all_4"]
    assert m["zhang2021_m2d_quads_valid"]
    assert m["zhang2021_seam_preflight_mode"] == "hard"
    assert m["structured_grid_unresolved"] == []
    # both copies of the cut carry the same number of grid edges
    sides = m["structured_grid_side_lengths"]
    assert len(sides) == 4 and sides[0] == sides[2] and sides[1] == sides[3]
