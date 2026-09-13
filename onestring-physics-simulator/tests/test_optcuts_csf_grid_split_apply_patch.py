from types import SimpleNamespace

from onestring_physics.optcuts_csf_grid_split_apply_patch import (
    DEFAULT_MAX_SPLITS,
    install_optcuts_csf_grid_split_apply_patch,
)


class _Mesh:
    def __init__(self):
        self.metrics = {}
        self.split_lines = []


class _Pipeline:
    pass


def test_csf_split_plan_is_propagated_to_m2d(monkeypatch):
    pipe = _Pipeline()
    pipe._build_m2d = lambda grid, domain, params=None: _Mesh()
    pipe._original = SimpleNamespace(_build_m2d=pipe._build_m2d)
    pipe.build_onestring_design = None
    pipe._ORIGINAL_BUILD_ONESTRING_DESIGN = None

    install_optcuts_csf_grid_split_apply_patch(pipe)

    domain = SimpleNamespace(
        split_lines=[("row", 0.5), ("col", -0.25)],
        csf_before=3.14,
        csf_after_split=1.91,
        csf_split_threshold=2.0,
    )
    params = SimpleNamespace(omega_parameterization_mode="optcuts_test")
    mesh = pipe._build_m2d(SimpleNamespace(), domain, params)

    assert mesh.metrics["csf_split_applied"] is True
    assert mesh.metrics["max_csf_before_split"] == 3.14
    assert mesh.metrics["max_csf_after_split"] == 1.91
    assert mesh.metrics["paper_style_grid_split_request_count"] == 2
    assert mesh.split_lines == [("row", 0.5), ("col", -0.25)]


def test_non_optcuts_test_is_unchanged():
    pipe = _Pipeline()
    pipe._build_m2d = lambda grid, domain, params=None: _Mesh()
    pipe._original = SimpleNamespace(_build_m2d=pipe._build_m2d)
    pipe.build_onestring_design = None
    pipe._ORIGINAL_BUILD_ONESTRING_DESIGN = None

    install_optcuts_csf_grid_split_apply_patch(pipe)
    params = SimpleNamespace(omega_parameterization_mode="bff")
    mesh = pipe._build_m2d(SimpleNamespace(), SimpleNamespace(), params)
    assert "csf_split_applied" not in mesh.metrics


def test_default_split_budget_is_large_enough_for_hierarchical_cutting():
    assert DEFAULT_MAX_SPLITS >= 64
