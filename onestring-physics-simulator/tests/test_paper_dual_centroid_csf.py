"""Regression tests for Fig.18 four-tile-centroid dual-area Split metric."""
from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics.paper_dual_csf import (
    build_dual_centroid_field, component_dual_stats,
)


def _grid(nx=4, ny=3, x_transform=None):
    uv = np.array([(float(x), float(y))
                   for y in range(ny+1) for x in range(nx+1)])
    xyz = np.column_stack([uv, np.zeros(len(uv))])
    if x_transform is not None:
        xyz[:, 0] = x_transform(uv[:, 0])
    def v(x, y):
        return y*(nx+1)+x
    quads = np.array([
        [v(x,y),v(x+1,y),v(x+1,y+1),v(x,y+1)]
        for y in range(ny) for x in range(nx)
    ], int)
    tri = np.array([
        t for f in quads for t in
        ([f[0],f[1],f[2]],[f[0],f[2],f[3]])
    ], int)
    parametrization = SimpleNamespace(
        uv_vertices_2d=uv, uv_faces=tri, surface_vertices_3d=xyz,
        surface_faces=tri, metrics={},
    )
    return uv, quads, parametrization


def test_complete_four_tile_centroid_dual_count_and_area():
    uv, quads, p = _grid(4,3,x_transform=lambda x: 2*x)
    field = build_dual_centroid_field(uv, quads, p)
    assert field["face_ids"].shape == (6,4)
    assert field["jacobian"] == pytest.approx(np.full(6,2.))
    assert field["area2"] == pytest.approx(np.ones(6))
    assert field["area3"] == pytest.approx(np.full(6,2.))
    s = component_dual_stats(np.arange(len(quads)), quads, field)
    assert s["measurable"]
    assert s["cell_count"] == 6
    assert s["span"] == pytest.approx(1.)


def test_dual_area_varies_across_centroid_cells():
    uv, quads, p = _grid(4,3,x_transform=lambda x: x**3)
    field = build_dual_centroid_field(uv, quads, p)
    assert sorted(set(field["jacobian"].tolist())) == pytest.approx([4.,13.,28.])
    whole = component_dual_stats(np.arange(len(quads)), quads, field)
    assert whole["span"] == pytest.approx(np.sqrt(7.))


def test_seam_crossing_duals_are_excluded_from_both_children():
    uv, quads, p = _grid(4,3,x_transform=lambda x: x**3)
    field = build_dual_centroid_field(uv, quads, p)
    left = np.array([i for i,f in enumerate(quads) if uv[f,0].mean() < 2], int)
    right = np.array([i for i,f in enumerate(quads) if uv[f,0].mean() > 2], int)
    # First test same-face-IDs filtering; cut duals spanning x=2 disappear.
    l = component_dual_stats(left, quads, field)
    r = component_dual_stats(right, quads, field)
    assert l["measurable"] and r["measurable"]
    assert l["cell_count"] == r["cell_count"] == 2
    assert l["span"] == pytest.approx(1.)
    assert r["span"] == pytest.approx(1.)
    assert set(l["used_dual_ids"]).isdisjoint(r["used_dual_ids"])
    # Even if a region's face IDs are artificially held together, a
    # duplicated central grid vertex makes the original dual invalid.
    cut_faces = quads.copy()
    next_id = len(uv)
    replacements = {}
    for fid in right:
        for k, vid in enumerate(cut_faces[fid]):
            if uv[vid,0] == 2.:
                if vid not in replacements:
                    replacements[vid] = next_id
                    next_id += 1
                cut_faces[fid,k] = replacements[vid]
    all_fan = component_dual_stats(np.arange(len(quads)), cut_faces, field)
    assert all_fan["cell_count"] == 4


def test_one_column_cut_child_is_unmeasurable_not_artificially_csf_one():
    uv, quads, p = _grid(3,3,x_transform=lambda x: x**3)
    field = build_dual_centroid_field(uv, quads, p)
    strip = np.array([i for i,f in enumerate(quads) if uv[f,0].mean() > 2.], int)
    assert len(strip) == 3
    missing = component_dual_stats(strip, quads, field)
    assert not missing["measurable"]
    assert missing["span"] is None
    assert missing["cell_count"] == 0


def test_one_quad_has_no_four_tile_centroid_dual_and_is_not_certified():
    uv, quads, p = _grid(1,1)
    field = build_dual_centroid_field(uv, quads, p)
    assert field["face_ids"].shape == (0,4)
    assert not component_dual_stats(np.array([0]), quads, field)["measurable"]
    assert not field["diagnostics"]["physical_auxetic_bound_certified"]


def test_split_pipeline_marks_single_quad_as_unmeasurable():
    from onestring_physics.paper_mesh_splitting import _split_mesh_single
    uv, quads, p = _grid(1,1)
    domain = SimpleNamespace(
        parameterization=p, csf_split_enabled=True,
        max_csf_splits=4, reference_grid_rotation_degrees=0.,
    )
    params = SimpleNamespace(
        paper_fidelity_mode=True,
        paper_csf_sampling_method="m2d_m3d_dual_centroid",
        paper_split_area_limit_experiment=False,
        paper_split_allow_neutral_cuts=True,
    )
    _vertices, out_faces, records, metrics = _split_mesh_single(
        np.column_stack([uv, np.zeros(len(uv))]), quads, domain, params
    )
    assert len(records) == 0
    assert len(out_faces) == 1
    assert metrics["csf_split_status"] == "uncertified_insufficient_dual_cells"
    assert metrics["csf_split_dual_unmeasured_region_count"] == 1
    assert not metrics["csf_split_candidate_linear_bound_satisfied"]
    assert not metrics["csf_split_candidate_area_bound_satisfied"]


def test_split_pipeline_uses_dual_field_and_preserves_original_grid():
    from onestring_physics.paper_mesh_splitting import _split_mesh_single
    uv, quads, p = _grid(4,3,x_transform=lambda x: x**3)
    domain = SimpleNamespace(
        parameterization=p, csf_split_enabled=True,
        max_csf_splits=3, reference_grid_rotation_degrees=0.,
    )
    params = SimpleNamespace(
        paper_fidelity_mode=True,
        paper_csf_sampling_method="m2d_m3d_dual_centroid",
        paper_split_area_limit_experiment=False,
        paper_split_allow_neutral_cuts=True,
    )
    vertices, cut_faces, records, metrics = _split_mesh_single(
        np.column_stack([uv, np.zeros(len(uv))]), quads, domain, params
    )
    assert metrics["csf_split_sampling_method"] == "m2d_m3d_dual_centroid"
    assert metrics["max_csf_before_split"] == pytest.approx(np.sqrt(7.))
    assert metrics["csf_split_dual_diagnostics"]["dual_cell_count"] == 6
    assert np.allclose(vertices[:len(uv),:2], uv)
    assert np.array_equal(cut_faces.shape, quads.shape)
    assert metrics["csf_split_bound_certified"] is False
    if records:
        assert metrics["csf_split_dual_unmeasured_region_count"] == 0
        assert metrics["max_csf_after_split"] <= metrics["max_csf_before_split"]
