"""Actual production M2D wrapper must reuse the exact Ω-fitted 50% mask.

The pre-existing unit tests installed only `paper_m2d` in isolation. In
production the clip wrapper is installed afterwards, so these exercises cover
that *last* builder and the later native Grid adapters as well.
"""
from types import SimpleNamespace
import numpy as np
import pytest

from onestring_physics.optcuts_area_grid_boundary import _grid_boundary
from onestring_physics.quad_grid import create_quad_grid
from onestring_physics.optcuts_test_boundary_clip_m2d_patch import (
    install_optcuts_test_boundary_clip_m2d_patch,
)


class QuadMesh:
    def __init__(self, vertices, faces, grid, stage, metrics=None, split_lines=None):
        self.vertices = np.asarray(vertices, float)
        self.faces = np.asarray(faces, int)
        self.grid = grid
        self.stage = stage
        self.metrics = metrics or {}
        self.split_lines = split_lines or []


def case():
    nx, ny = 4, 4
    mask = np.asarray([
        [False, True,  True,  False],
        [True,  True,  True,  False],
        [False, True, True, True],
        [False, False, True, True],
    ], bool)
    xx, yy = np.meshgrid(np.arange(nx+1), np.arange(ny+1), indexing="xy")
    uv = np.stack((xx, yy), axis=-1).reshape(-1, 2).astype(float)
    outline = _grid_boundary(mask).astype(float)
    domain = SimpleNamespace(
        boundary=np.vstack([outline, outline[0]]),
        uv_vertices=uv,
        overlay_nx=nx, overlay_ny=ny,
        _optcuts_area_grid_mask=mask.copy(),
        _optcuts_area_grid_coverage=mask.astype(float),
        split_lines=[], _optcuts_test_clip_boundary=True,
        parameterization=SimpleNamespace(metrics=dict(
            optcuts_area_grid_boundary_enabled=True,
            optcuts_area_grid_boundary_status="accepted_exact_grid_outline",
            optcuts_area_grid_grid_origin=[0., 0.],
            optcuts_area_grid_grid_step=[1., 1.],
            optcuts_area_grid_cell_count=int(mask.sum()),
        )),
    )
    grid=SimpleNamespace(nx=nx, ny=ny, tile_size=1., gap_size=0.)
    params=SimpleNamespace(
        omega_parameterization_mode="optcuts_test",
        optcuts_area_grid_boundary_enabled=True,
        optcuts_area_grid_coverage_threshold=.5,
    )
    return grid, domain, params


def owner_with_latest_clip():
    original=SimpleNamespace(QuadMesh=QuadMesh)
    def deprecated_crop(*args,**kwargs):
        raise AssertionError("legacy pre-crop must not be entered")
    owner=SimpleNamespace(
        _original=original,
        _build_m2d=deprecated_crop,
        _lift_m2d_to_m3d=lambda *_: ("lifted", "report"),
        _flatten_to_domain=lambda param, grid, params=None: param,
    )
    install_optcuts_test_boundary_clip_m2d_patch(owner)
    return owner


def exact_selected_ids(mesh, domain):
    """Audit IDs from the resulting actual face *positions*, not counts."""
    original=np.asarray(domain.uv_vertices,float)
    xy=np.asarray(mesh.vertices,float)
    ids=set()
    for face in mesh.faces:
        center=np.mean(xy[face,:2], axis=0)
        ij=np.floor(center+1e-9).astype(int)
        ids.add((int(ij[1]), int(ij[0])))
    return ids


def test_production_clip_wrapper_never_recrops_frozen_area_grid_mask(monkeypatch):
    import onestring_physics.optcuts_test_boundary_clip_m2d_patch as module
    grid,domain,params=case()
    # Strict containment must NEVER determine M2D in frozen-mask mode.
    monkeypatch.setattr(
        module,"_panel_inside_omega",
        lambda *_args,**_kw: (_ for _ in ()).throw(
            AssertionError("strict re-crop used in frozen mask mode")
        ),
    )
    owner=owner_with_latest_clip()
    m2d=owner._build_m2d(grid,domain,params)
    expected={(j,i) for j,i in np.argwhere(domain._optcuts_area_grid_mask)}
    assert exact_selected_ids(m2d,domain)==expected
    assert len(m2d.faces)==len(expected)
    assert m2d.metrics["m2d_authoritative_selection"]=="frozen_mask"
    assert m2d.metrics["optcuts_area_grid_mask_matched_cell_ids"] is True
    assert m2d.metrics["m2d_boundary_crossing_cells_discarded"] is False
    assert m2d.metrics["m2d_crop_policy"]=="frozen_original_omega_area_overlap_mask"
    assert all(len(face)==4 for face in m2d.faces)


def test_production_m2d_rejects_lattice_mismatch_instead_of_fabricating_cells():
    grid,domain,params=case()
    domain.uv_vertices[0,0] += .03
    owner=owner_with_latest_clip()
    with pytest.raises(RuntimeError, match="GRID_BOUNDARY_M2D_LATTICE_MISMATCH"):
        owner._build_m2d(grid,domain,params)


def test_production_m2d_rejects_mask_shape_and_missing_mask():
    grid,domain,params=case()
    owner=owner_with_latest_clip()
    domain._optcuts_area_grid_mask=np.ones((3,4),bool)
    with pytest.raises(RuntimeError, match="OPTCUTS_TEST_M2D_MASK_SHAPE_MISMATCH"):
        owner._build_m2d(grid,domain,params)
    del domain._optcuts_area_grid_mask
    with pytest.raises(RuntimeError, match="OPTCUTS_TEST_M2D_MISSING_FROZEN_GRID_MASK"):
        owner._build_m2d(grid,domain,params)


def test_latest_production_clip_after_other_m2d_wrappers_still_uses_frozen_mask():
    """Replicate relevant install order: paper -> clip -> native -> guard."""
    from onestring_physics.optcuts_grid_constrained_m2d_patch import (
        install_optcuts_grid_constrained_m2d_patch,
    )
    from onestring_physics.optcuts_grid_consistency_patch import (
        install_optcuts_grid_consistency_patch,
    )
    grid,domain,params=case()
    owner=owner_with_latest_clip()
    install_optcuts_grid_constrained_m2d_patch(owner)
    install_optcuts_grid_consistency_patch(owner)
    m2d=owner._build_m2d(grid,domain,params)
    want={(j,i) for j,i in np.argwhere(domain._optcuts_area_grid_mask)}
    assert exact_selected_ids(m2d,domain)==want
    assert m2d.metrics["m2d_authoritative_selection"]=="frozen_mask"


def test_ordinary_optcuts_test_still_uses_old_full_cell_containment():
    grid,domain,params=case()
    owner=owner_with_latest_clip()
    params.optcuts_area_grid_boundary_enabled=False
    del domain._optcuts_area_grid_mask
    domain.parameterization.metrics.clear()
    # Restrict Ω so only cell (1,1) is fully contained.
    domain.boundary=np.asarray([
        [1.,1.],[2.,1.],[2.,2.],[1.,2.],[1.,1.]
    ])
    m2d=owner._build_m2d(grid,domain,params)
    assert exact_selected_ids(m2d,domain)=={(1,1)}
    assert m2d.metrics["m2d_authoritative_selection"]=="strict_full_cell"
    assert m2d.metrics["optcuts_area_grid_boundary_applied_to_m2d"] is False
