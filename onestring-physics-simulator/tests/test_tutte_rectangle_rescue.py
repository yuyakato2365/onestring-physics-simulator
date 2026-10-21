"""Non-Paper emergency mapping is clearly labeled, injective, opt-in only."""
from __future__ import annotations

import numpy as np
import pytest

from onestring_physics.tutte_rectangle_rescue import rectangle_tutte_rescue
from onestring_physics.bff_rectangle_selection import choose_rectangle_by_source_boundary_fit
from onestring_physics.reference_bff import (
    ReferenceBFFError, validate_rectangular_boundary,
    triangle_jacobian_diagnostics
)


def _open_mesh(n=12):
    # Moderately curved, non-Lilium open disk with an interior vertex grid.
    v=np.array([[i/n,j/n,0.15*np.sin(3*i/n)*np.cos(5*j/n)]
                 for j in range(n+1) for i in range(n+1)],dtype=float)
    faces=[]
    for j in range(n):
        for i in range(n):
            a=j*(n+1)+i;b=a+1;c=a+(n+1);d=c+1
            faces.extend(((a,b,d),(a,d,c)))
    return v,np.asarray(faces,dtype=int)


def test_tutte_rectangle_is_valid_and_preserves_input_geometry():
    v,f=_open_mesh(12)
    original=v.copy()
    result=rectangle_tutte_rescue(v,f)
    np.testing.assert_array_equal(original,v)
    assert result.uv_vertices.shape==(len(v),2)
    assert result.metrics["bff_rectangle_verified"]
    assert result.metrics["nonpaper_rectangle_rescue_used"]
    assert result.metrics["paper_reference_equivalent"] is False
    assert "NOT_BFF" in result.metrics["parameterization_backend_name"]
    differential=triangle_jacobian_diagnostics(v,result.uv_vertices,f)
    assert differential["uv_triangle_flip_count"]==0
    assert differential["uv_degenerate_triangle_count"]==0
    assert all(np.isfinite(result.uv_vertices.ravel()))


def test_after_all_bff_candidates_fail_explicit_rescue_succeeds():
    v,f=_open_mesh(9)
    before=v.copy()
    invoked=[0]
    def rejecting_bff(*_,**__):
        invoked[0]+=1
        raise ValueError("Rectangular BFF candidate has mixed UV triangle orientations")
    result=choose_rectangle_by_source_boundary_fit(
        v,f,runner=rejecting_bff,allow_nonconformal_rescue=True,
    )
    assert invoked[0]>=20
    assert result.metrics["nonpaper_rectangle_rescue_used"]
    assert result.metrics["bff_corner_selection_rejected_count"]==invoked[0]
    assert result.metrics["fallbacks_used"]==[
        "explicit_nonconformal_tutte_rectangle_rescue"
    ]
    np.testing.assert_array_equal(v,before)


def test_default_never_substitutes_nonpaper_tutte():
    v,f=_open_mesh(9)
    def rejecting(*_,**__):
        raise ValueError("BFF invalid")
    with pytest.raises(ReferenceBFFError,match="original \\+ adaptive"):
        choose_rectangle_by_source_boundary_fit(
            v,f,runner=rejecting,allow_nonconformal_rescue=False
        )


def test_closed_mesh_is_not_silently_cut_to_disk():
    vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    faces=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]])
    with pytest.raises(ReferenceBFFError):
        rectangle_tutte_rescue(vertices,faces)


def test_strict_paper_path_rejects_rescue_even_if_flag_enabled():
    from pathlib import Path
    src=(Path(__file__).parents[1]/"src/onestring_physics/onestring_pipeline.py").read_text()
    assert 'and not bool(getattr(params,"paper_fidelity_mode",False))' in src
    assert "Nonconformal Tutte rectangle rescue is prohibited in strict Paper mode" in src
    assert '"NONPAPER_EXPERIMENTAL_NONCONFORMAL_TUTTE"' in src
