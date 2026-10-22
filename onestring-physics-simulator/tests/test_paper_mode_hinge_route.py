"""Paper-oriented mode uses the paper Eq.5 linkage K2D (and so the paper Dual Hinge solver)."""
from types import SimpleNamespace

from onestring_physics import lscm_latest_omega_hybrid_20260914_patch as hybrid


def test_paper_mode_selects_the_linkage_k2d_without_switching_omega():
    paper = SimpleNamespace(paper_fidelity_mode=True, omega_parameterization_mode="paper_reference_bff")
    off = SimpleNamespace(paper_fidelity_mode=False, omega_parameterization_mode="paper_reference_bff")
    assert hybrid._paper_mode_linkage(paper)
    assert not hybrid._hybrid_active(paper)          # Omega / M2D stay on the official-BFF route
    assert not hybrid._paper_mode_linkage(off)       # Paper OFF unchanged
