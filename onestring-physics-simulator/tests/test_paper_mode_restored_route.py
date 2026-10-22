"""Paper-oriented mode keeps the validated Omega/Split route (feature 5c406ba)."""
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_paper_mode_lobe_tips_is_a_single_rectangular_bff_solve():
    text = (ROOT / "src/onestring_physics/onestring_pipeline.py").read_text()
    head, _, rest = text.partition('if bool(getattr(params, "paper_fidelity_mode", False)):\n'
                                   '                # Paper-oriented mode: the validated route')
    assert rest, "paper-mode branch of the lobe-tip route is missing"
    paper_branch = rest.split("            else:", 1)[0]
    assert "rectangle_corner_positions=corner_xyz" in paper_branch
    assert 'boundary_policy="target_rectangle"' in paper_branch
    assert "choose_rectangle_by_source_boundary_fit" not in paper_branch
    assert "_verify_selected_bff_csf_compatible" not in paper_branch


def test_paper_mode_ui_defaults_to_lobe_tips_for_any_input():
    text = (ROOT / "app_runtime_base.py").read_text()
    assert "(3 if (_lilium_reference_input or paper_fidelity_enabled) else 2)" in text
