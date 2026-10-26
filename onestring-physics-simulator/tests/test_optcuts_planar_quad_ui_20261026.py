"""Regression tests for the *visible* dated Omega selector and its routing.

The 2026-10-26 option was initially injected only by an optional launcher
monkey-patch.  Keep the actual Streamlit sidebar's selectbox source correct
even if another wrapper or an entrypoint bypasses that patch.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODE = "optcuts_planar_quad_20261026"


def _ui_tree():
    return ast.parse((ROOT / "app_runtime_base.py").read_text(encoding="utf-8"))


def test_planar_quad_mode_is_in_real_omega_selectbox_options():
    calls = [
        node for node in ast.walk(_ui_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "selectbox"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "Omega parameterization mode"
    ]
    assert len(calls) == 1, "The authoritative Omega UI must have one selector"
    choices = ast.literal_eval(calls[0].args[1])
    assert choices.count(MODE) == 1
    assert "paper_reference_bff" in choices
    # Verify the date and name are not merely invisible internal identifiers.
    source = ast.unparse(calls[0])
    assert "2026-10-26" in source
    assert "Gmsh prototype" in source


@pytest.mark.parametrize(
    "selected,session_flag,paper_mode,expected",
    [
        (MODE, False, False, True),
        ("optcuts_test", True, False, True),
        ("optcuts_test", False, False, False),
        ("pca_debug", False, False, False),
        (MODE, False, True, False),
    ],
)
def test_planar_quad_routing_is_not_dependent_on_the_launcher(
    selected, session_flag, paper_mode, expected
):
    assignment = next(
        node for node in ast.walk(_ui_tree())
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "optcuts_planar_quad_20261026_enabled"
            for target in node.targets
        )
    )
    check = compile(ast.Expression(assignment.value), "<ui-flag>", "eval")
    scope = {
        "st": SimpleNamespace(session_state={
            "_onestring_planar_quad_20261026_selected": session_flag
        }),
        "paper_fidelity_enabled": paper_mode,
        "omega_parameterization_mode": selected,
    }
    assert bool(eval(check, {"__builtins__": {"bool": bool}}, scope)) is expected
