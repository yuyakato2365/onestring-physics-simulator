"""Guard against silently running a single T2D trial in full pipeline mode.

Static AST-only check: does not start Streamlit or expensive upstream stages.
"""
from __future__ import annotations

import ast
from pathlib import Path


def _app_ast():
    path = Path(__file__).resolve().parents[1] / "app_runtime_base.py"
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_full_pipeline_receives_all_t2d_search_options():
    tree = _app_ast()
    constructors = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "PipelineParameters"
    ]
    assert constructors, "No full-pipeline PipelineParameters constructor found"
    keys = {keyword.arg: keyword.value
            for keyword in constructors[0].keywords if keyword.arg is not None}
    for keyword_name, widget_name in {
        "hinge_layout_search_mode": "t2d_search_mode",
        "hinge_layout_collision_mode": "t2d_collision_mode",
        "hinge_layout_restart_count": "t2d_restart_count",
        "hinge_layout_perturb_translation": "t2d_perturb_translation",
        "hinge_layout_perturb_rotation_deg": "t2d_perturb_rotation_deg",
        "hinge_layout_random_seed": "t2d_search_seed",
    }.items():
        assert keyword_name in keys, f"Full pipeline omitted {keyword_name}"
        assert isinstance(keys[keyword_name], ast.Name)
        assert keys[keyword_name].id == widget_name
    assert isinstance(keys["hinge_layout_budget_mode"], ast.Constant)
    assert keys["hinge_layout_budget_mode"].value == "per_trial"


def test_t2d_only_uses_shared_budget_and_full_run_cache_key_tracks_settings():
    tree = _app_ast()
    assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute)
                and target.attr == "hinge_layout_budget_mode"
                for target in node.targets)
    ]
    assert any(isinstance(node.value, ast.Constant)
               and node.value.value == "total_shared" for node in assignments)
    functions = [node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef)
                 and node.name == "current_pipeline_key"]
    assert len(functions) == 1
    referenced = {node.id for node in ast.walk(functions[0])
                  if isinstance(node, ast.Name)}
    assert {
        "t2d_search_mode", "t2d_collision_mode", "t2d_restart_count",
        "t2d_perturb_translation", "t2d_perturb_rotation_deg",
        "t2d_search_seed",
    } <= referenced



def test_collision_split_configuration_is_wired_only_to_full_rebuild():
    tree=_app_ast()
    constructors=[node for node in ast.walk(tree)
                  if isinstance(node,ast.Call)
                  and isinstance(node.func,ast.Name)
                  and node.func.id=="PipelineParameters"]
    assert constructors
    kwargs={kw.arg:kw.value for kw in constructors[0].keywords
            if kw.arg is not None}
    controls={
        "collision_split_enabled":"collision_split_enabled",
        "collision_split_max_additional":"collision_split_max_additional",
        "collision_split_boundary_weight":"collision_split_boundary_weight",
        "collision_split_candidate_attempts":"collision_split_candidate_attempts",
        "collision_split_min_relative_improvement":"collision_split_min_improvement",
    }
    for param,widget in controls.items():
        assert param in kwargs
        assert isinstance(kwargs[param],ast.Name)
        assert kwargs[param].id==widget
    functions=[node for node in ast.walk(tree)
               if isinstance(node,ast.FunctionDef)
               and node.name=="current_pipeline_key"]
    referenced={node.id for node in ast.walk(functions[0])
                if isinstance(node,ast.Name)}
    assert set(controls.values())<=referenced
    # T2D-only rerun may not run false-M2D topology changes; it requires
    # a complete source mesh rebuild and is deliberately disabled there.
    assignments=[node for node in ast.walk(tree)
                 if isinstance(node,ast.Assign)
                 and any(isinstance(t,ast.Attribute)
                         and t.attr=="collision_split_enabled"
                         for t in node.targets)]
    assert any(isinstance(node.value,ast.Constant) and node.value.value is False
               for node in assignments)
