"""Any disk-type OBJ/STL/PLY is accepted; Lilium remains a special benchmark."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics.reference_bff import ReferenceMeshValidationError


def load_runner():
    path = Path(__file__).parents[1] / "scripts" / "run_reflatten_split.py"
    spec = importlib.util.spec_from_file_location("run_generic_reflatten_split", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mesh_path(tmp_path, name, closed=False):
    p=tmp_path / name
    # Flat oriented triangulated disk, with one interior vertex. No
    # shape-specific Lilium assumptions are encoded in this test mesh.
    if closed:
        p.write_text("""v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 3 2
f 1 2 4
f 2 3 4
f 3 1 4
""")
    else:
        p.write_text("""v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0.5 0.5 0
f 1 2 5
f 2 3 5
f 3 4 5
f 4 1 5
""")
    return p


def fake_run(monkeypatch, module, received):
    def init(target, params, progress_callback=None):
        received.append(params)
        return SimpleNamespace(
            mesh_2d_initial=SimpleNamespace(
                faces=np.zeros((4,4),dtype=int),
                metrics={
                    "csf_split_criterion":"reflatten_tile_area",
                    "csf_split_step_count":2,
                    "max_csf_before_split":2.4,
                    "max_csf_after_split":1.7,
                    "csf_split_status":"satisfied",
                    "csf_split_stopping_reason":"all components <= 2",
                    "reflatten_split_report":{"steps":[],"leaves":[]},
                }
            )
        )
    monkeypatch.setattr(module, "build_paper_reference_initialization", init)


def test_arbitrary_open_obj_can_run_and_reports_input_not_lilium(tmp_path, monkeypatch):
    mod=load_runner()
    args=mod.make_parser().parse_args([
        "--mesh",str(mesh_path(tmp_path,"ordinary_surface.obj")),
        "--grid","8", "--report",str(tmp_path/"result.json"),
    ])
    received=[]
    fake_run(monkeypatch,mod,received)
    report=mod.run(args)
    assert report["split_count"]==2
    assert report["corner_mode"]=="source_boundary_shape_fit"
    assert report["input_topology"]["disk_topology"]
    assert report["input_topology"]["boundary_loop_count"]==1
    assert received[0].paper_split_criterion=="reflatten_tile_area"
    assert received[0].nx==8
    assert (tmp_path/"result.json").is_file()


def test_lilium_corner_mode_keeps_existing_default(tmp_path,monkeypatch):
    mod=load_runner()
    args=mod.make_parser().parse_args([
        "--mesh",str(mesh_path(tmp_path,"lilium.obj")),
        "--report",str(tmp_path/"lilium_result.json"),
    ])
    received=[]
    fake_run(monkeypatch,mod,received)
    report=mod.run(args)
    assert report["corner_mode"]=="lobe_tips"
    assert received[0].bff_rectangle_selection_mode=="lobe_tips"


def test_closed_obj_rejected_explicitly_not_auto_repaired(tmp_path,monkeypatch):
    mod=load_runner()
    args=mod.make_parser().parse_args([
        "--mesh",str(mesh_path(tmp_path,"tetra.obj",closed=True)),
        "--report",str(tmp_path/"bad.json"),
    ])
    with pytest.raises(ReferenceMeshValidationError):
        mod.run(args)
    assert not (tmp_path/"bad.json").exists()


def test_generic_builtin_input_supported(tmp_path,monkeypatch):
    mod=load_runner()
    args=mod.make_parser().parse_args([
        "--builtin","dome","--grid","10",
        "--report",str(tmp_path/"dome.json"),
    ])
    received=[]
    fake_run(monkeypatch,mod,received)
    report=mod.run(args)
    assert report["input"]=="builtin:dome"
    assert report["corner_mode"]=="source_boundary_shape_fit"
    assert received[0].ny==10


def test_non_lilium_expect_count_never_forces_solver(tmp_path,monkeypatch):
    mod=load_runner()
    args=mod.make_parser().parse_args([
        "--mesh",str(mesh_path(tmp_path,"human_surface.obj")),
        "--expect-splits","1",
        "--report",str(tmp_path/"audit.json"),
    ])
    received=[]
    fake_run(monkeypatch,mod,received)
    with pytest.raises(AssertionError,match="Expected 1 Split"):
        mod.run(args)
    assert received and received[0].max_csf_splits==64


def test_streamlit_generic_corner_default_varies_with_uploaded_input():
    text=(Path(__file__).parents[1]/"app_runtime_base.py").read_text()
    # Paper OFF: the default varies with the uploaded input; Paper-oriented
    # mode keeps the validated lobe-tip corners for every input.
    assert "(3 if (_lilium_reference_input or paper_fidelity_enabled) else 2)" in text
    assert 'key=f"bff_rectangle_selection_for_{_corner_mesh_identity}"' in text
    assert 'validate_reference_mesh(result.points, result.faces)' in text
