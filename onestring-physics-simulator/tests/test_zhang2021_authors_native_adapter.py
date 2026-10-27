"""Unit tests of genuine authors' executable INTERFACE (not algorithm mocks)."""
from pathlib import Path
import importlib.util
import numpy as np
import pytest


MODULE = Path(__file__).resolve().parents[1] / "scripts" / "zhang2021_authors_native_adapter.py"
spec = importlib.util.spec_from_file_location("zhang2021_authors_native_adapter", MODULE)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def test_source_obj_preserves_independent_optcuts_uv_and_xyz_indices(tmp_path):
    path=tmp_path/"source.obj"
    src={
        "source_xyz_vertices":np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],float),
        "source_xyz_faces":np.array([[0,1,2],[0,2,3]],int),
        "source_uv_vertices":np.array([[0,0],[1,0],[1,1],[0,1],
                                       [.1,.1],[.9,.9]],float),
        "source_uv_faces":np.array([[4,1,5],[4,5,3]],int)
    }
    native._obj_export(src,path)
    text=path.read_text(encoding="utf-8")
    assert "v 0 0 0" in text
    assert "vt 0.10000000000000001 0.10000000000000001" in text
    assert "f 1/5 2/2 3/6" in text
    assert "f 1/5 3/6 4/4" in text


def test_3d_nonplanar_without_vt_is_rejected(tmp_path):
    path=tmp_path/"quad.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0.2\nv 0 1 0\nf 1 2 3 4\n")
    with pytest.raises(native.OriginalZhangExecutionError,
                       match="HAS_NO_UV_AND_IS_NOT_PLANAR"):
        native._parse_author_final_obj(path)


def test_all_quad_output_with_independent_texture_indices(tmp_path):
    path=tmp_path/"quads.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
                    "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
                    "f 1/1 2/2 3/3 4/4\n")
    uv,faces=native._parse_author_final_obj(path)
    assert np.array_equal(faces,[[0,1,2,3]])
    assert np.allclose(uv,[[0,0],[1,0],[1,1],[0,1]])


def test_original_3d_geometry_nonplanar_with_valid_uv_is_preserved(tmp_path):
    path=tmp_path/"quads.obj"
    path.write_text("v 0 0 0\nv 1 0 .4\nv 1 1 .7\nv 0 1 0\n"
                    "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
                    "f 1/1 2/2 3/3 4/4\n")
    uv,faces=native._parse_author_final_obj(path)
    assert np.allclose(uv,[[0,0],[1,0],[1,1],[0,1]])


def test_refuse_fake_executables(tmp_path):
    with pytest.raises(native.OriginalZhangExecutionError,
                       match="GENUINE_AUTHOR_EXECUTABLES_NOT_FOUND"):
        native.generate(tmp_path/"missing.npz",tmp_path/"out.npz",tmp_path,
                        35,.01,.05,True,10,[])


def test_invalid_target_not_silently_fixed(tmp_path):
    with pytest.raises(native.OriginalZhangExecutionError,
                       match="INVALID_NATIVE_TARGET_OR_BOUNDS"):
        native.generate(tmp_path/"missing.npz",tmp_path/"out.npz",tmp_path,
                        0,.01,.05,True,10,[])


def test_original_straighten_inverse_is_topology_preserving(tmp_path):
    source=tmp_path/"before.obj"
    after=tmp_path/"after.obj"
    original=[[0,0],[2,0],[2,1],[0,1]]
    stretched=[[0,0],[4,0],[4,2],[0,2]]
    for path,uv in [(source,original),(after,stretched)]:
        lines=[f"v {x} {y} 0" for x,y in uv]
        lines+=["f 1 2 3","f 1 3 4"]
        path.write_text("\n".join(lines)+"\n")
    q=np.array([[1.,.5],[2.,1.],[3.,1.5]])
    expected=np.array([[.5,.25],[1.,.5],[1.5,.75]])
    actual=native._map_after_to_before(source,after,q)
    assert np.allclose(actual,expected)


def test_original_straighten_inverse_refuses_different_faces(tmp_path):
    source=tmp_path/"before.obj"
    after=tmp_path/"after.obj"
    header="v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
    source.write_text(header+"f 1 2 3\nf 1 3 4\n")
    after.write_text(header+"f 1 2 3\nf 2 3 4\n")
    with pytest.raises(native.OriginalZhangExecutionError,
                       match="BEFORE_AFTER_TOPOLOGY_DIFFERS"):
        native._map_after_to_before(source,after,np.array([[.3,.3]]))


def test_independent_solver_does_not_replace_original_package(tmp_path):
    authors=tmp_path/"official"; authors.mkdir()
    original=authors/"libpardiso600-WIN-X86-64.dll"
    original.write_bytes(b"original authors software")
    work=tmp_path/"runtime";work.mkdir()
    (work/original.name).write_bytes(original.read_bytes())
    independent=tmp_path/"independent.dll"
    independent.write_bytes(b"synthetic test independent Eigen backend" * 200)
    sha=native._use_independent_sparse_backend(work,independent)
    assert (work/original.name).read_bytes()==independent.read_bytes()
    assert original.read_bytes()==b"original authors software"
    assert (work/"author_license_bound_solver_DO_NOT_LOAD.bak").read_bytes()==original.read_bytes()
    assert len(sha)==64
