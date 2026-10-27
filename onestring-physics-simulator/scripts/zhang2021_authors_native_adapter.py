#!/usr/bin/env python3
"""Run ALL THREE genuine Zhang et al. (CAD 2021) author executables.

This is NOT the alternative Gmsh/triangle-pair meshing algorithm. It invokes
the published, unchanged Straighten.exe -> Segment.exe -> QMSim.exe.

Prerequisites: download_zhang2021_executables.py --directory <path>,
Windows or configured Wine, plus EITHER a valid legitimately licensed
PARDISO 6 runtime OR an independently compiled Eigen sparse linear solver
DLL (--independent-eigen-solver-dll). The alternative DLL solves the SAME
linear systems without changing the authors' meshing/optimization logic.
No licensing credentials are copied, forged, patched, or bypassed.

Input/output NPZ protocol matches zhang2021_reference_bridge.py.
Results are rejected by the OneString bridge unless original OptCuts UV
triangle inversion + seam topology are valid. Never silently repair results.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import numpy as np


class OriginalZhangExecutionError(RuntimeError):
    pass

# SHA256 verified from authors' Code/Exe links (official ExeProgram.zip).
# A renamed mock executable must NEVER be reported as original Zhang code.
ORIGINAL_AUTHOR_EXE_SHA256 = {
    "Straighten.exe": "3c88f817ef9ac6bee4da571d78c9d038d7edb1dce545c63548f5c8a3f93c95fc",
    "Segment.exe": "f50287d3b3171bd0c4cab6959d5f7b05d01ae46637dfa8a3e00cdbe0f6ca5b5d",
    "QMSim.exe": "aa73bed0b07446fb20fb25da2035023ccd5e13b073f858f5778387592ce5c4fe",
}

def _verify_authors_executable_provenance(exe_root:Path) -> dict[str,str]:
    manifest = {}
    for executable, expected in ORIGINAL_AUTHOR_EXE_SHA256.items():
        path = exe_root / executable
        if not path.is_file():
            raise OriginalZhangExecutionError("ORIGINAL_AUTHOR_EXE_MISSING: " + executable)
        hasher = hashlib.sha256()
        with path.open("rb") as src:
            while block := src.read(1 << 20):
                hasher.update(block)
        sha = hasher.hexdigest()
        if sha != expected:
            raise OriginalZhangExecutionError(
                f"ORIGINAL_AUTHOR_EXE_HASH_MISMATCH: {executable} "
                f"sha256={sha}; not the verified public Zhang release")
        manifest[executable] = sha
    return manifest


def _obj_export(source:dict, dest:Path):
    """Export paired 3D and UV indices precisely as OBJ v/vt/f."""
    xyz=np.asarray(source["source_xyz_vertices"],float)
    xyz_faces=np.asarray(source["source_xyz_faces"],int)
    uv=np.asarray(source["source_uv_vertices"],float)
    uv_faces=np.asarray(source["source_uv_faces"],int)
    if (xyz.ndim!=2 or xyz.shape[1]!=3 or uv.ndim!=2 or uv.shape[1]!=2
            or xyz_faces.shape!=uv_faces.shape or xyz_faces.ndim!=2
            or xyz_faces.shape[1]!=3 or not len(xyz_faces)
            or not np.isfinite(xyz).all() or not np.isfinite(uv).all()
            or np.any(xyz_faces<0) or np.any(uv_faces<0)
            or np.any(xyz_faces>=len(xyz)) or np.any(uv_faces>=len(uv))):
        raise OriginalZhangExecutionError("INVALID_OPTCUTS_PAIRED_SOURCE")
    with dest.open("w",encoding="utf-8",newline="\n") as stream:
        stream.write("# Unmodified paired OptCuts 3D and UV triangles for Zhang et al.\n")
        for p in xyz: stream.write(f"v {p[0]:.17g} {p[1]:.17g} {p[2]:.17g}\n")
        for p in uv: stream.write(f"vt {p[0]:.17g} {p[1]:.17g}\n")
        for face3,face2 in zip(xyz_faces,uv_faces):
            stream.write("f "+" ".join(
                f"{int(a)+1}/{int(b)+1}" for a,b in zip(face3,face2))+"\n")


def _parse_author_final_obj(path:Path):
    """Interpret the original authors' quad OBJ, no guessed projection."""
    xyz=[]
    tex=[]
    rawfaces=[]
    with path.open("r",encoding="utf-8-sig",errors="replace") as f:
        for line in f:
            items=line.split()
            if not items: continue
            if items[0]=="v" and len(items)>=4:
                xyz.append([float(k) for k in items[1:4]])
            elif items[0]=="vt" and len(items)>=3:
                tex.append([float(k) for k in items[1:3]])
            elif items[0]=="f":
                rawfaces.append(items[1:])
    if not rawfaces or not xyz:
        raise OriginalZhangExecutionError("AUTHORS_FINAL_OBJ_EMPTY")
    if any(len(face)!=4 for face in rawfaces):
        raise OriginalZhangExecutionError("AUTHORS_FINAL_MESH_NOT_ALL_QUAD")
    # OBJ final texture coordinates may have seams: face corners have own vt.
    # Otherwise author vertices must be planar XY (no arbitrary XYZ->UV).
    tex_active=bool(tex) and all(
        len(item.split("/"))>=2 and item.split("/")[1] for face in rawfaces for item in face)
    if tex_active:
        points=np.asarray(tex,dtype=float)
        idxposition=1
    else:
        vertices=np.asarray(xyz,dtype=float)
        span=max(float(np.ptp(vertices,axis=0).max()),1e-12)
        if np.max(np.abs(vertices[:,2]-np.median(vertices[:,2])))>1e-8*span:
            raise OriginalZhangExecutionError(
                "AUTHORS_FINAL_OBJ_HAS_NO_UV_AND_IS_NOT_PLANAR")
        points=vertices[:,:2]
        idxposition=0
    faces=[]
    for face in rawfaces:
        row=[]
        for item in face:
            parts=item.split("/")
            index=int(parts[idxposition])
            if index==0: raise OriginalZhangExecutionError("OBJ_INDEX_ZERO")
            index=index-1 if index>0 else len(points)+index
            if index<0 or index>=len(points):
                raise OriginalZhangExecutionError("AUTHOR_OBJ_INVALID_INDEX")
            row.append(index)
        faces.append(row)
    return points,np.asarray(faces,dtype=int)


def _read_planar_triangle_obj(file_path:Path):
    """Read the genuine authors' Straighten before/after triangle charts."""
    vertices=[]
    faces=[]
    for line in file_path.read_text(encoding="utf-8-sig",errors="replace").splitlines():
        tokens=line.split()
        if not tokens: continue
        if tokens[0]=="v" and len(tokens)>=4:
            p=[float(v) for v in tokens[1:4]]
            vertices.append(p)
        if tokens[0]=="f":
            if len(tokens)!=4:
                raise OriginalZhangExecutionError("AUTHOR_STRAIGHTEN_FACE_NOT_TRIANGLE")
            indices=[]
            for part in tokens[1:]:
                idx=int(part.split("/")[0])
                if idx==0:
                    raise OriginalZhangExecutionError("AUTHOR_STRAIGHTEN_ZERO_OBJ_INDEX")
                indices.append(idx)
            faces.append(indices)
    vertices=np.asarray(vertices,float)
    faces=np.asarray(faces,int)-1
    if (not len(vertices) or not len(faces)
            or not np.isfinite(vertices).all()
            or np.any(faces<0) or np.any(faces>=len(vertices))):
        raise OriginalZhangExecutionError("INVALID_AUTHOR_STRAIGHTEN_CHART")
    if np.max(np.abs(vertices[:,2]))>1e-8*max(1.,np.ptp(vertices,axis=0).max()):
        raise OriginalZhangExecutionError("AUTHOR_STRAIGHTEN_CHART_NOT_PLANAR")
    return vertices[:,:2],faces


def _map_after_to_before(before:Path,after:Path,query:np.ndarray)->np.ndarray:
    """Invert ONLY the real authors' piecewise Straighten deformation.

    The authors' output before/after charts must have identical connectivity
    and vertex order. No closest-point projection or guessed UV adjustment.
    """
    before_uv,before_faces=_read_planar_triangle_obj(before)
    after_uv,after_faces=_read_planar_triangle_obj(after)
    if (before_uv.shape!=after_uv.shape
            or not np.array_equal(before_faces,after_faces)):
        raise OriginalZhangExecutionError(
            "AUTHOR_STRAIGHTEN_BEFORE_AFTER_TOPOLOGY_DIFFERS")
    from onestring_physics.optcuts_planar_quad_20261026 import _triangle_inverse
    before_3d=np.column_stack((before_uv,np.zeros(len(before_uv))))
    try:
        mapping=_triangle_inverse(
            after_uv,after_faces,before_3d,before_faces,query)
    except Exception as exc:
        raise OriginalZhangExecutionError(
            "AUTHOR_STRAIGHTEN_PIECEWISE_INVERSE_UNDEFINED: "+str(exc)) from exc
    return mapping[:,:2]


def _use_independent_sparse_backend(working_root:Path,dll:Path):
    """Replace ONLY a temporary work-copy's numerical PARDISO backend.

    Never modify original software package or bypass/license-patch PARDISO.
    The alternative Eigen solver must be built separately from our source.
    """
    import hashlib
    dll=dll.resolve()
    if not dll.is_file() or dll.stat().st_size<4096:
        raise OriginalZhangExecutionError("INDEPENDENT_SPARSE_BACKEND_DLL_INVALID")
    target=working_root/"libpardiso600-WIN-X86-64.dll"
    if not target.is_file():
        raise OriginalZhangExecutionError("ORIGINAL_SOLVER_DLL_MISSING")
    target.rename(working_root/"author_license_bound_solver_DO_NOT_LOAD.bak")
    shutil.copy2(dll,target)
    if hashlib.sha256(target.read_bytes()).digest()!=hashlib.sha256(dll.read_bytes()).digest():
        raise OriginalZhangExecutionError("INDEPENDENT_SPARSE_BACKEND_COPY_CORRUPTED")
    return hashlib.sha256(dll.read_bytes()).hexdigest()


def _execute(program:Path,args:list[str],cwd:Path,timeout_seconds:int,wine:list[str]):
    run=([str(program)] if platform.system()=="Windows" else wine+[program.name])+args
    if not wine and platform.system()!="Windows":
        raise OriginalZhangExecutionError(
            "AUTHORS_NATIVE_WINDOWS_ONLY: supply --wine-command for a configured Wine host")
    try:
        completed=subprocess.run(run,cwd=cwd,stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,text=True,errors="replace",timeout=timeout_seconds,
            shell=False,check=False,env=dict(os.environ,QT_QPA_PLATFORM="offscreen"))
    except (OSError,subprocess.TimeoutExpired) as e:
        raise OriginalZhangExecutionError(
            f"AUTHOR_ORIGINAL_EXECUTION_FAILED_{program.name}: {e}") from e
    if completed.returncode:
        raise OriginalZhangExecutionError(
            f"AUTHOR_ORIGINAL_STAGE_FAILED_{program.name}: exit={completed.returncode}; "+
            (completed.stderr+"\n"+completed.stdout)[-2200:])
    return {"program":program.name,"exit_code":completed.returncode,
            "stdout_tail":completed.stdout[-2000:],
            "stderr_tail":completed.stderr[-1000:]}


def generate(source_npz:Path,output_npz:Path,exe_root:Path,target_quads:int,
             hausdorff_bound:float,sheet_width:float,keep_features:bool,
             timeout_seconds:int,wine:list[str],
             independent_eigen_solver_dll:Path|None=None,
             final_coordinate_chart:str="before"):
    if target_quads<4 or hausdorff_bound<=0 or sheet_width<=0:
        raise OriginalZhangExecutionError("INVALID_NATIVE_TARGET_OR_BOUNDS")
    exe_root=exe_root.resolve()
    if not all((exe_root/name).is_file()
               for name in ("Straighten.exe","Segment.exe","QMSim.exe","WholePipeline.bat")):
        raise OriginalZhangExecutionError("GENUINE_AUTHOR_EXECUTABLES_NOT_FOUND")
    original_exe_sha256 = _verify_authors_executable_provenance(exe_root)
    with np.load(source_npz,allow_pickle=False) as npz:
        src={key:npz[key].copy() for key in npz.files}
    # Separate runtime dir even from original author package; never alter it.
    with tempfile.TemporaryDirectory(prefix="zhang2021_native_") as tdir:
        root=Path(tdir)/"ExeProgram"
        shutil.copytree(exe_root,root)
        independent_solver_sha256=None
        if independent_eigen_solver_dll is not None:
            independent_solver_sha256=_use_independent_sparse_backend(
                root,independent_eigen_solver_dll)
        for p in ("Source/Input","Source/before","Source/after",
                  "Result","Result/finalOut"):
            (root/p).mkdir(parents=True,exist_ok=True)
        basename="onestring_optcuts"
        inputfile=root/"Source/Input"/f"{basename}.obj"
        _obj_export(src,inputfile)
        before=f"Source/before/{basename}.obj"
        after=f"Source/after/{basename}_axis.obj"
        result=f"Result/{basename}.obj"
        final=f"Result/finalOut/{basename}_out.obj"
        stages=[]
        stages.append(_execute(root/"Straighten.exe",
            [str(inputfile.relative_to(root)),before,after],root,timeout_seconds,wine))
        if not (root/after).is_file():
            raise OriginalZhangExecutionError("STRAIGHTEN_OUTPUT_MISSING")
        stages.append(_execute(root/"Segment.exe",
            [after,"Result/",before,str(target_quads),str(hausdorff_bound)],
            root,timeout_seconds,wine))
        if not (root/result).is_file():
            raise OriginalZhangExecutionError("SEGMENT_OUTPUT_MISSING")
        stages.append(_execute(root/"QMSim.exe",
            [result,final,before,"0" if keep_features else "1",
             str(target_quads),str(hausdorff_bound),str(sheet_width)],
            root,timeout_seconds,wine))
        if not (root/final).is_file():
            raise OriginalZhangExecutionError("QMSIM_OUTPUT_MISSING")
        points,faces=_parse_author_final_obj(root/final)
        if final_coordinate_chart=="after":
            points=_map_after_to_before(root/before,root/after,points)
        elif final_coordinate_chart!="before":
            raise OriginalZhangExecutionError("UNKNOWN_FINAL_COORDINATE_CHART")
        output_npz.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output_npz,vertices_uv=points,faces=faces)
        manifest={
            "engine":"Zhang et al. 2021 ORIGINAL authors' binaries",
            "authors_executable_source":"ExeProgram.zip from official research page",
            "reference_backend_is_genuine":True,
            "authors_original_executable_sha256":original_exe_sha256,
            "target_quads":target_quads,"actual_quads":int(len(faces)),
            "hausdorff_bound":hausdorff_bound,
            "sheet_width":sheet_width,"keep_features":keep_features,
            "sparse_solver_backend":(
                "independent_eigen_sparse_lu"
                if independent_solver_sha256 is not None
                else "original_proprietary_pardiso_6"),
            "independent_sparse_solver_sha256":independent_solver_sha256,
            "final_coordinate_chart":final_coordinate_chart,
            "stages":stages,
        }
        output_npz.with_suffix(".provenance.json").write_text(
            json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
        return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--exe-root",type=Path,
        default=Path(os.environ.get("ONESTRING_ZHANG2021_EXE_ROOT","")))
    p.add_argument("--target-quads",type=int,default=35)
    p.add_argument("--hausdorff-bound",type=float,default=.01)
    p.add_argument("--sheet-width",type=float,default=.05)
    p.add_argument("--keep-features",action="store_true",default=True)
    p.add_argument("--disable-feature-preservation",action="store_false",dest="keep_features")
    p.add_argument("--timeout-seconds",type=int,default=900)
    p.add_argument("--wine-command",nargs="+",default=[])
    p.add_argument("--independent-eigen-solver-dll",type=Path)
    p.add_argument("--final-coordinate-chart",choices=("before","after"),
                   default="before",
                   help="Use 'after' ONLY if original author QMSim output is "
                        "in straightened chart, then invert exact original "
                        "before/after triangle mapping. Default is 'before'.")
    a=p.parse_args()
    if not str(a.exe_root) or str(a.exe_root)==".":
        p.error("Provide --exe-root pointing at the authors' genuine ExeProgram directory")
    manifest=generate(a.input,a.output,a.exe_root,a.target_quads,
                      a.hausdorff_bound,a.sheet_width,a.keep_features,
                      a.timeout_seconds,a.wine_command,
                      a.independent_eigen_solver_dll,
                      a.final_coordinate_chart)
    print(json.dumps(manifest,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
