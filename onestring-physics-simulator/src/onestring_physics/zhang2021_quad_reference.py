"""Runner for the authors' reference implementation of Zhang et al. (CAD 2021).

    C. Zhang, S. Chai, L. Liu, X.-M. Fu,
    "Quad meshing with coarse layouts for planar domains",
    Computer-Aided Design 140 (2021) 103084, doi:10.1016/j.cad.2021.103084.

This module does NOT re-implement the paper.  It drives the three programs of
the authors' published code (Fu lab share "Code_new.zip"), built from source
by ``scripts/setup_zhang2021_quad_official.py`` with build/portability
patches only (no algorithmic change; see
``third_party_patches/zhang2021_quad_coarse_layout/README.md``):

=========================  ==============================================
program (authors' name)    paper section
=========================  ==============================================
``z21_straighten``         3.2.1 boundary edge clustering + straightening
(``Straighten.exe``)       (bijective, scaffold-based, low distortion)
``z21_segment``            3.2.2 field-guided segmentation (dividing lines,
(``Segment.exe``)          ECVs), 3.3.1 template quad meshing, 3.3.2
                           T-junction elimination + poly-chord subdivision
                           with feature points and Hausdorff sampling
``z21_qmsim``              3.3.3 flip removal, modified poly-chord collapse
(``QMSim.exe``)            with width limit, SD geometric optimisation,
                           split/collapse to approach l_tar, Hausdorff bound
=========================  ==============================================

The command lines reproduce ``WholePipeline.bat`` of the authors'
executable package exactly::

    Straighten in.obj before.obj after_axis.obj
    Segment    after_axis.obj Result/ before.obj  <N_tar> <eps>
    QMSim      Result/before.obj final.obj before.obj 0 <N_tar> <eps> <l_collapse>

with the paper defaults N_tar = 4000, eps = 0.01 (x bbox diagonal),
l_collapse = 0.05 (x bbox diagonal), and "0" = keep feature points.

Input: a triangle mesh whose texture coordinates (``vt``) form the planar
domain (authors' readme).  ``v`` may be the planar domain itself or the 3D
surface the UV chart parameterizes (the authors' sample D1_00071_input.obj is
3D); the straightening distortion is measured w.r.t. ``v``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Sequence

import numpy as np

from .zhang2021_quad_metrics import (
    boundary_loops,
    paper_metrics,
    read_obj_polygons,
    scaled_jacobians,
)

PAPER_DOI = "10.1016/j.cad.2021.103084"
PAPER_DEFAULT_TARGET_QUADS = 4000
PAPER_DEFAULT_HAUSDORFF_RATIO = 0.01
PAPER_DEFAULT_COLLAPSE_WIDTH_RATIO = 0.05

_PROGRAMS = {
    "straighten": "z21_straighten",
    "segment": "z21_segment",
    "qmsim": "z21_qmsim",
}


class Zhang2021Error(RuntimeError):
    """Explicit failure of the reference pipeline (never a silent fallback)."""


class Zhang2021UnavailableError(Zhang2021Error):
    """The reference binaries are not built on this machine."""


@dataclass(frozen=True)
class Zhang2021Parameters:
    """Parameters of the authors' pipeline (WholePipeline.bat semantics)."""

    target_quads: int = PAPER_DEFAULT_TARGET_QUADS
    hausdorff_ratio: float = PAPER_DEFAULT_HAUSDORFF_RATIO
    collapse_width_ratio: float = PAPER_DEFAULT_COLLAPSE_WIDTH_RATIO
    keep_features: bool = True
    timeout_seconds: float = 3600.0
    keep_workdir: bool = False

    def validate(self) -> None:
        if int(self.target_quads) < 1:
            raise ValueError("ZHANG2021_TARGET_QUADS_MUST_BE_POSITIVE")
        if not (0.0 < float(self.hausdorff_ratio) < 1.0):
            raise ValueError("ZHANG2021_HAUSDORFF_RATIO_OUT_OF_RANGE")
        if not (0.0 < float(self.collapse_width_ratio) < 1.0):
            raise ValueError("ZHANG2021_COLLAPSE_WIDTH_RATIO_OUT_OF_RANGE")


@dataclass
class Zhang2021Result:
    quad_vertices: np.ndarray          # (n, 2) final quad mesh Q in input-UV frame
    quads: np.ndarray                  # (m, 4)
    initial_quad_vertices: np.ndarray  # Segment output (initial mesh, Sec. 3.3.2)
    initial_quads: np.ndarray
    before_vertices: np.ndarray        # pre-processed domain mesh M (UV positions)
    before_faces: list[list[int]]
    after_vertices: np.ndarray         # straightened domain M^s (same connectivity)
    metrics: dict[str, object] = field(default_factory=dict)
    logs: dict[str, str] = field(default_factory=dict)
    workdir: str | None = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_install_root() -> Path:
    return _project_root() / "third_party" / "Zhang2021_QuadCoarseLayout_official"


def _candidate_bin_dirs() -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("ONESTRING_ZHANG2021_BIN_DIR", "").strip()
    if env:
        out.append(Path(env).expanduser())
    root = default_install_root()
    out += [root / "build" / "onestring_port", root / "build" / "bin", root / "bin"]
    return out


def resolve_reference_binaries(bin_dir: str | Path | None = None) -> dict[str, Path]:
    """Locate z21_straighten / z21_segment / z21_qmsim (fail closed)."""
    dirs = [Path(bin_dir).expanduser()] if bin_dir else _candidate_bin_dirs()
    tried: list[str] = []
    for base in dirs:
        found: dict[str, Path] = {}
        for key, name in _PROGRAMS.items():
            for candidate in (base / name, base / key / name, base / f"{name}.exe", base / key / f"{name}.exe"):
                tried.append(str(candidate))
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    found[key] = candidate.resolve()
                    break
        if len(found) == len(_PROGRAMS):
            return found
    raise Zhang2021UnavailableError(
        "ZHANG2021_REFERENCE_BINARIES_NOT_FOUND: build them with "
        "'python3 scripts/setup_zhang2021_quad_official.py' (downloads the "
        "authors' Code_new.zip, applies build-only portability patches) or set "
        "ONESTRING_ZHANG2021_BIN_DIR. Tried: " + ", ".join(tried[:9]) + " ...")


def write_textured_obj(path: Path, xyz: np.ndarray, xyz_faces: np.ndarray,
                       uv: np.ndarray, uv_faces: np.ndarray) -> None:
    """``v`` = xyz, ``vt`` = uv, ``f v/vt`` (LF line endings, 17 digits)."""
    xyz = np.asarray(xyz, float)
    if xyz.shape[1] == 2:
        xyz = np.column_stack((xyz, np.zeros(len(xyz))))
    uv = np.asarray(uv, float)
    f3 = np.asarray(xyz_faces, int)
    f2 = np.asarray(uv_faces, int)
    if f3.shape != f2.shape or f3.ndim != 2 or f3.shape[1] != 3:
        raise ValueError("ZHANG2021_INPUT_FACE_PAIRING_INVALID")
    lines = ["# OneString -> Zhang et al. 2021 reference pipeline\n"]
    lines += [f"v {x:.17g} {y:.17g} {z:.17g}\n" for x, y, z in xyz[:, :3]]
    lines += [f"vt {u:.17g} {v:.17g}\n" for u, v in uv]
    lines += [f"f {a+1}/{p+1} {b+1}/{q+1} {c+1}/{r+1}\n" for (a, b, c), (p, q, r) in zip(f3, f2)]
    path.write_text("".join(lines), encoding="utf-8")


def _signed_areas(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    p = np.asarray(points, float)[np.asarray(tris, int)]
    return 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                  - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))


def _run(cmd: Sequence[str], cwd: Path, log: Path, timeout: float) -> tuple[int, float, str]:
    started = time.perf_counter()
    try:
        with log.open("w", encoding="utf-8", errors="replace") as fh:
            proc = subprocess.run(list(map(str, cmd)), cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT,
                                  timeout=timeout, check=False)
        code = proc.returncode
    except subprocess.TimeoutExpired as exc:
        raise Zhang2021Error(f"ZHANG2021_TIMEOUT: {Path(cmd[0]).name} exceeded {timeout:.0f}s") from exc
    text = log.read_text(encoding="utf-8", errors="replace")
    return code, time.perf_counter() - started, text


def run_reference_pipeline(
    xyz: np.ndarray,
    xyz_faces: np.ndarray,
    uv: np.ndarray,
    uv_faces: np.ndarray,
    params: Zhang2021Parameters | None = None,
    *,
    bin_dir: str | Path | None = None,
    workdir: str | Path | None = None,
) -> Zhang2021Result:
    """Run Straighten -> Segment -> QMSim exactly as WholePipeline.bat does.

    ``uv``/``uv_faces`` define the planar domain; ``xyz``/``xyz_faces`` the
    surface the straightening distortion is measured against (pass the UV
    itself for a purely planar domain).  The UV chart is passed with
    counter-clockwise triangles; a globally clockwise chart is mirrored
    (u -> -u) before and after, which is what the authors' pre-process
    ``flip_neg_charts`` would otherwise do implicitly.
    """
    params = params or Zhang2021Parameters()
    params.validate()
    binaries = resolve_reference_binaries(bin_dir)
    uv = np.asarray(uv, float)
    uv_faces = np.asarray(uv_faces, int)
    signed = _signed_areas(uv, uv_faces)
    if np.any(np.abs(signed) <= 0.0):
        raise Zhang2021Error("ZHANG2021_INPUT_UV_DEGENERATE_TRIANGLE")
    if np.any(signed > 0) and np.any(signed < 0):
        raise Zhang2021Error(
            "ZHANG2021_INPUT_UV_FOLDOVER: mixed triangle orientation "
            f"(positive={int(np.sum(signed > 0))}, negative={int(np.sum(signed < 0))})")
    mirrored = bool(np.all(signed < 0))
    uv_in = uv * np.array([-1.0, 1.0]) if mirrored else uv.copy()

    own_tmp = workdir is None
    work = Path(tempfile.mkdtemp(prefix="onestring_zhang2021_")) if own_tmp else Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    (work / "Result").mkdir(exist_ok=True)
    logs: dict[str, str] = {}
    timings: dict[str, float] = {}
    try:
        write_textured_obj(work / "input.obj", xyz, xyz_faces, uv_in, uv_faces)
        code, timings["straighten_seconds"], logs["straighten"] = _run(
            [binaries["straighten"], "input.obj", "before.obj", "after_axis.obj"],
            work, work / "straighten.log", params.timeout_seconds)
        if code != 0 or not (work / "after_axis.obj").is_file():
            tail = "\n".join(logs["straighten"].splitlines()[-12:])
            raise Zhang2021Error(
                "ZHANG2021_STRAIGHTENING_FAILED (Sec. 3.2.1; the authors' program would "
                f"continue with an unstraightened domain - refused): exit={code}\n{tail}")
        eps = f"{float(params.hausdorff_ratio):.17g}"
        n_tar = str(int(params.target_quads))
        code, timings["segment_seconds"], logs["segment"] = _run(
            [binaries["segment"], "after_axis.obj", "./Result/", "before.obj", n_tar, eps],
            work, work / "segment.log", params.timeout_seconds)
        initial = work / "Result" / "before.obj"
        if code != 0 or not initial.is_file():
            tail = "\n".join(logs["segment"].splitlines()[-12:])
            raise Zhang2021Error(f"ZHANG2021_SEGMENT_FAILED (Sec. 3.2.2-3.3.2): exit={code}\n{tail}")
        shutil.copy(initial, work / "initial_quad.obj")
        # QMSim deletes its input file (remove(argv[1])); hand it a copy.
        shutil.copy(initial, work / "qmsim_input.obj")
        code, timings["qmsim_seconds"], logs["qmsim"] = _run(
            [binaries["qmsim"], "qmsim_input.obj", "final_quad.obj", "before.obj",
             "0" if params.keep_features else "1", n_tar, eps,
             f"{float(params.collapse_width_ratio):.17g}"],
            work, work / "qmsim.log", params.timeout_seconds)
        if code != 0 or not (work / "final_quad.obj").is_file():
            tail = "\n".join(logs["qmsim"].splitlines()[-12:])
            raise Zhang2021Error(f"ZHANG2021_QMSIM_FAILED (Sec. 3.3.3): exit={code}\n{tail}")

        before_v, before_f, _, _ = read_obj_polygons(work / "before.obj")
        after_v, after_f, _, _ = read_obj_polygons(work / "after_axis.obj")
        init_v, init_f, _, _ = read_obj_polygons(work / "initial_quad.obj")
        final_v, final_f, _, _ = read_obj_polygons(work / "final_quad.obj")
        if after_f != before_f:
            raise Zhang2021Error("ZHANG2021_STRAIGHTENED_CONNECTIVITY_MISMATCH")
        if any(len(f) != 4 for f in final_f):
            bad = sum(len(f) != 4 for f in final_f)
            raise Zhang2021Error(f"ZHANG2021_OUTPUT_NOT_ALL_QUAD: non_quads={bad}")
        quads = np.asarray(final_f, int)
        qv = final_v[:, :2].copy()
        iv = init_v[:, :2].copy()
        bv = before_v[:, :2].copy()
        av = after_v[:, :2].copy()

        # The pre-process may split cut-bridging edges, appending vertices;
        # the first len(uv) "before" vertices must equal the input chart.
        n_uv = len(uv_in)
        if len(bv) < n_uv:
            raise Zhang2021Error("ZHANG2021_PREPROCESS_LOST_VERTICES")
        frame_error = float(np.max(np.abs(bv[:n_uv] - uv_in))) if n_uv else 0.0
        span = float(np.max(np.ptp(uv_in, axis=0)))
        if frame_error > 1e-9 * max(span, 1.0):
            raise Zhang2021Error(
                "ZHANG2021_PREPROCESS_CHANGED_CHART_FRAME: max |before - input uv| = "
                f"{frame_error:.3e} (mirror_neg_charts or a scale change was applied)")
        if mirrored:
            # Undo u -> -u.  Faces keep their index order, so all elements
            # recover the (clockwise) winding of the caller's source chart.
            for arr in (qv, iv, bv, av):
                arr[:, 0] *= -1.0
        init_quads = np.asarray(init_f, int) if all(len(f) == 4 for f in init_f) else None

        tri_faces = before_f
        m = paper_metrics(qv, quads, bv, tri_faces)
        sj = scaled_jacobians(qv, quads)
        metrics: dict[str, object] = {
            "zhang2021_paper_doi": PAPER_DOI,
            "zhang2021_engine": "authors' reference implementation (Straighten/Segment/QMSim), portable build",
            "zhang2021_binaries": {k: str(v) for k, v in binaries.items()},
            "zhang2021_target_quads": int(params.target_quads),
            "zhang2021_hausdorff_ratio": float(params.hausdorff_ratio),
            "zhang2021_collapse_width_ratio": float(params.collapse_width_ratio),
            "zhang2021_keep_features": bool(params.keep_features),
            "zhang2021_input_uv_mirrored_for_ccw": mirrored,
            "zhang2021_preprocess_added_vertices": int(len(bv) - n_uv),
            "zhang2021_domain_boundary_loops": int(len(boundary_loops(tri_faces))),
            "zhang2021_quad_boundary_loops": int(len(boundary_loops(quads))),
            "zhang2021_n_patches": int(m.n_patches),
            "zhang2021_n_quads": int(m.n_quads),
            "zhang2021_j_min": float(m.j_min),
            "zhang2021_j_avg": float(m.j_avg),
            "zhang2021_d_h": float(m.d_h),
            "zhang2021_d_h_percent_bbox": float(m.d_h_percent_bbox),
            "zhang2021_hausdorff_bound_satisfied": bool(m.d_h <= float(params.hausdorff_ratio) * m.bbox_diagonal * (1 + 1e-9)),
            "zhang2021_flipped_quads": int(np.sum(sj <= 0.0)),
            "zhang2021_initial_quads": int(len(init_f)),
            **timings,
        }
        return Zhang2021Result(
            quad_vertices=qv, quads=quads,
            initial_quad_vertices=iv,
            initial_quads=init_quads if init_quads is not None else np.zeros((0, 4), int),
            before_vertices=bv, before_faces=before_f, after_vertices=av,
            metrics=metrics, logs=logs,
            workdir=str(work) if (params.keep_workdir or not own_tmp) else None,
        )
    finally:
        if own_tmp and not params.keep_workdir:
            shutil.rmtree(work, ignore_errors=True)
