#!/usr/bin/env python3
"""Fetch, patch and build the authors' reference implementation of

    C. Zhang, S. Chai, L. Liu, X.-M. Fu,
    "Quad meshing with coarse layouts for planar domains",
    Computer-Aided Design 140 (2021) 103084, doi:10.1016/j.cad.2021.103084.

The authors' code is NOT redistributed by this repository (their archive has
no license file).  This script

1. downloads ``Code_new.zip`` from the authors' public share linked on
   Xiao-Ming Fu's publication page (or uses ``--archive``),
2. verifies its SHA-256,
3. normalises text encodings exactly as recorded in the patch README
   (UTF-16 / UTF-8-BOM -> UTF-8, CRLF -> LF, C/C++ sources and .txt only),
4. applies ``third_party_patches/zhang2021_quad_coarse_layout/onestring_port.patch``
   (build-system and platform portability only; see the README next to it),
5. builds ``z21_straighten``, ``z21_segment``, ``z21_qmsim`` with CMake,
6. runs a smoke test through ``onestring_physics.zhang2021_quad_reference``.

Requirements: CMake >= 3.16, a C++17 compiler (GCC or Clang), Eigen 3.3+,
CGAL 5+ (with Boost/GMP/MPFR), OpenMP (recommended), git or patch.

macOS (Homebrew):  brew install cmake eigen cgal libomp
                   python3 scripts/setup_zhang2021_quad_official.py \\
                       --cmake-arg=-DOpenMP_ROOT=$(brew --prefix libomp)
Ubuntu/Debian:     sudo apt install cmake g++ libeigen3-dev libcgal-dev
                   python3 scripts/setup_zhang2021_quad_official.py
Windows:           use WSL (Ubuntu) with the Linux instructions.
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
import sys
import tempfile
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR = ROOT / "third_party_patches" / "zhang2021_quad_coarse_layout"
PATCH = PATCH_DIR / "onestring_port.patch"
DEFAULT_INSTALL_ROOT = ROOT / "third_party" / "Zhang2021_QuadCoarseLayout_official"

SHARE_URL = "https://rec.ustc.edu.cn/share/8081ab10-dd29-11eb-b5e1-5fc36251b03b"
SHARE_NUMBER = "8081ab10-dd29-11eb-b5e1-5fc36251b03b"
REC_API = "https://recapi.ustc.edu.cn/api/v2"
ARCHIVE_NAME = "Code_new.zip"
ARCHIVE_SHA256 = "49b70af5e6272c96aec2c3cb3aec84709477c9e51d033e24924218709917f4b3"
ARCHIVE_TOP = "Code/"
NORMALISED_EXTENSIONS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".txt"}
PROGRAMS = ("straighten/z21_straighten", "segment/z21_segment", "qmsim/z21_qmsim")


def log(msg: str) -> None:
    print(f"[zhang2021-setup] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------- download
def _rec_post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(REC_API + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8-sig"))


def _find_share_resource(name: str, folder: str | None = None):
    data = _rec_post("/share/target/resource/list", {
        "share_number": SHARE_NUMBER, "share_resource_number": folder,
        "is_rec": "false", "share_constraint": {"password": None}})
    entity = data.get("entity")
    items = entity
    if isinstance(entity, dict):
        items = next((entity[k] for k in ("datas", "data", "list", "rows") if k in entity), [entity])
    for item in items or []:
        label = str(item.get("name", ""))
        ext = item.get("file_ext")
        full = label if (not ext or label.endswith("." + ext)) else f"{label}.{ext}"
        if item.get("type") == "folder":
            found = _find_share_resource(name, item.get("number"))
            if found:
                return found
        elif full == name:
            return item
    return None


def download_archive(dest: Path) -> Path:
    log(f"downloading {ARCHIVE_NAME} from the authors' share {SHARE_URL}")
    item = _find_share_resource(ARCHIVE_NAME)
    if not item:
        raise RuntimeError(f"{ARCHIVE_NAME} not found in the share")
    data = _rec_post("/share/download", {"share_number": SHARE_NUMBER,
                                          "share_constraint": {"password": None},
                                          "share_resources_list": [item["number"]]})
    url = data["entity"][item["number"]] + "&download=download"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    last = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=600) as r, tmp.open("wb") as fh:
                shutil.copyfileobj(r, fh, 1 << 20)
            tmp.replace(dest)
            return dest
        except Exception as exc:  # noqa: BLE001 - retried, then reported
            last = exc
            log(f"download attempt {attempt + 1} failed: {exc}; retrying")
            time.sleep(3 + 3 * attempt)
    raise RuntimeError(f"download failed: {last}")


# ------------------------------------------------------- extract/normalise
def normalise(data: bytes) -> bytes:
    if data[:2] == b"\xff\xfe":
        data = data[2:].decode("utf-16-le").encode("utf-8")
    elif data[:2] == b"\xfe\xff":
        data = data[2:].decode("utf-16-be").encode("utf-8")
    elif data[:3] == b"\xef\xbb\xbf":
        data = data[3:]
    return data.replace(b"\r\n", b"\n")


def extract(archive: Path, src: Path) -> int:
    count = 0
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            name = info.filename
            if not name.startswith(ARCHIVE_TOP) or name.endswith("/"):
                continue
            rel = Path(name[len(ARCHIVE_TOP):])
            if rel.is_absolute() or ".." in rel.parts:
                raise RuntimeError(f"unsafe path in archive: {name}")
            out = src / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            data = z.read(info)
            if rel.suffix.lower() in NORMALISED_EXTENSIONS:
                data = normalise(data)
            out.write_bytes(data)
            count += 1
    return count


def apply_patch(src: Path) -> str:
    env = dict(os.environ)
    if shutil.which("git"):
        cmd = ["git", "apply", "--whitespace=nowarn", "-p1", str(PATCH)]
        # src/ usually lies inside the OneString checkout: stop git from
        # discovering that repository (it would then silently skip paths).
        env["GIT_CEILING_DIRECTORIES"] = str(src.parent)
    elif shutil.which("patch"):
        cmd = ["patch", "-p1", "--forward", "-i", str(PATCH)]
    else:
        raise RuntimeError("neither git nor patch is available")
    subprocess.run(cmd, cwd=src, check=True, env=env)
    for required in ("CMakeLists.txt", "onestring_port/include/z21_compat.h",
                     "onestring_port/straighten/headless_main.cpp"):
        if not (src / required).is_file():
            raise RuntimeError(f"patch did not create {required}")
    return " ".join(cmd[:2])


# ----------------------------------------------------------------- build
def build(src: Path, build_dir: Path, build_type: str, jobs: int, cmake_args: list[str]) -> dict:
    build_dir.mkdir(parents=True, exist_ok=True)
    configure = ["cmake", "-S", str(src), "-B", str(build_dir), f"-DCMAKE_BUILD_TYPE={build_type}", *cmake_args]
    log("configure: " + " ".join(configure))
    proc = subprocess.run(configure, capture_output=True, text=True)
    (build_dir / "onestring_configure.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:] + proc.stderr[-4000:])
        raise RuntimeError("cmake configure failed (see onestring_configure.log)")
    openmp = "OpenMP not found" not in (proc.stdout + proc.stderr)
    cmd = ["cmake", "--build", str(build_dir), "--parallel", str(jobs)]
    log("build: " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (build_dir / "onestring_build.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-6000:] + proc.stderr[-6000:])
        raise RuntimeError("build failed (see onestring_build.log)")
    out = build_dir / "onestring_port"
    for prog in PROGRAMS:
        if not (out / prog).is_file():
            raise RuntimeError(f"missing program after build: {out / prog}")
    return {"bin_dir": str(out), "openmp_for_segment": bool(openmp), "configure": configure}


# ------------------------------------------------------------ smoke test
def smoke_test(bin_dir: Path) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    from onestring_physics.zhang2021_quad_reference import Zhang2021Parameters, run_reference_pipeline

    nx, ny = 40, 20
    xs, ys = np.meshgrid(np.linspace(0.0, 2.0, nx + 1), np.linspace(0.0, 1.0, ny + 1))
    uv = np.c_[xs.ravel(), ys.ravel()]
    tris = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            b, c, d = a + 1, a + nx + 2, a + nx + 1
            tris += [[a, b, c], [a, c, d]]
    tris = np.asarray(tris)
    with tempfile.TemporaryDirectory(prefix="z21_smoke_") as tmp:
        res = run_reference_pipeline(np.c_[uv, np.zeros(len(uv))], tris, uv, tris,
                                     Zhang2021Parameters(target_quads=200, hausdorff_ratio=0.01),
                                     bin_dir=bin_dir, workdir=tmp)
    m = res.metrics
    ok = (m["zhang2021_n_quads"] > 0 and m["zhang2021_flipped_quads"] == 0
          and m["zhang2021_quad_boundary_loops"] == 1 and m["zhang2021_n_patches"] == 1)
    report = {k: m[k] for k in ("zhang2021_n_quads", "zhang2021_n_patches", "zhang2021_j_min",
                                "zhang2021_d_h_percent_bbox", "zhang2021_flipped_quads")}
    report["passed"] = bool(ok)
    if not ok:
        raise RuntimeError(f"smoke test failed (a 2x1 rectangle must give one patch): {report}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, help=f"local copy of the authors' {ARCHIVE_NAME}")
    ap.add_argument("--install-root", type=Path, default=DEFAULT_INSTALL_ROOT)
    ap.add_argument("--build-type", default="Release")
    ap.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--cmake-arg", action="append", default=[], help="extra CMake argument (repeatable)")
    ap.add_argument("--force", action="store_true", help="delete an existing install root first")
    ap.add_argument("--skip-smoke-test", action="store_true")
    args = ap.parse_args()

    root = args.install_root.resolve()
    if root.exists():
        if not args.force:
            log(f"{root} exists; use --force to rebuild from scratch")
            return 2
        shutil.rmtree(root)
    root.mkdir(parents=True)
    archive = args.archive.resolve() if args.archive else root / "download" / ARCHIVE_NAME
    if not args.archive:
        try:
            download_archive(archive)
        except Exception as exc:  # noqa: BLE001
            log(f"automatic download failed: {exc}")
            log(f"download {ARCHIVE_NAME} manually from {SHARE_URL} and rerun with --archive PATH")
            return 3
    digest = sha256(archive)
    if digest != ARCHIVE_SHA256:
        log(f"SHA-256 mismatch for {archive}: {digest} (expected {ARCHIVE_SHA256}); refusing to patch")
        return 4
    src = root / "src"
    n = extract(archive, src)
    log(f"extracted {n} files to {src} (C/C++/.txt encodings normalised)")
    tool = apply_patch(src)
    log(f"applied {PATCH.name} with {tool}")
    info = build(src, root / "build", args.build_type, args.jobs, list(args.cmake_arg))
    manifest = {
        "paper": "Zhang, Chai, Liu, Fu, Quad meshing with coarse layouts for planar domains, CAD 140 (2021) 103084",
        "source_share": SHARE_URL,
        "archive_sha256": digest,
        "patch_sha256": sha256(PATCH),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        **info,
    }
    if not args.skip_smoke_test:
        manifest["smoke_test"] = smoke_test(Path(info["bin_dir"]))
        log(f"smoke test passed: {manifest['smoke_test']}")
    (root / "ONESTRING_ZHANG2021_SETUP.json").write_text(json.dumps(manifest, indent=2))
    log(f"done. Programs in {info['bin_dir']} (found automatically by the OneString runner)")
    if not info["openmp_for_segment"]:
        log("WARNING: Segment was built without OpenMP (see onestring_configure.log)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
