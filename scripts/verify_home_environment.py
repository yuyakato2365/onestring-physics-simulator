from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXPECTED_BASE = ROOT / "src_backup_before_sideface_contact" / "onestring_physics" / "onestring_pipeline.py"
REQUIRED_FILES = [
    ROOT / "app.py",
    ROOT / "app_backup_before_mitered_t3d.py",
    SRC / "onestring_physics" / "onestring_pipeline.py",
    SRC / "onestring_physics" / "visualization.py",
    EXPECTED_BASE,
    ROOT / "requirements-local-cu128-lock.txt",
    ROOT / "requirements-gpu-cu128.txt",
    ROOT / "requirements.txt",
    ROOT / "pyproject.toml",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a cloned OneString runtime and Python environment.")
    parser.add_argument("--require-cuda", action="store_true", help="Fail when the CUDA 12.8 reference backend is unavailable.")
    args = parser.parse_args()

    missing = [str(path.relative_to(ROOT)) for path in REQUIRED_FILES if not path.is_file()]
    sys.path.insert(0, str(SRC))
    import onestring_physics.onestring_pipeline as pipeline

    runtime_wrapper = Path(pipeline.__file__).resolve()
    runtime_base = Path(pipeline.SIDEFACE_CONTACT_PATCH_ORIGINAL_PATH).resolve()
    override_source = Path(inspect.getsourcefile(pipeline._build_surface_parameterization) or "").resolve()

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        torch_version = str(torch.__version__)
        torch_cuda_runtime = str(torch.version.cuda)
        gpu_name = str(torch.cuda.get_device_name(0)) if cuda_available else "none"
    except Exception as exc:
        cuda_available = False
        torch_version = f"error:{type(exc).__name__}"
        torch_cuda_runtime = "none"
        gpu_name = "none"

    report = {
        "repository_root": str(ROOT),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "runtime_pipeline_wrapper": str(runtime_wrapper),
        "runtime_pipeline_base": str(runtime_base),
        "runtime_override_source": str(override_source),
        "expected_pipeline_base": str(EXPECTED_BASE.resolve()),
        "runtime_base_matches_expected": runtime_base == EXPECTED_BASE.resolve(),
        "runtime_override_comes_from_wrapper": override_source == runtime_wrapper,
        "required_files_missing": missing,
        "required_file_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in REQUIRED_FILES
            if path.is_file()
        },
        "torch_version": torch_version,
        "torch_cuda_runtime": torch_cuda_runtime,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "versions": {
            name: _version(name)
            for name in ("numpy", "scipy", "streamlit", "plotly", "trimesh", "pytest")
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    failed = bool(missing) or runtime_base != EXPECTED_BASE.resolve() or override_source != runtime_wrapper
    if args.require_cuda:
        failed = failed or not cuda_available or torch_cuda_runtime != "12.8" or "+cu128" not in torch_version
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
