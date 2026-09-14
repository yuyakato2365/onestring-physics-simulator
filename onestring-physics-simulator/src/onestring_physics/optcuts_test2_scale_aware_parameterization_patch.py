"""Use the dedicated source-modified OptCuts binary for ``optcuts_test2``.

The dedicated build uses an unnormalized OneString scale-violation SUM on the
physical surface. The scale term is intentionally not area-normalized: every
violating triangle contributes directly.

Runtime activation is proven by a marker file written from the exact C++
physical-energy path. The test2 route fails closed if that file is not written;
it never silently accepts a plain-SD OptCuts run.
"""
from __future__ import annotations

from dataclasses import replace
import csv
import os
from pathlib import Path
from typing import Any

import numpy as np

from .optcuts_backend import OptCutsConfig, OptCutsUnavailableError, OptCutsError
from . import optcuts_backend
from . import optcuts_pipeline_patch as optcuts_pipeline


def _is_test2() -> bool:
    return os.environ.get("ONESTRING_OPTCUTS_TEST_VARIANT", "0").strip() == "2"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _onestring_binary() -> Path:
    root = _project_root()
    stamp = root / ".onestring_optcuts_binary"
    candidates: list[Path] = []
    if stamp.is_file():
        try:
            value = stamp.read_text(encoding="utf-8").strip()
            if value:
                candidates.append(Path(value).expanduser())
        except Exception:
            pass
    build = root / "third_party" / "OptCuts" / "build_onestring"
    candidates.extend([
        build / "OptCuts_onestring_runner",
        build / "OptCuts_bin",
        build / "OptCuts_bin.exe",
        build / "Release" / "OptCuts_bin",
        build / "Release" / "OptCuts_bin.exe",
    ])
    for path in candidates:
        if path.is_file():
            resolved = path.resolve()
            if resolved.name == "OptCuts_onestring_runner" and not os.access(resolved, os.X_OK):
                try:
                    resolved.chmod(resolved.stat().st_mode | 0o111)
                except OSError as exc:
                    raise OptCutsUnavailableError(
                        f"OneString OptCuts runner exists but is not executable and chmod failed: "
                        f"{resolved}: {exc}"
                    ) from exc
            if not os.access(resolved, os.X_OK):
                raise OptCutsUnavailableError(
                    f"OneString OptCuts executable is not executable: {resolved}. "
                    "Re-run `python3 scripts/enable_optcuts_scale_sum.py`."
                )
            return resolved
    raise OptCutsUnavailableError(
        "The source-modified OneString OptCuts binary is not built. Run:\n"
        "  python3 scripts/enable_optcuts_scale_sum.py\n"
        "This builds third_party/OptCuts/build_onestring/OptCuts_bin with the "
        "physical OneString scale objective force-enabled."
    )


def _hard_scale_audit(result: Any, bound: float) -> tuple[float, bool]:
    sigma2 = np.asarray(result.metrics.get("per_triangle_sigma2", []), dtype=float)
    sigma2 = sigma2[np.isfinite(sigma2) & (sigma2 > 1.0e-15)]
    if len(sigma2) == 0:
        return float("inf"), False
    scale_range = float(np.max(sigma2) / np.min(sigma2))
    return scale_range, bool(scale_range <= float(bound) + 1.0e-9)


def _prepare_scale_diagnostics() -> Path:
    path = _project_root() / "logs" / "optcuts_scale_objective.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    os.environ["ONESTRING_OPTCUTS_SCALE_DIAG_PATH"] = str(path)
    return path


def _prepare_activation_proof() -> Path:
    path = _project_root() / "logs" / "optcuts_scale_objective.active"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    os.environ["ONESTRING_OPTCUTS_ACTIVE_PATH"] = str(path)
    return path


def _read_diag_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _f(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def _summarize_scale_diagnostics(path: Path) -> dict[str, Any]:
    rows = _read_diag_rows(path)
    global_rows = [r for r in rows if r.get("scope") == "global"]
    local_rows = [r for r in rows if r.get("scope") == "local"]
    summary: dict[str, Any] = {
        "path": str(path),
        "rows": len(rows),
        "global_rows": len(global_rows),
        "local_rows": len(local_rows),
    }
    if not global_rows:
        print(
            "[OPTCUTS-SCALE-DIAG-WARNING] no global diagnostic rows were written; "
            f"path={path}"
        )
        summary["status"] = "missing"
        return summary

    first = global_rows[0]
    last = global_rows[-1]
    first_energy_ratio = _f(first, "scale_to_sd")
    last_energy_ratio = _f(last, "scale_to_sd")
    first_grad_ratio = _f(first, "scale_grad_to_sd_grad")
    last_grad_ratio = _f(last, "scale_grad_to_sd_grad")
    first_range = _f(first, "scale_range")
    last_range = _f(last, "scale_range")
    first_viol = int(float(first.get("violating", "0") or 0))
    last_viol = int(float(last.get("violating", "0") or 0))

    summary.update({
        "status": "ok",
        "first_energy_ratio": first_energy_ratio,
        "last_energy_ratio": last_energy_ratio,
        "first_grad_ratio": first_grad_ratio,
        "last_grad_ratio": last_grad_ratio,
        "first_range": first_range,
        "last_range": last_range,
        "first_violating": first_viol,
        "last_violating": last_viol,
        "first_reference_center": _f(first, "reference_center"),
        "last_reference_center": _f(last, "reference_center"),
    })

    print(
        "[OPTCUTS-SCALE-DIAG] "
        f"global_rows={len(global_rows)} local_samples={len(local_rows)} "
        f"energy_ratio(scale/SD)={first_energy_ratio:.6g}->{last_energy_ratio:.6g} "
        f"grad_ratio(scale/SD)={first_grad_ratio:.6g}->{last_grad_ratio:.6g} "
        f"range={first_range:.6g}->{last_range:.6g} "
        f"violating={first_viol}->{last_viol} path={path}"
    )

    peak_energy_ratio = max(
        (_f(r, "scale_to_sd") for r in global_rows), default=float("nan")
    )
    peak_grad_ratio = max(
        (_f(r, "scale_grad_to_sd_grad") for r in global_rows), default=float("nan")
    )
    summary["peak_energy_ratio"] = peak_energy_ratio
    summary["peak_grad_ratio"] = peak_grad_ratio
    if (
        np.isfinite(peak_energy_ratio)
        and np.isfinite(peak_grad_ratio)
        and peak_energy_ratio < 1.0e-4
        and peak_grad_ratio < 1.0e-4
    ):
        summary["status"] = "negligible"
        print(
            "[OPTCUTS-SCALE-DIAG-WARNING] scale term is numerically negligible "
            f"relative to SD: peak_energy_ratio={peak_energy_ratio:.6g} "
            f"peak_grad_ratio={peak_grad_ratio:.6g}."
        )
    elif np.isfinite(first_range) and np.isfinite(last_range) and last_range >= first_range - 1.0e-8:
        summary["status"] = "active_but_no_range_improvement"
        print(
            "[OPTCUTS-SCALE-DIAG-WARNING] scale term is active, but the logged "
            f"global optimization did not reduce scale range ({first_range:.6g} -> {last_range:.6g})."
        )
    else:
        print(
            "[OPTCUTS-SCALE-DIAG-OK] scale objective is active and the logged "
            "global optimization reduced the scale range."
        )
    return summary


def install_optcuts_test2_scale_aware_parameterization_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_test2_scale_aware_parameterization_installed", False):
        return

    original_run = optcuts_pipeline.run_official_optcuts
    if getattr(original_run, "_onestring_source_modified_dispatch", False):
        pipeline._onestring_test2_scale_aware_parameterization_installed = True
        return

    def run_dispatch(surface_vertices, surface_faces, config=None):
        cfg = config if config is not None else OptCutsConfig()
        if _is_test2():
            bound = float(os.environ.get("ONESTRING_OPTCUTS_SCALE_BOUND", "2.0"))
            weight = float(os.environ.get("ONESTRING_OPTCUTS_SCALE_WEIGHT", "1.0"))
            os.environ["ONESTRING_OPTCUTS_SCALE_BOUND"] = str(bound)
            os.environ["ONESTRING_OPTCUTS_SCALE_WEIGHT"] = str(weight)
            diag_path = _prepare_scale_diagnostics()
            active_path = _prepare_activation_proof()
            binary = _onestring_binary()

            os.environ["ONESTRING_OPTCUTS_BIN"] = str(binary)
            os.environ["ONESTRING_OPTCUTS_EXECUTABLE"] = str(binary)
            cfg = replace(cfg, executable=str(binary))

            # Several legacy OptCuts patches in this app can replace the backend
            # resolver at runtime.  That is why an explicit cfg.executable could
            # still resolve to OptCuts_official.  Test2 no longer asks that mutable
            # resolver which binary to use.  For this one call we pin the backend's
            # resolver itself to the already validated source-modified runner.
            original_resolver = optcuts_backend.resolve_optcuts_executable
            def _test2_exact_resolver(explicit=None):
                return binary
            optcuts_backend.resolve_optcuts_executable = _test2_exact_resolver
            try:
                print(
                    "[OPTCUTS-TEST2-SOURCE-MODIFIED] "
                    f"binary={binary} exact_backend_pin=true scale_bound={bound:g} "
                    f"scale_weight={weight:g} "
                    "scale_model=sum_residual_squared_no_area_normalization "
                    f"diag={diag_path} active_proof={active_path}"
                )
                print(
                    "[OPTCUTS-TEST2-DIRECT-BACKEND] resolver_bypassed=true "
                    f"executable={binary}"
                )
                result = optcuts_backend.run_official_optcuts(
                    surface_vertices, surface_faces, cfg
                )
            finally:
                optcuts_backend.resolve_optcuts_executable = original_resolver

            actual = Path(str(result.metrics.get("optcuts_executable", ""))).expanduser()
            if not actual.is_absolute():
                actual = actual.resolve()
            if actual.resolve() != binary.resolve():
                raise OptCutsError(
                    "OneString Test2 backend executed the wrong binary despite exact pin: "
                    f"expected={binary.resolve()} actual={actual.resolve()}"
                )

            if not active_path.is_file():
                raise OptCutsError(
                    "OneString OptCuts scale objective FAILED TO ACTIVATE at runtime. "
                    f"Expected activation proof file was not written: {active_path}. "
                    f"backend_executable={actual}. Re-run "
                    "`python3 scripts/enable_optcuts_scale_sum.py` and restart Streamlit."
                )
            try:
                activation_text = active_path.read_text(encoding="utf-8").strip()
            except Exception:
                activation_text = "active"
            print(
                "[OPTCUTS-ONESTRING-PHYSICAL-OBJECTIVE-CONFIRMED] "
                + activation_text.replace("\n", " ")
            )

            diag = _summarize_scale_diagnostics(diag_path)
            scale_range, hard_feasible = _hard_scale_audit(result, bound)
            result.metrics.update({
                "optcuts_internal_scale_factor_enabled": True,
                "optcuts_internal_scale_factor_model": (
                    "E_physical = lambda_SD * E_SD + weight * SUM(residual^2), "
                    "no area normalization; scale term is outside the OptCuts SD "
                    "multiplier; global scale center frozen per global Newton iteration "
                    "and shared by candidate-local relaxation; local OptCuts area "
                    "bookkeeping compensated; scaffold excluded; scale gradient "
                    "included in candidate discovery"
                ),
                "optcuts_internal_scale_factor_bound": bound,
                "optcuts_internal_scale_factor_weight": weight,
                "optcuts_source_modified_binary": str(binary),
                "optcuts_runtime_source_patch_used": True,
                "optcuts_outer_multi_run_selector_used": False,
                "optcuts_cpp_activation_marker_seen": True,
                "optcuts_cpp_activation_proof_path": str(active_path),
                "optcuts_cpp_activation_proof": activation_text,
                "optcuts_internal_scale_factor_final_range": scale_range,
                "optcuts_internal_scale_factor_final_hard_feasible": hard_feasible,
                "optcuts_scale_objective_diagnostics": diag,
                "optcuts_scale_objective_diagnostics_path": str(diag_path),
            })
            print(
                "[OPTCUTS-INTERNAL-SCALE-FINAL] "
                f"range={scale_range:.9g} bound={bound:.9g} hard_feasible={hard_feasible} "
                f"diag_status={diag.get('status', 'unknown')}"
            )
            return result
        return original_run(surface_vertices, surface_faces, cfg)

    run_dispatch._onestring_source_modified_dispatch = True
    run_dispatch._onestring_original_run_official_optcuts = original_run
    optcuts_pipeline.run_official_optcuts = run_dispatch

    pipeline._onestring_test2_scale_aware_parameterization_installed = True
    print(
        "[OPTCUTS-TEST2-SOURCE-MODIFIED-ROUTE] installed; test2 uses a dedicated "
        "OptCuts binary with unnormalized scale-violation SUM objective, hard "
        "runtime activation proof, exact backend binary pinning, shared global/local "
        "center, scale-aware candidate discovery, scaffold exclusion, and automatic "
        "contribution diagnostics"
    )


__all__ = ["install_optcuts_test2_scale_aware_parameterization_patch"]
