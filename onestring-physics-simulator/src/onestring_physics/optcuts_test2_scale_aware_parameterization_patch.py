"""Use the dedicated source-modified OptCuts binary for ``optcuts_test2``.

The dedicated build uses

    E_geom = E_SD + scale_weight * SUM_i residual_i^2

for the physical surface. The OneString scale term is intentionally not
area-normalized: every violating triangle contributes directly. The global UV
optimizer refreshes a global scale-band center once per Newton iteration and
freezes it through that iteration; candidate-local optimizers reuse the same
reference. Candidate discovery also receives the scale-gradient contribution.

OptCuts' local SD score bookkeeping rescales by local/global surface area. The
C++ patch compensates this internally so the OneString scale SUM remains an
unnormalized sum in candidate-local comparisons. Scaffold/air-mesh evaluations
are excluded from the OneString scale term.

Each test2 run writes ``logs/optcuts_scale_objective.csv`` and prints a summary
comparing SD vs scale energy/gradient contributions.
"""
from __future__ import annotations

from dataclasses import replace
import csv
import os
from pathlib import Path
from typing import Any

import numpy as np

from .optcuts_backend import OptCutsConfig, OptCutsUnavailableError
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
        build / "OptCuts_bin",
        build / "OptCuts_bin.exe",
        build / "Release" / "OptCuts_bin",
        build / "Release" / "OptCuts_bin.exe",
    ])
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise OptCutsUnavailableError(
        "The source-modified OneString OptCuts binary is not built. Run:\n"
        "  python3 scripts/build_optcuts_onestring.py\n"
        "This applies the repository-tracked OptCuts source change and builds "
        "third_party/OptCuts/build_onestring/OptCuts_bin."
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
            "[OPTCUTS-SCALE-DIAG-WARNING] scale term is present, but the logged "
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
            # The old area-averaged objective needed a large multiplier. The new
            # objective is an unnormalized SUM, so 1.0 is already much stronger.
            weight = float(os.environ.get("ONESTRING_OPTCUTS_SCALE_WEIGHT", "1.0"))
            os.environ["ONESTRING_OPTCUTS_SCALE_BOUND"] = str(bound)
            os.environ["ONESTRING_OPTCUTS_SCALE_WEIGHT"] = str(weight)
            diag_path = _prepare_scale_diagnostics()
            binary = _onestring_binary()
            cfg = replace(cfg, executable=str(binary))
            print(
                "[OPTCUTS-TEST2-SOURCE-MODIFIED] "
                f"binary={binary} scale_bound={bound:g} scale_weight={weight:g} "
                "scale_model=sum_residual_squared_no_area_normalization "
                f"diag={diag_path}"
            )
            result = original_run(surface_vertices, surface_faces, cfg)

            stderr_tail = str(result.metrics.get("optcuts_stderr_tail", ""))
            cpp_marker = next(
                (line for line in stderr_tail.splitlines()
                 if "[OPTCUTS-SCALE-CPP-ACTIVE]" in line),
                "",
            )
            if cpp_marker:
                print(cpp_marker)
                print("[OPTCUTS-SCALE-CPP-CONFIRMED] patched C++ objective executed")
            else:
                print(
                    "[OPTCUTS-SCALE-CPP-WARNING] C++ activation marker was not found "
                    "in OptCuts stderr tail"
                )

            diag = _summarize_scale_diagnostics(diag_path)
            scale_range, hard_feasible = _hard_scale_audit(result, bound)
            result.metrics.update({
                "optcuts_internal_scale_factor_enabled": True,
                "optcuts_internal_scale_factor_model": (
                    "E_geom = E_SD + weight * SUM(residual^2), no area normalization; "
                    "global scale center frozen per global Newton iteration and shared "
                    "by candidate-local relaxation; local OptCuts area bookkeeping "
                    "compensated; scaffold excluded; scale gradient included in "
                    "candidate discovery"
                ),
                "optcuts_internal_scale_factor_bound": bound,
                "optcuts_internal_scale_factor_weight": weight,
                "optcuts_source_modified_binary": str(binary),
                "optcuts_runtime_source_patch_used": False,
                "optcuts_outer_multi_run_selector_used": False,
                "optcuts_cpp_activation_marker_seen": bool(cpp_marker),
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
        "OptCuts binary with unnormalized scale-violation SUM objective, shared "
        "global/local center, scale-aware candidate discovery, scaffold exclusion, "
        "and automatic contribution diagnostics"
    )


__all__ = ["install_optcuts_test2_scale_aware_parameterization_patch"]
