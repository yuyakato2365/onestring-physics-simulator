#!/usr/bin/env python3
"""Summarize the latest OneString OptCuts requirement diagnostic run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any


def _default_log() -> Path:
    return Path(__file__).resolve().parents[1] / "logs" / "optcuts_requirement.jsonl"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=_default_log())
    args = parser.parse_args()
    path = args.log.expanduser().resolve()
    if not path.is_file():
        print(f"No diagnostic log yet: {path}")
        return 1

    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except Exception:
            continue
        if isinstance(item, dict) and item.get("run_id") not in {None, "unscoped"}:
            records.append(item)
    if not records:
        print(f"No scoped OptCuts run found in: {path}")
        return 1

    latest_run = records[-1]["run_id"]
    run = [r for r in records if r.get("run_id") == latest_run]
    by_event: dict[str, dict[str, Any]] = {}
    for r in run:
        by_event[str(r.get("event"))] = r

    print(f"run_id: {latest_run}")
    print(f"log: {path}")
    print("required sequence: Official OptCuts -> seam -> Omega straight -> grid align -> M2D")
    print()

    start = by_event.get("official_optcuts_start")
    finish = by_event.get("official_optcuts_finish")
    error = by_event.get("official_optcuts_error")
    heartbeat = by_event.get("official_optcuts_heartbeat")
    if finish:
        print(f"[1] Official OptCuts: PASS ({_fmt(finish.get('elapsed_seconds'))} s)")
    elif error:
        print(f"[1] Official OptCuts: FAIL ({error.get('exception_type')}: {error.get('exception_message')})")
    elif start:
        elapsed = heartbeat.get("elapsed_seconds") if heartbeat else time.time() - float(start.get("ts_unix", time.time()))
        print(f"[1] Official OptCuts: RUNNING ({_fmt(float(elapsed))} s elapsed)")
    else:
        print("[1] Official OptCuts: NOT OBSERVED")

    omega = by_event.get("omega_requirement_verdict")
    if omega:
        diag = dict(omega.get("diagnostics") or {})
        result = "PASS" if omega.get("passed") else "FAIL"
        print(
            f"[2] Omega seam straight + grid align: {result} "
            f"(line {_fmt(float(diag.get('max_final_chain_line_error', float('nan'))))} / "
            f"{_fmt(float(diag.get('line_tolerance', float('nan'))))}, "
            f"grid {_fmt(float(diag.get('max_final_grid_alignment_error', float('nan'))))} / "
            f"{_fmt(float(diag.get('grid_alignment_tolerance', float('nan'))))})"
        )
    else:
        print("[2] Omega seam straight + grid align: NOT REACHED")

    m2d = by_event.get("m2d_requirement_verdict")
    if m2d:
        result = "PASS" if m2d.get("passed") else "FAIL"
        print(
            f"[3] M2D straight grid seam: {result} "
            f"(paths={m2d.get('grid_path_count', '?')}, "
            f"nonstraight={m2d.get('nonstraight_grid_path_count', '?')})"
        )
    else:
        print("[3] M2D straight grid seam: NOT REACHED")

    complete = bool(finish and omega and omega.get("passed") and m2d and m2d.get("passed"))
    print()
    if complete:
        print("FINAL REQUIREMENT VERDICT: PASS")
        return 0
    if error or (omega and not omega.get("passed")) or (m2d and not m2d.get("passed")):
        print("FINAL REQUIREMENT VERDICT: FAIL")
        return 2
    print("FINAL REQUIREMENT VERDICT: INCOMPLETE / STILL RUNNING")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
