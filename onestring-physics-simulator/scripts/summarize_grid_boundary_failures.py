#!/usr/bin/env python3
"""Compare failed area-grid UV continuation attempts without rerunning OptCuts.

Usage:
    python3 scripts/summarize_grid_boundary_failures.py

Reads logs/grid_boundary_diagnostics/*/diagnosis.json and creates a ranked
human-readable report, complete CSV and a shareable evidence ZIP containing
up to six representative full-resolution failures. Never deletes evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics
from zipfile import ZIP_DEFLATED, ZipFile


def _rows(root, limit):
    found = sorted(root.glob("*/diagnosis.json"),
                   key=lambda p: p.parent.name, reverse=True)
    data = []
    for p in found[:limit]:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as e:
            print(f"[GRID-DIAG-SUMMARY] ignored unreadable {p}: {e}")
            continue
        d["_folder"] = p.parent
        data.append(d)
    return data


def _number(d, key, default=0):
    value = d.get(key, default)
    return default if value is None else value


def _group_key(d):
    meta = d.get("metadata", {})
    return (str(meta.get("official_seam_start_index", "?")),
            str(meta.get("uv_rotation_degrees", "?")))


def _most_common_worst_triangles(rows):
    frequency = Counter()
    top1 = Counter()
    for row in rows:
        hot = row.get("ranked_worst_triangles", [])
        if hot:
            top1[int(hot[0]["triangle_id"])] += 1
        frequency.update(int(x["triangle_id"]) for x in hot[:10])
    return frequency.most_common(10), top1.most_common(10)


def _recommended_next_checks(rows):
    crossing = [
        d for d in rows if _number(
            d.get("proper_boundary_intersections", {})
             .get("target_at_failed_fraction", {}),
            "count", 0
        ) > 0
    ]
    reflex = [
        d for d in rows
        if float(_number(d, "reflex_hotspot_enrichment", 0)) >= 2.
        and float(_number(d,
            "worst_one_percent_near_reflex_within_two_tiles_fraction", 0)) >= .5
    ]
    common, top1 = _most_common_worst_triangles(rows)
    relevant = []
    if crossing:
        relevant.append(
            f"- **Boundary motion is suspect:** {len(crossing)}/{len(rows)} "
            "failed steps contain proper nonadjacent boundary crossings. "
            "Investigate non-linear boundary homotopy or landmark matching. "
            "This does not prove the final prescribed contour is impossible."
        )
    if reflex:
        relevant.append(
            f"- **Reflex-corner concentration:** {len(reflex)}/{len(rows)} "
            "failures have >=2x concentration of severely compressed UV "
            "triangles within two cells of 270-degree corners. "
            "Inspect corner-aware correspondence, envelope, then local seam "
            "cuts without changing the 50% mask."
        )
    if top1 and top1[0][1] >= max(2, len(rows)//2):
        relevant.append(
            f"- **Recurring hot triangle:** triangle {top1[0][0]} is the "
            f"single worst in {top1[0][1]}/{len(rows)} failures. "
            "If the same source mesh topology is used, inspect its local "
            "triangulation, Jacobian and proximity to target corners."
        )
    if not relevant:
        relevant.append(
            "- **Mixed/distributed evidence:** no dominant proper boundary "
            "crossing, reflex-corner cluster or recurrent worst triangle. "
            "Inspect the full UV arrays, initialization and interior area "
            "barrier before committing to a seam topology change."
        )
    relevant.append(
        "- **Do not relax the existing signed-area guard or accept "
        "overlapping/inverted triangles based only on these diagnostics.**"
    )
    return relevant


def summarize(root, *, last=48, max_bundles=6):
    root = Path(root)
    rows = _rows(root, last)
    if not rows:
        raise FileNotFoundError(
            f"No diagnosis.json snapshots under {root}; first rerun "
            "the experimental Ω mode with diagnostics enabled."
        )
    metrics = []
    for d in rows:
        meta = d.get("metadata", {})
        ints = d.get("proper_boundary_intersections", {})
        crossing = ints.get("target_at_failed_fraction", {}).get("count", None)
        metrics.append({
            "source_cut": meta.get("official_seam_start_index", "?"),
            "rotation_deg": meta.get("uv_rotation_degrees", "?"),
            "correspondence_attempt": meta.get("correspondence_attempt", "?"),
            "phase": meta.get("phase", "?"),
            "method": meta.get("method", "?"),
            "last_valid_fraction": d["boundary_fraction_last_accepted"],
            "rejected_fraction": d["boundary_fraction_rejected"],
            "rejected_min_signed_area_ratio": d[
                "rejected_min_signed_area_ratio"],
            "below_0_005_count": d[
                "rejected_below_existing_0_005_area_guard_count"],
            "reflex_hotspot_enrichment": d["reflex_hotspot_enrichment"],
            "boundary_crossing_count": crossing,
            "worst_triangle_id": d["ranked_worst_triangles"][0]["triangle_id"],
            "diagnostic_hypothesis": d["diagnostic_hypothesis"],
            "snapshot": str(d["_folder"]),
        })
    sorted_rows = sorted(rows, key=lambda d: (
        -float(d["boundary_fraction_last_accepted"]),
        int(_number(d, "rejected_below_existing_0_005_area_guard_count", 0)),
    ))
    high = sorted_rows[0]
    per_source = defaultdict(list)
    for d in rows:
        per_source[_group_key(d)].append(d)
    outline = [
        "# OptCuts → Grid boundary failure evidence",
        "",
        f"Analyzed **{len(rows)}** failed correspondences. "
        "All are numerical failures, not mathematical impossibility proofs.",
        "",
        "## Best attempted correspondence",
        "",
        f"- **Source cut / rotation:** {_group_key(high)[0]} / "
        f"{_group_key(high)[1]}°",
        f"- **Phase / matching:** {high['metadata'].get('phase','?')} / "
        f"{high['metadata'].get('method','?')}",
        "- **Last accepted fraction:** "
        f"{high['boundary_fraction_last_accepted']:.6f}",
        "- **Rejected minimum triangle signed-area ratio:** "
        f"{high['rejected_min_signed_area_ratio']:.8g}",
        f"- **Worst UV triangle ID:** "
        f"{high['ranked_worst_triangles'][0]['triangle_id']}",
        f"- **Failure snapshot:** `{high['_folder']}`",
        "",
        "## Across cut sources and UV orientations",
        "",
        "| Source cut | Rotation | Attempts | Best progress | "
        "Median reflex hotspot enrichment | Boundary crossings |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, entries in sorted(per_source.items()):
        enrich = statistics.median(float(_number(
            d, "reflex_hotspot_enrichment", 0)) for d in entries)
        crossing_count = sum(
            int(_number(d.get("proper_boundary_intersections", {})
                .get("target_at_failed_fraction", {}), "count", 0))
            for d in entries
        )
        outline.append(
            f"| {key[0]} | {key[1]}° | {len(entries)} | "
            f"{max(d['boundary_fraction_last_accepted'] for d in entries):.6f}"
            f" | {enrich:.3f}× | {crossing_count} |"
        )
    common, top1 = _most_common_worst_triangles(rows)
    outline += [
        "",
        "## Consistently compressed triangles",
        "",
        "Top-10 membership across failed attempts (same triangle IDs only "
        "comparable when source cut topology and vertex IDs are unchanged):",
        "",
        ", ".join(f"#{tri} ({n} attempts)" for tri,n in common) or "None",
        "",
        "## Next experiments justified by the evidence",
        "",
        *_recommended_next_checks(rows),
        "",
        "## How to inspect/reproduce",
        "",
        "Each snapshot contains `overview.svg`, `worst_triangle_zoom.svg`, "
        "`diagnosis.json` and `reproduce_uv.npz` (source, accepted, rejected "
        "and harmonic UVs, full M2D mask, correspondence, triangle face IDs).",
        "",
        "The accompanying ZIP includes this report, a full CSV, and up to "
        f"{max_bundles} representative failure snapshots, without "
        "changing any mesh or solver parameters.",
    ]
    summary = root/"analysis_summary.md"
    summary.write_text("\n".join(outline)+"\n", encoding="utf-8")
    csv_path = root/"comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    # Diversify ZIP across UV rotations where possible, then fill by progress.
    selected = []
    seen = set()
    for d in sorted_rows:
        key = _group_key(d)
        if key not in seen and len(selected) < max_bundles:
            selected.append(d)
            seen.add(key)
    for d in sorted_rows:
        if len(selected) >= max_bundles:
            break
        if d not in selected:
            selected.append(d)
    archive = root/"grid_failure_evidence.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED,
                 compresslevel=6) as z:
        z.write(summary, summary.name)
        z.write(csv_path, csv_path.name)
        for d in selected:
            p = d["_folder"]
            for name in ("diagnosis.json", "reproduce_uv.npz",
                         "overview.svg", "worst_triangle_zoom.svg"):
                child = p/name
                if child.exists():
                    z.write(child, arcname=f"snapshots/{p.name}/{name}")
    return {
        "attempts": len(rows),
        "summary": str(summary),
        "csv": str(csv_path),
        "zip": str(archive),
        "best_fraction": float(high["boundary_fraction_last_accepted"]),
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default="logs/grid_boundary_diagnostics",
        help="Folder created by experimental area-Grid Ω on failure")
    parser.add_argument("--last", type=int, default=48,
                        help="Analyze the latest N failed correspondences")
    parser.add_argument("--max-bundles", type=int, default=6,
                        help="Include up to N complete snapshots in ZIP")
    options=parser.parse_args(argv)
    if options.last < 1 or options.max_bundles < 1:
        parser.error("--last and --max-bundles must be >=1")
    try:
        result=summarize(options.root, last=options.last,
                         max_bundles=options.max_bundles)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print("[GRID-DIAG-SUMMARY] analyzed=", result["attempts"])
    print("[GRID-DIAG-SUMMARY] best_fraction=", result["best_fraction"])
    print("[GRID-DIAG-SUMMARY] report=", result["summary"])
    print("[GRID-DIAG-SUMMARY] table=", result["csv"])
    print("[GRID-DIAG-SUMMARY] share_this_zip=", result["zip"])


if __name__ == "__main__":
    main()
