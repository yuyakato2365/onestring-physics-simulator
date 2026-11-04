"""2026-11-04: readable Markdown summary of a diagnostic trace (docs/TRACE_LOG_FORMAT.md).
usage: python3 trace_summary.py TRACE.jsonl > summary.md"""
import json, sys
from collections import OrderedDict

R = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
out = [f"# Trace summary: {sys.argv[1]}", "", f"{len(R)} records, {R[0]['wall']} – {R[-1]['wall']}" if R else "empty", ""]
run = [r for r in R if r["stage"] == "run"]
for r in run:
    if r["event"] == "start":
        out += ["## Run", "", f"- mesh: `{r.get('mesh')}`", f"- out: `{r.get('out')}`",
                "- switches: " + ", ".join(f"`{k}={v}`" for k, v in sorted(r.get("env", {}).items())), ""]
    elif r["event"] == "failed":
        out += ["## Run FAILED", "", f"{r.get('error_type')}: {r.get('error', '')[:2000]}", "", "```", r.get("traceback", "")[-4000:], "```", ""]
    elif r["event"] == "done":
        out += [f"Run finished in {r.get('seconds', 0):.0f} s.", ""]
prog = OrderedDict()
for r in R:
    if r["stage"] == "progress":
        prog.setdefault(r["event"], []).append(r)
if prog:
    out += ["## Progress (first / last message of each stage)", "", "| stage | messages | first | last |", "|---|---|---|---|"]
    for k, v in prog.items():
        f = v[0].get("detail", "").replace("|", "/")[:120]; l = v[-1].get("detail", "").replace("|", "/")[:160]
        out.append(f"| {k} | {len(v)} | t={v[0]['t']:.0f}s {f} | t={v[-1]['t']:.0f}s {l} |")
    out.append("")
ms = [r for r in R if r["stage"] == "optcuts.multistart"]
if ms:
    out += ["## OptCuts multi-start", "", "| budget | start | finished | frames | loops | feasible | distortion | seam cost | error |", "|---|---|---|---|---|---|---|---|---|"]
    for r in ms:
        if r["event"] == "branch":
            out.append(f"| {r['budget']} | {r['start']} | {r['finished']} | {r['frames']} | {r['last_boundary_loops']} | {r['feasible']} | "
                       f"{r['distortion']:.4g} | {r['seam_cost']:.4g} | {(r.get('error') or '')[:80]} |")
    for r in ms:
        if r["event"] in ("final", "final_refused"):
            out += ["", f"**{r['event']}**: " + json.dumps({k: v for k, v in r.items() if k not in ('t', 'wall', 'pid', 'stage', 'event')}, ensure_ascii=False)[:600]]
    out.append("")
gm = [r for r in R if r["stage"] == "grid.message"]
if gm:
    out += ["## Grid planner messages", ""] + [f"- t={r['t']:.0f}s {r['text'][:600]}" for r in gm] + [""]
ld = [r for r in R if r["stage"] == "grid.logical_domain"]
for r in ld:
    if r["event"] == "start":
        kinds = r.get("feature_kind", [])
        out += [f"## Logical polygon search, min side ratio {r['min_ratio']}", "",
                f"- features m = {r['features']} ({sum(k == 'tip' for k in kinds)} seam tips), max rounds {r['max_rounds']}, zipper {r['zipper']}, "
                f"turn reference {r['turn_reference']}, seam pairs {len(r.get('seam_pairs') or [])}",
                f"- side targets L (panels): min {min(r['side_target_panels']):.2f}, max {max(r['side_target_panels']):.2f}, "
                f"sum {sum(r['side_target_panels']):.1f}; sides with n_lo >= 2: {sum(1 for x in r['side_min_panels'] if x >= 2)}", "",
                "| round | lengths found | status | zipper check | separated pairs | MILP |", "|---|---|---|---|---|---|"]
    elif r["event"] == "attempt":
        out.append(f"| {r.get('round')} | {r.get('lengths') is not None} | {str(r.get('status', ''))[:90]} | {str(r.get('zipper_check', ''))[:90]} | "
                   f"{r.get('separated_pairs', '')} | {str(r.get('milp_status', ''))[:50]} |")
    elif r["event"] == "result":
        out += ["", f"**result**: found {r['found']} after {r['rounds']} rounds", ""]
gp = [r for r in R if r["stage"] == "grid.plan" and r["event"] == "failed"]
for r in gp:
    out += ["## Grid planner FAILED", "", r.get("error", "")[:3000], ""]
for st in ("connect", "resolve"):
    rs = [r for r in R if r["stage"] == st]
    if rs:
        out += [f"## {st}", ""] + ["- " + r["event"] + ": " + json.dumps({k: v for k, v in r.items() if k not in ('t', 'wall', 'pid', 'stage', 'event', 'path')}, ensure_ascii=False)[:500] for r in rs] + [""]
print("\n".join(out))
