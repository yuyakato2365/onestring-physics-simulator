# Diagnostic trace log (2026-11-04)

Purpose: every long run leaves a machine-readable record of what each stage saw and decided, so that a person or
another AI can find out *why* a run failed or produced a poor result without rerunning it.

## Where it is written

| Run | Trace directory (default) |
|---|---|
| `tools/optcuts_soft_fit_20261104/run_shape.py MESH OUT` (pipeline S → … → T2D) | `OUT/trace/` |
| `tools/optcuts_soft_fit_20261104/optcuts_multistart.py MESH OUT` (seam start search) | `OUT/trace/` |
| `tools/optcuts_soft_fit_20261104/connect_interior_slits.py DUMP MESH W OUT` | `OUT/trace/` |
| `tools/optcuts_soft_fit_20261104/resolve_cut_with_optcuts.py DIR MESH` | `DIR/trace/` |

Set `ONESTRING_TRACE_DIR=/some/dir` to override; unset it (and do not use the drivers above) to disable tracing.
`src/onestring_physics/run_trace.py` implements it; writing never raises, so a broken log never breaks a run.

Files:

* `trace.jsonl`: one JSON object per line, appended in time order.
* `grid_planner_input.pkl` (pipeline runs): the exact input of the structured-grid planner
  (`xyz`, `f3`, `uv`, `f2`, plain-valued `params`), so a planner failure can be replayed offline with
  `structured_grid_20261028.plan_structured_grid(xyz, f3, uv, f2, params)`.

Readable summary: `python3 tools/optcuts_soft_fit_20261104/trace_summary.py OUT/trace/trace.jsonl > OUT/trace/summary.md`.

## Record layout

```json
{"t": 12.3, "wall": "2026-11-04T09:44:38", "pid": 229, "stage": "grid.logical_domain", "event": "attempt", ...}
```

`t` = seconds since the first record of that process; `stage` / `event` say what the record is; the remaining keys are
the data.  NumPy arrays are written as lists in full.

## Stages and events

| stage | event | data |
|---|---|---|
| `run` | `start` | mesh, out, argv, the driver's own environment switches (`ONESTRING_*`, `MS_*` and the switches `run_shape.py` reads; nothing else of the environment), plain-valued params |
| `run` | `done` / `failed` | seconds / error_type, error, traceback |
| `progress` | *stage name* | fraction, detail: every progress message of the pipeline (OptCuts, grid, K3D iterations, K2D iterations, T2D trials, collision repair, …) |
| `grid.plan` | `start` | uv_vertices, faces, boundary_edges, zipper, turn_reference, panels, preset |
| `grid.plan` | `done` / `failed` | – / error_type, error (full text), traceback |
| `grid.message` | `SG-…` tag | text: every `[SG-…]` message of the planner in full |
| `grid.logical_domain` | `start` | min_ratio, max_rounds, zipper, features (number of boundary features m), tips (indices of seam tips), cone_pairs, turn_reference, side_target_panels (L_k = 3D length / panel edge), side_min_panels (n_lo), turning_deg (measured turning at each feature), feature_kind, feature_uv, seam_pairs (the two UV copies of each cut branch), milp_time_limit |
| `grid.logical_domain` | `attempt` | one per search round: round, turns (−2 = seam tip U-turn, −1 / 0 / +1 quarter turns), dirs (0 E, 1 N, 2 W, 3 S), lengths (null = no integer side lengths), status / separated_pairs from the length MILP, zipper_check (why a candidate polygon was rejected), milp_status |
| `grid.logical_domain` | `result` | found, rounds, chosen_turns, chosen_lengths |
| `grid.logical_domain` | `ratio_failed` | min_ratio, error (full text of SG_LOGICAL_POLYGON_NOT_FOUND with every attempt) |
| `optcuts.multistart` | `start` | mesh, OptCuts environment switches |
| `optcuts.multistart` | `branch` | one per branch and round: budget (topology changes allowed), start (boundary vertex), finished, error, frames, last_uv_vertices, last_boundary_loops (2+ = a seam inside the surface that does not reach the boundary), feasible, distortion, seam_cost |
| `optcuts.multistart` | `final` / `final_refused` | winner and its score / winner, the bridge's error, frames |
| `connect` | `input`, `slit`, `resolve_lbfgs`, `outer_boundary` | interior slit connection (path, length) and the UV re-solve |
| `resolve` | `start`, `done` | OptCuts re-solve of the connected cut: uv vertices, boundary loops, simple outer boundary, seam length, SD, area span |

## Typical questions and where to look

* *Why did the grid planner fail?* `grid.logical_domain` `start` (how many features / tips, the side targets) and every
  `attempt`: if `lengths` is null in every round, the turns chosen in stage 1 admit no integer side lengths with every
  side ≥ `min_ratio` × its 3D length (see `status`); if lengths exist but `zipper_check` rejects them, the polygon
  self-touches along a seam.  The search stops after `max_rounds` rounds per ratio.
* *Why does OptCuts leave a seam inside the surface?* `optcuts.multistart` `branch` records with
  `last_boundary_loops` ≥ 2, and the winner's frames (`f*.bin`).
* *Why are T2D hinges open?* `progress` records of the dual-hinge stage: every trial's energy and collisions, the
  opening-angle filter, each collision-repair attempt and the selected trial.
