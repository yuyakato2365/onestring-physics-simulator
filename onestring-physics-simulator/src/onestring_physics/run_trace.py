"""2026-11-04: machine-readable progress log for diagnosis (by people or other AIs).

Enabled by the environment variable ONESTRING_TRACE_DIR (a directory).  Every call of ``trace(stage, event, **data)``
appends one JSON object per line to ``<dir>/trace.jsonl``:

    {"t": seconds since the first event, "wall": ISO time, "pid": process id, "stage": "...", "event": "...", ...data}

Values are converted to plain JSON (numpy arrays -> lists, numbers -> float/int); long lists are kept whole, because
the point is to let a later reader reconstruct what the optimiser saw.  Tracing never raises: a failure to write is
ignored so that a run is never broken by its log.  See docs/TRACE_LOG_FORMAT.md for the stages and events.
"""
from __future__ import annotations

import datetime as _dt
import json as _json
import os as _os
import time as _time

_T0 = None


def enabled() -> bool:
    return bool(_os.environ.get("ONESTRING_TRACE_DIR"))


def _plain(x):
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
    except Exception:  # noqa: BLE001
        pass
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_plain(v) for v in x]
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return str(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


def trace(stage: str, event: str, **data) -> None:
    global _T0
    d = _os.environ.get("ONESTRING_TRACE_DIR")
    if not d:
        return
    try:
        _os.makedirs(d, exist_ok=True)
        now = _time.time()
        if _T0 is None:
            _T0 = now
        rec = {"t": round(now - _T0, 3), "wall": _dt.datetime.now().isoformat(timespec="seconds"), "pid": _os.getpid(),
               "stage": stage, "event": event, **_plain(data)}
        with open(_os.path.join(d, "trace.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(_json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 - the log must never break a run
        pass
