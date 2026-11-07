"""2026-11-05: the pipeline runs in a background thread, so it no longer depends on the browser connection.

Streamlit stops a script run the moment its browser connection drops (sleep, background tab, network blip) or a widget
is touched.  The thread runs without the page's script context and reads a snapshot of the session state (any
Streamlit call of the page's context there would raise StopException once the page's run has been stopped).  The calculation therefore runs in its own thread; the script run only waits for it and shows its progress
and finished stages.  If the script run is stopped, the thread keeps going; when the page is opened again the app
finds the job here (one registry per server process), shows everything finished so far and adopts the result.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid

_LOCK = threading.Lock()
_JOBS: dict[str, dict] = {}


def _new_job(meta: dict) -> dict:
    job = dict(id=uuid.uuid4().hex[:8], status="running", started=time.time(), finished=None, meta=dict(meta),
               progress=[], stages=[], result=None, error=None, error_text="", consumed=False)
    with _LOCK:
        _JOBS[job["id"]] = job
    return job


class _SessionSnapshot(dict):
    """attribute access like st.session_state"""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


class _ThreadAwareSessionState:
    """st.session_state that, inside a pipeline thread, is that job's snapshot of the page's session state.

    The pipeline thread must not touch the real session state: Streamlit raises StopException there as soon as the
    page's script run has been stopped (connection lost, widget touched), which killed the calculation."""
    def __init__(self, real):
        object.__setattr__(self, "_real", real)

    def _target(self):
        snap = _THREAD_STATE.get(threading.get_ident())
        return snap if snap is not None else object.__getattribute__(self, "_real")

    def __getattr__(self, k):
        return getattr(self._target(), k)

    def __setattr__(self, k, v):
        setattr(self._target(), k, v)

    def __delattr__(self, k):
        delattr(self._target(), k)

    def __getitem__(self, k):
        return self._target()[k]

    def __setitem__(self, k, v):
        self._target()[k] = v

    def __delitem__(self, k):
        del self._target()[k]

    def __contains__(self, k):
        return k in self._target()

    def __iter__(self):
        return iter(self._target())

    def __len__(self):
        return len(self._target())

    def get(self, k, default=None):
        return self._target().get(k, default)

    def setdefault(self, k, default=None):
        t = self._target()
        if k not in t:
            t[k] = default
        return t[k]

    def pop(self, k, *default):
        return self._target().pop(k, *default)

    def keys(self):
        return self._target().keys()

    def items(self):
        return self._target().items()

    def values(self):
        return self._target().values()

    def to_dict(self):
        t = self._target()
        return dict(t) if isinstance(t, dict) else t.to_dict()


_THREAD_STATE: dict[int, _SessionSnapshot] = {}


def _install_session_proxy():
    import streamlit as st
    if not isinstance(st.session_state, _ThreadAwareSessionState):
        st.session_state = _ThreadAwareSessionState(st.session_state)


def start_job(fn, meta: dict | None = None) -> dict:
    """fn(progress_callback, stage_sink) -> result, run in a daemon thread WITHOUT the page's script context (its
    Streamlit calls are ignored) and with a snapshot of the page's session state taken now."""
    job = _new_job(meta or {})
    snapshot = _SessionSnapshot()
    try:
        import streamlit as st
        real = st.session_state._real if isinstance(st.session_state, _ThreadAwareSessionState) else st.session_state
        for k in list(real.keys()):
            try:
                snapshot[k] = real[k]
            except Exception:   # noqa: BLE001
                pass
        _install_session_proxy()
    except Exception:   # noqa: BLE001
        pass

    def progress(stage, fraction, detail=""):
        job["progress"].append((time.time(), str(stage), float(fraction), str(detail)))

    def run():
        _THREAD_STATE[threading.get_ident()] = snapshot
        try:
            job["result"] = fn(progress, job["stages"])
            job["status"] = "done"
        except BaseException as exc:   # noqa: BLE001 - handed back to the script run
            job["error"] = exc
            job["error_text"] = traceback.format_exc()
            job["status"] = "error"
            print(f"[BACKGROUND-RUN] job {job['id']} failed: {type(exc).__name__}: {exc}", flush=True)
        finally:
            _THREAD_STATE.pop(threading.get_ident(), None)
            job["finished"] = time.time()
            print(f"[BACKGROUND-RUN] job {job['id']} {job['status']} after {job['finished'] - job['started']:.0f} s", flush=True)

    th = threading.Thread(target=run, name=f"onestring-pipeline-{job['id']}", daemon=True)
    th.start()
    print(f"[BACKGROUND-RUN] job {job['id']} started", flush=True)
    return job


def active_job() -> dict | None:
    """the newest job that is still running, or finished but not yet taken over by a page"""
    with _LOCK:
        jobs = sorted(_JOBS.values(), key=lambda j: j["started"], reverse=True)
    for j in jobs:
        if j["status"] == "running" or not j["consumed"]:
            return j
    return None


def wait_job(job: dict, on_progress=None, on_stage=None, poll_seconds: float = 0.5):
    """Block the script run until the job ends, replaying its progress and finished stages through the callbacks
    (from the start, so a page opened later shows everything).  Returns the result or re-raises the job's error."""
    seen_p, seen_s = 0, 0
    while True:
        p = job["progress"]
        if on_progress is not None and len(p) > seen_p:
            _, stage, frac, detail = p[-1]
            on_progress(stage, frac, detail)
            seen_p = len(p)
        s = job["stages"]
        while on_stage is not None and seen_s < len(s):
            on_stage(seen_s, s[seen_s])
            seen_s += 1
        if job["status"] != "running" and seen_s >= len(job["stages"]):
            break
        time.sleep(poll_seconds)
    job["consumed"] = True
    if job["status"] == "error":
        raise job["error"]
    return job["result"]
