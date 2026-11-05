"""2026-11-05: the app's version number, raised automatically whenever its implementation changes.

The implementation is every .py file of the app scripts (app*.py), src/onestring_physics/ and tools/.  Their content
hash is looked up in APP_VERSIONS.json (project root); a hash that is not there yet gets the next number, the time
and the names of the files that changed since the previous number.  The per-file hashes of the latest version are
kept in .app_version_cache.json (not needed for the numbering).  Hashing is redone only when a file's size or
modification time changed, so Streamlit reruns stay cheap.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

_MEMO: dict = {}


def _files(root: Path) -> list[Path]:
    out = sorted(root.glob("app*.py"))
    for sub in ("src/onestring_physics", "tools"):
        d = root / sub
        if d.is_dir():
            out += sorted(p for p in d.rglob("*.py") if "__pycache__" not in p.parts)
    return out


def current_version(root: str | Path | None = None) -> dict:
    """{'n': number, 'time': 'YYYY-MM-DD HH:MM', 'changed': [file names changed since the previous number]}"""
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    files = _files(root)
    stamp = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files)
    if _MEMO.get("stamp") == stamp:
        return _MEMO["version"]
    per_file = {str(p.relative_to(root)): hashlib.sha1(p.read_bytes()).hexdigest() for p in files}
    total = hashlib.sha1("".join(f"{k}:{v}\n" for k, v in sorted(per_file.items())).encode()).hexdigest()
    reg_path, cache_path = root / "APP_VERSIONS.json", root / ".app_version_cache.json"
    try:
        reg = json.loads(reg_path.read_text())
    except Exception:   # noqa: BLE001
        reg = []
    hit = next((e for e in reg if e.get("hash") == total), None)
    if hit is None:
        try:
            prev = json.loads(cache_path.read_text()).get("files", {})
        except Exception:   # noqa: BLE001
            prev = {}
        changed = sorted(k for k in set(per_file) | set(prev) if per_file.get(k) != prev.get(k)) if prev else []
        hit = dict(n=(max((e.get("n", 0) for e in reg), default=0) + 1), hash=total,
                   time=time.strftime("%Y-%m-%d %H:%M"), changed=changed[:20])
        reg.append(hit)
        try:
            reg_path.write_text(json.dumps(reg, ensure_ascii=False, indent=1))
            cache_path.write_text(json.dumps({"hash": total, "files": per_file}))
        except Exception as e:   # noqa: BLE001
            print(f"[APP-VERSION] could not save the version list: {e}", flush=True)
    _MEMO.update(stamp=stamp, version=hit)
    return hit


def version_caption(root: str | Path | None = None) -> str:
    v = current_version(root)
    ch = v.get("changed") or []
    tail = ""
    if ch:
        names = ", ".join(os.path.basename(c) for c in ch[:4]) + (f" ほか {len(ch) - 4} 件" if len(ch) > 4 else "")
        tail = f"　前の版から変わったファイル：{names}"
    return f"バージョン {v['n']}（{v['time']}）{tail}"
