#!/usr/bin/env python3
"""Supply missing original Qt5 QPA plugin for author's Windows Straighten.exe.

The author's official ExeProgram.zip contains Qt5Core/Gui/Widgets.dll but
NO platforms/qwindows.dll or qoffscreen.dll; Qt5's QApplication can therefore
block on a GUI loader error even before the algorithm runs. This script
retrieves exact-era Qt 5.10 plugins from the *unmodified* official PyPI PyQt5
5.10 Windows wheel, hash-checked against PyPI metadata. It installs into a
local ExeProgram copy; do NOT redistribute Qt/PyQt5 binaries from this repo.
Respect LGPL/GPL and PyPI distribution terms.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile


def install_plugins(exe_root:Path, qt_pyqt_version:str="5.10"):
    exe_root=exe_root.resolve()
    if not (exe_root/"Straighten.exe").is_file():
        raise ValueError("Missing ORIGINAL authors Straighten.exe")
    with urllib.request.urlopen(f"https://pypi.org/pypi/PyQt5/{qt_pyqt_version}/json",timeout=40) as r:
        data=json.load(r)
    files=[x for x in data["urls"] if x["filename"].lower().endswith("win_amd64.whl")
           and x["packagetype"]=="bdist_wheel"]
    if len(files)!=1:
        raise RuntimeError("Exact-era PyQt5 Qt wheel for Windows x64 not uniquely found")
    record=files[0]
    with urllib.request.urlopen(record["url"],timeout=120) as r:
        wheel=r.read()
    actual=hashlib.sha256(wheel).hexdigest()
    if actual!=record["digests"]["sha256"]:
        raise RuntimeError("Official PyPI PyQt5 wheel failed SHA256 verification")
    output=exe_root/"platforms"
    output.mkdir(exist_ok=True)
    extracted=[]
    with zipfile.ZipFile(io.BytesIO(wheel)) as z:
        for filename in z.namelist():
            name=filename.replace("\\","/").lower()
            if name.endswith("/platforms/qwindows.dll") or name.endswith("/platforms/qoffscreen.dll"):
                dest=output/Path(filename).name
                dest.write_bytes(z.read(filename))
                extracted.append(dest.name)
    if "qwindows.dll" not in [x.lower() for x in extracted]:
        raise RuntimeError("Correct Qt platform plugin was missing from the official PyQt5 wheel")
    result={"wheel":record["filename"],"sha256":actual,
            "installed_plugins":extracted,"destination":str(output),
            "note":"Qt runtime plugin only; Zhang authors' algorithm binaries unchanged"}
    (exe_root/"qt_platform_install_provenance.json").write_text(
        json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe-root",type=Path,required=True)
    ap.add_argument("--qt-pyqt-version",default="5.10")
    args=ap.parse_args()
    print(json.dumps(install_plugins(args.exe_root,args.qt_pyqt_version),indent=2))
