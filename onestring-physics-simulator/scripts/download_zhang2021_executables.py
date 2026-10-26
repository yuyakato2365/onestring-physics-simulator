#!/usr/bin/env python3
"""Download ORIGINAL authors' published Windows executables, not a replacement.

Source: https://rec.ustc.edu.cn/share/911f9e50-dd29-11eb-aa39-130e36684f1f
Do not commit the proprietary binaries to a source repository.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import urllib.request
from urllib.parse import urlparse
import zipfile

SHARE="911f9e50-dd29-11eb-aa39-130e36684f1f"
EXPECTED_SHA256="be7971f41821800ce32e0e2c0c9a53e081b002ec33e60521ccb4e2c9be848d37"
RESOURCE="287d8cc0-dd29-11eb-95dd-410793957484"
API="https://recapi.ustc.edu.cn/api/v2"


def _post(path,payload):
    req=urllib.request.Request(
        API+path,data=json.dumps(payload).encode("utf8"),
        headers={"Content-Type":"application/json","Origin":"https://rec.ustc.edu.cn"},
        method="POST")
    with urllib.request.urlopen(req,timeout=40) as r:
        response=json.loads(r.read().decode("utf-8-sig"))
    if response.get("status_code")!=200:
        raise RuntimeError(f"Original authors' EXE share rejected {path}: {response.get('message')}")
    return response["entity"]


def download(directory:Path):
    directory=directory.resolve()
    directory.mkdir(parents=True,exist_ok=True)
    info=_post("/share/target/info",{"share_number":SHARE})
    if info.get("share_title")!="Exe" or info.get("share_number")!=SHARE:
        raise RuntimeError("Unexpected author executable share")
    files=_post("/share/target/resource/list",{"share_number":SHARE})
    expected=[x for x in files if x.get("number")==RESOURCE
              and x.get("name")=="ExeProgram" and x.get("file_ext")=="zip"]
    if len(expected)!=1:
        raise RuntimeError("Original author executable archive no longer available")
    link=_post("/share/download",{
        "share_number":SHARE,
        "share_constraint":{"password":None},
        "share_resources_list":[RESOURCE]})[RESOURCE]
    if urlparse(link).scheme!="https" or urlparse(link).hostname!="recstore.ustc.edu.cn":
        raise RuntimeError("Unexpected author download host")
    archive_path=directory/"Zhang2021_authors_ExeProgram.zip"
    digest=hashlib.sha256()
    with urllib.request.urlopen(link,timeout=120) as r,archive_path.open("wb") as out:
        while p:=r.read(1<<20):
            digest.update(p)
            out.write(p)
    if archive_path.stat().st_size!=int(expected[0]["bytes"]):
        raise RuntimeError("Original executable archive size mismatch")
    if digest.hexdigest()!=EXPECTED_SHA256:
        raise RuntimeError("Original executable archive SHA256 did not match verified official release")
    target=directory/"authors_executables"
    target.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive_path) as z:
        if z.testzip():
            raise RuntimeError("Corrupted official author executable ZIP")
        for member in z.infolist():
            relative=Path(member.filename)
            dest=(target/relative).resolve()
            if relative.is_absolute() or ".." in relative.parts or not dest.is_relative_to(target.resolve()):
                raise RuntimeError("Unsafe entry in author executable ZIP")
            if member.is_dir():
                dest.mkdir(parents=True,exist_ok=True)
            else:
                dest.parent.mkdir(parents=True,exist_ok=True)
                with z.open(member) as src,dest.open("wb") as dst:
                    shutil.copyfileobj(src,dst)
    record={"official_authors_exe":True,
            "source_share":SHARE,
            "sha256":digest.hexdigest(),
            "bytes":archive_path.stat().st_size,
            "destination":str(target)}
    (directory/"official_exe_provenance.json").write_text(
        json.dumps(record,indent=2)+"\n",encoding="utf-8")
    return record


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(download(a.directory),indent=2))
