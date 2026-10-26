#!/usr/bin/env python3
"""Download the genuine Zhang et al. 2021 authors' public C++ package.

Author source is *not* vendored into the OneString repository. Public guest-
share API survives the 2026 REC shutdown notice. This script uses only
author-published public share metadata and refuses arbitrary redirect hosts.

Source: https://rec.ustc.edu.cn/share/8081ab10-dd29-11eb-b5e1-5fc36251b03b
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request
from urllib.parse import urlparse
import zipfile

SHARE = "8081ab10-dd29-11eb-b5e1-5fc36251b03b"
RESOURCE = "29b9f530-dd29-11eb-b95c-8d3989e040ed"
BASE = "https://recapi.ustc.edu.cn/api/v2"
KNOWN_HOST = "recstore.ustc.edu.cn"


def _post(endpoint, data):
    req = urllib.request.Request(
        BASE + endpoint,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Origin": "https://rec.ustc.edu.cn"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=40) as handle:
        info = json.loads(handle.read().decode("utf-8-sig"))
    if info.get("status_code") != 200:
        raise RuntimeError(f"Author public share API {endpoint} refused: {info.get('message')}")
    return info["entity"]


def download_authors_source(directory: Path, *, no_extract: bool = False) -> dict:
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    record = _post("/share/target/info", {"share_number": SHARE})
    if record.get("share_title") != "Code" or record.get("share_number") != SHARE:
        raise RuntimeError("Unexpected public share identity; refusing download")
    names = _post("/share/target/resource/list", {"share_number": SHARE})
    matching = [x for x in names if x.get("number") == RESOURCE
                and x.get("name") == "Code_new" and x.get("file_ext") == "zip"]
    if len(matching) != 1:
        raise RuntimeError("Public author Code_new.zip no longer found at known resource")
    req = {"share_number": SHARE,
           "share_constraint": {"password": None},
           "share_resources_list": [RESOURCE]}
    link = _post("/share/download", req).get(RESOURCE)
    if not isinstance(link, str) or urlparse(link).scheme != "https" or urlparse(link).hostname != KNOWN_HOST:
        raise RuntimeError("Unexpected author archive download host")
    output = directory / "Zhang2021_authors_Code_new.zip"
    sha = hashlib.sha256()
    with urllib.request.urlopen(link, timeout=100) as response, output.open("wb") as handle:
        while part := response.read(1 << 20):
            sha.update(part)
            handle.write(part)
    size = output.stat().st_size
    if size != int(matching[0]["bytes"]):
        raise RuntimeError(f"Archive truncated: {size} != {matching[0]['bytes']}")
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError("Author archive corrupted at " + bad)
        if not no_extract:
            extract_dir = directory / "authors_source"
            extract_dir.mkdir(parents=True, exist_ok=True)
            for member in archive.infolist():
                relative = Path(member.filename)
                if (relative.is_absolute() or ".." in relative.parts
                        or member.is_dir() and not relative.parts):
                    raise RuntimeError("Unsafe author archive entry")
                dest = (extract_dir / relative).resolve()
                if not dest.is_relative_to(extract_dir.resolve()):
                    raise RuntimeError("Unsafe author archive traversal")
                if member.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as src, dest.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    result = {"source": "original_Zhang_Chai_Liu_Fu_2021_public_Code_new",
              "share_number": SHARE, "resource_number": RESOURCE,
              "archive_sha256": sha.hexdigest(), "archive_bytes": size,
              "archive": str(output),
              "source_root": str(directory / "authors_source" / "Code")}
    (directory / "authors_download_provenance.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()
    print(json.dumps(download_authors_source(args.directory, no_extract=args.no_extract),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
