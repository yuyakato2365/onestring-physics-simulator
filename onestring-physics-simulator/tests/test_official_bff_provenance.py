"""Regression: never record the OneString checkout's HEAD as official BFF's SHA."""
import hashlib
import json

from onestring_physics.reference_bff import (
    _discover_git_commit, OFFICIAL_BFF_CLI_COMMIT, OFFICIAL_BFF_REPOSITORY,
)


def test_unknown_rectangle_binary_has_no_fabricated_source_commit(tmp_path):
    exe=tmp_path/"bff-rect-command-line"
    exe.write_bytes(b"fake binary")
    assert _discover_git_commit(exe)==OFFICIAL_BFF_CLI_COMMIT


def test_verified_manifest_returns_official_bff_source_commit(tmp_path):
    exe=tmp_path/"bff-rect-command-line"
    exe.write_bytes(b"fake binary")
    sha="a"*40
    path=tmp_path/"bff-rect-command-line.provenance.json"
    path.write_text(json.dumps({
        "official_bff_repository": OFFICIAL_BFF_REPOSITORY,
        "official_bff_source_commit_sha": sha,
        "bff_rect_executable_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
    }))
    assert _discover_git_commit(exe)==sha


def test_stale_manifest_cannot_claim_commit_of_modified_executable(tmp_path):
    exe=tmp_path/"bff-rect-command-line"
    exe.write_bytes(b"fake binary")
    path=tmp_path/"bff-rect-command-line.provenance.json"
    path.write_text(json.dumps({
        "official_bff_repository": OFFICIAL_BFF_REPOSITORY,
        "official_bff_source_commit_sha": "a"*40,
        "bff_rect_executable_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
    }))
    exe.write_bytes(b"DIFFERENT")
    assert _discover_git_commit(exe)==OFFICIAL_BFF_CLI_COMMIT
