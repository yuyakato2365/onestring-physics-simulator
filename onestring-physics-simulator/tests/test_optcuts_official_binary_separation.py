from pathlib import Path
from types import SimpleNamespace

import pytest

from onestring_physics import optcuts_official_binary_patch as official
from onestring_physics.optcuts_backend import OptCutsUnavailableError

from onestring_physics.optcuts_official_binary_patch import (
    _grid_optcuts_root,
    _official_optcuts_root,
    official_executable_candidates,
)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def test_default_official_candidates_never_point_to_grid_tree(monkeypatch):
    monkeypatch.delenv("ONESTRING_OPTCUTS_OFFICIAL_EXECUTABLE", raising=False)
    candidates = official_executable_candidates(None)
    grid_root = _grid_optcuts_root()
    official_root = _official_optcuts_root()
    assert candidates
    assert all(not _is_within(path, grid_root) for path in candidates)
    assert any(_is_within(path, official_root) for path in candidates)


def test_legacy_grid_executable_is_rejected_from_official_candidates(monkeypatch):
    monkeypatch.delenv("ONESTRING_OPTCUTS_OFFICIAL_EXECUTABLE", raising=False)
    grid_binary = _grid_optcuts_root() / "build" / "OptCuts_bin"
    candidates = official_executable_candidates(str(grid_binary))
    assert grid_binary not in candidates
    assert all(not official._is_grid_optcuts_path(path) for path in candidates)
    assert any(_is_within(path, _official_optcuts_root()) for path in candidates)


def test_explicit_external_official_binary_remains_supported(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("ONESTRING_OPTCUTS_OFFICIAL_EXECUTABLE", raising=False)
    external = tmp_path / "OptCuts_bin"
    candidates = official_executable_candidates(str(external))
    assert candidates[0] == external

def test_existing_official_binary_in_sibling_worktree_is_discovered(monkeypatch, tmp_path: Path):
    active = tmp_path / "current" / "onestring-physics-simulator"
    sibling = tmp_path / "other-worktree"
    official_root = sibling / "onestring-physics-simulator" / "third_party" / "OptCuts_official"
    binary = official_root / "build" / "OptCuts_bin"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"example official executable")
    monkeypatch.setattr(official, "_project_root", lambda: active)
    monkeypatch.delenv("ONESTRING_OPTCUTS_OFFICIAL_EXECUTABLE", raising=False)
    def fake_run(command, **kwargs):
        assert command[:2] == ["git", "-C"]
        assert command[-3:] == ["worktree", "list", "--porcelain"]
        return SimpleNamespace(
            returncode=0,
            stdout=f"worktree {tmp_path / 'current'}\\nHEAD abc\\n\\n"
                   f"worktree {sibling}\\nHEAD def\\n",
        )
    monkeypatch.setattr(official.subprocess, "run", fake_run)
    assert official.resolve_official_optcuts_executable() == binary.resolve()


def test_grid_fork_in_another_worktree_is_never_accepted(monkeypatch, tmp_path: Path):
    sibling_grid = tmp_path / "sibling" / "onestring-physics-simulator" / "third_party" / "OptCuts" / "build" / "OptCuts_bin"
    sibling_grid.parent.mkdir(parents=True)
    sibling_grid.write_bytes(b"grid fork")
    monkeypatch.setattr(official, "_project_root", lambda: tmp_path / "current")
    monkeypatch.setattr(official, "_worktree_official_roots", lambda: [])
    monkeypatch.setenv("ONESTRING_OPTCUTS_OFFICIAL_EXECUTABLE", str(sibling_grid))
    assert sibling_grid not in official.official_executable_candidates(str(sibling_grid))
    with pytest.raises(OptCutsUnavailableError, match="clean Official OptCuts"):
        official.resolve_official_optcuts_executable(str(sibling_grid))


def test_no_binary_does_not_silently_switch_algorithms(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(official, "_project_root", lambda: tmp_path / "empty")
    monkeypatch.setattr(official, "_worktree_official_roots", lambda: [])
    monkeypatch.delenv("ONESTRING_OPTCUTS_OFFICIAL_EXECUTABLE", raising=False)
    with pytest.raises(OptCutsUnavailableError, match="setup_optcuts_official.py"):
        official.resolve_official_optcuts_executable()
