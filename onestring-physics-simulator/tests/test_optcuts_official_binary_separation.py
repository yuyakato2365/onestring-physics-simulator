from pathlib import Path

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
    assert all(_is_within(path, _official_optcuts_root()) for path in candidates)


def test_explicit_external_official_binary_remains_supported(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("ONESTRING_OPTCUTS_OFFICIAL_EXECUTABLE", raising=False)
    external = tmp_path / "OptCuts_bin"
    candidates = official_executable_candidates(str(external))
    assert candidates[0] == external
