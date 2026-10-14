"""Regression: isolated Mac worktree can reuse binaries built in old OneString."""
from pathlib import Path

import pytest
import app_postcut_rebff as app
from onestring_physics.reference_bff import ReferenceBFFUnavailableError


@pytest.mark.parametrize("rectangle,relative", [
    (False, "third_party/boundary-first-flattening/build/bff-command-line"),
    (True, "third_party/bff_rectangle_build/bff-rect-command-line"),
])
def test_old_worktree_official_bff_binary_is_found(
        tmp_path, monkeypatch, rectangle, relative):
    original = (tmp_path / "Documents" / "OldStudyProject" /
                "onestring-physics-simulator" / relative)
    original.parent.mkdir(parents=True)
    original.write_text("#!/bin/sh\nexit 0\n")
    original.chmod(0o755)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    # Simulate the new worktree being missing its BFF executable, and verify
    # that a valid executable from a sibling checkout is selected instead.
    def find_cli(explicit=None):
        if explicit is None:
            raise ReferenceBFFUnavailableError("new worktree has no BFF")
        path = Path(explicit)
        if path.is_file() and path.stat().st_mode & 0o111:
            return path.resolve()
        raise ReferenceBFFUnavailableError("not executable")

    fn = ("find_rectangular_bff_executable" if rectangle
          else "find_official_bff_executable")
    monkeypatch.setattr(app, fn, find_cli)
    assert app._locate_bff_cli("", rectangle=rectangle) == original.resolve()


def test_missing_binary_explains_rebuild(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        app, "find_official_bff_executable",
        lambda explicit=None: (_ for _ in ()).throw(
            ReferenceBFFUnavailableError("no official BFF")
        ),
    )
    with pytest.raises(ReferenceBFFUnavailableError,
                       match="repair_postcut_bff_mac.sh"):
        app._locate_bff_cli("", rectangle=False)
