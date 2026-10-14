"""Verify extra Split is decided from *post-reBFF* CSF of each piece."""
import numpy as np
import postcut_rebff_iterative as mod


def _fake_seed():
    xyz = np.zeros((4, 3))
    uv = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    cut = dict(axis="row", coordinate=0.5, origin="test")
    return xyz, uv, faces, cut


def _fake_split(patch, axis, coordinate, rebff):
    """First cut leaves one CSF violator; second makes both children pass."""
    if patch["side"] == "initial":
        values = (1.3, 2.8)
    elif patch["side"] == "initial_positive":
        values = (1.5, 1.8)
    else:
        raise AssertionError("Compliant patch was incorrectly split")
    children = []
    for i, value in enumerate(values):
        children.append({
            "side": patch["side"] + ("_negative" if i == 0 else "_positive"),
            "xyz": patch["xyz"], "new_uv": patch["new_uv"],
            "original_uv": patch["new_uv"], "faces": patch["faces"],
            "pre": {"linear_span": 2.8},
            "post": {"linear_span": value}, "bff_metadata": {},
        })
    return children


def _run(monkeypatch, budget=8, candidate=True):
    monkeypatch.setattr(mod, "_split_and_rebff", _fake_split)
    monkeypatch.setattr(
        mod, "candidate_lines",
        lambda *a, **k: ([dict(axis="col", coordinate=0.5)]
                         if candidate else []),
    )
    xyz, uv, faces, cut = _fake_seed()
    return mod.iterative_postcut_rebff(
        xyz, uv, faces, first_cut=cut,
        rebff=lambda *a: None, grid=14, csf_limit=2.,
        max_splits=budget,
    )


def test_cut_only_violating_piece_and_repeat(monkeypatch):
    result = _run(monkeypatch)
    assert result["split_count"] == 2
    assert result["all_pass"]
    assert result["stop_reason"] == "all_patches_pass"
    assert [s["parent"] for s in result["split_history"]] == [
        "initial", "initial_positive",
    ]
    assert len(result["patches"]) == 3
    assert result["split_history"][0]["violating_patch_count"] == 1
    assert result["split_history"][1]["violating_patch_count"] == 0


def test_stop_at_budget_with_unresolved_csf(monkeypatch):
    result = _run(monkeypatch, budget=1)
    assert result["split_count"] == 1
    assert not result["all_pass"]
    assert result["stop_reason"] == "max_split_budget_exhausted"
    assert result["max_linear_span"] == 2.8


def test_missing_candidate_does_not_silently_pass(monkeypatch):
    result = _run(monkeypatch, candidate=False)
    assert result["split_count"] == 1
    assert not result["all_pass"]
    assert result["stop_reason"] == "no_curvature_guided_interior_grid_candidate"
