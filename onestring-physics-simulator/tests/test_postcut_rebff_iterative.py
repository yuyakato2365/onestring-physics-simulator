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
        initial_linear_span=3.0,
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
    assert result["stop_reason"] == "no_splittable_violating_regions"
    assert result["unsplittable_region_count"] == 1
    assert result["unsplittable_regions"][0]["reason"] == "no_interior_full_grid_cut_candidate"
    assert result["violating_patch_count"] == 1


def test_initial_bff_below_limit_never_cuts(monkeypatch):
    def no_split(*args, **kwargs):
        raise AssertionError('No Split should occur after a passing initial BFF')
    monkeypatch.setattr(mod, '_split_and_rebff', no_split)
    xyz, uv, faces, _ = _fake_seed()
    result = mod.iterative_postcut_rebff(
        xyz, uv, faces, first_cut=None, rebff=no_split, grid=14,
        csf_limit=2., max_splits=8, initial_linear_span=1.7,
        initial_bff_metadata={'bff_rectangle_verified': True},
    )
    assert result['all_pass']
    assert result['split_count'] == 0
    assert result['stop_reason'] == 'initial_bff_satisfies_limit'
    assert result['max_linear_span'] == 1.7
    assert len(result['patches']) == 1
    assert result['patches'][0]['bff_metadata']['bff_rectangle_verified']


def test_first_cut_when_no_existing_grid_cut_record(monkeypatch):
    monkeypatch.setattr(mod, '_split_and_rebff', _fake_split)
    monkeypatch.setattr(
        mod, 'candidate_lines',
        lambda *a, **k: [dict(axis='row', coordinate=0.5)],
    )
    xyz, uv, faces, _ = _fake_seed()
    result = mod.iterative_postcut_rebff(
        xyz, uv, faces, first_cut=None, rebff=lambda *a: None,
        grid=14, csf_limit=2., max_splits=1, initial_linear_span=3.,
    )
    assert result['split_count'] == 1
    assert result['split_history'][0]['axis'] == 'row'


def test_grid_candidates_on_zero_anchored_grid():
    # The initial UV boundary begins at 0.13, not at a gridline.
    xyz = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0],
                    [0, 1, 0], [.5, .5, .3]], float)
    uv = np.array([[.13, .17], [1.13, .17], [1.13, 1.17],
                   [.13, 1.17], [.63, .67]], float)
    faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    cuts = mod.candidate_lines(xyz, uv, faces, grid=10)
    assert cuts
    assert all(abs(c['coordinate']*10 - round(c['coordinate']*10)) < 1e-8
               for c in cuts)


def test_unsplittable_single_cell_does_not_block_other_region(monkeypatch):
    calls=[]
    def fake_cut(patch, axis, coordinate, rebff):
        calls.append(patch["side"])
        if patch["side"]=="initial":
            spans=(3.0, 2.6)
        elif patch["side"]=="initial_positive":
            spans=(1.2,1.5)
        else:
            raise AssertionError("One-cell region must not be split")
        return [
            dict(
                side=patch["side"]+("_negative" if i==0 else "_positive"),
                xyz=patch["xyz"],new_uv=patch["new_uv"],faces=patch["faces"],
                pre={"linear_span":float(patch["post"]["linear_span"])},
                post={"linear_span":span},bff_metadata={},
            ) for i,span in enumerate(spans)
        ]
    monkeypatch.setattr(mod,"_split_and_rebff",fake_cut)
    # Initial negative has a higher CSF than the positive region but its
    # chart is already a single cell; the positive region is splittable.
    # Mark only the negative child as a single cell by exposing a unique
    # source shape. This ensures the planner really skips that region.
    def cut_with_single_cell(patch,axis,coordinate,rebff):
        children=fake_cut(patch,axis,coordinate,rebff)
        if patch["side"]=="initial":
            children[0]["faces"]=np.empty((0,3),int)
        return children
    monkeypatch.setattr(mod,"_split_and_rebff",cut_with_single_cell)
    def finite_cut_candidates(xyz,uv,faces,grid,**kwargs):
        return [] if len(faces)==0 else [dict(axis="col",coordinate=.5)]
    monkeypatch.setattr(mod,"candidate_lines",finite_cut_candidates)
    xyz,uv,faces,first=_fake_seed()
    result=mod.iterative_postcut_rebff(
        xyz,uv,faces,first_cut=first,rebff=lambda *a:None,
        grid=14,max_splits=64,initial_linear_span=3.,
    )
    assert calls==["initial","initial_positive"]
    assert result["split_count"]==2
    assert result["stop_reason"]=="no_splittable_violating_regions"
    assert not result["all_pass"]
    assert result["max_linear_span"]==3.0
    assert result["violating_patch_count"]==1
    assert result["unsplittable_region_count"]==1
    assert result["unsplittable_regions"][0]["side"]=="initial_negative"
    assert result["unsplittable_regions"][0]["linear_span"]==3.0


def test_all_failed_candidates_in_worst_region_do_not_block_other(monkeypatch):
    tried=[]
    def fake_cut(patch,axis,coordinate,rebff):
        tried.append(patch["side"])
        if patch["side"]=="initial":
            spans=(3.0,2.5)
        elif patch["side"]=="initial_negative":
            raise ValueError("Split creates invalid BFF child")
        elif patch["side"]=="initial_positive":
            spans=(1.3,1.4)
        else:
            raise AssertionError("Unexpected region was split")
        return [
            dict(
                side=patch["side"]+("_negative" if i==0 else "_positive"),
                xyz=patch["xyz"],new_uv=patch["new_uv"],faces=patch["faces"],
                pre={"linear_span":float(patch["post"]["linear_span"])},
                post={"linear_span":span},bff_metadata={},
            ) for i,span in enumerate(spans)
        ]
    monkeypatch.setattr(mod,"_split_and_rebff",fake_cut)
    monkeypatch.setattr(mod,"candidate_lines",lambda *a,**kw: [
        dict(axis="row",coordinate=.5)
    ])
    xyz,uv,faces,first=_fake_seed()
    result=mod.iterative_postcut_rebff(
        xyz,uv,faces,first_cut=first,rebff=lambda *a:None,
        grid=14,max_splits=64,initial_linear_span=3.,
    )
    assert tried==["initial","initial_negative","initial_positive"]
    assert result["split_count"]==2
    assert result["stop_reason"]=="no_splittable_violating_regions"
    assert not result["all_pass"]
    assert result["unsplittable_region_count"]==1
    assert result["unsplittable_regions"][0]["attempted_candidates"]==1
    assert result["unsplittable_regions"][0]["reason"]=="all_candidate_cuts_failed"
    assert len(result["rejected_candidates"])==1


def test_no_first_cut_no_candidates_is_unconverged(monkeypatch):
    monkeypatch.setattr(mod,"candidate_lines",lambda *a,**kw: [])
    xyz,uv,faces,_=_fake_seed()
    result=mod.iterative_postcut_rebff(
        xyz,uv,faces,first_cut=None,rebff=lambda *a:None,
        grid=14,max_splits=64,initial_linear_span=3.,
    )
    assert not result["all_pass"]
    assert result["split_count"]==0
    assert result["max_linear_span"]==3.
    assert result["unsplittable_region_count"]==1
    assert result["stop_reason"]=="no_splittable_violating_regions"


def test_fixed_global_grid_has_no_internal_cut_in_single_cell():
    # The old algorithm recalculated span / grid on EACH child and could
    # invent extra Split lines inside a single original grid cell.
    xyz=np.array([
        [.12,.15,0.],[.22,.15,0.],[.22,.25,0.],[.12,.25,0.],
        [.17,.20,.05],
    ],float)
    uv=xyz[:,:2].copy()
    faces=np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]],int)
    assert mod.candidate_lines(xyz,uv,faces,grid=14)
    cuts=mod.candidate_lines(
        xyz,uv,faces,grid=14,fixed_spacing=.5,
        fixed_origin=np.array([0.,0.])
    )
    assert cuts==[]


def test_fixed_grid_respects_original_origin():
    xyz=np.array([
        [.12,.15,0.],[.72,.15,0.],[.72,.75,0.],[.12,.75,0.],
        [.42,.45,.2],
    ],float)
    uv=xyz[:,:2].copy()
    faces=np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]],int)
    cuts=mod.candidate_lines(
        xyz,uv,faces,grid=14,fixed_spacing=.5,
        fixed_origin=np.array([.12,.15])
    )
    assert cuts
    for cut in cuts:
        dim=1 if cut["axis"]=="row" else 0
        origin=[.12,.15][dim]
        assert abs((cut["coordinate"]-origin)/.5-1.)<1e-9


def test_invalid_initial_preferred_cut_can_try_a_different_legal_cut(monkeypatch):
    attempts=[]
    def simulate(patch,axis,coordinate,rebff):
        attempts.append((patch["side"],axis))
        if axis=="row":
            raise ValueError("Initial preferred BFF cut was invalid")
        return [
            dict(
                side=patch["side"]+suffix,
                xyz=patch["xyz"],new_uv=patch["new_uv"],faces=patch["faces"],
                pre={"linear_span":3.0},post={"linear_span":1.5},
                bff_metadata={}
            ) for suffix in ("_negative","_positive")
        ]
    monkeypatch.setattr(mod,"_split_and_rebff",simulate)
    monkeypatch.setattr(mod,"candidate_lines",
                        lambda *args,**kw:[dict(axis="col",coordinate=.5)])
    xyz,uv,faces,first=_fake_seed()
    result=mod.iterative_postcut_rebff(
        xyz,uv,faces,first_cut=first,rebff=lambda *a:None,
        grid=14,max_splits=3,initial_linear_span=3.,
    )
    assert attempts==[("initial","row"),("initial","col")]
    assert result["split_count"]==1
    assert result["all_pass"]
    assert result["unsplittable_region_count"]==0
    assert len(result["rejected_candidates"])==1


def test_743_csf_regression_is_rejected_and_next_grid_line_is_tried(monkeypatch):
    """An almost-collapsed registered chart must not be committed as a Split."""
    attempts=[]
    def artificial_cut(patch, axis, coordinate, rebff):
        attempts.append(axis)
        values=(1.4,743.245) if axis=="row" else (1.4,1.8)
        return [
            dict(side=patch["side"]+suffix,
                 xyz=patch["xyz"],new_uv=patch["new_uv"],faces=patch["faces"],
                 pre={"linear_span":3.0},
                 post={"linear_span":v},
                 independent_bff_post={"linear_span":1.6},
                 postcut_joint_atlas=True,
                 bff_metadata={})
            for suffix,v in zip(("_negative","_positive"),values)
        ]
    monkeypatch.setattr(mod,"_split_and_rebff",artificial_cut)
    monkeypatch.setattr(mod,"candidate_lines",lambda *a,**k: [
        dict(axis="col",coordinate=.5)
    ])
    xyz,uv,faces,first=_fake_seed()
    result=mod.iterative_postcut_rebff(
        xyz,uv,faces,first_cut=first,rebff=lambda *a:None,
        grid=14,max_splits=1,initial_linear_span=3.,
    )
    assert attempts==["row","col"]
    assert result["all_pass"]
    assert result["split_count"]==1
    assert result["max_linear_span"]==1.8
    assert len(result["rejected_candidates"])==1
    assert "CSF regression" in result["rejected_candidates"][0]["reason"]
    assert "743.245" in result["rejected_candidates"][0]["reason"]


def test_all_csf_regressive_grid_cuts_leave_parent_unchanged(monkeypatch):
    """Failed reBFF is honestly unresolved, never a fabricated successful cut."""
    monkeypatch.setattr(mod,"_split_and_rebff",lambda patch,axis,coordinate,rebff: [
        dict(side=patch["side"]+suffix,
             xyz=patch["xyz"],new_uv=patch["new_uv"],faces=patch["faces"],
             post={"linear_span":v},
             postcut_joint_atlas=True)
        for suffix,v in zip(("_negative","_positive"),(2.2,20.0))
    ])
    monkeypatch.setattr(mod,"candidate_lines",lambda *a,**k: [
        dict(axis="row",coordinate=.5)
    ])
    xyz,uv,faces,_=_fake_seed()
    result=mod.iterative_postcut_rebff(
        xyz,uv,faces,first_cut=None,rebff=lambda *a:None,
        grid=14,max_splits=8,initial_linear_span=3.0,
    )
    assert result["split_count"]==0
    assert not result["all_pass"]
    assert result["max_linear_span"]==3.0
    assert result["stop_reason"]=="no_splittable_violating_regions"
    assert result["unsplittable_regions"][0]["reason"]=="all_candidate_cuts_failed"
    assert "CSF regression" in result["rejected_candidates"][0]["reason"]
