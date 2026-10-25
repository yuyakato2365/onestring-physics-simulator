"""Holonomy regression: global discrete choices, not seam stiffness, repair this."""
from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics import optcuts_seamless_joint_transitions as j
from onestring_physics import optcuts_seamless_lattice_mode as mode


def test_group_composition_and_zero_translation_cycle():
    state=j.LatticeUnion(3)
    state.relation(0,1,0,(1,0))
    state.relation(1,2,0,(0,2))
    state.relation(0,2,0,(1,2))
    ra,ka,ta=state.find(2)
    assert ra==state.find(0)[0]
    assert ka==0
    np.testing.assert_array_equal(ta,[1,2])
    with pytest.raises(j._Inconsistent,match="TRANSLATIONAL_HOLONOMY"):
        state.relation(0,2,0,(2,2))


def test_global_discrete_search_repairs_conflicting_greedy_rounding(monkeypatch):
    # The first-choice transforms have a nonzero loop translation:
    # 0 -> 1 shift +1; 1 -> 0 shift +1.
    # The SECOND choice for the second chain instead closes the loop.
    chains=[
        ((100,101),(0,),(1,)),
        ((200,201),(1,),(0,)),
    ]
    def candidates(uv,chain,h,*,per_chain=24):
        if chain[0][0]==100:
            return [
                {"physical":chain[0],"a":chain[1],"b":chain[2],
                 "k":0,"steps":(1,0),"rms":0.01},
                {"physical":chain[0],"a":chain[1],"b":chain[2],
                 "k":0,"steps":(0,0),"rms":0.9},
            ]
        return [
            {"physical":chain[0],"a":chain[1],"b":chain[2],
             "k":0,"steps":(1,0),"rms":0.01},
            {"physical":chain[0],"a":chain[1],"b":chain[2],
             "k":0,"steps":(-1,0),"rms":0.2},
        ]
    monkeypatch.setattr(j,"candidate_transitions",candidates)
    selected,audit=j.select_joint_transitions(
        np.zeros((2,2)),chains,1.,beam_width=4
    )
    assert selected[0]["steps"]==(1,0)
    assert selected[1]["steps"]==(-1,0)
    assert audit["seamless_lattice_joint_selection_enabled"]
    assert audit["seamless_lattice_joint_rejected_by_reason"][
        "TRANSLATIONAL_HOLONOMY"
    ]>=1
    # Old source-UV graph preflight must agree independently.
    mode._global_lattice_constraints(2,selected,1.)


def test_all_inconsistent_transition_choices_fail_no_relaxation(monkeypatch):
    chains=[
        ((100,101),(0,),(1,)),
        ((200,201),(1,),(0,)),
    ]
    def candidates(uv,chain,h,*,per_chain=24):
        return [{"physical":chain[0],"a":chain[1],"b":chain[2],
                 "k":0,"steps":(1,0),"rms":0.}]
    monkeypatch.setattr(j,"candidate_transitions",candidates)
    with pytest.raises(
        mode.SeamlessLatticeError,
        match="SEAMLESS_JOINT_TRANSITION_INFEASIBLE"
    ) as err:
        j.select_joint_transitions(np.zeros((2,2)),chains,1.)
    assert "TRANSLATIONAL_HOLONOMY" in str(err.value)


def test_90_degree_cone_fixed_point_is_not_translational_holonomy():
    state=j.LatticeUnion(2)
    state.relation(0,1,1,(2,0))
    state.relation(0,0,1,(2,0))
    assert len(state.fixed)==1
    np.testing.assert_array_equal(next(iter(state.fixed.values())),[1,1])
    # Additional compatible equations around that vertex remain valid.
    state.relation(0,0,2,(2,2))
    with pytest.raises(j._Inconsistent,match="CONE_ANCHOR_CONFLICT"):
        state.relation(0,0,1,(4,0))


def test_three_edge_rotation_loop_consistency_with_old_preflight():
    # Building a cycle with prescribed C4 lattice transformations.
    chosen=[
        {"physical":(1,2),"a":(0,),"b":(1,),
         "k":1,"steps":(3,2)},
        {"physical":(2,3),"a":(1,),"b":(2,),
         "k":0,"steps":(-1,1)},
        {"physical":(3,1),"a":(2,),"b":(0,),
         "k":3,"steps":(-3,2)},
    ]
    new=j.LatticeUnion(3)
    errors=[]
    for c in chosen:
        try:j._add_chain(new,c)
        except j._Inconsistent as e:errors.append(e.reason)
    try:
        mode._global_lattice_constraints(3,chosen,1.)
        old_fails=False
    except mode.SeamlessLatticeError:
        old_fails=True
    assert bool(errors)==old_fails
