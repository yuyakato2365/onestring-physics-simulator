"""Globally consistent discrete C4 / integer-lattice seam transitions.

A greedy fit of each physical OptCuts chain can assign contradictory integer
translations around seam junctions.  This module performs a bounded joint
search and incrementally checks exact graph holonomy, rather than relaxing
inconsistent equations, welding arbitrary vertices or dropping any cut.

All transform offsets are represented as INTEGER multiples of the ACTUAL h.
The currently supported cone holonomy has its fixed point on that grid.
An infeasible cut graph fails with diagnostics; a new cut / cone-placement
algorithm is a separate research problem.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .optcuts_seamless_lattice_mode import ROT, SeamlessLatticeError


class _Inconsistent(Exception):
    def __init__(self, reason, detail):
        self.reason=reason
        self.detail=detail
        super().__init__(f"{reason}: {detail}")


class LatticeUnion:
    """Dynamic graph of u_child=R^k*u_parent+h*t, t in Z².

    The stored translations are in lattice units; optional cone anchors
    (root UV coordinates / h) are integers and exact.  Copying the state
    allows beam-search branching without discarding an earlier constraint.
    """
    def __init__(self, n):
        self.parent=np.arange(n,dtype=np.int64)
        self.turn=np.zeros(n,dtype=np.int8)
        self.shift=np.zeros((n,2),dtype=np.int64)
        self.fixed={}

    def clone(self):
        out=object.__new__(LatticeUnion)
        out.parent=self.parent.copy()
        out.turn=self.turn.copy()
        out.shift=self.shift.copy()
        out.fixed={i:pt.copy() for i,pt in self.fixed.items()}
        return out

    def find(self,v):
        v=int(v)
        p=int(self.parent[v])
        if p==v:
            return v,0,np.zeros(2,dtype=np.int64)
        root,pk,pt=self.find(p)
        k=int(self.turn[v])
        new_k=(k+pk)%4
        new_t=ROT[k]@pt+self.shift[v]
        self.parent[v]=root
        self.turn[v]=new_k
        self.shift[v]=new_t
        return root,new_k,new_t

    def relation(self,a,b,k,translation):
        """Accept equation x_b=R[k] x_a + translation, or raise."""
        ra,ka,ta=self.find(a)
        rb,kb,tb=self.find(b)
        R=ROT[k]
        target_rot=(k+ka)%4
        target_shift=R@ta+np.asarray(translation,dtype=np.int64)
        if ra!=rb:
            # Solve for root_b as a function of root_a.
            rk=(target_rot-kb)%4
            delta=ROT[(-kb)%4]@(target_shift-tb)
            self.parent[rb]=ra
            self.turn[rb]=rk
            self.shift[rb]=delta
            if rb in self.fixed:
                pb=self.fixed.pop(rb)
                pa=ROT[(-rk)%4]@(pb-delta)
                if ra in self.fixed and not np.array_equal(
                    self.fixed[ra],pa
                ):
                    raise _Inconsistent(
                        "CONE_ANCHOR_CONFLICT",
                        {"a":int(a),"b":int(b),
                         "existing":self.fixed[ra].tolist(),
                         "required":pa.tolist()}
                    )
                self.fixed[ra]=pa
            return
        if kb==target_rot:
            diff=target_shift-tb
            if np.any(diff):
                raise _Inconsistent(
                    "TRANSLATIONAL_HOLONOMY",
                    {"a":int(a),"b":int(b),"steps":diff.tolist()}
                )
            return
        # A rotational cone around a fixed grid vertex can close this loop.
        D=ROT[kb]-ROT[target_rot]
        delta=target_shift-tb
        determinant=round(float(np.linalg.det(D)))
        if determinant==0:
            raise _Inconsistent(
                "UNSUPPORTED_CONE",
                {"a":int(a),"b":int(b),"rotations":[int(kb),int(target_rot)]}
            )
        # Exact integer solution; no floating point pivot drift.
        numerator=np.array([
            D[1,1]*delta[0]-D[0,1]*delta[1],
            -D[1,0]*delta[0]+D[0,0]*delta[1]
        ],dtype=np.int64)
        if np.any(numerator%determinant):
            raise _Inconsistent(
                "CONE_OFF_INTEGER_LATTICE",
                {"a":int(a),"b":int(b),
                 "numerator":numerator.tolist(),
                 "denominator":int(determinant)}
            )
        pivot=numerator//determinant
        if ra in self.fixed and not np.array_equal(self.fixed[ra],pivot):
            raise _Inconsistent(
                "CONE_ANCHOR_CONFLICT",
                {"a":int(a),"b":int(b),
                 "existing":self.fixed[ra].tolist(),
                 "required":pivot.tolist()}
            )
        self.fixed[ra]=pivot


def candidate_transitions(uv,chain,h,*,per_chain=24):
    """Candidate integer transitions near *every* reasonable C4 fit.

    Conical cut tips (both copies have one UV index) choose translations
    from nearby integer fixed points.  Non-tip chains get translations
    near their least-squares offset.  No blind zero translation fallback.
    """
    physical,a,b=chain
    a=tuple(map(int,a));b=tuple(map(int,b))
    A=np.asarray(uv,float)[list(a)]
    B=np.asarray(uv,float)[list(b)]
    tip=[i for i,(u,v) in enumerate(zip(a,b)) if u==v]
    possibilities={}
    for k,R in enumerate(ROT):
        if tip:
            shifts=[]
            for i in tip:
                origin=np.rint(A[i]/h).astype(np.int64)
                for dx in (-2,-1,0,1,2):
                    for dy in (-2,-1,0,1,2):
                        gridpoint=origin+np.array([dx,dy],dtype=np.int64)
                        shifts.append((np.eye(2,dtype=int)-R)@gridpoint)
        else:
            nearest=np.rint(np.mean(B-A@R.T,axis=0)/h).astype(np.int64)
            shifts=[nearest+np.array([dx,dy],dtype=np.int64)
                    for dx in (-1,0,1) for dy in (-1,0,1)]
        for shift in shifts:
            key=(k,int(shift[0]),int(shift[1]))
            if key in possibilities:continue
            difference=B-(A@R.T+h*shift)
            sq=float(np.mean(np.sum(difference*difference,axis=1)))
            possibilities[key]=sq
    ranked=sorted(possibilities.items(),key=lambda x:(x[1],x[0]))
    chosen=ranked[:per_chain]
    # Keep at least the locally best candidate of each rotation, otherwise
    # the only globally feasible orientation may be accidentally excluded.
    for k in range(4):
        options=[v for v in ranked if v[0][0]==k]
        if options and options[0] not in chosen:
            chosen.append(options[0])
    chosen.sort(key=lambda x:(x[1],x[0]))
    return [
        {"physical":tuple(map(int,physical)),"a":a,"b":b,
         "k":int(key[0]),"steps":(int(key[1]),int(key[2])),
         "rms":float(np.sqrt(sq))}
        for key,sq in chosen
    ]


def _add_chain(union,transition):
    for a,b in zip(transition["a"],transition["b"]):
        union.relation(
            int(a),int(b),int(transition["k"]),transition["steps"]
        )


def select_joint_transitions(uv,chains,h,*,beam_width=24,
                             candidates_per_chain=24):
    """Bounded, globally screened search for compatible chain assignments.

    The full graph is checked after EVERY candidate, not after a greedy
    final assignment. Preserves exact constraints and reports the first
    unsatisfiable frontier, never silently ignores problematic chains.
    """
    if not chains:
        raise SeamlessLatticeError("SEAMLESS_NO_PHYSICAL_OPT_CUT_CHAIN")
    if h<=0 or not np.isfinite(h):
        raise SeamlessLatticeError("SEAMLESS_INVALID_LATTICE_SPACING")
    vertex_ids=sorted(set(int(v) for _,a,b in chains for v in (*a,*b)))
    compact={v:i for i,v in enumerate(vertex_ids)}
    options=[]
    occurrences=Counter(v for _,a,b in chains for v in set((*a,*b)))
    for original_chain in chains:
        choices=candidate_transitions(
            uv,original_chain,h,per_chain=candidates_per_chain
        )
        if not choices:
            raise SeamlessLatticeError("SEAMLESS_NO_DISCRETE_TRANSITION_CANDIDATE")
        options.append(choices)
    order=sorted(range(len(chains)),key=lambda i:(
        -sum(occurrences[v]>1 for v in set((*chains[i][1],*chains[i][2]))),
        -len(chains[i][1]), i
    ))
    init=LatticeUnion(len(vertex_ids))
    # States: (cost, chosen mapping, union). Fixed beam width controls runtime.
    beam=[(0.,{},init)]
    rejected=Counter()
    rejection_examples=[]
    explored=0
    for depth,ci in enumerate(order):
        next_states=[]
        for cost,selection,union in beam:
            for c in options[ci]:
                trial=union.clone()
                packed={**c,
                    "a":tuple(compact[v] for v in c["a"]),
                    "b":tuple(compact[v] for v in c["b"])}
                try:
                    _add_chain(trial,packed)
                except _Inconsistent as exc:
                    rejected[exc.reason]+=1
                    if len(rejection_examples)<6:
                        rejection_examples.append({
                            "source_chain":list(c["physical"][:12]),
                            "k":int(c["k"]),
                            "steps":list(c["steps"]),
                            "reason":exc.reason,
                            "detail":exc.detail,
                        })
                    continue
                explored+=1
                newsel={**selection,ci:c}
                # Correspondence error is the discrete objective; preserve
                # chain length support but compare RMS on actual source UV.
                score=cost+len(c["a"])*c["rms"]**2/max(h*h,1e-24)
                next_states.append((score,newsel,trial))
        if not next_states:
            source=chains[ci][0]
            raise SeamlessLatticeError(
                "SEAMLESS_JOINT_TRANSITION_INFEASIBLE: "
                f"frontier={depth+1}/{len(order)} "
                f"source_chain={source[:15]} "
                f"beam_width={beam_width} "
                f"candidates={len(options[ci])} "
                f"rejected={dict(rejected)} "
                f"first_conflicts={rejection_examples[:4]}. "
                "Global transition choices conflict; new cone/cut "
                "topology or wider discrete search may be required. "
                "No local seam constraints were discarded."
            )
        next_states.sort(key=lambda t:t[0])
        beam=next_states[:max(1,int(beam_width))]
    cost,selected,_=beam[0]
    result=[selected[i] for i in range(len(chains))]
    return result,{
        "seamless_lattice_joint_selection_enabled":True,
        "seamless_lattice_joint_candidate_states_explored":int(explored),
        "seamless_lattice_joint_rejected_by_reason":dict(rejected),
        "seamless_lattice_joint_rejection_examples":rejection_examples,
        "seamless_lattice_joint_beam_width":int(beam_width),
        "seamless_lattice_joint_objective":float(cost),
        "seamless_lattice_joint_chain_count":len(chains),
        "seamless_lattice_joint_naive_local_independent":False,
    }


__all__=["select_joint_transitions","LatticeUnion","candidate_transitions"]
