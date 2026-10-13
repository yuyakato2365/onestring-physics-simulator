"""Sparse global step for OneString Appendix-E Projective Dynamics.

Given the inertial prediction and local Kabsch rigid projections, solve
( I + wr I + H' Wh H + S' Ws S + L' Wl L ) x
  = x_pred + wr x_rigid + L' Wl lift_targets

where H enforces point hinges, S paired side-face midpoint coincidence,
L selected lift-point positions. Weights are exposed as independent
implementation parameters, not asserted to be authors' exact values.

Discretized collision constraints are handled in a separate nonlinear
projection stage. This is not an IPC/CCD no-penetration guarantee.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import splu


def _hinge_vertex(h, key):
    k=int(getattr(h,key))
    if str(getattr(h,"surface","top")).lower()=="bottom" and 0<=k<4:
        k+=4
    if not 0<=k<8:
        raise ValueError("Hinge index outside eight-corner panel")
    return k


class PaperPDGlobal:
    def __init__(self, tile_count, hinges, snaps, lifts,
                 *, rigid_weight, hinge_weight, snap_weight, lift_weight):
        self.tile_count=int(tile_count)
        self.vertex_count=8*self.tile_count
        self.rigid_penalty=max(0.0,float(rigid_weight))*12.
        h_weight=max(0.0,float(hinge_weight))*24.
        s_weight=max(0.0,float(snap_weight))*12.
        l_weight=max(0.0,float(lift_weight))*12.
        # Linear scalar difference rows; each applies identically to XYZ.
        rows=[]; columns=[]; coeff=[]; weights=[]; lift_indices=[]
        def constraint(entries, weight, lift=False):
            k=len(weights)
            for vid,value in entries:
                rows.append(k);columns.append(int(vid));coeff.append(float(value))
            weights.append(weight)
            if lift:
                lift_indices.append(k)
        def vid(t,c):
            return 8*int(t)+int(c)
        for h in hinges:
            if h.tile_a == h.tile_b:
                continue
            a,b=int(h.tile_a),int(h.tile_b)
            if not (0<=a<self.tile_count and 0<=b<self.tile_count):
                raise ValueError("Hinge body outside PD tile range")
            ia,ib=_hinge_vertex(h,"local_vertex_a"),_hinge_vertex(h,"local_vertex_b")
            constraint([(vid(a,ia),1.),(vid(b,ib),-1.)],h_weight)
        for s in snaps:
            terms=[(vid(s.a,k),.25) for k in s.face_a]
            terms += [(vid(s.b,k),-.25) for k in s.face_b]
            constraint(terms,s_weight)
        for lift in lifts:
            coefficient=1./sum(len(f) for f in lift.faces)
            terms=[(vid(t,k),coefficient)
                   for t,f in zip(lift.tiles,lift.faces) for k in f]
            constraint(terms,l_weight,True)
        self.lift_indices=lift_indices
        self.weights=np.asarray(weights,dtype=float)
        self.matrix=coo_matrix(
            (coeff,(rows,columns)),shape=(len(weights),self.vertex_count)
        ).tocsr()
        gram=(self.matrix.T@diags(self.weights,format="csr")@self.matrix)
        diagonal=diags(np.full(self.vertex_count,1.+self.rigid_penalty))
        self.factor=splu((diagonal+gram).tocsc())

    def step(self,predicted,rigid_projection,lift_targets):
        n=self.vertex_count
        pred=np.asarray(predicted,float).reshape(n,3)
        rigid=np.asarray(rigid_projection,float).reshape(n,3)
        goals=np.asarray(lift_targets,float)
        if goals.shape!=(len(self.lift_indices),3):
            raise ValueError("Lift target count disagrees with the global PD matrix")
        rhs=pred+self.rigid_penalty*rigid
        if len(self.weights):
            projected=np.zeros((len(self.weights),3),float)
            projected[self.lift_indices]=goals
            rhs+=self.matrix.T@(self.weights[:,None]*projected)
        solved=np.column_stack([self.factor.solve(rhs[:,j]) for j in range(3)])
        if not np.isfinite(solved).all():
            raise FloatingPointError("Global PD produced nonfinite tile positions")
        return solved.reshape(self.tile_count,8,3)
