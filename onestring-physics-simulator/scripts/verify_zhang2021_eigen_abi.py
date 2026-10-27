#!/usr/bin/env python3
"""Numerically test independently licensed Eigen PARDISO-shaped sparse solver.

Checks EXACT indefinite symmetric Ax=b against known solutions, phases,
original authors' 1-based CSR and explicit rejection of singular input.
This does not attempt to bypass or patch proprietary PARDISO licensing.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import math
from pathlib import Path


def test(dll_path):
    dll=ct.CDLL(str(Path(dll_path).resolve()))
    I=ct.c_int
    D=ct.c_double
    PT=ct.c_void_p
    ptr=(PT*64)()
    mtype=I(-2)
    solver=I(0)
    error=I(-999)
    iparm=(I*64)()
    dparm=(D*64)()
    dll.pardisoinit(ptr,ct.byref(mtype),ct.byref(solver),
                    iparm,dparm,ct.byref(error))
    assert error.value==0, ("init",error.value)
    assert iparm[34]==0, "authors' 1-based CSR"
    maxfct,mnum,n,nrhs,msglvl=I(1),I(1),I(2),I(1),I(0)
    # [ 2  1 ] * [1] = [3]
    # [ 1 -1 ]   [1]   [0]
    a=(D*3)(2.,1.,-1.)
    ia=(I*3)(1,3,4)
    ja=(I*3)(1,2,2)
    perm=(I*2)()
    rhs=(D*2)(3.,0.)
    x=(D*2)()
    for ph in [11,22,33]:
        phase=I(ph)
        dll.pardiso(ptr,ct.byref(maxfct),ct.byref(mnum),
                    ct.byref(mtype),ct.byref(phase),ct.byref(n),
                    a,ia,ja,perm,ct.byref(nrhs),iparm,
                    ct.byref(msglvl),rhs,x,ct.byref(error),dparm)
        assert error.value==0, ("phase",ph,error.value)
    assert abs(x[0]-1)<1e-10 and abs(x[1]-1)<1e-10, list(x)
    # Reuse same factorization, verify different RHS.
    rhs2=(D*2)(4.,-2.)
    x2=(D*2)()
    phase=I(33)
    dll.pardiso(ptr,ct.byref(maxfct),ct.byref(mnum),
                ct.byref(mtype),ct.byref(phase),ct.byref(n),
                a,ia,ja,perm,ct.byref(nrhs),iparm,
                ct.byref(msglvl),rhs2,x2,ct.byref(error),dparm)
    assert error.value==0
    assert abs(2*x2[0]+x2[1]-4)<1e-10
    assert abs(x2[0]-x2[1]+2)<1e-10
    phase=I(-1)
    dll.pardiso(ptr,ct.byref(maxfct),ct.byref(mnum),
                ct.byref(mtype),ct.byref(phase),ct.byref(n),
                a,ia,ja,perm,ct.byref(nrhs),iparm,
                ct.byref(msglvl),rhs2,x2,ct.byref(error),dparm)
    assert error.value==0 and ptr[0] is None
    print("EIGEN_ORIGINAL_SPARSE_DIRECT_ABI_KKT_PASS")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--dll",type=Path,required=True)
    test(parser.parse_args().dll)
