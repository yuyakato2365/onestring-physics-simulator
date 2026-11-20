"""Normalise the input mesh exactly as onestring-physics-simulator does before OptCuts (load_mesh -> normalize_shape),
so that every later length (FAIR_R, sizes) refers to the same scale as the OptCuts output.
usage: normalize_mesh.py IN.obj|IN.stl OUT.obj   (env ONESTRING_REPO)"""
import os, sys, numpy as np
R = os.environ['ONESTRING_REPO']; sys.path.insert(0, R + '/src'); sys.path.insert(0, R + '/tools/optcuts_soft_fit_20261104')
from optcuts_multistart import load_mesh
V, F = load_mesh(os.path.abspath(sys.argv[1]))
with open(sys.argv[2], 'w') as f:
    for v in V: f.write('v %.17g %.17g %.17g\n' % tuple(v))           # full double precision: OptCuts reacts to 1e-8 changes
    for t in np.asarray(F) + 1: f.write('f %d %d %d\n' % tuple(t))
print('normalised', V.shape, F.shape, 'bbox', V.min(0).round(4), V.max(0).round(4))
