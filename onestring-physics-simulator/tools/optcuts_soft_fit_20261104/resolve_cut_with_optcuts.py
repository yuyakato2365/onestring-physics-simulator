"""2026-11-04: second step after connect_interior_slits.py.  The plain L-BFGS re-solve there keeps every triangle's
orientation but not the global injectivity of the UV map (the outer boundary can overlap itself once the new cut opens).
Here the cut surface (the OptCuts seams + the connecting cut) is given to OptCuts itself as its input mesh, with the same
energy settings (environment, e.g. ONESTRING_ANGLE_W=16 ONESTRING_SEAM_BEND_W=2) and bijectivity on, so the UV map is
solved by OptCuts' own bijective solver; OptCuts may add further cuts only if the distortion bound is still violated.
The chart is written in the original surface indexing (f3 refers to the vertices of MESH).

usage: python3 resolve_cut_with_optcuts.py CONNECT_DIR MESH.obj
"""
import sys, os, glob, json, pickle, shutil
import numpy as np, trimesh
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from optcuts_multistart import load_mesh, EXE
from scipy.spatial import cKDTree
d, mesh = sys.argv[1], sys.argv[2]
T = pickle.load(open(os.path.join(d, "connect_topology.pkl"), "rb")); V0, F0 = load_mesh(mesh)
vmap, Fc = np.asarray(T["vmap"]), np.asarray(T["F2"]); Vc = V0[vmap]
os.chdir(str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("ONESTRING_TRACE_DIR", os.path.join(d, "trace"))
from onestring_physics.run_trace import trace
trace("resolve", "start", uv_vertices=int(len(Vc)), faces=int(len(Fc)))
from onestring_physics import optcuts_backend as ob
dump = os.path.join(d, "resolve_frames"); shutil.rmtree(dump, ignore_errors=True); os.makedirs(dump)
os.environ.update({"ONESTRING_CSF_MODE": "0", "ONESTRING_DUMP_DIR": dump})
for k in ("ONESTRING_FORCE_START_VERTEX", "ONESTRING_TOPO_BUDGET", "ONESTRING_PRINT_START_CANDS", "ONESTRING_NO_BRANCH"): os.environ.pop(k, None)
cfg = ob.OptCutsConfig(executable=EXE, csf_tail_refine=False, timeout_seconds=86400.0)
r = ob._run_official_optcuts_once(Vc, Fc, cfg)
xyz = np.asarray(r.surface_vertices_3d); f3 = np.asarray(r.surface_faces); uv = np.asarray(r.uv_vertices_2d); f2 = np.asarray(r.uv_faces)
dist, idx = cKDTree(Vc).query(xyz); assert dist.max() < 1e-6
f3o = vmap[idx[f3]]                                             # back to the original surface's vertex ids
assert {tuple(sorted(f)) for f in f3o.tolist()} == {tuple(sorted(f)) for f in F0.tolist()}, "faces do not match the surface"
chart = dict(xyz=V0, f3=f3o, uv=uv, f2=f2)
pickle.dump(chart, open(os.path.join(d, "optcuts_chart.pkl"), "wb"))
Vq = np.zeros((len(uv), 3)); Vq[f2.ravel()] = V0[f3o.ravel()]
trimesh.Trimesh(Vq, f2, process=False).export(os.path.join(d, "cut.obj")); np.save(os.path.join(d, "uv.npy"), uv)
from shapely.geometry import Polygon
dE = {}
for f in f2:
    for a, c in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])): dE[(a, c)] = True
nb = {a: c for (a, c) in dE if (c, a) not in dE}; s = next(iter(nb)); ring = [s]
while nb[ring[-1]] != s: ring.append(nb[ring[-1]])
P = Polygon(uv[ring])
from igloo_export_lib import tri_stats, seams_of
st, K, ar = tri_stats(Vq, f2, uv); _, slen = seams_of(Vq, f2)
info = dict(uv_vertices=int(len(uv)), boundary_loops=len(r.boundary_loops), simple=bool(P.is_valid), seam_length=slen, frames=len(glob.glob(dump + "/f*.bin")), **st)
json.dump(info, open(os.path.join(d, "resolve.json"), "w"))
trace("resolve", "done", **info)
print("[RESOLVE]", info, flush=True)
