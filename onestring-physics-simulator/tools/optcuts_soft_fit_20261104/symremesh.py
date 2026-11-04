"""2026-11-03: mirror-symmetric isotropic remesh.
The mirror plane is found as in optcuts_multistart (PCA axes refined by mirroring the surface onto itself); the mesh is
cut along the plane, one half is remeshed with Botsch-Kobbelt (gpytoolbox.remesh_botsch, the remesher of the Large Steps
/ Mitsuba 3 examples) one pass at a time until vertex count, mean edge length, edge-length spread and the 1 % smallest
angle all change by < TOL for 5 passes in a row; boundary edges (the cut and the rim) are feature edges (resampled but
kept on their polylines), the sharp rim corners and the rim/plane crossings are fixed.  The half is mirrored and welded
along the plane, so the plane carries a row of edges.   usage: symremesh.py IN OUT [TOL]"""
import sys, numpy as np, trimesh, gpytoolbox as gpy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv += [] if len(sys.argv) > 3 else ["0.005"]
src, dst, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])
import optcuts_multistart as M
m = trimesh.load(src, force="mesh", process=False); V = np.asarray(m.vertices, float); F = np.asarray(m.faces, int)
n, d, err = M.detect_mirror_plane(V, F)
print(f"mirror plane n {np.round(n, 5)} d {d:.5f} (mirror error {err:.1e} of the diagonal)", flush=True)
h = float(trimesh.Trimesh(V, F, process=False).edges_unique_length.mean())
# the vertex rows just beside the plane (and their mirror images) are moved onto it and welded, so the plane becomes a
# row of edges; otherwise the cut leaves vertices a fraction of an edge away from it, which the remesher cannot
# collapse onto the cut (it is a feature line) and which stay as slivers
dist0 = V @ n - d
sel = np.abs(dist0) < 0.3 * h
V1 = V.copy(); V1[sel] -= np.outer(dist0[sel], n)
w = trimesh.Trimesh(V1, F, process=False); w.merge_vertices(digits_vertex=7)
Fw = np.asarray(w.faces); Vw = np.asarray(w.vertices, float)
P_ = Vw[Fw]; area = 0.5 * np.linalg.norm(np.cross(P_[:, 1] - P_[:, 0], P_[:, 2] - P_[:, 0]), axis=1)
Fw = Fw[(area > 1e-12 * h * h) & (Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 0] != Fw[:, 2])]
print(f"snapped {int(sel.sum())} vertices within 0.3 edge of the plane onto it; {len(Fw)} faces left", flush=True)
# keep the half on the positive side of the plane (no cap: the cut becomes a boundary on the plane)
half = trimesh.intersections.slice_mesh_plane(trimesh.Trimesh(Vw, Fw, process=False), plane_normal=n, plane_origin=n * d, cap=False)
half.merge_vertices(); Vh = np.asarray(half.vertices, float); Fh = np.asarray(half.faces, np.int32)
dist = Vh @ n - d
Vh[np.abs(dist) < 1e-9] -= np.outer(dist[np.abs(dist) < 1e-9], n)        # exactly on the plane
def bedges(F_):
    E = np.sort(np.concatenate([F_[:, [0, 1]], F_[:, [1, 2]], F_[:, [2, 0]]]), 1); u, c = np.unique(E, axis=0, return_counts=True); return u[c == 1]
def corners(V_, F_):
    """boundary vertices where the boundary turns by > 50 deg, plus the vertices on the plane that end the cut"""
    be = bedges(F_); nb = {}
    for a, b in be: nb.setdefault(a, []).append(b); nb.setdefault(b, []).append(a)
    out = []
    for v, ns in nb.items():
        if len(ns) != 2: out.append(v); continue
        a, b = V_[ns[0]] - V_[v], V_[ns[1]] - V_[v]
        turn = 180 - np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))
        on_plane = abs(V_[v] @ n - d) < 1e-9
        other_on = [abs(V_[w] @ n - d) < 1e-9 for w in ns]
        if turn > 50 or (on_plane and not all(other_on)): out.append(v)
    return np.array(sorted(set(out)), int)
def stats(V_, F_):
    t = trimesh.Trimesh(V_, F_, process=False); L = t.edges_unique_length
    return np.array([len(V_), L.mean(), L.std() / L.mean(), np.degrees(np.percentile(t.face_angles, 1))])
fix = corners(Vh, Fh)
print(f"half: {len(Vh)} v {len(Fh)} f, boundary edges {len(bedges(Fh))}, fixed corner vertices {len(fix)}", flush=True)
def collapse_short_boundary(V_, F_, fixed, thr):
    """collapse boundary edges shorter than thr (the remesher leaves very short pieces on its feature lines), keeping a
    fixed endpoint when there is one; only edges whose endpoints share exactly one neighbour (link condition)"""
    F_ = F_.copy(); fixed = set(int(x) for x in fixed); total = 0
    while True:
        be = bedges(F_); L = np.linalg.norm(V_[be[:, 0]] - V_[be[:, 1]], axis=1)
        nbrs = {}
        for f in F_:
            for k_ in range(3):
                nbrs.setdefault(int(f[k_]), set()).update(int(x) for x in f if x != f[k_])
        touched = set(); did = 0
        for e in np.argsort(L):
            if L[e] >= thr: break
            a_, b_ = int(be[e, 0]), int(be[e, 1])
            if a_ in touched or b_ in touched or (a_ in fixed and b_ in fixed): continue
            if len(nbrs[a_] & nbrs[b_]) != 1: continue
            keep_, drop_ = (b_, a_) if a_ not in fixed and b_ in fixed else (a_, b_)
            F_[F_ == drop_] = keep_
            F_ = F_[np.array([len(set(f)) == 3 for f in F_])]
            touched |= {a_, b_} | nbrs[a_] | nbrs[b_]; did += 1
        total += did
        if not did: break
    used = np.unique(F_); remap = -np.ones(len(V_), int); remap[used] = np.arange(len(used))
    return V_[used], remap[F_].astype(np.int32), total
prev = stats(Vh, Fh); calm = 0; k = 0
while calm < 5 and k < 1000:            # 1000 only guards against a loop that never settles
    fe = bedges(Fh)
    Vh, Fh = gpy.remesh_botsch(Vh, Fh, i=1, h=h, project=True, feature=fix, feature_edges=fe)
    Fh = np.asarray(Fh, np.int32)
    fix = corners(Vh, Fh)
    Vh, Fh, ncol = collapse_short_boundary(Vh, Fh, fix, 0.5 * h)
    fix = corners(Vh, Fh)
    k += 1; cur = stats(Vh, Fh); ch = np.abs(cur - prev) / np.maximum(np.abs(prev), 1e-12)
    calm = calm + 1 if ch.max() < tol else 0; prev = cur
    print(f"pass {k:3d}: {int(cur[0])} v, edge mean {cur[1]:.4f} spread {cur[2]:.3f}, 1% angle {cur[3]:.1f} deg, max change {ch.max():.4f}", flush=True)
def converge(Vh, Fh, fix, label):
    prev = stats(Vh, Fh); calm = 0; k = 0
    while calm < 5 and k < 1000:
        fe = bedges(Fh)
        Vh, Fh = gpy.remesh_botsch(Vh, Fh, i=1, h=h, project=True, feature=fix, feature_edges=fe)
        Fh = np.asarray(Fh, np.int32); fix = np.arange(len(fix))
        k += 1; cur = stats(Vh, Fh); ch = np.abs(cur - prev) / np.maximum(np.abs(prev), 1e-12)
        calm = calm + 1 if ch.max() < tol else 0; prev = cur
    print(f"{label}: {k} passes, {int(cur[0])} v, edge spread {cur[2]:.3f}, 1% angle {cur[3]:.1f} deg, min angle {np.degrees(trimesh.Trimesh(Vh, Fh, process=False).face_angles.min()):.1f}", flush=True)
    return Vh, Fh, fix
dist = Vh @ n - d
onp = np.abs(dist) < 1e-6 * h * 100
print(f"half after remesh: max |distance to plane| of cut vertices {np.abs(dist[onp]).max() if onp.any() else -1:.2e}, cut vertices {onp.sum()}", flush=True)
Vh[onp] -= np.outer(dist[onp], n)
# mirror and weld
Vm = Vh - 2 * np.outer(Vh @ n - d, n); Fm = Fh[:, ::-1] + len(Vh)
Vall = np.vstack([Vh, Vm]); Fall = np.vstack([Fh, Fm])
full = trimesh.Trimesh(Vall, Fall, process=False); full.merge_vertices(merge_tex=True, merge_norm=True)
Vf, Ff = np.asarray(full.vertices), np.asarray(full.faces)
E = np.sort(np.concatenate([Ff[:, [0, 1]], Ff[:, [1, 2]], Ff[:, [2, 0]]]), 1); u, c = np.unique(E, axis=0, return_counts=True)
loops = trimesh.Trimesh(Vf, Ff, process=False).outline().entities
st0, st1 = stats(V, F), stats(Vf, Ff)
print(f"original: {int(st0[0])} v, edge spread {st0[2]:.3f}, 1% angle {st0[3]:.1f} deg, min angle {np.degrees(trimesh.Trimesh(V, F, process=False).face_angles.min()):.1f}", flush=True)
print(f"remeshed: {len(Vf)} v {len(Ff)} f, edge spread {st1[2]:.3f}, 1% angle {st1[3]:.1f} deg, min angle {np.degrees(trimesh.Trimesh(Vf, Ff, process=False).face_angles.min()):.1f}, "
      f"euler {len(Vf) - len(u) + len(Ff)}, non-manifold edges {(c > 2).sum()}, boundary loops {len(loops)}, vertices on the plane {int((np.abs(Vf @ n - d) < 1e-9).sum())}", flush=True)
trimesh.Trimesh(Vf, Ff, process=False).export(dst)
_, dd, _ = trimesh.proximity.closest_point(trimesh.Trimesh(V, F, process=False), Vf)
print(f"distance of the new vertices to the original surface: max {dd.max() / h:.4f} x mean edge", flush=True)
