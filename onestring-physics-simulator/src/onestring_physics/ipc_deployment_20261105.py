"""2026-11-05: deployment simulation (flat T2D -> assembled) with Affine Body Dynamics + Incremental Potential Contact.

Method
* Tiles are affine bodies (ABD, Lan et al. 2022): each tile has 12 unknowns q = (p, A), its vertices are
  x = p + A (X - c); a stiff orthogonality energy k_o |A^T A - I|^2 keeps it rigid.  This is the light way to get
  IPC-quality contact for rigid parts (12 unknowns per tile instead of 3 per vertex).
* Time stepping is the incremental potential of implicit Euler: each step minimises
      1/(2 h^2) |q - q_pred|_M^2 + orthogonality + hinges + snap + lift + contact barrier + ground barrier
  with projected Newton, and every Newton step is shortened by continuous collision detection (ipctk), so tiles
  never pass through each other or through the ground (IPC, Li et al. 2020).  Velocities come from the actual motion
  v = (q_new - q_old) / h.
* Actuation, one of
  - "snap_lift" as the paper (Sec. 5.4): Snap pulls each pair of side-face midpoints of a routed gap together, Lift
    pulls each lift point to its assembled position.  Both are ramped together over the actuation time.
  - "drawstring" (2026-11-06): one closed string through the outer boundary (drawstring_20261106), pulled with a
    tension ramped up over the actuation time and then held.  The string slides through the tiles, so the tension is
    the same along it; its potential is tension x length of the gap crossings.  Nothing else steers the tiles: the
    shape comes from hinges, contact, gravity and the floor.  Contact geometry is shrunk only 0.2 % with a 0.2 %
    contact distance, so faces that meet in the assembled shape can really close.
* The assembled shape is first placed on the flat layout (rigid fit about the vertical axis of the tile centres,
  bottom on the ground), so the lift targets and the comparison with T3D are in the flat layout's frame.
* Hinges: a stiff ball joint at the midpoint of the two hinged corners, held in both tiles' frames (the printed hinge
  spans their small T2D gap), so it is exactly satisfied at the start; the contact geometry is shrunk slightly, so
  hinged corners never register as a contact.
* The run continues after the ramp until the tiles stop moving (kinetic energy and step size below tolerance).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

_BOX_TRIS = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])


@dataclass
class IPCSettings:
    mm_per_unit: float = 113.0        # print scale (for gravity in model units)
    dt: float = 1.0 / 60.0
    actuation_seconds: float = 4.0
    settle_tolerance: float = 1e-4    # max vertex speed (x tile edge / s) to call the structure settled
    max_seconds: float = 30.0         # simulated-time guard
    gravity_on: bool = True
    dhat_rel: float = 1e-2            # contact activation distance, x median tile edge (0.12 mm at 113 mm/unit)
    hinge_stiffness: float = 1e4      # x tile mass / h^2
    snap_stiffness: float = 2e2
    lift_stiffness: float = 2e2
    orth_stiffness: float = 1e4
    collision_shrink: float = 0.03    # contact geometry = tile shrunk by this fraction about its centre: starts
                                      # intersection-free, and closed gaps / hinges stay outside dhat (about 1.5 % edge per side)
    newton_velocity_tol: float = 1e-2 # Newton stops when |search direction|_inf / dt < this x scene diagonal / s (IPC's
                                      # criterion; the 2026-11-05 version used the tile edge, 30 x stricter)
    actuation: str = "snap_lift"      # "snap_lift" (paper Sec. 5.4) or "drawstring" (outer boundary drawstring)
    string_tension: float = 20.0      # drawstring: final tension, x the weight of ONE tile
    string_route: str = "seam"        # drawstring: "seam" (laces the two sides of each cut together) or "loop"
                                      # (once around the whole outline, down one side of a cut and back up the other)
    hinged_pairs_collide: bool = True # two tiles joined by a hinge still collide with each other: with dual hinges
                                      # this contact is what lets each hinge fold one way only
    frames: int = 120
    progress: object = None


def _rigid_place_target(T2: np.ndarray, T3: np.ndarray):
    """T3D tile centres fitted to the T2D tile centres by a rotation about z + translation; bottom on z = 0."""
    c2, c3 = T2.mean(1)[:, :2], T3.mean(1)[:, :2]
    a, b = c2 - c2.mean(0), c3 - c3.mean(0)
    U, _, Vt = np.linalg.svd(b.T @ a)
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T

    def apply(P):
        P = np.asarray(P, float)
        q = P.copy()
        q[..., :2] = (P[..., :2] - c3.mean(0)) @ R.T + c2.mean(0)
        return q
    T3p = apply(T3)
    dz = -T3p[..., 2].min()
    T3p[..., 2] += dz

    def apply_point(P):
        q = apply(P)
        q[..., 2] += dz
        return q

    def to_t3d(P):
        """layout frame -> the T3D frame (inverse of the placement)"""
        P = np.asarray(P, float)
        q = P.copy()
        q[..., 2] -= dz
        q[..., :2] = (P[..., :2] - c2.mean(0)) @ R + c3.mean(0)
        return q
    return T3p, apply_point, to_t3d


def _all_gap_actuation(state):
    """Snap every interior gap (when the run has no valid Sec. 5.3 route) and lift the selected lift points."""
    from .paper_pd_deployment import _Snap, _Lift, _opposite_side_pairs, _material_gap_face
    rest = np.asarray(state.tiles_2d_dual_hinge.vertices, float)
    snaps, lifts = [], []
    for gap in state.gap_graph.gaps:
        if gap.boundary:
            continue
        tiles = sorted(set(map(int, gap.surrounding_tiles)))
        try:
            material = {t: _material_gap_face(gap, t) for t in tiles}
        except Exception:   # noqa: BLE001
            material = None
        for a, b, fa, fb in _opposite_side_pairs(rest, tiles, np.asarray(gap.centroid_2d, float)[:2], material):
            snaps.append(_Snap(int(gap.id), a, b, fa, fb))
    gaps = {int(g.id): g for g in state.gap_graph.gaps}
    for point in state.lift_points:
        gap = gaps[int(point.gap_id)]
        owned = tuple(sorted(set(map(int, gap.surrounding_tiles))))
        try:
            faces = tuple(_material_gap_face(gap, t)[:2] for t in owned)
        except Exception:   # noqa: BLE001
            faces = tuple((0, 1) for _ in owned)
        ref = np.mean([rest[t, list(f)].mean(axis=0) for t, f in zip(owned, faces)], axis=0)
        lifts.append(_Lift(int(gap.id), owned, faces, ref, np.asarray(point.position_3d, float)))
    return snaps, lifts


def simulate_ipc_deployment(state, cfg: IPCSettings | None = None) -> dict:
    import ipctk
    from .paper_pd_deployment import _validate_route
    cfg = cfg or IPCSettings()
    t_wall = time.perf_counter()
    T2 = np.asarray(state.tiles_2d_dual_hinge.vertices, float).copy()
    T3 = np.asarray(state.tiles_3d.vertices, float)
    n = len(T2)
    T2[..., 2] -= T2[..., 2].min()                                    # flat layout on the ground
    T2[..., 2] += 0.5 * cfg.dhat_rel * float(np.median(np.linalg.norm(np.roll(T2[:, :4], -1, 1) - T2[:, :4], axis=2)))
    T3p, place, to_t3d = _rigid_place_target(T2, T3)
    segs = []
    if cfg.actuation == "drawstring":
        from .drawstring_20261106 import drawstring_segments
        segs = drawstring_segments(state, cfg.string_route)
        snaps, lifts = [], []
        n_seam = sum(1 for sg in segs if sg.get("seam"))
        route_note = (f"outer-boundary drawstring, {len(segs)} gap crossings"
                      + (f", laced across the seam ({n_seam} crossings)" if n_seam else ", once around the outline"))
    else:
        try:
            snaps, lifts, _ = _validate_route(state)
            route_note = "Sec. 5.3 string route"
        except Exception as exc:   # noqa: BLE001
            snaps, lifts = _all_gap_actuation(state)
            route_note = f"no valid string route ({exc}); every interior gap is snapped"
    edge = float(np.median(np.linalg.norm(np.roll(T2[:, :4], -1, 1) - T2[:, :4], axis=2)))
    dhat = cfg.dhat_rel * edge
    h = cfg.dt
    g = 9810.0 / cfg.mm_per_unit if cfg.gravity_on else 0.0

    # ---- affine bodies -------------------------------------------------------------------------------------
    c = T2.mean(1)                                                   # (n,3) rest centres
    r = T2 - c[:, None, :]                                           # (n,8,3) rest offsets
    V = n * 8
    def jac(rr):
        rows, cols, vals = [], [], []
        for i in range(n):
            for k in range(8):
                v = i * 8 + k
                for d in range(3):
                    rows.append(3 * v + d); cols.append(12 * i + d); vals.append(1.0)            # p
                    for e in range(3):
                        rows.append(3 * v + d); cols.append(12 * i + 3 + 3 * d + e); vals.append(rr[i, k, e])  # A[d,e]
        return sp.csr_matrix((vals, (rows, cols)), shape=(3 * V, 12 * n))
    J = jac(r)
    # contact geometry: each tile shrunk by cfg.collision_shrink about its centre, so the T2D layout (which allows
    # contact up to 0.1 % of the edge) starts intersection-free, as IPC requires
    Jc = jac(r * (1.0 - cfg.collision_shrink))
    q = np.zeros(12 * n)
    for i in range(n):
        q[12 * i:12 * i + 3] = c[i]
        q[12 * i + 3:12 * i + 12] = np.eye(3).ravel()
    mass_v = np.repeat(np.full(n, 1.0 / 8.0), 8)                       # unit mass per tile
    Mv = sp.diags(np.repeat(mass_v, 3))
    M = (J.T @ Mv @ J).tocsr()
    m_tile = 1.0
    w0 = m_tile / h ** 2

    def verts(qv):
        return (J @ qv).reshape(V, 3)

    def cverts(qv):
        return (Jc @ qv).reshape(V, 3)

    # ---- linear actuation / hinge maps (point selectors) ---------------------------------------------------
    def sel(pairs):
        """rows: points = mean of listed vertices; returns sparse (3P x 3V)"""
        R_, C_, X_ = [], [], []
        for row, vlist in enumerate(pairs):
            for v in vlist:
                for d in range(3):
                    R_.append(3 * row + d); C_.append(3 * v + d); X_.append(1.0 / len(vlist))
        return sp.csr_matrix((X_, (R_, C_)), shape=(3 * len(pairs), 3 * V))
    hinges = list(state.hinge_graph.hinges)
    # hinge = ball joint at the midpoint of the two hinged corners (the printed hinge spans their small T2D gap), held
    # in both tiles' frames: p_a + A_a r_a = p_b + A_b r_b, exactly satisfied at the start
    R_, C_, X_ = [], [], []
    for row, h_ in enumerate(hinges):
        a_, b_ = int(h_.tile_a), int(h_.tile_b)
        mid = 0.5 * (T2[a_, int(h_.local_vertex_a)] + T2[b_, int(h_.local_vertex_b)])
        for tile, sign in ((a_, 1.0), (b_, -1.0)):
            off = mid - c[tile]
            for d in range(3):
                R_.append(3 * row + d); C_.append(12 * tile + d); X_.append(sign)
                for e in range(3):
                    R_.append(3 * row + d); C_.append(12 * tile + 3 + 3 * d + e); X_.append(sign * off[e])
    Hd = sp.csr_matrix((X_, (R_, C_)), shape=(3 * len(hinges), 12 * n))
    Sa = sel([[s.a * 8 + f for f in s.face_a] for s in snaps]) @ J
    Sb = sel([[s.b * 8 + f for f in s.face_b] for s in snaps]) @ J
    Sd = (Sa - Sb).tocsr()
    # drawstring gap crossings (face centre of tile a -> face centre of tile b), as linear maps of q
    Dd = (sel([[sg["a"] * 8 + f for f in sg["face_a"]] for sg in segs]) @ J
          - sel([[sg["b"] * 8 + f for f in sg["face_b"]] for sg in segs]) @ J).tocsr()
    T_full = cfg.string_tension * 1.0 * 9810.0 / cfg.mm_per_unit           # x the weight of one tile (unit mass)
    snap_rest = Sd @ q                                               # flat-layout separations
    Lm = sel([[t * 8 + f for t, face in zip(L.tiles, L.faces) for f in face] for L in lifts]) @ J
    lift_ref = Lm @ q
    lift_tgt = np.concatenate([place(L.target) for L in lifts]) if lifts else np.zeros(0)
    k_h = cfg.hinge_stiffness * w0
    Hh = (Hd.T @ Hd).tocsr() * k_h

    # ---- collision mesh -------------------------------------------------------------------------------------
    F = (np.asarray(_BOX_TRIS)[None] + 8 * np.arange(n)[:, None, None]).reshape(-1, 3).astype(np.int32)
    E = ipctk.edges(F) if hasattr(ipctk, "edges") else None
    if E is None:
        es = set()
        for f in F:
            for a_, b_ in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                es.add((min(a_, b_), max(a_, b_)))
        E = np.array(sorted(es), dtype=np.int32)
    X0 = cverts(q)
    mesh = ipctk.CollisionMesh(np.asfortranarray(X0), np.asfortranarray(E), np.asfortranarray(F))
    # Optionally, two tiles joined by a hinge can be kept from colliding with each other (cfg.hinged_pairs_collide =
    # False; filters ipctk's broad-phase candidates with a table of hinged pairs, a few ms per call).  By default they
    # do collide: with dual hinges that contact is what makes each hinge fold one way only (2026-11-06; the 2026-11-05
    # evening version excluded them, after hinged pairs were found pressed together - that pressing came from the
    # Snap targets and the weak barrier, not from the contact itself).
    hinged = np.zeros((n, n), bool)
    for h_ in hinges:
        hinged[int(h_.tile_a), int(h_.tile_b)] = hinged[int(h_.tile_b), int(h_.tile_a)] = True
    e_tile = E[:, 0] // 8
    f_tile = F[:, 0] // 8

    def candidates(Xa, Xb=None, inflation=0.0):
        c_ = ipctk.Candidates()
        if Xb is None:
            c_.build(mesh, np.asfortranarray(Xa), inflation)
        else:
            c_.build(mesh, np.asfortranarray(Xa), np.asfortranarray(Xb), inflation)
        if cfg.hinged_pairs_collide:
            return c_
        ee_ = c_.ee_candidates
        if ee_:
            ids = np.array([(x.edge0_id, x.edge1_id) for x in ee_])
            keep = ~hinged[e_tile[ids[:, 0]], e_tile[ids[:, 1]]]
            c_.ee_candidates = [x for x, k in zip(ee_, keep) if k]
        fv_ = c_.fv_candidates
        if fv_:
            ids = np.array([(x.face_id, x.vertex_id) for x in fv_])
            keep = ~hinged[f_tile[ids[:, 0]], ids[:, 1] // 8]
            c_.fv_candidates = [x for x, k in zip(fv_, keep) if k]
        ev_ = c_.ev_candidates
        if ev_:
            ids = np.array([(x.edge_id, x.vertex_id) for x in ev_])
            keep = ~hinged[e_tile[ids[:, 0]], ids[:, 1] // 8]
            c_.ev_candidates = [x for x, k in zip(ev_, keep) if k]
        return c_

    def contacts(X):
        col_ = ipctk.NormalCollisions()
        col_.build(candidates(X, None, dhat), mesh, np.asfortranarray(X), dhat)
        return col_
    # barrier stiffness, IPC's own rule (ipctk.initial_barrier_stiffness): start at the minimum stiffness for this
    # scene size and mass, raise it to balance the other forces when contacts appear, and double it while a contact
    # keeps closing in below dhat/10; never above IPC's maximum.  (2026-11-05 first version: a weight-based stiffness
    # 36 x below IPC's minimum - hinge and Snap forces then pressed hinged tiles to 1e-11 apart, and every collision-
    # free step size took 0.8 s.)
    bbox_diag = float(np.linalg.norm(X0.max(0) - X0.min(0)))
    kappa_min, kappa_max = ipctk.initial_barrier_stiffness(bbox_diag, ipctk.ClampedLogBarrier(), dhat, 1.0 / 8.0,
                                                           np.zeros(12 * n), np.zeros(12 * n))
    kappa = [kappa_min]
    prev_dmin = [float('inf')]
    barrier = ipctk.BarrierPotential(dhat, 1.0)
    # continuous collision detection: Tight Inclusion with tolerance 1e-3 and at most 1000 iterations per query (the
    # default 1e-6 / 1e7 costs 3-4 x as much and, with two elements nearly touching at the start of a step, minutes
    # for one step).  When a query hits the limit it returns a smaller, still collision-free step.
    ccd = ipctk.TightInclusionCCD(1e-3, 1000)

    # ---- energy ---------------------------------------------------------------------------------------------
    def orth_terms(qv, need_h):
        A = qv.reshape(n, 12)[:, 3:].reshape(n, 3, 3)
        S = np.einsum("nji,njk->nik", A, A) - np.eye(3)
        k_o = cfg.orth_stiffness * w0 * edge ** 2
        e = k_o * np.sum(S ** 2)
        G = 4 * k_o * np.einsum("nij,njk->nik", A, S)
        g_full = np.zeros((n, 12)); g_full[:, 3:] = G.reshape(n, 9)
        if not need_h:
            return e, g_full.ravel(), None
        blocks = np.zeros((n, 9, 9))
        for idx in range(9):
            dA = np.zeros((n, 3, 3)); dA.reshape(n, 9)[:, idx] = 1.0
            dS = np.einsum("nji,njk->nik", dA, A) + np.einsum("nji,njk->nik", A, dA)
            dG = 4 * k_o * (np.einsum("nij,njk->nik", dA, S) + np.einsum("nij,njk->nik", A, dS))
            blocks[:, :, idx] = dG.reshape(n, 9)
        blocks = 0.5 * (blocks + blocks.transpose(0, 2, 1))
        w_, Q_ = np.linalg.eigh(blocks)
        blocks = np.einsum("nij,nj,nkj->nik", Q_, np.maximum(w_, 0.0), Q_)
        R_, C_ = np.meshgrid(np.arange(9), np.arange(9), indexing="ij")
        rr = (12 * np.arange(n)[:, None, None] + 3 + R_[None]).ravel()
        cc = (12 * np.arange(n)[:, None, None] + 3 + C_[None]).ravel()
        Hm = sp.csr_matrix((blocks.ravel(), (rr, cc)), shape=(12 * n, 12 * n))
        return e, g_full.ravel(), Hm

    def ground_terms(X, need_h):
        z = X[:, 2]
        act = z < dhat
        e = 0.0; gx = np.zeros_like(X); hz = np.zeros(V)
        if act.any():
            d = np.maximum(z[act], 1e-12)
            k = kappa[0]
            e = float(-k * np.sum((d - dhat) ** 2 * np.log(d / dhat)))
            gz = -k * (2 * (d - dhat) * np.log(d / dhat) + (d - dhat) ** 2 / d)
            gx[act, 2] = gz
            hz[act] = np.maximum(-k * (2 * np.log(d / dhat) + 4 * (d - dhat) / d - (d - dhat) ** 2 / d ** 2), 0.0)
        Hx = sp.diags(np.column_stack([np.zeros(V), np.zeros(V), hz]).ravel()) if need_h else None
        return e, gx.ravel(), Hx

    def string_terms(qv, T_s, need_h):
        """tension x length of the drawstring's gap crossings (the part inside the tiles is constant)"""
        if not len(segs) or T_s <= 0:
            return 0.0, 0.0, None
        d = (Dd @ qv).reshape(-1, 3)
        L = np.linalg.norm(d, axis=1)
        Le = np.maximum(L, 1e-3 * edge)
        u = d / Le[:, None]
        e = T_s * float(L.sum())
        g_ = T_s * (Dd.T @ u.ravel())
        if not need_h:
            return e, g_, None
        B = (np.eye(3)[None] - u[:, :, None] * u[:, None, :]) * (T_s / Le)[:, None, None]
        S_ = len(segs)
        rr = (3 * np.arange(S_)[:, None, None] + np.arange(3)[None, :, None] + 0 * np.arange(3)[None, None, :]).ravel()
        cc = (3 * np.arange(S_)[:, None, None] + 0 * np.arange(3)[None, :, None] + np.arange(3)[None, None, :]).ravel()
        Bm = sp.csr_matrix((B.ravel(), (rr, cc)), shape=(3 * S_, 3 * S_))
        return e, g_, (Dd.T @ Bm @ Dd)

    def energy(qv, q_pred, ks, kl, snap_t, lift_t, need_g=True, need_h=True, collisions=None, T_s=0.0):
        X = cverts(qv)
        dq = qv - q_pred
        e = 0.5 * w0 * dq @ (M @ dq)
        gq = w0 * (M @ dq) if need_g else None
        Hq = w0 * M if need_h else None
        eo, go, Ho = orth_terms(qv, need_h)
        e += eo
        hv = Hd @ qv
        e += 0.5 * k_h * hv @ hv
        sv = Sd @ qv - snap_t
        e += 0.5 * ks * sv @ sv
        lv = Lm @ qv - lift_t if len(lift_t) else np.zeros(0)
        e += 0.5 * kl * lv @ lv
        if collisions is None:
            collisions = contacts(X)
        eb = kappa[0] * barrier(collisions, mesh, np.asfortranarray(X)) if len(collisions) else 0.0
        eg, gg, Hg = ground_terms(X, need_h)
        es, gs, Hs = string_terms(qv, T_s, need_h)
        e += eb + eg + es
        if need_g:
            gq = gq + go + k_h * (Hd.T @ hv) + ks * (Sd.T @ sv) + (kl * (Lm.T @ lv) if len(lv) else 0.0) + gs
            gx = gg.copy()
            if len(collisions):
                gx += kappa[0] * barrier.gradient(collisions, mesh, np.asfortranarray(X))
            gq = gq + Jc.T @ gx
        if need_h:
            Hx = Hg
            if len(collisions):
                Hx = Hx + kappa[0] * barrier.hessian(collisions, mesh, np.asfortranarray(X),
                                                     project_hessian_to_psd=ipctk.PSDProjectionMethod.CLAMP)
            Hq = Hq + Ho + Hh + ks * (Sd.T @ Sd) + (kl * (Lm.T @ Lm) if len(lv) else 0) + Jc.T @ Hx @ Jc
            if Hs is not None:
                Hq = Hq + Hs
        return e, gq, Hq, collisions

    # ---- time stepping ----------------------------------------------------------------------------------------
    vq = np.zeros_like(q)
    grav_q = np.zeros_like(q)
    grav_q.reshape(n, 12)[:, 2] = -g                                  # gravity acts on the translation only
    frames, times, curves = [verts(q).reshape(n, 8, 3).copy()], [0.0], dict(snap=[], lift=[], hinge=[], dmin=[], newton=[], string=[])
    L0 = float(np.linalg.norm((Dd @ q).reshape(-1, 3), axis=1).sum()) if len(segs) else 0.0
    t = 0.0
    step = 0
    settled = False
    save_every = max(1, int(round((cfg.actuation_seconds + 1.0) / h / cfg.frames)))
    while t < cfg.max_seconds:
        phase = min(1.0, t / cfg.actuation_seconds) if cfg.actuation_seconds > 0 else 1.0
        ks = cfg.snap_stiffness * w0 * min(1.0, 2 * phase + 1e-3)
        kl = cfg.lift_stiffness * w0 * min(1.0, 2 * phase + 1e-3)
        T_s = T_full * phase if len(segs) else 0.0
        snap_t = (1.0 - phase) * snap_rest
        lift_t = (1.0 - phase) * lift_ref + phase * lift_tgt
        q_pred = q + h * vq + h * h * grav_q
        q_old = q.copy()
        for it in range(100):
            e0, gq, Hq, col = energy(q, q_pred, ks, kl, snap_t, lift_t, T_s=T_s)
            if it == 0 and len(col):
                # IPC: when contacts are present, raise kappa so the barrier balances the other forces
                Xc = np.asfortranarray(cverts(q))
                gb = Jc.T @ barrier.gradient(col, mesh, Xc)
                ge = gq - kappa[0] * gb
                k_new, _ = ipctk.initial_barrier_stiffness(bbox_diag, ipctk.ClampedLogBarrier(), dhat, 1.0 / 8.0, ge, gb)
                if k_new > kappa[0]:
                    kappa[0] = min(k_new, kappa_max)
                    e0, gq, Hq, col = energy(q, q_pred, ks, kl, snap_t, lift_t, T_s=T_s)
            Hq = (Hq + sp.eye(12 * n) * 1e-9 * w0).tocsc()
            dq = -spla.spsolve(Hq, gq)
            X0 = cverts(q)
            X1 = cverts(q + dq)
            alpha = candidates(X0, X1).compute_collision_free_stepsize(mesh, np.asfortranarray(X0),
                                                                       np.asfortranarray(X1), 0.0, ccd)
            dz = X1[:, 2] - X0[:, 2]
            down = dz < 0
            if down.any():
                alpha = min(alpha, float(np.min(np.where(down, (X0[:, 2] - 0.1 * np.minimum(X0[:, 2], dhat)) / np.maximum(-dz, 1e-30), 1.0))))
            alpha = min(1.0, 0.9 * alpha if alpha < 1.0 else 1.0)
            while alpha > 1e-10:
                e1 = energy(q + alpha * dq, q_pred, ks, kl, snap_t, lift_t, need_g=False, need_h=False, T_s=T_s)[0]
                if e1 <= e0 + 1e-4 * alpha * gq @ dq:
                    break
                alpha *= 0.5
            q = q + alpha * dq
            if np.max(np.abs(J @ dq)) / h < cfg.newton_velocity_tol * bbox_diag:
                break
        X = cverts(q)
        col = contacts(X)
        # ipctk returns the SQUARED minimum distance
        dmin2 = float(col.compute_minimum_distance(mesh, np.asfortranarray(X))) if len(col) else float("inf")
        dmin = float(np.sqrt(dmin2))
        # adaptive stiffness as in IPC: double kappa while a contact keeps closing in, here below dhat/100 (dhat/10
        # drove kappa up to IPC's maximum and Newton to 100 iterations per step; IPC's own 1e-9 x scene diagonal let
        # contacts get so close that one collision-free step took minutes)
        if dmin < 1e-2 * dhat and dmin2 < prev_dmin[0]:
            kappa[0] = min(2.0 * kappa[0], kappa_max)
        prev_dmin[0] = dmin2
        vq = (q - q_old) / h
        t += h
        step += 1
        curves["snap"].append(float(np.sqrt(np.mean((Sd @ q) ** 2)) / edge) if len(snaps) else 0.0)
        curves["string"].append(float(np.linalg.norm((Dd @ q).reshape(-1, 3), axis=1).sum()) / L0 if L0 > 0 else 0.0)
        curves["lift"].append(float(np.sqrt(np.mean((Lm @ q - lift_tgt) ** 2)) / edge) if len(lift_tgt) else 0.0)
        curves["hinge"].append(float(np.sqrt(np.mean((Hd @ q) ** 2)) / edge))
        curves["dmin"].append(float(min(dmin, X[:, 2].min())) / edge)
        curves["newton"].append(it + 1)
        if step % save_every == 0:
            frames.append(verts(q).reshape(n, 8, 3).copy()); times.append(t)
        if cfg.progress is not None:
            what = (f"gap crossings {curves['string'][-1]:.3f} of their start length" if len(segs)
                    else f"snap {curves['snap'][-1]:.3f}")
            cfg.progress(min(0.99, t / (cfg.actuation_seconds + 2.0)), f"t = {t:.2f} s, {what}, Newton {it + 1}")
        vmax = np.max(np.abs(J @ vq)) / edge
        if phase >= 1.0 and vmax < cfg.settle_tolerance:
            settled = True
            break
    frames.append(verts(q).reshape(n, 8, 3).copy()); times.append(t)
    final = verts(q).reshape(n, 8, 3)
    # error to T3D after the best rigid fit (Kabsch): the gathered shape may end up shifted or turned on the floor
    A_ = final.reshape(-1, 3) - final.reshape(-1, 3).mean(0)
    B_ = T3p.reshape(-1, 3) - T3p.reshape(-1, 3).mean(0)
    U_, _, Vt_ = np.linalg.svd(A_.T @ B_)
    D_ = np.diag([1.0, 1.0, np.sign(np.linalg.det(U_ @ Vt_))])
    fit_err = float(np.sqrt(np.mean(np.sum((A_ @ (U_ @ D_ @ Vt_) - B_) ** 2, axis=1))) / edge)
    return dict(frames=frames, frames_t3d=[to_t3d(f) for f in frames], times=times, curves=curves, target=T3p, final=final, settled=settled,
                simulated_seconds=t, steps=step, wall_seconds=time.perf_counter() - t_wall,
                error_to_target=float(np.sqrt(np.mean((final - T3p) ** 2)) / edge), edge=edge, dhat=dhat,
                route=route_note, n_snaps=len(snaps), n_lifts=len(lifts), error_to_target_best_fit=fit_err,
                n_string_segments=len(segs))


def simulate_for_app(state, progress_callback=None, **settings):
    """App entry: runs the simulation and returns the app's DeploymentResult (frames in the T3D frame, so the
    translucent T3D target of the Assembly Animation view lines up)."""
    from .legacy_pipeline_base import DeploymentResult
    cfg = IPCSettings(**settings)
    if progress_callback is not None:
        cfg.progress = lambda f, s: progress_callback("ABD + IPC", f, s)
    r = simulate_ipc_deployment(state, cfg)
    frames = [np.asarray(f, float) for f in r["frames_t3d"]]
    metrics = dict(physics_backend="ipc_python", actual_backend="ABD + IPC (ipctk, CPU)",
                   simulation_runtime_seconds=float(r["wall_seconds"]), simulated_seconds=float(r["simulated_seconds"]),
                   time_steps=int(r["steps"]), settled=bool(r["settled"]),
                   final_error_to_T3D_rel_edge=float(r["error_to_target"]),
                   final_error_to_T3D_best_rigid_fit_rel_edge=float(r["error_to_target_best_fit"]),
                   drawstring_crossings_length_ratio=(float(r["curves"]["string"][-1])
                                                      if r["curves"]["string"] and r["n_string_segments"] else None),
                   min_contact_distance_rel_edge=float(min(r["curves"]["dmin"])) if r["curves"]["dmin"] else None,
                   snap_residual_rel_edge=float(r["curves"]["snap"][-1]) if r["curves"]["snap"] else None,
                   hinge_residual_rel_edge=float(r["curves"]["hinge"][-1]) if r["curves"]["hinge"] else None,
                   max_newton_iterations=int(max(r["curves"]["newton"])) if r["curves"]["newton"] else 0,
                   actuation=r["route"], snap_pairs=int(r["n_snaps"]), lift_points=int(r["n_lifts"]),
                   contact="IPC barrier + CCD line search (no penetration); contact geometry shrunk by "
                           f"{cfg.collision_shrink:.0%}", dhat_rel_edge=cfg.dhat_rel)
    return DeploymentResult(frames=frames, final_tiles=frames[-1], metrics=metrics, collision_counts=[0] * len(frames))
