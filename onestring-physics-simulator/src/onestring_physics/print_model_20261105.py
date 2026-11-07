"""2026-11-05: preview of the printed part made from the T2D Dual Hinge result (H2D, PLA tiles + TPU hinges).

Layout: the flat T2D tiles, scaled so the whole layout fits the build plate (the larger of the two orientations, a
margin kept), bottom faces on the plate.

Hinges are a TPU membrane, one per face (top / bottom), as in the paper's prints: on each face the membrane covers the
whole face of every tile that has a hinge on that face, and the tiles are joined only at their hinges, where the two
tiles' corners touch.  There the membrane fills the acute tip of the gap and ends in a smooth curve tangent to both tile
edges; on the obtuse side it follows the tile corners exactly (see _hinge_fill).  The gaps between tiles stay open in
the membrane too, so they can close when the string is pulled; where two tiles have no hinge on a face, the membrane is
not joined at all.  Hinges whose corners do not quite touch in the final layout are still joined up to a set distance:
the joint is lengthened along the edges and widened until it bridges them.  The membrane takes the outermost layer of the tile
(the PLA tile is cut back by the membrane thickness on that face and the membrane fills the cut), so the part height
is unchanged and every tile stands on the build plate, with or without a bottom membrane.

Each tile gets a cross-shaped string channel: from the middle of every side face, at mid-height, to the tile centre;
its diameter is a percentage of that tile's mean edge.  This is a preview (the parts are shown overlapping, no
boolean cut yet); the printable export comes next.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_BOX = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
                 [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]])


@dataclass
class PrintSettings:
    bed_x_mm: float = 300.0          # H2D, both nozzles
    bed_y_mm: float = 320.0
    margin_mm: float = 10.0
    hole_pct_of_edge: float = 25.0   # string channel diameter, % of the tile's mean edge
    membrane_thickness_mm: float = 0.6   # paper 6.1 (TPU hinge thickness)
    hinge_neck_mm: float = 1.04          # narrowest width of the membrane joint between two tiles at a hinge
                                         # (1.3 x the paper's 0.8 mm hinge width)
    hinge_max_gap_mm: float = 2.0        # hinge corners up to this far apart are still joined (the joint is lengthened
                                         # and widened to bridge them); farther ones are left unjoined


def build_print_model(state, cfg: PrintSettings) -> dict:
    T = np.asarray(state.tiles_2d_dual_hinge.vertices, float)          # (n, 8, 3): 0-3 top, 4-7 bottom
    hinges = list(state.hinge_graph.hinges)
    lo, hi = T.reshape(-1, 3).min(0), T.reshape(-1, 3).max(0)
    ext = hi - lo
    W, H = cfg.bed_x_mm - 2 * cfg.margin_mm, cfg.bed_y_mm - 2 * cfg.margin_mm
    s0, s1 = min(W / ext[0], H / ext[1]), min(W / ext[1], H / ext[0])
    rotate = s1 > s0
    s = max(s0, s1)
    P = (T - lo) * s                                                    # mm, bottom of the layout at z = 0
    if rotate:
        P = np.stack([P[..., 1], ext[0] * s - P[..., 0], P[..., 2]], -1)
    P[..., 0] += cfg.margin_mm
    P[..., 1] += cfg.margin_mm
    top, bot = P[:, :4], P[:, 4:]
    centre = P[:, :8].reshape(len(P), 8, 3).mean(1)
    edge = np.linalg.norm(np.roll(top, -1, 1) - top, axis=2)            # (n, 4)
    thick = np.linalg.norm(top - bot, axis=2).mean(1)
    # string channels: side-face centres -> tile centre
    holes = []
    for i in range(len(P)):
        d = cfg.hole_pct_of_edge / 100.0 * edge[i].mean()
        for k in range(4):
            side = (top[i, k] + top[i, (k + 1) % 4] + bot[i, k] + bot[i, (k + 1) % 4]) / 4.0
            holes.append((side, centre[i], d))
    # hinge membranes: per face, the faces of the tiles that carry a hinge there + one joint at each hinge
    t = cfg.membrane_thickness_mm
    membrane = {"top": dict(tiles=set(), bridges=[], bridge_z=[]), "bottom": dict(tiles=set(), bridges=[], bridge_z=[])}
    gaps_joined, gaps_left = [], []
    for h in hinges:
        a, b = int(h.tile_a), int(h.tile_b)
        side = "top" if str(getattr(h, "surface", "top")).lower() == "top" else "bottom"
        face = top if side == "top" else bot
        va, vb = int(h.local_vertex_a) % 4, int(h.local_vertex_b) % 4
        membrane[side]["tiles"].update((a, b))
        poly, gap = _hinge_fill(face[a, :, :2], va, face[b, :, :2], vb, cfg.hinge_neck_mm, cfg.hinge_max_gap_mm)
        if poly is None:
            gaps_left.append(gap)
        else:
            gaps_joined.append(gap)
            membrane[side]["bridges"].append(poly)
            # the joint spans the membrane layers of both tiles (their faces may sit at slightly different heights)
            za, zb = face[a, va, 2], face[b, vb, 2]
            membrane[side]["bridge_z"].append((min(za, zb) - t, max(za, zb)) if side == "top"
                                              else (min(za, zb), max(za, zb) + t))
    # the membrane is not added onto the tile: the PLA tile is cut back by the membrane thickness on that face and the
    # membrane fills the cut, so every tile keeps its full height and stands on the plate whether it has a bottom
    # membrane or not.  The cut follows the tile's side edges (tiles may have slanted sides).
    P_pla = P.copy()
    height = np.maximum(top[:, :, 2] - bot[:, :, 2], 1e-9)                # (n, 4) along each side edge
    frac = np.clip(t / height, 0.0, 0.45)[..., None]
    for side, face in (("top", top), ("bottom", bot)):
        m = membrane[side]
        m["tile_boxes"] = []                                            # (8, 3): 0-3 upper face, 4-7 lower face
        for i in sorted(m["tiles"]):
            if side == "top":
                cut = top[i] - (top[i] - bot[i]) * frac[i]
                P_pla[i, :4] = cut
                m["tile_boxes"].append(np.vstack([top[i], cut]))
            else:
                cut = bot[i] + (top[i] - bot[i]) * frac[i]
                P_pla[i, 4:] = cut
                m["tile_boxes"].append(np.vstack([cut, bot[i]]))
        m["polygons"] = [face[i, :, :2] for i in sorted(m["tiles"])] + m["bridges"]
        z = float(face[:, :, 2].mean())
        m["z"] = (z - t, z) if side == "top" else (z, z + t)
    hole_d = np.array([x[2] for x in holes]).reshape(len(P), 4).mean(1)
    covered = np.zeros(len(P))
    for side in ("top", "bottom"):
        for i in membrane[side]["tiles"]:
            covered[i] = max(covered[i], t)
    wall = (thick - hole_d) / 2.0 - covered        # PLA left between a channel and the membrane / outer face
    return dict(P=P, P_pla=P_pla, holes=holes, membrane=membrane, scale=s, rotated=rotate,
                size_mm=(P[..., :2].reshape(-1, 2).max(0) - P[..., :2].reshape(-1, 2).min(0)).tolist() + [float(P[..., 2].max())],
                edge=edge, thick=thick, hole_d=hole_d, wall=wall, n_hinges=len(hinges),
                gaps_joined=np.array(gaps_joined), gaps_left=np.array(gaps_left))


def _seg_dist(p, a, b):
    ab = b - a
    t = 0.0 if ab @ ab < 1e-18 else float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _hinge_fill(qa, va, qb, vb, width, max_gap, seg=16):
    """membrane that joins tiles a and b at a hinge (qa, qb: their face corners (4, 2); va, vb: the hinge corners).
    Returns (polygon, gap between the two corners), or (None, gap) when the corners are farther apart than max_gap.

    Around the hinge corners the gap between the two tiles has an acute side and an obtuse side.  On the obtuse side
    nothing is added beyond the straight line between the two corners, so the membrane follows the tile corners.  On the
    acute side the gap is filled and the fill ends in a smooth curve that is tangent to both tile edges, so the gap
    keeps a rounded tip.  The curve is a cubic Bezier whose four points all lie on the two tile edges (from the point d
    along each edge, back half way towards the corner), so it always stays between the two edges - also when the
    corners do not quite touch and the edges are nearly parallel (the 2026-11-05 version used the point where the edge
    lines meet, which is extremely far away for nearly parallel edges and blew the preview up to millions of mm).
    When the corners do not quite touch, the same fill bridges the space between them.  The fill is lengthened along
    the edges until its narrowest part - from the obtuse-side line to the curve - is at least `width` (or until it
    reaches 45 % of an edge)."""
    a0, b0 = qa[va], qb[vb]
    gap = float(np.linalg.norm(b0 - a0))
    if gap > max_gap:
        return None, gap
    ua = [qa[(va - 1) % 4] - a0, qa[(va + 1) % 4] - a0]
    ub = [qb[(vb - 1) % 4] - b0, qb[(vb + 1) % 4] - b0]
    unit = lambda v: v / max(np.linalg.norm(v), 1e-12)
    # the acute side: the pair of edges (one per tile) with the smallest angle between them
    _, ea, eb = min(((np.arccos(np.clip(unit(x) @ unit(y), -1, 1)), x, y) for x in ua for y in ub), key=lambda z: z[0])
    theta = np.arccos(np.clip(unit(ea) @ unit(eb), -1, 1))
    cap = 0.45 * min(np.linalg.norm(ea), np.linalg.norm(eb))
    # for touching corners the curve's middle is 0.625 d cos(theta/2) from the corner
    d = min(1.6 * width / max(np.cos(theta / 2.0), 0.2), cap)
    u = np.linspace(0.0, 1.0, seg)[:, None]
    while True:
        p0, p1 = a0 + d * unit(ea), a0 + 0.5 * d * unit(ea)
        p2, p3 = b0 + 0.5 * d * unit(eb), b0 + d * unit(eb)
        curve = (1 - u) ** 3 * p0 + 3 * u * (1 - u) ** 2 * p1 + 3 * u ** 2 * (1 - u) * p2 + u ** 3 * p3
        narrowest = min(_seg_dist(c, a0, b0) for c in curve)
        if narrowest >= width or d >= cap:
            break
        d = min(d * 1.15, cap)
    pts = [a0] + list(curve) + ([b0] if gap > 1e-9 else [])
    return np.array(pts), gap


def _prism(poly, z0, z1):
    """a flat polygon (k, 2) extruded from z0 to z1, fan-triangulated from its first point (every polygon here is
    star-shaped from it)"""
    poly = np.asarray(poly, float)
    k = len(poly)
    V = np.vstack([np.c_[poly, np.full(k, z1)], np.c_[poly, np.full(k, z0)]])
    F = [[0, j, j + 1] for j in range(1, k - 1)] + [[k, k + j + 1, k + j] for j in range(1, k - 1)]
    for j in range(k):
        j2 = (j + 1) % k
        F += [[j, k + j, k + j2], [j, k + j2, j2]]
    return V, np.array(F)


def _tube(a, b, d, seg=10):
    ax = b - a
    L = np.linalg.norm(ax)
    if L < 1e-9:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    ax /= L
    u = np.cross(ax, [0, 0, 1.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(ax, [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(ax, u)
    t = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ring = (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v) * d / 2.0
    V = np.vstack([a + ring, b + ring])
    F = []
    for k in range(seg):
        k2 = (k + 1) % seg
        F += [[k, k2, seg + k2], [k, seg + k2, seg + k]]
    return V, np.array(F)


def preview_figure(model: dict, cfg: PrintSettings, show_holes: bool = True):
    import plotly.graph_objects as go
    P = model["P_pla"]                       # the PLA tiles, already cut back where a membrane takes their outer layer
    n = len(P)
    V = P.reshape(-1, 3)
    F = (_BOX[None] + 8 * np.arange(n)[:, None, None]).reshape(-1, 3)
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2], color="#e08a3c",
                            opacity=0.45 if show_holes else 1.0, flatshading=True, name="タイル（PLA）",
                            showlegend=True, hoverinfo="skip"))
    # hinge membranes, one trace per face: the layer cut out of each tile + the joints between tiles, flush with the
    # tile faces
    for side, label, color in (("bottom", "下面の膜（TPU）", "#1d8f6a"), ("top", "上面の膜（TPU）", "#2bb38a")):
        m = model["membrane"][side]
        z0, z1 = m["z"]
        hv, hf = [], []
        for box in m["tile_boxes"]:
            hf.append(_BOX + len(hv))
            hv += np.asarray(box).tolist()
        for poly, (z0, z1) in zip(m["bridges"], m["bridge_z"]):
            v, f = _prism(poly, z0, z1)
            hf.append(f + len(hv))
            hv += v.tolist()
        if hv:
            hv, hf = np.array(hv), np.vstack(hf)
            fig.add_trace(go.Mesh3d(x=hv[:, 0], y=hv[:, 1], z=hv[:, 2], i=hf[:, 0], j=hf[:, 1], k=hf[:, 2],
                                    color=color, flatshading=True, name=label, showlegend=True,
                                    lighting=dict(ambient=0.8), hoverinfo="skip"))
    if show_holes:
        tv, tf = [], []
        for a, b, d in model["holes"]:
            v, f = _tube(np.asarray(a), np.asarray(b), d)
            if len(v):
                tf.append(f + len(tv))
                tv += v.tolist()
        if tv:
            tv, tf = np.array(tv), np.vstack(tf)
            fig.add_trace(go.Mesh3d(x=tv[:, 0], y=tv[:, 1], z=tv[:, 2], i=tf[:, 0], j=tf[:, 1], k=tf[:, 2],
                                    color="#2f6db5", flatshading=True, name="紐の穴", showlegend=True, hoverinfo="skip"))
    bx = [0, cfg.bed_x_mm, cfg.bed_x_mm, 0, 0]
    by = [0, 0, cfg.bed_y_mm, cfg.bed_y_mm, 0]
    fig.add_trace(go.Scatter3d(x=bx, y=by, z=[0] * 5, mode="lines", line=dict(color="#888", width=3),
                               name="造形範囲", hoverinfo="skip"))
    fig.update_layout(height=640, margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h"),
                      scene=dict(aspectmode="data", xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)"))
    return fig


def render_print_preview(st, state) -> None:
    """Streamlit section: settings, figures and the numbers that matter for printing."""
    c1, c2, c3, c4 = st.columns(4)
    cfg = PrintSettings(
        bed_x_mm=c1.number_input("造形範囲 x (mm)", 50.0, 400.0, 300.0, 5.0, key="print_bed_x"),
        bed_y_mm=c2.number_input("造形範囲 y (mm)", 50.0, 400.0, 320.0, 5.0, key="print_bed_y"),
        margin_mm=c3.number_input("余白 (mm)", 0.0, 50.0, 10.0, 1.0, key="print_margin"),
        hole_pct_of_edge=c4.number_input("紐の穴の径（タイルの辺に対する %）", 0.0, 80.0, 25.0, 1.0, key="print_hole_pct"),
    )
    d1, d2, d3 = st.columns(3)
    cfg.membrane_thickness_mm = d1.number_input("ヒンジの膜の厚さ (mm)", 0.1, 5.0, 0.6, 0.1, key="print_membrane_t")
    cfg.hinge_neck_mm = d2.number_input("ヒンジで膜がタイル同士をつなぐ幅 (mm)", 0.1, 10.0, 1.04, 0.05, key="print_hinge_neck_v2")
    cfg.hinge_max_gap_mm = d2.number_input("角どうしがこの距離まで離れていてもヒンジをつなぐ (mm)", 0.0, 20.0, 2.0, 0.25,
                                           key="print_hinge_max_gap")
    show_holes = d3.checkbox("紐の穴を表示する（タイルを半透明にする）", value=True, key="print_show_holes")
    m = build_print_model(state, cfg)
    mt, mb = m["membrane"]["top"], m["membrane"]["bottom"]
    sx, sy, sz = m["size_mm"]
    st.caption(f"倍率 {m['scale']:.1f} mm / 単位{'（90° 回転して配置）' if m['rotated'] else ''}　・　配置の大きさ {sx:.0f} × {sy:.0f} mm、高さ {sz:.1f} mm　・　"
               f"タイル {len(m['P'])} 枚：辺 中央値 {np.median(m['edge']):.1f} mm（最小 {m['edge'].min():.1f}・最大 {m['edge'].max():.1f}）、厚さ 中央値 {np.median(m['thick']):.1f} mm　・　"
               f"穴の径 {m['hole_d'].min():.1f}〜{m['hole_d'].max():.1f} mm、穴の上下に残る肉 最小 {m['wall'].min():.2f} mm　・　"
               f"ヒンジ {m['n_hinges']} 個　・　上面の膜：タイル {len(mt['tiles'])} 枚・つなぎ目 {len(mt['bridges'])}　・　"
               f"下面の膜：タイル {len(mb['tiles'])} 枚・つなぎ目 {len(mb['bridges'])}　・　"
               f"膜は足すのではなく、PLA のタイルを膜の厚さの分だけ削ってそこを膜にしています（高さは変わらず、どのタイルも造形台に接します）")
    gj, gl = m["gaps_joined"], m["gaps_left"]
    far = gj[gj > 0.05] if len(gj) else gj
    if len(far):
        st.caption(f"角どうしが離れていたヒンジ {len(far)} 個は、膜をのばしてつないでいます（離れ方 最大 {far.max():.2f} mm）。")
    if len(gl):
        st.warning(f"角どうしが {cfg.hinge_max_gap_mm:.2f} mm より離れているヒンジが {len(gl)} 個あり、つないでいません"
                   f"（最大 {gl.max():.2f} mm）。つなぐ距離の上限を上げるとつながります。")
    if m["wall"].min() < 1.0:
        st.warning(f"穴と膜（または外面）のあいだに残る PLA が 1 mm 未満のタイルが {int((m['wall'] < 1.0).sum())} 枚あります。"
                   "穴の径か膜の厚さを小さくしてください。")
    st.plotly_chart(preview_figure(m, cfg, show_holes), use_container_width=True, key="print_preview_fig")
    render_stl_download(st, m, cfg, state)


# ---- 2026-11-05: STL export (PLA tiles and TPU membranes as two files, for one two-material print) ---------------
def _stl_bytes(V, F) -> bytes:
    """binary STL"""
    import struct
    V = np.asarray(V, np.float32)
    F = np.asarray(F, np.int64)
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(F), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, tri
    return b"OneString print export".ljust(80, b" ") + struct.pack("<I", len(F)) + rec.tobytes()


def _manifold_mesh(man):
    m = man.to_mesh()
    return np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts)


def _cylinder(a, b, r, seg=24):
    """a closed cylinder from point a to point b as a Manifold"""
    from manifold3d import Manifold
    a, b = np.asarray(a, float), np.asarray(b, float)
    ax = b - a
    L = float(np.linalg.norm(ax))
    z = ax / L
    x = np.cross(z, [0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.cross(z, [1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    M = np.c_[x, y, z, a]                                  # 3 x 4: columns map the cylinder's x, y, z axes; last = origin
    return Manifold.cylinder(L, r, -1.0, seg).transform(M.tolist())


def _ccw(poly):
    poly = np.asarray(poly, float)
    area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    return poly if area >= 0 else poly[::-1]


def export_parts(model: dict) -> dict:
    """{'PLA': (V, F), 'TPU': (V, F)} in mm, positioned on the build plate.

    PLA: every tile, cut back where a membrane takes its outer layer, with its cross-shaped string channel removed
    (each arm runs from just outside the side face to just past the tile centre).
    TPU: per face, the membrane layer of every hinged tile and the joints between tiles merged into one solid."""
    from manifold3d import Manifold, CrossSection, OpType
    P = model["P_pla"]
    n = len(P)
    holes = model["holes"]
    tiles = []
    for i in range(n):
        tile = Manifold.hull_points(np.asarray(P[i], float).tolist())
        arms = []
        for a, c, d in holes[4 * i:4 * i + 4]:
            if d <= 0:
                continue
            u = (np.asarray(a) - np.asarray(c))
            u /= max(np.linalg.norm(u), 1e-12)
            arms.append(_cylinder(np.asarray(c) - u * d / 2.0, np.asarray(a) + u * 1.0, d / 2.0))
        if arms:
            tile = tile - Manifold.batch_boolean(arms, OpType.Add)
        tiles.append(tile)
    pla = Manifold.compose(tiles)
    tpu_faces = []
    for side in ("top", "bottom"):
        m = model["membrane"][side]
        parts = [Manifold.hull_points(np.asarray(b, float).tolist()) for b in m["tile_boxes"]]
        for poly, (z0, z1) in zip(m["bridges"], m["bridge_z"]):
            cs = CrossSection([_ccw(poly)])
            parts.append(Manifold.extrude(cs, float(z1 - z0)).translate([0.0, 0.0, float(z0)]))
        if parts:
            tpu_faces.append(Manifold.batch_boolean(parts, OpType.Add))
    tpu = Manifold.compose(tpu_faces) if tpu_faces else None
    out = {"PLA": _manifold_mesh(pla)}
    if tpu is not None:
        out["TPU"] = _manifold_mesh(tpu)
    return out


_README = """OneString 印刷用ファイル（{date}、アプリのバージョン {version}）

PLA_tiles.stl   PLA のタイル（ヒンジの膜の分を削り、十字の紐の穴を開けたもの）
TPU_membrane.stl  TPU のヒンジの膜（上面と下面。タイル同士のつなぎ目を含む）
単位は mm。2 つは同じ座標で作ってあるので、重ねればそのまま 1 つの部品になります。

Bambu Studio（H2D）での読み込み方
1. 2 つのファイルを同時に選んで読み込み、「1 つのオブジェクトとして読み込みますか（複数部品）」で「はい」を選ぶ。
2. オブジェクトの中の部品ごとに、PLA_tiles に PLA、TPU_membrane に TPU（95A HF）のフィラメントを割り当てる。
3. 造形台への配置は動かさずにそのままスライスする（下面の膜は造形台の高さ 0 から始まります）。

設定：造形範囲 {bed}、膜の厚さ {t} mm、つなぎ目の幅 {neck} mm、紐の穴の径 タイルの辺の {hole} %
"""


def export_zip(model: dict, cfg: PrintSettings) -> bytes:
    import io
    import zipfile
    parts = export_parts(model)
    try:
        from .app_version_20261105 import current_version
        version = current_version()["n"]
    except Exception:   # noqa: BLE001
        version = "?"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("PLA_tiles.stl", _stl_bytes(*parts["PLA"]))
        if "TPU" in parts:
            z.writestr("TPU_membrane.stl", _stl_bytes(*parts["TPU"]))
        z.writestr("README.txt", _README.format(
            date=__import__("time").strftime("%Y-%m-%d %H:%M"), version=version,
            bed=f"{cfg.bed_x_mm:.0f} × {cfg.bed_y_mm:.0f} mm", t=cfg.membrane_thickness_mm,
            neck=cfg.hinge_neck_mm, hole=cfg.hole_pct_of_edge))
    return buf.getvalue()


def render_stl_download(st, model: dict, cfg: PrintSettings, state) -> None:
    import time as _time
    st.markdown("#### ヒンジ付きの STL をダウンロード")
    try:
        import manifold3d  # noqa: F401
    except Exception:   # noqa: BLE001
        st.warning("STL を作るには manifold3d が必要です。アプリを止めて、ターミナルで `pip install manifold3d` を実行してから起動し直してください。")
        return
    key = repr(cfg) + f"|{id(state)}"
    if st.button("STL を作る（PLA と TPU の 2 ファイルを zip にまとめる）", key="print_make_stl"):
        with st.spinner("タイルに紐の穴を開け、膜をまとめています…"):
            t0 = _time.perf_counter()
            data = export_zip(model, cfg)
        st.session_state["print_stl_zip"] = (key, data, _time.strftime("%Y%m%d-%H%M"), _time.perf_counter() - t0)
    saved = st.session_state.get("print_stl_zip")
    if saved and saved[0] == key:
        _, data, stamp, secs = saved
        st.download_button(f"zip をダウンロード（{len(data) / 1e6:.1f} MB、{secs:.0f} 秒で作成）", data,
                           file_name=f"onestring_print_{stamp}.zip", mime="application/zip", key="print_stl_download")
        st.caption("zip の中の README.txt に、Bambu Studio で PLA と TPU を割り当てる手順を書いてあります。")
    elif saved:
        st.caption("設定が変わったので、もう一度「STL を作る」を押してください。")
