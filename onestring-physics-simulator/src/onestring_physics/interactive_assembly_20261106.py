"""2026-11-06: interactive assembly bench - the T2D tiles as rigid bodies in the browser (Rapier physics, three.js view).

Tiles are convex rigid bodies of their own T2D shape (collision shape 2 % smaller, so hinged corners start apart);
every hinge is a ball joint at the midpoint of its two hinged corners.  Hinged tiles still collide with each other:
with dual hinges that contact makes each hinge fold one way only.  The outer-boundary drawstring of
drawstring_20261106 can be pulled with a set tension; tiles can be grabbed and pulled with the mouse or a finger;
gravity, the floor and friction can be switched.  The assembled shape (T3D) can be shown on top, fitted to the current
shape, with the remaining error.

This is game physics (fast, approximate: small overlaps and joint slack can occur) for trying things out; the IPC
simulation (ipc_deployment_20261105) is the accurate one.

Units on the page: one tile edge (median) = 1, y up; gravity = 9.81 m/s^2 in those units at the print scale.
"""
from __future__ import annotations

import json

import numpy as np


def assembly_data(state, mm_per_unit: float = 128.0) -> dict:
    from .ipc_deployment_20261105 import _rigid_place_target
    from .drawstring_20261106 import drawstring_segments
    T2 = np.asarray(state.tiles_2d_dual_hinge.vertices, float).copy()
    T3 = np.asarray(state.tiles_3d.vertices, float)
    T2[..., 2] -= T2[..., 2].min()
    T3p, _, _ = _rigid_place_target(T2, T3)
    edge = float(np.median(np.linalg.norm(np.roll(T2[:, :4], -1, 1) - T2[:, :4], axis=2)))
    centre = T2.reshape(-1, 3).mean(0)
    centre[2] = 0.0

    def to_page(P):
        P = (np.asarray(P, float) - centre) / edge
        return np.stack([P[..., 0], P[..., 2], -P[..., 1]], axis=-1)       # z up -> y up

    tiles = to_page(T2)
    hinges = []
    for h in state.hinge_graph.hinges:
        a, b = int(h.tile_a), int(h.tile_b)
        va, vb = int(h.local_vertex_a), int(h.local_vertex_b)
        if str(getattr(h, "surface", "top")).lower() == "bottom":
            va, vb = (va + 4 if va < 4 else va), (vb + 4 if vb < 4 else vb)
        p = 0.5 * (tiles[a, va] + tiles[b, vb])
        hinges.append([a, b, *np.round(p, 5).tolist()])
    strings = {route: [[s["a"], list(s["face_a"]), s["b"], list(s["face_b"])] for s in drawstring_segments(state, route)]
               for route in ("seam", "loop")}
    return dict(tiles=np.round(tiles, 5).tolist(), target=np.round(to_page(T3p), 5).tolist(), hinges=hinges,
                strings=strings, edge_mm=edge * mm_per_unit, n=int(len(tiles)))


def page_html(state, mm_per_unit: float = 128.0, mode: str = "app") -> str:
    """mode: "app" (inside the Streamlit page), "artifact" (published page; the host adds the document skeleton) or
    "file" (a complete .html file to open in any browser)"""
    data = json.dumps(assembly_data(state, mm_per_unit), separators=(",", ":"))
    html = _TEMPLATE.replace("__DATA__", data)
    if mode == "app":
        html = html.replace("<title>OneString 組み上げ実験台</title>", "")
    if mode == "file":
        html = ('<!doctype html><html lang="ja"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"></head><body>'
                + html + "</body></html>")
    return html


def render_in_app(st, state) -> None:
    import streamlit.components.v1 as components
    mm = float(st.session_state.get("print_mm_per_unit_for_bench", 128.0))
    st.caption("ブラウザの物理エンジン（Rapier）で、タイルをつかんで引っぱったり、外周の紐を絞ったりして試せます。"
               "手早く試すためのもので、わずかなめり込みや関節のゆるみが起きることがあります。きちんと確かめるのは IPC のシミュレーションです。")
    components.html(page_html(state, mm), height=820, scrolling=False)


_TEMPLATE = r"""<title>OneString 組み上げ実験台</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Zen+Kaku+Gothic+New:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: workbench — the 3D bench fills the main area, a narrow control rail beside it (below it on phones) */
:root{
  --bg:#eef0ec; --panel:#f8f9f6; --line:#d3d8cf; --fg:#1f2620; --muted:#5d685f;
  --pla:#d98b4a; --tpu:#2f8f6e; --string:#b8322a; --ghost:#3a6fb0; --scene:#e4e8e1; --pin:#6a4fa3;
  --font-ui:"Zen Kaku Gothic New","Hiragino Sans","Noto Sans JP",system-ui,sans-serif;
  --font-num:"IBM Plex Mono",ui-monospace,"SFMono-Regular",Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#151a17; --panel:#1c221e; --line:#2f3832; --fg:#e4e9e3; --muted:#9aa69c;
  --pla:#e09a5c; --tpu:#47b38c; --string:#e0574d; --ghost:#6f9ee0; --scene:#1a201c; --pin:#a28ce0; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#151a17; --panel:#1c221e; --line:#2f3832; --fg:#e4e9e3; --muted:#9aa69c;
  --pla:#e09a5c; --tpu:#47b38c; --string:#e0574d; --ghost:#6f9ee0; --scene:#1a201c; --pin:#a28ce0; color-scheme:dark}
html,body{height:100%}
[hidden]{display:none!important}
body{background:var(--bg);color:var(--fg);font-family:var(--font-ui);font-size:14px;
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:12px;box-sizing:border-box}
.wrap{display:grid;grid-template-columns:minmax(0,1fr) 272px;gap:14px;height:100%;min-height:560px}
.bench{position:relative;min-width:0;border:1px solid var(--line);border-radius:6px;overflow:hidden;background:var(--scene);touch-action:none}
.bench canvas{display:block;width:100%;height:100%}
.hint{position:absolute;left:10px;bottom:8px;font-size:12px;color:var(--muted);pointer-events:none}
.status{position:absolute;inset:0;display:grid;place-items:center;color:var(--muted);font-size:13px}
.rail{display:flex;flex-direction:column;gap:14px;min-width:0;overflow-y:auto}
h1{font-size:17px;font-weight:700;margin:0;text-wrap:balance;letter-spacing:.02em}
.sub{font-size:12px;color:var(--muted);margin-top:2px}
.group{display:flex;flex-direction:column;gap:8px;padding:12px;background:var(--panel);border:1px solid var(--line);border-radius:6px}
.label{font-size:11px;letter-spacing:.08em;color:var(--muted)}
.row{display:flex;align-items:center;justify-content:space-between;gap:8px}
.num{font-family:var(--font-num);font-variant-numeric:tabular-nums;font-size:13px}
.big{font-family:var(--font-num);font-variant-numeric:tabular-nums;font-size:22px;font-weight:500}
button{font:inherit;font-weight:500;padding:8px 10px;border-radius:5px;border:1px solid var(--line);background:var(--bg);color:var(--fg);cursor:pointer}
button.primary{background:var(--string);border-color:var(--string);color:#fff}
button:focus-visible,input:focus-visible{outline:2px solid var(--ghost);outline-offset:2px}
.buttons{display:grid;grid-template-columns:1fr 1fr;gap:8px}
input[type=range]{width:100%;accent-color:var(--string)}
label.check{display:flex;align-items:center;gap:8px;cursor:pointer}
.swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
@media (max-width:760px){.wrap{grid-template-columns:1fr;grid-template-rows:62vh auto;height:auto}}
@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
</style>
<div class="wrap">
  <div class="bench" id="bench"><div class="status" id="status">物理エンジンを読み込んでいます…</div>
    <div class="hint">タイルをドラッグでつかむ ・ 何もないところをドラッグで視点を回す ・ ホイール／ピンチで拡大</div></div>
  <div class="rail">
    <div><h1>組み上げ実験台</h1><div class="sub" id="sub"></div></div>
    <div class="group">
      <div class="label">外周の巾着の紐</div>
      <div class="row" role="radiogroup" aria-label="紐の通し方">
        <label class="check"><input type="radio" name="route" id="routeSeam" value="seam" checked> 切れ目でかがる</label>
        <label class="check"><input type="radio" name="route" id="routeLoop" value="loop"> 外周を一周</label>
      </div>
      <button class="primary" id="pull">紐を絞る</button>
      <div class="row"><span>引く力（タイル 1 枚の重さの倍）</span><span class="num" id="tensionv"></span></div>
      <input type="range" id="tension" min="0" max="100" step="1" value="20">
      <div class="sub" id="tensionN"></div>
      <div class="row"><span><span class="swatch" style="background:var(--string)"></span>すき間を渡る紐の長さ</span><span class="num" id="slen">100%</span></div>
    </div>
    <div class="group">
      <div class="label">つかんだパネル</div>
      <label class="check"><input type="checkbox" id="pinOnRelease" checked> <span><span class="swatch" style="background:var(--pin)"></span>離したらその位置で固定する</span></label>
      <button id="unpinAll" disabled>固定を全部解除</button>
      <div class="sub">固定したパネルはもう一度つかめば動かせます。ダブルクリック（ダブルタップ）で 1 枚だけ解除します。</div>
    </div>
    <div class="group">
      <div class="label">目標形状（T3D）とのずれ</div>
      <div class="row"><span class="big" id="err">–</span><span class="num">× 辺</span></div>
      <label class="check"><input type="checkbox" id="ghost"> <span><span class="swatch" style="background:var(--ghost)"></span>目標形状を重ねて表示</span></label>
      <div class="sub">今の形に一番よく重なる位置に置いて測ります（位置や向きのずれは数えません）。</div>
    </div>
    <div class="group">
      <div class="label">環境</div>
      <label class="check"><input type="checkbox" id="gravity" checked> 重力</label>
      <label class="check"><input type="checkbox" id="floor" checked> 床（机）</label>
      <div class="row"><span>摩擦</span><span class="num" id="frictionv"></span></div>
      <input type="range" id="friction" min="0" max="1" step="0.05" value="0.4">
      <div class="buttons"><button id="pause">一時停止</button><button id="reset">最初に戻す</button></div>
      <div class="row"><span class="sub">1 コマの計算</span><span class="num" id="ms">–</span></div>
    </div>
  </div>
</div>
<script type="importmap">{"imports":{
  "three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
  "three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/",
  "rapier":"https://cdn.jsdelivr.net/npm/@dimforge/rapier3d-compat@0.12.0/rapier.es.js"}}</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import RAPIER from "rapier";
const D = __DATA__;
const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const N = D.n, G = 9810 / D.edge_mm;           // gravity in tile edges / s^2
$("sub").textContent = `タイル ${N} 枚 ・ ヒンジ ${D.hinges.length} 個 ・ 辺 ${D.edge_mm.toFixed(1)} mm`;
$("status").textContent = "物理エンジンを読み込んでいます…";
await RAPIER.init();
$("status").hidden = true;

// ---- view
const bench = $("bench");
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(2, devicePixelRatio));
bench.prepend(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 500);
let span = 0; for (const t of D.tiles) for (const v of t) span = Math.max(span, Math.hypot(v[0], v[2]));
camera.position.set(span * 1.2, span * 1.5, span * 1.9);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1, 0); controls.update();
scene.add(new THREE.HemisphereLight(0xffffff, 0x445544, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.4); sun.position.set(30, 60, 20); scene.add(sun);
const floorMesh = new THREE.Mesh(new THREE.CircleGeometry(span * 3, 64).rotateX(-Math.PI / 2),
  new THREE.MeshStandardMaterial({roughness:1}));
floorMesh.position.y = -0.002; scene.add(floorMesh);
function themeColors(){
  scene.background = new THREE.Color(css("--scene"));
  floorMesh.material.color = new THREE.Color(css("--panel"));
  plaMat.color = new THREE.Color(css("--pla")); ghostMat.color = new THREE.Color(css("--ghost")); pinMat.color = new THREE.Color(css("--pin"));
  stringMat.color = new THREE.Color(css("--string"));
}
const BOX = [[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]];
const plaMat = new THREE.MeshStandardMaterial({roughness:.7, flatShading:true, side:THREE.DoubleSide});
const pinMat = new THREE.MeshStandardMaterial({roughness:.7, flatShading:true, side:THREE.DoubleSide});
const ghostMat = new THREE.MeshBasicMaterial({transparent:true, opacity:.22, depthWrite:false, side:THREE.DoubleSide});
const stringMat = new THREE.LineBasicMaterial();
function boxGeometry(P, c){
  const pos = []; for (const f of BOX) for (const k of f) pos.push(P[k][0]-c[0], P[k][1]-c[1], P[k][2]-c[2]);
  const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3)); g.computeVertexNormals(); return g;
}
const centres = D.tiles.map(P => [0,1,2].map(d => P.reduce((s, v) => s + v[d], 0) / 8));
const meshes = D.tiles.map((P, i) => { const m = new THREE.Mesh(boxGeometry(P, centres[i]), plaMat); m.userData.i = i; scene.add(m); return m; });
// target (T3D), one merged mesh moved as a whole onto the current shape
const ghostGeo = (() => { const pos = []; for (const P of D.target) for (const f of BOX) for (const k of f) pos.push(...P[k]);
  const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3)); return g; })();
const ghost = new THREE.Mesh(ghostGeo, ghostMat); ghost.matrixAutoUpdate = false; ghost.visible = false; scene.add(ghost);
const stringLine = new THREE.LineSegments(new THREE.BufferGeometry(), stringMat); scene.add(stringLine);
themeColors();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", themeColors);
new MutationObserver(themeColors).observe(document.documentElement, {attributes:true, attributeFilter:["data-theme"]});

// ---- physics
let world, bodies, floorCol, colliders;
const faceCentre = (i, f) => [0,1,2].map(d => f.reduce((s, k) => s + D.tiles[i][k][d], 0) / 4 - centres[i][d]);
// the string's gap crossings for the chosen route ("seam": laced across the cut, "loop": once around the outline)
let strLocal = [];
function setRoute(){ const r = $("routeSeam").checked ? "seam" : "loop";
  strLocal = D.strings[r].map(([a, fa, b, fb]) => [a, faceCentre(a, fa), b, faceCentre(b, fb)]);
  // the shown length is relative to the flat layout, so switching the route mid-pull stays comparable
  L0 = strLocal.reduce((s, [a, la, b, lb]) => s + Math.hypot(...[0,1,2].map(d => centres[b][d] + lb[d] - centres[a][d] - la[d])), 0); }
function build(){
  if (world) world.free();
  world = new RAPIER.World({x:0, y: $("gravity").checked ? -G : 0, z:0});
  world.timestep = 1 / 480;
  if ("numSolverIterations" in world.integrationParameters) world.integrationParameters.numSolverIterations = 16;
  colliders = [];
  bodies = D.tiles.map((P, i) => {
    const c = centres[i];
    const b = world.createRigidBody(RAPIER.RigidBodyDesc.dynamic().setTranslation(c[0], c[1], c[2])
      .setLinearDamping(3).setAngularDamping(3));
    const pts = new Float32Array(P.flatMap(v => [0,1,2].map(d => (v[d] - c[d]) * 0.98)));
    const cd = RAPIER.ColliderDesc.convexHull(pts) || RAPIER.ColliderDesc.cuboid(.5, .4, .5);
    colliders.push(world.createCollider(cd.setDensity(1).setFriction(+$("friction").value), b));
    return b;
  });
  for (const [a, b, x, y, z] of D.hinges) {
    const ca = centres[a], cb = centres[b];
    const j = world.createImpulseJoint(RAPIER.JointData.spherical({x:x-ca[0], y:y-ca[1], z:z-ca[2]}, {x:x-cb[0], y:y-cb[1], z:z-cb[2]}), bodies[a], bodies[b], true);
  }
  setFloor();
  for (const m of meshes) m.position.set(...centres[m.userData.i]);
  pull = 0;
  pinned.clear(); for (const m of meshes) m.material = plaMat; showPins();
}
function setFloor(){
  if (floorCol) { world.removeCollider(floorCol, false); floorCol = null; }
  if ($("floor").checked) floorCol = world.createCollider(RAPIER.ColliderDesc.cuboid(span * 4, 1, span * 4).setTranslation(0, -1, 0).setFriction(+$("friction").value));
  floorMesh.visible = $("floor").checked;
}
const tileWeight = () => { let m = 0; for (const b of bodies) m += b.mass(); return m / N * G; };
let pulling = false, pull = 0, paused = false, L0 = null;
const tmp = new THREE.Vector3(), q = new THREE.Quaternion();
function worldPoint(b, local){ const t = b.translation(), r = b.rotation(); q.set(r.x, r.y, r.z, r.w);
  return tmp.set(local[0], local[1], local[2]).applyQuaternion(q).add(new THREE.Vector3(t.x, t.y, t.z)).clone(); }
let grab = null;   // {body, local, target}
function applyForces(){
  for (const b of bodies) { b.resetForces(true); b.resetTorques(true); }   // addForceAtPoint adds a torque too
  pull += ((pulling ? 1 : 0) - pull) * 0.004;   // ease the tension in and out over about a second
  const T = +$("tension").value * tileWeight() * pull;
  let L = 0; const pts = [];
  for (const [a, la, b, lb] of strLocal) {
    const pa = worldPoint(bodies[a], la), pb = worldPoint(bodies[b], lb);
    const d = pb.clone().sub(pa), len = d.length(); L += len; pts.push(pa, pb);
    if (T > 0 && len > 1e-6) { d.multiplyScalar(T / len);
      bodies[a].addForceAtPoint({x:d.x, y:d.y, z:d.z}, pa, true);
      bodies[b].addForceAtPoint({x:-d.x, y:-d.y, z:-d.z}, pb, true); }
  }
  if (L0 === null) L0 = L;
  if (grab) {
    const b = grab.body, p = worldPoint(b, grab.local), v = b.linvel(), m = b.mass();
    const k = 60 * m * G, c = 2 * Math.sqrt(k * m);
    const f = grab.target.clone().sub(p).multiplyScalar(k).sub(new THREE.Vector3(v.x, v.y, v.z).multiplyScalar(c));
    b.addForceAtPoint({x:f.x, y:f.y, z:f.z}, p, true);
  }
  return {L, pts};
}
// ---- error to the target after the best rigid fit (Horn's quaternion method)
function bestFit(){
  const A = [], B = [];
  for (let i = 0; i < N; i++) { const b = bodies[i];
    for (let k = 0; k < 8; k++) { const P = D.tiles[i][k], c = centres[i];
      A.push(worldPoint(b, [P[0]-c[0], P[1]-c[1], P[2]-c[2]])); const T = D.target[i][k]; B.push(new THREE.Vector3(T[0], T[1], T[2])); } }
  const ca = new THREE.Vector3(), cb = new THREE.Vector3();
  A.forEach(v => ca.add(v)); B.forEach(v => cb.add(v)); ca.divideScalar(A.length); cb.divideScalar(B.length);
  const S = [[0,0,0],[0,0,0],[0,0,0]];
  for (let i = 0; i < A.length; i++) { const a = B[i].clone().sub(cb).toArray(), b = A[i].clone().sub(ca).toArray();
    for (let r = 0; r < 3; r++) for (let s = 0; s < 3; s++) S[r][s] += a[r] * b[s]; }
  const [[xx,xy,xz],[yx,yy,yz],[zx,zy,zz]] = S;
  const M = [[xx+yy+zz, yz-zy, zx-xz, xy-yx],[yz-zy, xx-yy-zz, xy+yx, zx+xz],[zx-xz, xy+yx, -xx+yy-zz, yz+zy],[xy-yx, zx+xz, yz+zy, -xx-yy+zz]];
  let shift = 0; for (const r of M) shift += Math.abs(r[0]) + Math.abs(r[1]) + Math.abs(r[2]) + Math.abs(r[3]);
  let x = [1, 0.1, 0.1, 0.1];
  for (let it = 0; it < 200; it++) { const y = M.map((r, i) => r.reduce((s, m, j) => s + m * x[j], 0) + shift * x[i]);
    const nrm = Math.hypot(...y); x = y.map(v => v / nrm); }
  const rot = new THREE.Quaternion(x[1], x[2], x[3], x[0]).normalize();      // maps target onto current
  let e = 0; for (let i = 0; i < A.length; i++) e += B[i].clone().sub(cb).applyQuaternion(rot).add(ca).distanceToSquared(A[i]);
  const m = new THREE.Matrix4().makeTranslation(ca.x, ca.y, ca.z).multiply(new THREE.Matrix4().makeRotationFromQuaternion(rot))
    .multiply(new THREE.Matrix4().makeTranslation(-cb.x, -cb.y, -cb.z));
  return {err: Math.sqrt(e / A.length), m};
}
// ---- input
const ray = new THREE.Raycaster(), ndc = new THREE.Vector2(), plane = new THREE.Plane();
function pointer(e){ const r = renderer.domElement.getBoundingClientRect(); ndc.set((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1); ray.setFromCamera(ndc, camera); }
// pinned panels: kinematic bodies that keep the pose they had when released
const pinned = new Set();
function showPins(){ $("unpinAll").disabled = pinned.size === 0;
  $("unpinAll").textContent = pinned.size ? `固定を全部解除（${pinned.size} 枚）` : "固定を全部解除"; }
function pin(i){ const b = bodies[i]; b.setBodyType(RAPIER.RigidBodyType.KinematicPositionBased, true);
  b.setLinvel({x:0, y:0, z:0}, true); b.setAngvel({x:0, y:0, z:0}, true); pinned.add(i); meshes[i].material = pinMat; showPins(); }
function unpin(i){ if (!pinned.has(i)) return; const b = bodies[i]; b.setBodyType(RAPIER.RigidBodyType.Dynamic, true);
  b.wakeUp(); pinned.delete(i); meshes[i].material = plaMat; showPins(); }
renderer.domElement.addEventListener("pointerdown", e => {
  pointer(e); const hit = ray.intersectObjects(meshes)[0]; if (!hit) return;
  const i = hit.object.userData.i; unpin(i);
  const b = bodies[i], t = b.translation(), r = b.rotation();
  const local = hit.point.clone().sub(new THREE.Vector3(t.x, t.y, t.z)).applyQuaternion(new THREE.Quaternion(r.x, r.y, r.z, r.w).invert());
  grab = {i, body: b, local: local.toArray(), target: hit.point.clone(), start: hit.point.clone()};
  plane.setFromNormalAndCoplanarPoint(camera.getWorldDirection(new THREE.Vector3()).negate(), hit.point);
  controls.enabled = false; renderer.domElement.setPointerCapture(e.pointerId); b.wakeUp();
});
renderer.domElement.addEventListener("pointermove", e => { if (!grab) return; pointer(e); ray.ray.intersectPlane(plane, grab.target); });
// a panel that was actually moved (not just clicked) stays where it was let go
const release = () => { if (grab && $("pinOnRelease").checked && grab.target.distanceTo(grab.start) > 0.2) pin(grab.i);
  grab = null; controls.enabled = true; };
renderer.domElement.addEventListener("dblclick", e => { pointer(e); const hit = ray.intersectObjects(meshes)[0]; if (hit) unpin(hit.object.userData.i); });
$("unpinAll").onclick = () => { for (const i of [...pinned]) unpin(i); };
renderer.domElement.addEventListener("pointerup", release); renderer.domElement.addEventListener("pointercancel", release);
$("pull").onclick = () => { pulling = !pulling; $("pull").textContent = pulling ? "紐をゆるめる" : "紐を絞る"; for (const b of bodies) b.wakeUp(); };
// one tile weighs about (edge x edge x thickness) of PLA; show the pull in newtons for a feel of the force
const tileGrams = (() => { const P = D.tiles[0], c = centres[0]; let h = 0; for (let k = 0; k < 4; k++) h += P[k][1] - P[k + 4][1];
  return 1.24 * Math.pow(D.edge_mm / 10, 3) * (h / 4); })();
const showT = () => { $("tensionv").textContent = (+$("tension").value).toFixed(0);
  $("tensionN").textContent = `約 ${(+$("tension").value * tileGrams * 9.81 / 1000).toFixed(2)} N（タイル 1 枚 約 ${tileGrams.toFixed(1)} g、PLA として）`; };
const showF = () => $("frictionv").textContent = (+$("friction").value).toFixed(2);
$("tension").oninput = showT; showT();
$("friction").oninput = () => { showF(); for (const c of colliders) c.setFriction(+$("friction").value); if (floorCol) floorCol.setFriction(+$("friction").value); }; showF();
$("gravity").onchange = () => { world.gravity = {x:0, y: $("gravity").checked ? -G : 0, z:0}; for (const b of bodies) b.wakeUp(); };
$("floor").onchange = setFloor;
$("routeSeam").onchange = $("routeLoop").onchange = () => { setRoute(); for (const b of bodies) b.wakeUp(); };
$("ghost").onchange = () => ghost.visible = $("ghost").checked;
$("pause").onclick = () => { paused = !paused; $("pause").textContent = paused ? "再開" : "一時停止"; };
$("reset").onclick = () => { pulling = false; $("pull").textContent = "紐を絞る"; build(); };
// ---- loop
function resize(){ const w = bench.clientWidth, h = bench.clientHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); }
new ResizeObserver(resize).observe(bench); resize();
setRoute();
build();
let frame = 0, msAvg = 0;
function loop(){
  requestAnimationFrame(loop);
  let st = null;
  if (!paused) { const t0 = performance.now();
    for (let s = 0; s < 8; s++) { st = applyForces(); world.step(); }
    msAvg = msAvg * 0.9 + (performance.now() - t0) * 0.1; }
  else st = applyForces();
  for (const m of meshes) { const b = bodies[m.userData.i], t = b.translation(), r = b.rotation(); m.position.set(t.x, t.y, t.z); m.quaternion.set(r.x, r.y, r.z, r.w); }
  stringLine.geometry.setFromPoints(st.pts);
  $("slen").textContent = (100 * st.L / L0).toFixed(1) + "%";
  if (frame++ % 15 === 0) { const f = bestFit(); $("err").textContent = f.err.toFixed(3); ghost.matrix.copy(f.m); $("ms").textContent = msAvg.toFixed(1) + " ms"; }
  controls.update(); renderer.render(scene, camera);
}
loop();
</script>
"""
