"""Paper-grounded UI layer for the 2026-09-16 OneString/OptCuts view.

This is intentionally presentation-only: it does not alter numerical solvers.
It annotates controls with the paper equations, adds a Figure-5-derived pipeline
map above View stage, and augments Streamlit progress bars with a live glass
pipeline indicator while preserving the ordinary percentage bar below it.
"""
from __future__ import annotations

import html
from typing import Any


def _formula_card(title: str, equation: str, note: str) -> str:
    return f"""
<div class='os-eq-card'><div class='os-eq-kicker'>{html.escape(title)}</div>
<div class='os-eq'>{equation}</div><div class='os-eq-note'>{html.escape(note)}</div></div>
"""


def _pipeline_html(active: str = "", fraction: float | None = None, detail: str = "") -> str:
    nodes = [
        ("S", "Target"), ("Omega", "Ω"), ("M2D", "M2D"), ("M3D", "M3D"),
        ("K3D", "K3D"), ("T3D", "T3D"), ("K2D", "K2D"),
        ("T2D_TOP", "T2D Top"), ("T2D_DUAL", "T2D Dual"), ("HINGE", "Hinge Opt."),
    ]
    p = 0.0 if fraction is None else max(0.0, min(1.0, float(fraction)))
    cards = []
    for key, label in nodes:
        is_active = key == active
        cls = "os-node active" if is_active else "os-node"
        style = f"--p:{p*360:.1f}deg" if is_active else "--p:0deg"
        cards.append(f"<div class='{cls}' style='{style}'><div class='os-ring'><div class='os-inner'>{label}</div></div></div>")
    top = "".join(cards[:6])
    bottom = "".join(cards[6:])
    detail_html = f"<div class='os-live-detail'>{html.escape(detail)}</div>" if detail else ""
    return f"""
<div class='os-pipeline-wrap'>
  <div class='os-pipeline-title'>Paper Fig. 5 · Surface rationalization pipeline</div>
  <div class='os-pipeline-row'>{top}</div>
  <div class='os-branch'>↳ flat branch: M2D → K2D → extrusion → hinge placement / optimization</div>
  <div class='os-pipeline-row'>{bottom}</div>{detail_html}
</div>
"""


def _stage_from_progress(value: Any, text: str) -> tuple[str, float, str]:
    try:
        p = float(value)
        if p > 1.0:
            p /= 100.0
    except Exception:
        p = 0.0
    t = (text or "").lower()
    if "omega" in t or "parameter" in t or p < .18: return "Omega", p/.18 if p < .18 else .5, text
    if "m2d" in t or p < .30: return "M2D", (p-.18)/.12, text
    if "m3d" in t or p < .40: return "M3D", (p-.30)/.10, text
    if "k3d" in t or "planar" in t or p < .52: return "K3D", (p-.40)/.12, text
    if "t3d" in t or "extrusion" in t or p < .62: return "T3D", (p-.52)/.10, text
    if "k2d" in t or "edge" in t or p < .76: return "K2D", (p-.62)/.14, text
    if "hinge" in t or "t2d" in t or p < .90: return "HINGE", (p-.76)/.14, text
    return "T2D_DUAL", (p-.90)/.10, text


def install_paper_ui_20260916_patch() -> None:
    try:
        import streamlit as st
    except Exception:
        return
    if getattr(st, "_onestring_paper_ui_20260916", False):
        return

    st.markdown("""
<style>
.os-eq-card{padding:12px 14px;margin:8px 0 12px;border:1px solid rgba(120,130,150,.20);border-radius:14px;background:linear-gradient(135deg,rgba(255,255,255,.68),rgba(220,228,240,.26));backdrop-filter:blur(16px);box-shadow:0 8px 26px rgba(30,45,70,.06)}
.os-eq-kicker{font-size:11px;letter-spacing:.08em;text-transform:uppercase;opacity:.58;font-weight:700}.os-eq{font-family:Georgia,serif;font-size:17px;margin:5px 0}.os-eq-note{font-size:12px;opacity:.68}
.os-pipeline-wrap{padding:15px;margin:8px 0 14px;border:1px solid rgba(120,130,150,.18);border-radius:18px;background:linear-gradient(145deg,rgba(255,255,255,.74),rgba(222,230,242,.28));backdrop-filter:blur(18px);box-shadow:0 12px 32px rgba(25,40,70,.07)}
.os-pipeline-title{font-size:12px;font-weight:700;letter-spacing:.04em;opacity:.68;margin-bottom:10px}.os-pipeline-row{display:flex;gap:8px;flex-wrap:wrap}.os-node{min-width:78px}.os-ring{padding:2px;border-radius:12px;background:rgba(130,140,155,.25)}.os-node.active .os-ring{background:conic-gradient(from -90deg,rgba(37,143,255,.78) 0 var(--p),rgba(130,140,155,.24) var(--p) 360deg);box-shadow:0 0 18px rgba(60,155,255,.16)}.os-inner{padding:9px 10px;border-radius:10px;background:rgba(248,250,253,.86);text-align:center;font-size:12px;font-weight:650}.os-branch{font-size:11px;opacity:.55;margin:8px 0}.os-live-detail{font-size:11px;opacity:.65;margin-top:9px}
</style>""", unsafe_allow_html=True)

    original_markdown = st.markdown
    original_selectbox = st.selectbox
    original_progress = st.progress

    # Equation group cards are inserted immediately before the first control in
    # each relevant group.  They distinguish paper terms from implementation knobs.
    shown: set[str] = set()
    def _show_once(key: str, title: str, equation: str, note: str) -> None:
        if key in shown: return
        shown.add(key)
        original_markdown(_formula_card(title, equation, note), unsafe_allow_html=True)

    control_fns = {}
    annotations = {
        "K3D AL anchor weight (w_a)": ("k3d", "Paper Sec. 4.2 · Eq. (1)", "E<sub>Assembled</sub>(v)=ω₁E<sub>Planar</sub>+ω₂E<sub>Square</sub>+ω₃E<sub>Surface</sub>", "w_a is an implementation-side augmented-Lagrangian stabilization knob; it is not one of the paper's ω₁,ω₂,ω₃."),
        "max 3D iterations": ("k3d", "Paper Sec. 4.2 · Eq. (1)", "E<sub>Assembled</sub>(v)=ω₁E<sub>Planar</sub>+ω₂E<sub>Square</sub>+ω₃E<sub>Surface</sub>", "Iteration count controls the numerical solve; it is not a paper energy coefficient."),
        "max 2D iterations": ("k2d", "Paper Sec. 4.3 · Eq. (5)", "E<sub>Flat</sub>(v)=ω₁E<sub>Edge</sub>+ω₂E<sub>Collision</sub>+ω₃E<sub>Fab</sub>", "K2D matches K3D edge lengths while avoiding overlap and respecting fabrication gap-angle bounds."),
        "hinge layout iterations": ("hinge", "Paper Sec. 4.4 · Hinge placement / optimization", "T2D Top → T2D Dual → hinge optimization", "This is downstream of K2D. Hinge coincidence is not a K2D objective in the paper."),
    }

    for fn_name in ("number_input", "slider", "checkbox", "text_input"):
        base = getattr(st, fn_name, None)
        if not callable(base): continue
        control_fns[fn_name] = base
        def make_wrapper(base_fn):
            def wrapped(label, *args, **kwargs):
                info = annotations.get(str(label))
                if info: _show_once(*info)
                return base_fn(label, *args, **kwargs)
            return wrapped
        setattr(st, fn_name, make_wrapper(base))

    def selectbox(label, options, *args, **kwargs):
        if label == "View stage":
            original_markdown(_pipeline_html(), unsafe_allow_html=True)
            original_markdown("<div style='font-size:11px;opacity:.62;margin-top:-8px'>Fig. 5をUI用に再構成した工程図。S→Ω→M2D、3D branch: M3D→K3D→T3D、flat branch: K2D→T2D→hinge optimization の対応を表示します。</div>", unsafe_allow_html=True)
        return original_selectbox(label, options, *args, **kwargs)
    st.selectbox = selectbox

    def progress(value=0, *args, **kwargs):
        text = str(kwargs.get("text", ""))
        placeholder = st.empty()
        stage, local, detail = _stage_from_progress(value, text)
        placeholder.markdown(_pipeline_html(stage, local, detail), unsafe_allow_html=True)
        bar = original_progress(value, *args, **kwargs)
        original_bar_progress = bar.progress
        def update(v, *a, **kw):
            tx = str(kw.get("text", ""))
            s, lp, d = _stage_from_progress(v, tx)
            placeholder.markdown(_pipeline_html(s, lp, d), unsafe_allow_html=True)
            return original_bar_progress(v, *a, **kw)
        bar.progress = update
        return bar
    st.progress = progress
    st._onestring_paper_ui_20260916 = True


__all__ = ["install_paper_ui_20260916_patch"]
