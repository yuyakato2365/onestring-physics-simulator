"""Paper Figure-5 + equation-first UI for the 2026-09-16 OptCuts launcher.

This layer changes presentation only.  It deliberately uses the ORIGINAL paper
Figure 5 rather than a redrawn pipeline.  The figure is fetched from the authors'
PDF once and cached locally, then the Figure-5 strip is cropped from page 5.
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import Any

PAPER_PDF = "https://onestringtopullthemall.github.io/static/pdfs/onestringpull_authors_version_compressed.pdf"


def _card(title: str, equation: str, mapping: str) -> str:
    return f"""<div class='os-eq-card'><div class='os-eq-title'>{html.escape(title)}</div>
<div class='os-eq'>{equation}</div><div class='os-eq-map'>{mapping}</div></div>"""


SECTION_CARDS = {
    "Target Input": ("Paper Fig. 5 · S → Ω → M₂D", "S \u2192<sup>c</sup> Ω \u2192 M<sub>2D</sub>", "入力曲面 S を parameterize して Ω を得て、規則 quad grid を重ね M₂D を構成します。以下の値はこの入力・離散化段階のパラメータです。"),
    "Pipeline Optimization": ("Paper Sec. 4.2 / 4.3 · K₃D and K₂D", "E<sub>Assembled</sub>(v)=ω₁E<sub>Planar</sub>+ω₂E<sub>Square</sub>+ω₃E<sub>Surface</sub><br>E<sub>Flat</sub>(v)=ω₁E<sub>Edge</sub>+ω₂E<sub>Collision</sub>+ω₃E<sub>Fab</sub>", "3D controls は M₃D→K₃D、2D controls は M₂D→K₂D に対応します。iteration/time budget は数値解法側の設定であり、論文の ω ではありません。"),
    "Hinge Layout Optimization": ("Paper Sec. 4.4 · Hinge Optimization", "E<sub>Hinge</sub>(v)=ω₁E<sub>Rigid</sub>+ω₂E<sub>Collision</sub>+ω₃E<sub>Conn</sub>", "T₂D (Top Hinge) → T₂D (Dual Hinge) の後段です。connection/collision weights はこの目的関数に対応し、anchor・time budget・candidate cap は実装上の安定化/計算量制御です。"),
    "Compute Backend": ("Implementation-only numerical settings", "paper energy unchanged", "CPU/CUDA・dtype は論文の目的関数を変更せず、実装上の計算 backend/精度だけを変更します。"),
    "Actuation Simulation": ("Deployment simulation · downstream of Fig. 5", "x<sup>k+1</sup> = Solver(x<sup>k</sup>, constraints)", "Figure 5 の surface rationalization 後の deployment simulation 用設定です。論文 Sec. 4.2/4.3 の ω とは別物です。"),
    "High-Fidelity Physical Mode": ("Implementation-only physical extensions", "E = E<sub>base</sub> + E<sub>gravity</sub> + E<sub>friction</sub> + E<sub>hinge</sub>", "研究用の追加近似です。One String 論文 Figure 5 の surface-rationalization objective そのものではありません。"),
}

CONTROL_CARDS = {
    "K3D AL anchor weight (w_a)": ("K₃D · implementation stabilization", "E<sub>AL</sub>=E<sub>Assembled</sub> + w<sub>a</sub> E<sub>anchor</sub> + AL(planarity)", "wₐ は原論文 Eq.(1) の ω₁,ω₂,ω₃ ではありません。現在の Augmented-Lagrangian 実装で K₃D を初期形状近傍に保つ追加項です。"),
    "w_planar / EPlanar": ("K₃D · Paper Eq. (1),(2)", "ω₁ E<sub>Planar</sub>", "この値が K₃D の planarity 項の重みです。"),
    "w_square / ESquare": ("K₃D · Paper Eq. (1)", "ω₂ E<sub>Square</sub>", "この値が square/similarity 項の重みです。"),
    "w_surface / ESurface": ("K₃D · Paper Eq. (1)", "ω₃ E<sub>Surface</sub>", "この値が target surface closeness 項の重みです。"),
    "2D optimization iterations": ("K₂D · Paper Sec. 4.3 Eq. (5)", "E<sub>Flat</sub>=ω₁E<sub>Edge</sub>+ω₂E<sub>Collision</sub>+ω₃E<sub>Fab</sub>", "K₂D solve の反復上限。係数そのものではなく solver budget です。"),
    "K2D strict solver time budget": ("K₂D · implementation budget", "min E<sub>Flat</sub>(v)  subject to time ≤ T", "T は実装上の時間上限で、論文パラメータではありません。"),
    "hinge connection weight": ("T₂D · Paper Sec. 4.4", "ω₃ E<sub>Conn</sub>", "対応 hinge vertices を一致させる項の重みです。"),
    "hinge collision weight": ("T₂D · Paper Sec. 4.4", "ω₂ E<sub>Collision</sub>", "厚み付き tile の overlap を避ける項の重みです。"),
    "hinge layout anchor weight": ("T₂D · implementation stabilization", "E<sub>Hinge,impl</sub>=E<sub>Hinge</sub>+w<sub>a</sub>E<sub>anchor</sub>", "原論文の3項にはない、初期 flat layout への追加 trust/anchor 項です。"),
}


def _paper_figure5_png() -> bytes | None:
    """Return the original Figure 5 strip from the authors' PDF, cached locally."""
    try:
        import urllib.request
        import fitz
        root = Path(__file__).resolve().parents[2]
        cache = root / ".paper_cache"
        cache.mkdir(exist_ok=True)
        pdf_path = cache / "onestringpull_authors_version_compressed.pdf"
        png_path = cache / "figure5.png"
        if png_path.exists():
            return png_path.read_bytes()
        if not pdf_path.exists():
            urllib.request.urlretrieve(PAPER_PDF, pdf_path)
        doc = fitz.open(pdf_path)
        page = doc[4]  # printed page containing Fig. 5
        # Figure 5 occupies the full-width strip at the top of page 5.
        r = page.rect
        clip = fitz.Rect(r.x0, r.y0, r.x1, r.y0 + r.height * 0.285)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), clip=clip, alpha=False)
        data = pix.tobytes("png")
        png_path.write_bytes(data)
        return data
    except Exception:
        return None


def _render_fig5(st, *, caption: str, active: str = "", fraction: float | None = None) -> None:
    data = _paper_figure5_png()
    if data:
        st.image(data, use_container_width=True, caption=caption)
    else:
        st.warning("Original Figure 5 could not be loaded. Run `pip install -r requirements.txt` and retry.")
    if active:
        p = 0 if fraction is None else int(max(0.0, min(1.0, float(fraction))) * 100)
        st.markdown(f"<div class='os-stage'><b>{html.escape(active)}</b><span>{p}%</span><div class='os-track'><i style='width:{p}%'></i></div></div>", unsafe_allow_html=True)


def _stage(value: Any, text: str) -> tuple[str, float]:
    try:
        p = float(value); p = p / 100.0 if p > 1 else p
    except Exception: p = 0.0
    t = (text or "").lower()
    if "omega" in t or p < .18: return "S → Ω", max(0,p/.18)
    if "m2d" in t or p < .30: return "Ω → M₂D", max(0,(p-.18)/.12)
    if "m3d" in t or p < .40: return "M₂D → M₃D", max(0,(p-.30)/.10)
    if "k3d" in t or "planar" in t or p < .54: return "M₃D → K₃D · 3D Optimization", max(0,(p-.40)/.14)
    if "t3d" in t or "extrusion" in t or p < .64: return "K₃D → T₃D · Extrusion / Face Planarity", max(0,(p-.54)/.10)
    if "k2d" in t or "edge" in t or p < .78: return "M₂D → K₂D · 2D Optimization", max(0,(p-.64)/.14)
    if "hinge" in t or "t2d" in t or p < .94: return "K₂D → T₂D → Hinge Optimization", max(0,(p-.78)/.16)
    return "T₂D (Dual Hinge)", max(0,(p-.94)/.06)


def install_paper_ui_20260916_patch() -> None:
    try: import streamlit as st
    except Exception: return
    if getattr(st, "_onestring_paper_ui_20260916", False): return

    original_markdown, original_header = st.markdown, st.header
    original_selectbox, original_progress = st.selectbox, st.progress
    st.markdown("""<style>
.os-eq-card{padding:14px 16px;margin:7px 0 13px;border:1px solid rgba(100,115,140,.18);border-radius:15px;background:linear-gradient(145deg,rgba(255,255,255,.82),rgba(225,233,244,.34));box-shadow:0 8px 28px rgba(20,35,60,.06)}
.os-eq-title{font-size:12px;font-weight:750;letter-spacing:.035em;opacity:.68}.os-eq{font-family:Georgia,'Times New Roman',serif;font-size:18px;line-height:1.55;margin:6px 0}.os-eq-map{font-size:12px;line-height:1.5;opacity:.68}
.os-stage{margin:7px 0 8px;padding:10px 13px;border:1px solid rgba(80,140,220,.18);border-radius:13px;background:rgba(232,241,252,.42);display:grid;grid-template-columns:1fr auto;gap:6px 12px;font-size:12px}.os-stage span{font-variant-numeric:tabular-nums}.os-track{grid-column:1/3;height:5px;border-radius:99px;background:rgba(120,130,145,.18);overflow:hidden}.os-track i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,rgba(80,165,255,.55),rgba(30,125,245,.9));transition:width .25s ease}
</style>""", unsafe_allow_html=True)

    shown_sections: set[str] = set()
    def header(body, *a, **kw):
        out = original_header(body, *a, **kw)
        key = str(body)
        if key in SECTION_CARDS and key not in shown_sections:
            shown_sections.add(key); original_markdown(_card(*SECTION_CARDS[key]), unsafe_allow_html=True)
        return out
    st.header = header

    shown_controls: set[str] = set()
    def wrap_control(base):
        def wrapped(label, *a, **kw):
            key = str(label)
            if key in CONTROL_CARDS and key not in shown_controls:
                shown_controls.add(key); original_markdown(_card(*CONTROL_CARDS[key]), unsafe_allow_html=True)
            return base(label, *a, **kw)
        return wrapped
    for name in ("number_input","slider","checkbox","toggle","text_input"):
        base = getattr(st, name, None)
        if callable(base): setattr(st, name, wrap_control(base))

    def selectbox(label, options, *a, **kw):
        if str(label) == "View stage":
            _render_fig5(st, caption="Original paper Figure 5 · selected View stage corresponds to a node/process in this pipeline.")
        return original_selectbox(label, options, *a, **kw)
    st.selectbox = selectbox

    def progress(value=0, *a, **kw):
        ph = st.empty(); name, local = _stage(value, str(kw.get("text", "")))
        with ph.container(): _render_fig5(st, caption="Original Figure 5 · current computation stage", active=name, fraction=local)
        bar = original_progress(value, *a, **kw); base_update = bar.progress
        def update(v, *aa, **kk):
            n, lp = _stage(v, str(kk.get("text", "")))
            with ph.container(): _render_fig5(st, caption="Original Figure 5 · current computation stage", active=n, fraction=lp)
            return base_update(v, *aa, **kk)
        bar.progress = update; return bar
    st.progress = progress
    st._onestring_paper_ui_20260916 = True

__all__ = ["install_paper_ui_20260916_patch"]
