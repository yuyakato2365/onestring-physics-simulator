"""Original Figure-5 + equation-first UI for the 2026-09-16 OptCuts launcher."""
from __future__ import annotations

import base64
import html
import importlib
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

PAPER_PDF = "https://onestringtopullthemall.github.io/static/pdfs/onestringpull_authors_version_compressed.pdf"


def _card(title: str, equation: str, mapping: str) -> str:
    return f"""<div class='os-eq-card'><div class='os-eq-title'>{html.escape(title)}</div>
<div class='os-eq'>{equation}</div><div class='os-eq-map'>{mapping}</div></div>"""


SECTION_CARDS = {
    "Target Input": ("Paper Fig. 5 · S → Ω → M₂D", "S →<sup>c</sup> Ω → M<sub>2D</sub>", "入力曲面 S を parameterize して Ω を得て、regular quad grid を重ねて M₂D を構成する段階。"),
    "Pipeline Optimization": ("Paper Sec. 4.2 / 4.3 · K₃D and K₂D", "E<sub>Assembled</sub>(v)=ω₁E<sub>Planar</sub>+ω₂E<sub>Square</sub>+ω₃E<sub>Surface</sub><br>E<sub>Flat</sub>(v)=ω₁E<sub>Edge</sub>+ω₂E<sub>Collision</sub>+ω₃E<sub>Fab</sub>", "3D controls は M₃D→K₃D、2D controls は M₂D→K₂D。iteration/time budget は solver 設定であり論文の ω ではない。"),
    "Hinge Layout Optimization": ("Paper Sec. 4.4 · Hinge Optimization", "E<sub>Hinge</sub>(v)=ω₁E<sub>Rigid</sub>+ω₂E<sub>Collision</sub>+ω₃E<sub>Conn</sub>", "T₂D (Top Hinge) → T₂D (Dual Hinge) の後段。anchor/time budget/candidate cap は implementation-only。"),
    "Compute Backend": ("Implementation-only numerical settings", "Paper objective: unchanged", "CPU/CUDA・dtype は数値 backend/precision のみを変更する。"),
    "Actuation Simulation": ("Deployment simulation · downstream of Fig. 5", "x<sup>k+1</sup>=Solver(x<sup>k</sup>, constraints)", "Figure 5 の surface rationalization 後の deployment simulation。Sec. 4.2/4.3 の ω とは別。"),
    "High-Fidelity Physical Mode": ("Implementation-only physical extensions", "E=E<sub>base</sub>+E<sub>gravity</sub>+E<sub>friction</sub>+E<sub>hinge</sub>", "One String 論文 Figure 5 の objective そのものではない追加近似。"),
}

CONTROL_CARDS = {
    "w_planar / EPlanar": ("K₃D · Paper Eq. (1),(2)", "ω₁ E<sub>Planar</sub>", "この UI 値 = ω₁。各 quad の best-fit plane からの deviation を抑える。"),
    "w_square / ESquare": ("K₃D · Paper Eq. (1)", "ω₂ E<sub>Square</sub>", "この UI 値 = ω₂。quad の side-length disparity / square-like similarity を抑える。"),
    "w_surface / ESurface": ("K₃D · Paper Eq. (1)", "ω₃ E<sub>Surface</sub>", "この UI 値 = ω₃。target surface S への closeness を制御する。"),
    "K3D AL anchor weight (w_a)": ("K₃D · implementation-only stabilization", "E<sub>impl</sub>=E<sub>Assembled</sub>+w<sub>a</sub>E<sub>anchor</sub>+AL(planarity)", "wₐ は原論文 Eq.(1) の ω ではない。"),
    "3D optimization iterations": ("K₃D · solver budget", "min E<sub>Assembled</sub>(v)", "反復上限。目的関数の係数ではない。"),
    "2D optimization iterations": ("K₂D · Paper Sec. 4.3 Eq. (5)", "E<sub>Flat</sub>=ω₁E<sub>Edge</sub>+ω₂E<sub>Collision</sub>+ω₃E<sub>Fab</sub>", "K₂D solve の反復上限。"),
    "K2D strict solver time budget": ("K₂D · implementation-only budget", "min E<sub>Flat</sub>(v), time≤T", "T は solver の時間上限。"),
    "hinge connection weight": ("T₂D · Paper Sec. 4.4", "ω₃ E<sub>Conn</sub>", "この UI 値は hinge vertex coincidence 項の重み。"),
    "hinge collision weight": ("T₂D · Paper Sec. 4.4", "ω₂ E<sub>Collision</sub>", "この UI 値は tile overlap 回避項の重み。"),
    "hinge layout anchor weight": ("T₂D · implementation-only stabilization", "E<sub>impl</sub>=E<sub>Hinge</sub>+w<sub>a</sub>E<sub>anchor</sub>", "原論文 Sec. 4.4 の3項にはない trust/anchor 項。"),
}


def _ensure_fitz():
    try:
        return importlib.import_module("fitz")
    except Exception:
        # app_optcuts.py should remain runnable after only `git pull`; install the
        # one small renderer dependency on first use if the environment predates it.
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "pymupdf>=1.24"], check=True, timeout=120)
            return importlib.import_module("fitz")
        except Exception:
            return None


def _download_pdf(path: Path) -> bool:
    req = urllib.request.Request(PAPER_PDF, headers={"User-Agent": "Mozilla/5.0 OneStringSimulator/2026-09-16"})
    for context in (ssl.create_default_context(), ssl._create_unverified_context()):
        try:
            with urllib.request.urlopen(req, context=context, timeout=45) as src, path.open("wb") as dst:
                dst.write(src.read())
            if path.stat().st_size > 100_000:
                return True
        except Exception:
            path.unlink(missing_ok=True)
    return False


def _paper_figure5_png() -> bytes | None:
    """Load the exact Fig.5 strip from the authors' PDF; no redrawn substitute."""
    root = Path(__file__).resolve().parents[2]
    cache = root / ".paper_cache"
    cache.mkdir(exist_ok=True)
    pdf_path = cache / "onestringpull_authors_version_compressed.pdf"
    png_path = cache / "figure5.png"
    if png_path.exists() and png_path.stat().st_size > 10_000:
        return png_path.read_bytes()
    fitz = _ensure_fitz()
    if fitz is None:
        return None
    if not pdf_path.exists() and not _download_pdf(pdf_path):
        return None
    try:
        doc = fitz.open(pdf_path)
        page = doc[4]
        # Exact full-width Figure-5 artwork, excluding paper body text/caption.
        clip = fitz.Rect(45, 72, 570, 258)
        pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0), clip=clip, alpha=False)
        data = pix.tobytes("png")
        png_path.write_bytes(data)
        return data
    except Exception:
        png_path.unlink(missing_ok=True)
        return None


# percentage coordinates on the exact Figure-5 crop above.
STAGE_BOXES = {
    "S → Ω": (1.5, 4.0, 18.0, 91.0),
    "Ω → M₂D": (14.0, 47.0, 29.0, 49.0),
    "M₂D → M₃D": (25.0, 3.0, 20.0, 92.0),
    "M₃D → K₃D · 3D Optimization": (40.0, 2.0, 24.0, 48.0),
    "K₃D → T₃D · Extrusion / Face Planarity": (59.0, 2.0, 23.0, 49.0),
    "M₂D → K₂D · 2D Optimization": (40.0, 48.0, 24.0, 48.0),
    "K₂D → T₂D → Hinge Optimization": (61.0, 47.0, 38.0, 50.0),
    "T₂D (Dual Hinge)": (82.0, 44.0, 17.0, 53.0),
}


def _figure_html(data: bytes, active: str = "", fraction: float | None = None) -> str:
    uri = "data:image/png;base64," + base64.b64encode(data).decode("ascii")
    overlay = ""
    if active in STAGE_BOXES:
        x, y, w, h = STAGE_BOXES[active]
        deg = max(0.0, min(1.0, float(fraction or 0.0))) * 360.0
        overlay = f"<div class='os-fig-highlight' style='left:{x}%;top:{y}%;width:{w}%;height:{h}%;--progress:{deg}deg'></div>"
    return f"<div class='os-fig5'><img src='{uri}' alt='Original One String paper Figure 5'>{overlay}</div>"


def _render_fig5(st, *, caption: str, active: str = "", fraction: float | None = None) -> None:
    data = _paper_figure5_png()
    if data is None:
        st.error("Figure 5 の取得に失敗しました。自作図への fallback は行いません。ネットワーク接続を確認して再実行してください。")
        return
    st.markdown(_figure_html(data, active, fraction), unsafe_allow_html=True)
    st.caption(caption)
    if active:
        p = int(max(0.0, min(1.0, float(fraction or 0.0))) * 100)
        st.markdown(f"<div class='os-stage'><b>{html.escape(active)}</b><span>{p}%</span></div>", unsafe_allow_html=True)


def _stage(value: Any, text: str) -> tuple[str, float]:
    try:
        p = float(value); p = p / 100.0 if p > 1 else p
    except Exception:
        p = 0.0
    p = max(0.0, min(1.0, p))
    t = (text or "").lower()
    # Text wins over global percentage. This prevents labels such as "S→Ω 11%"
    # from being formed by an unrelated global progress value.
    if "hinge" in t or "dual" in t: return "K₂D → T₂D → Hinge Optimization", p
    if "t2d" in t: return "K₂D → T₂D → Hinge Optimization", p
    if "k2d" in t or "edge" in t or "2d optim" in t: return "M₂D → K₂D · 2D Optimization", p
    if "t3d" in t or "extrusion" in t: return "K₃D → T₃D · Extrusion / Face Planarity", p
    if "k3d" in t or "planar" in t or "3d optim" in t: return "M₃D → K₃D · 3D Optimization", p
    if "m3d" in t or "lift" in t: return "M₂D → M₃D", p
    if "m2d" in t or "grid" in t or "crop" in t: return "Ω → M₂D", p
    if "omega" in t or "parameter" in t or "bff" in t: return "S → Ω", p
    if p < .18: return "S → Ω", p/.18
    if p < .30: return "Ω → M₂D", (p-.18)/.12
    if p < .40: return "M₂D → M₃D", (p-.30)/.10
    if p < .54: return "M₃D → K₃D · 3D Optimization", (p-.40)/.14
    if p < .64: return "K₃D → T₃D · Extrusion / Face Planarity", (p-.54)/.10
    if p < .78: return "M₂D → K₂D · 2D Optimization", (p-.64)/.14
    return "K₂D → T₂D → Hinge Optimization", (p-.78)/.22


def install_paper_ui_20260916_patch() -> None:
    try:
        import streamlit as st
    except Exception:
        return
    if getattr(st, "_onestring_paper_ui_20260916", False):
        return

    original_markdown, original_header = st.markdown, st.header
    original_selectbox, original_progress = st.selectbox, st.progress
    st.markdown("""<style>
.os-eq-card{padding:14px 16px;margin:7px 0 13px;border:1px solid rgba(100,115,140,.18);border-radius:15px;background:linear-gradient(145deg,rgba(255,255,255,.82),rgba(225,233,244,.34));box-shadow:0 8px 28px rgba(20,35,60,.06)}
.os-eq-title{font-size:12px;font-weight:750;letter-spacing:.035em;opacity:.68}.os-eq{font-family:Georgia,'Times New Roman',serif;font-size:18px;line-height:1.55;margin:6px 0}.os-eq-map{font-size:12px;line-height:1.5;opacity:.68}
.os-fig5{position:relative;width:100%;padding:8px;border:1px solid rgba(120,130,145,.16);border-radius:18px;background:rgba(245,248,252,.72);box-sizing:border-box}.os-fig5 img{display:block;width:100%;height:auto;border-radius:12px}.os-fig-highlight{position:absolute;box-sizing:border-box;border-radius:15px;padding:3px;background:conic-gradient(from -90deg,rgba(40,145,255,.82) 0 var(--progress),rgba(145,155,170,.36) var(--progress) 360deg);box-shadow:0 0 22px rgba(50,150,255,.22);opacity:.94;pointer-events:none;mask:linear-gradient(#000 0 0) content-box exclude,linear-gradient(#000 0 0);-webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);-webkit-mask-composite:xor}.os-stage{margin:7px 0 8px;padding:8px 12px;border:1px solid rgba(80,140,220,.18);border-radius:12px;background:rgba(232,241,252,.42);display:flex;justify-content:space-between;font-size:12px}
</style>""", unsafe_allow_html=True)

    current_section = {"name": ""}
    shown_sections: set[str] = set()
    def header(body, *a, **kw):
        out = original_header(body, *a, **kw)
        key = str(body); current_section["name"] = key
        if key in SECTION_CARDS and key not in shown_sections:
            shown_sections.add(key)
            original_markdown(_card(*SECTION_CARDS[key]), unsafe_allow_html=True)
        return out
    st.header = header

    shown_controls: set[str] = set()
    def wrap_control(base):
        def wrapped(label, *a, **kw):
            key = str(label)
            if key in CONTROL_CARDS and key not in shown_controls:
                shown_controls.add(key)
                original_markdown(_card(*CONTROL_CARDS[key]), unsafe_allow_html=True)
            return base(label, *a, **kw)
        return wrapped
    for name in ("number_input", "slider", "checkbox", "toggle", "text_input"):
        base = getattr(st, name, None)
        if callable(base):
            setattr(st, name, wrap_control(base))

    def selectbox(label, options, *a, **kw):
        if str(label) == "View stage":
            _render_fig5(st, caption="Original paper Figure 5 · View stage が論文 pipeline のどこに対応するかを確認できます。")
        return original_selectbox(label, options, *a, **kw)
    st.selectbox = selectbox

    def progress(value=0, *a, **kw):
        ph = st.empty()
        name, local = _stage(value, str(kw.get("text", "")))
        with ph.container():
            _render_fig5(st, caption="Original Figure 5 · 青い glass ring が現在処理中の範囲と進捗を示します。", active=name, fraction=local)
        bar = original_progress(value, *a, **kw)
        base_update = bar.progress
        def update(v, *aa, **kk):
            n, lp = _stage(v, str(kk.get("text", "")))
            with ph.container():
                _render_fig5(st, caption="Original Figure 5 · 青い glass ring が現在処理中の範囲と進捗を示します。", active=n, fraction=lp)
            return base_update(v, *aa, **kk)
        bar.progress = update
        return bar
    st.progress = progress
    st._onestring_paper_ui_20260916 = True


__all__ = ["install_paper_ui_20260916_patch"]
