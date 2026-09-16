"""Equation-first sidebar + Figure-5 stage overlay for the new UI branch.

Presentation only: this module does not change OneString numerical solvers.
"""
from __future__ import annotations

import base64
import html
import ssl
import threading
import urllib.request
from pathlib import Path
from typing import Any

PAPER_PDF = "https://onestringtopullthemall.github.io/static/pdfs/onestringpull_authors_version_compressed.pdf"


def _card(title: str, equation: str, mapping: str, badge: str = "Paper mapping") -> str:
    return f"""<div class='os-eq-card'><div class='os-eq-top'><span>{html.escape(title)}</span><em>{html.escape(badge)}</em></div><div class='os-eq'>{equation}</div><div class='os-eq-map'>{mapping}</div></div>"""


SECTION_CARDS = {
    "Target Input": ("S → Ω → M₂D", "c:S→Ω, &nbsp; M<sub>2D</sub>=Grid(Ω,s)", "入力曲面 S の parameterization と fabrication grid の生成。tile/grid 等はこの段階の離散化パラメータ。"),
    "Pipeline Optimization": ("K₃D / K₂D optimization", "E<sub>Assembled</sub>(v)=ω₁E<sub>Planar</sub>+ω₂E<sub>Square</sub>+ω₃E<sub>Surface</sub><br>E<sub>Flat</sub>(v)=ω₁E<sub>Edge</sub>+ω₂E<sub>Collision</sub>+ω₃E<sub>Fab</sub>", "上段が Paper Sec.4.2 の M₃D→K₃D、下段が Sec.4.3 の M₂D→K₂D。iteration/time budget は ω ではなく solver 設定。"),
    "Hinge Layout Optimization": ("T₂D hinge optimization", "E<sub>Hinge</sub>(v)=ω₁E<sub>Rigid</sub>+ω₂E<sub>Collision</sub>+ω₃E<sub>Conn</sub>", "Paper Sec.4.4。connection/collision はこの式の項。anchor、candidate cap、time budget は実装側の安定化・計算量設定。"),
    "Compute Backend": ("Numerical backend", "arg min E &nbsp; (objective unchanged)", "CPU/CUDA、dtype、worker 数は論文の energy parameter ではない。", "Implementation"),
    "Actuation Simulation": ("Deployment simulation", "x<sup>k+1</sup>=Solver(x<sup>k</sup>, constraints)", "Figure 5 の rationalization 後の simulation 設定。K₃D/K₂D の ω とは別。", "Downstream"),
    "High-Fidelity Physical Mode": ("Physical extensions", "E=E<sub>base</sub>+E<sub>gravity</sub>+E<sub>friction</sub>+E<sub>hinge</sub>", "現 simulator の追加近似であり Figure 5 の三つの paper objective そのものではない。", "Implementation"),
}

CONTROL_CARDS = {
    "w_planar / EPlanar": ("K₃D · EAssembled", "<b>ω₁</b>E<sub>Planar</sub> + ω₂E<sub>Square</sub> + ω₃E<sub>Surface</sub>", "この値 → ω₁。quad planarity 項。"),
    "w_square / ESquare": ("K₃D · EAssembled", "ω₁E<sub>Planar</sub> + <b>ω₂</b>E<sub>Square</sub> + ω₃E<sub>Surface</sub>", "この値 → ω₂。square-like shape 項。"),
    "w_surface / ESurface": ("K₃D · EAssembled", "ω₁E<sub>Planar</sub> + ω₂E<sub>Square</sub> + <b>ω₃</b>E<sub>Surface</sub>", "この値 → ω₃。target surface 追従項。"),
    "K3D AL anchor weight (w_a)": ("K₃D · AL stabilization", "E<sub>impl</sub>=E<sub>Assembled</sub>+<b>w<sub>a</sub></b>E<sub>anchor</sub>+AL(planarity)", "wₐ は paper Eq.(1) の ω ではなく現実装の stabilization。", "Implementation"),
    "3D optimization iterations": ("K₃D solver budget", "min<sub>v</sub> E<sub>Assembled</sub>(v)", "反復回数。energy の係数ではない。", "Solver"),
    "2D optimization iterations": ("K₂D solver budget", "min<sub>v</sub> E<sub>Flat</sub>(v)", "EFlat solve の反復回数。", "Solver"),
    "K2D strict solver time budget": ("K₂D solver budget", "min E<sub>Flat</sub>(v), &nbsp; time≤<b>T</b>", "T は計算時間上限。", "Solver"),
    "hinge connection weight": ("T₂D · EHinge", "ω₁E<sub>Rigid</sub>+ω₂E<sub>Collision</sub>+<b>ω₃E<sub>Conn</sub></b>", "hinge vertex coincidence / connection 項。"),
    "hinge collision weight": ("T₂D · EHinge", "ω₁E<sub>Rigid</sub>+<b>ω₂E<sub>Collision</sub></b>+ω₃E<sub>Conn</sub>", "tile overlap 回避項。"),
    "hinge layout anchor weight": ("T₂D · stabilization", "E<sub>impl</sub>=E<sub>Hinge</sub>+<b>w<sub>a</sub>E<sub>anchor</sub></b>", "Paper Sec.4.4 の三項にはない実装側 anchor。", "Implementation"),
}

_FIG_LOCK = threading.Lock(); _FIG_THREAD: threading.Thread | None = None; _FIG_ERROR = ""

def _paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[2]; cache = root / ".paper_cache"; cache.mkdir(exist_ok=True)
    return cache / "onestringpull_authors_version_compressed.pdf", cache / "figure5.png"

def _build_figure5_cache() -> None:
    global _FIG_ERROR
    pdf_path, png_path = _paths()
    try:
        if png_path.exists() and png_path.stat().st_size > 10000: return
        import fitz
        if not pdf_path.exists() or pdf_path.stat().st_size < 100000:
            req = urllib.request.Request(PAPER_PDF, headers={"User-Agent":"Mozilla/5.0 OneStringSimulator"})
            last = None
            for ctx in (ssl.create_default_context(), ssl._create_unverified_context()):
                try:
                    with urllib.request.urlopen(req, context=ctx, timeout=25) as src, pdf_path.open("wb") as dst: dst.write(src.read())
                    last=None; break
                except Exception as exc: last=exc; pdf_path.unlink(missing_ok=True)
            if last is not None: raise last
        doc=fitz.open(pdf_path); page=doc[4]; clip=fitz.Rect(45,72,570,258)
        pix=page.get_pixmap(matrix=fitz.Matrix(3,3),clip=clip,alpha=False); png_path.write_bytes(pix.tobytes("png")); _FIG_ERROR=""
    except Exception as exc: _FIG_ERROR=f"{type(exc).__name__}: {exc}"

def _kick() -> None:
    global _FIG_THREAD
    _, png=_paths()
    if png.exists() and png.stat().st_size>10000:return
    with _FIG_LOCK:
        if _FIG_THREAD is None or not _FIG_THREAD.is_alive():
            _FIG_THREAD=threading.Thread(target=_build_figure5_cache,daemon=True); _FIG_THREAD.start()

def _figure() -> bytes|None:
    _,png=_paths()
    if png.exists() and png.stat().st_size>10000:
        try:return png.read_bytes()
        except Exception:pass
    _kick(); return None

# Coordinates are percentages of the cropped original Figure 5.
STAGE_BOXES={
 "S":(0.5,1,13,96), "Omega":(10,45,14,52), "M2D":(23,44,13,53), "M3D":(32,0,15,50),
 "K3D":(44,0,15,50), "T3D":(57,0,15,50), "K2D":(44,47,15,51), "T2D Top":(57,47,15,51),
 "T2D Dual":(69,44,16,54), "Hinge Optimization":(82,44,17,54),
}
VIEW_TO_STAGE={"S":"S","Split Map":"M2D","Omega":"Omega","M2D":"M2D","M3D":"M3D","K3D":"K3D","T3D":"T3D","K2D":"K2D","T2D Top":"T2D Top","T2D Dual":"T2D Dual","T2D":"T2D Dual"}

def _fig_html(data:bytes, active:str="", fraction:float|None=None, indeterminate:bool=False)->str:
    uri="data:image/png;base64,"+base64.b64encode(data).decode("ascii"); overlay=""
    if active in STAGE_BOXES:
        x,y,w,h=STAGE_BOXES[active]
        if indeterminate:
            overlay=f"<div class='os-fig-highlight os-indeterminate' style='left:{x}%;top:{y}%;width:{w}%;height:{h}%'></div>"
        else:
            deg=max(0,min(1,float(fraction or 0)))*360
            overlay=f"<div class='os-fig-highlight' style='left:{x}%;top:{y}%;width:{w}%;height:{h}%;--progress:{deg}deg'></div>"
    return f"<div class='os-fig5'><img src='{uri}'>{overlay}</div>"

def _render(st, caption:str, active:str="", fraction:float|None=None, indeterminate:bool=False)->None:
    data=_figure()
    if data is None:
        detail=html.escape(_FIG_ERROR) if _FIG_ERROR else "Figure 5 をバックグラウンドで準備中"
        st.markdown(f"<div class='os-fig-loading'><span></span><b>{detail}</b><small>pipeline 計算とは独立しています。</small></div>",unsafe_allow_html=True)
    else:
        st.markdown(_fig_html(data,active,fraction,indeterminate),unsafe_allow_html=True); st.caption(caption)
    if active:
        pct="processing" if indeterminate else f"{int(max(0,min(1,float(fraction or 0)))*100)}%"
        st.markdown(f"<div class='os-stage'><b>{html.escape(active)}</b><span>{pct}</span></div>",unsafe_allow_html=True)

def _stage(value:Any,text:str)->tuple[str,float,bool]:
    try:p=float(value);p=p/100 if p>1 else p
    except Exception:p=0
    p=max(0,min(1,p));t=(text or "").lower()
    rules=[(("hinge","dual","t2d"),"Hinge Optimization"),(("k2d","2d optim","edge"),"K2D"),(("t3d","extrusion"),"T3D"),(("k3d","3d optim","planar"),"K3D"),(("m3d","lift"),"M3D"),(("m2d","grid","crop"),"M2D"),(("omega","parameter","bff","optcuts"),"Omega")]
    for keys,name in rules:
        if any(k in t for k in keys): return name,p,False
    bands=[(.12,"S"),(.24,"Omega"),(.34,"M2D"),(.45,"M3D"),(.60,"K3D"),(.70,"T3D"),(.84,"K2D"),(1.01,"Hinge Optimization")]
    lo=0
    for hi,name in bands:
        if p<hi:return name,(p-lo)/max(hi-lo,1e-9),False
        lo=hi
    return "Hinge Optimization",1,False

def install_paper_ui_20260916_patch()->None:
    try:import streamlit as st
    except Exception:return
    if getattr(st,"_onestring_paper_ui_20260916",False):return
    md,header,subheader=st.markdown,st.header,getattr(st,"subheader",None); selectbox,progress=st.selectbox,st.progress
    md("""<style>
section[data-testid="stSidebar"] .os-eq-card{padding:12px 13px;margin:5px 0 12px;border:1px solid rgba(91,118,153,.18);border-radius:14px;background:linear-gradient(145deg,rgba(255,255,255,.78),rgba(225,235,247,.40));box-shadow:0 7px 24px rgba(30,55,90,.06)}
.os-eq-top{display:flex;justify-content:space-between;gap:8px;font-size:11px;font-weight:750;opacity:.75}.os-eq-top em{font-style:normal;font-size:9px;padding:2px 6px;border-radius:999px;background:rgba(60,135,230,.10)}.os-eq{font-family:Georgia,'Times New Roman',serif;font-size:16px;line-height:1.5;margin:7px 0}.os-eq-map{font-size:11px;line-height:1.45;opacity:.70}.os-eq b{color:rgb(22,111,214)}
.os-fig5{position:relative;width:100%;padding:8px;border:1px solid rgba(120,130,145,.16);border-radius:18px;background:rgba(245,248,252,.72);box-sizing:border-box}.os-fig5 img{display:block;width:100%;height:auto;border-radius:12px}.os-fig-highlight{position:absolute;box-sizing:border-box;border-radius:15px;padding:4px;background:conic-gradient(from -90deg,rgba(35,145,255,.88) 0 var(--progress),rgba(145,155,170,.34) var(--progress) 360deg);box-shadow:0 0 24px rgba(40,145,255,.25);pointer-events:none;-webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);-webkit-mask-composite:xor;mask-composite:exclude}.os-indeterminate{background:conic-gradient(from var(--spin),rgba(145,155,170,.30) 0 255deg,rgba(35,145,255,.92) 300deg,rgba(145,155,170,.30) 360deg);animation:osOrbit 1.25s linear infinite}.os-stage{margin:7px 0 8px;padding:8px 12px;border:1px solid rgba(80,140,220,.18);border-radius:12px;background:rgba(232,241,252,.42);display:flex;justify-content:space-between;font-size:12px}.os-fig-loading{min-height:120px;border:1px solid rgba(120,130,145,.16);border-radius:18px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px}.os-fig-loading span{width:23px;height:23px;border-radius:50%;border:3px solid rgba(130,145,165,.22);border-top-color:rgba(40,145,255,.8);animation:osSpin .8s linear infinite}@property --spin{syntax:'<angle>';initial-value:0deg;inherits:false}@keyframes osOrbit{to{--spin:360deg}}@keyframes osSpin{to{transform:rotate(360deg)}}
</style>""",unsafe_allow_html=True)
    shown_sections:set[str]=set();shown_controls:set[str]=set()
    def heading(base):
        def wrapped(body,*a,**kw):
            out=base(body,*a,**kw);key=str(body)
            if key in SECTION_CARDS and key not in shown_sections:
                shown_sections.add(key);md(_card(*SECTION_CARDS[key]),unsafe_allow_html=True)
            return out
        return wrapped
    st.header=heading(header)
    if callable(subheader):st.subheader=heading(subheader)
    def wrap(base):
        def wrapped(label,*a,**kw):
            key=str(label)
            if key in CONTROL_CARDS and key not in shown_controls:
                shown_controls.add(key);md(_card(*CONTROL_CARDS[key]),unsafe_allow_html=True)
            return base(label,*a,**kw)
        return wrapped
    for n in ("number_input","slider","checkbox","toggle","text_input"):
        base=getattr(st,n,None)
        if callable(base):setattr(st,n,wrap(base))
    def sel(label,options,*a,**kw):
        value=selectbox(label,options,*a,**kw)
        if str(label)=="View stage":
            stage=VIEW_TO_STAGE.get(str(value),"")
            _render(st,"Original paper Figure 5 · blue glass frame = displayed result",stage,1.0 if stage else None,False)
        return value
    st.selectbox=sel
    class ProgressProxy:
        def __init__(self,bar,ph):self._bar=bar;self._ph=ph
        def progress(self,v,*a,**kw):
            name,local,ind=_stage(v,str(kw.get("text","")));self._ph.empty()
            with self._ph.container():_render(st,"Original paper Figure 5 · current computation",name,local,ind)
            self._bar.progress(v,*a,**kw);return self
        def empty(self):self._ph.empty();return self._bar.empty()
        def __getattr__(self,n):return getattr(self._bar,n)
    def prog(value=0,*a,**kw):
        ph=st.empty();name,local,ind=_stage(value,str(kw.get("text","")))
        with ph.container():_render(st,"Original paper Figure 5 · current computation",name,local,ind)
        return ProgressProxy(progress(value,*a,**kw),ph)
    st.progress=prog;st._onestring_paper_ui_20260916=True

__all__=["install_paper_ui_20260916_patch"]
