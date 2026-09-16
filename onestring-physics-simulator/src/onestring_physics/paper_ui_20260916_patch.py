"""Persistent equation-first sidebar + single Figure-5 process overlay for 2026-09-16 UI."""
from __future__ import annotations
import base64, html, ssl, threading, urllib.request
from pathlib import Path
PAPER_PDF="https://onestringtopullthemall.github.io/static/pdfs/onestringpull_authors_version_compressed.pdf"
def _card(t,e,m,b="Paper"):
 return f"<div class='os-eq-card'><div class='os-eq-top'><b>{html.escape(t)}</b><em>{b}</em></div><div class='os-eq'>{e}</div><div class='os-eq-map'>{m}</div></div>"
SECTION_CARDS={"Target Input":("S → Ω → M₂D","c:S→Ω &nbsp;&nbsp; M<sub>2D</sub>=Grid(Ω,s)","Parameterization と fabrication grid の生成。"),"Pipeline Optimization":("Optimization objectives","E<sub>Assembled</sub>(v)=ω₁E<sub>Planar</sub>+ω₂E<sub>Square</sub>+ω₃E<sub>Surface</sub><br><br>E<sub>Flat</sub>(v)=ω₁E<sub>Edge</sub>+ω₂E<sub>Collision</sub>+ω₃E<sub>Fab</sub>","上: M₃D→K₃D (Sec.4.2) / 下: M₂D→K₂D (Sec.4.3)。iterations/time budget は solver 設定。"),"Hinge Layout Optimization":("Hinge objective","E<sub>Hinge</sub>(v)=ω₁E<sub>Rigid</sub>+ω₂E<sub>Collision</sub>+ω₃E<sub>Conn</sub>","Paper Sec.4.4。anchor/candidate/time budget は implementation-only。"),"Compute Backend":("Numerical backend","arg min E — objective unchanged","CPU/CUDA・dtype は数値実装設定。","Implementation"),"Actuation Simulation":("Deployment simulation","x<sup>k+1</sup>=Solver(x<sup>k</sup>,constraints)","Figure 5 後段の simulation。","Downstream"),"High-Fidelity Physical Mode":("Physical extensions","E=E<sub>base</sub>+E<sub>gravity</sub>+E<sub>friction</sub>+E<sub>hinge</sub>","Paper objective そのものではない追加近似。","Implementation")}
CONTROL_CARDS={"w_planar / EPlanar":("K₃D · EAssembled","<b>ω₁</b>E<sub>Planar</sub> + ω₂E<sub>Square</sub> + ω₃E<sub>Surface</sub>","この値 = ω₁ (planarity)。"),"w_square / ESquare":("K₃D · EAssembled","ω₁E<sub>Planar</sub> + <b>ω₂</b>E<sub>Square</sub> + ω₃E<sub>Surface</sub>","この値 = ω₂ (square-like shape)。"),"w_surface / ESurface":("K₃D · EAssembled","ω₁E<sub>Planar</sub> + ω₂E<sub>Square</sub> + <b>ω₃</b>E<sub>Surface</sub>","この値 = ω₃ (surface fitting)。"),"K3D AL anchor weight (w_a)":("K₃D stabilization","E<sub>impl</sub>=E<sub>Assembled</sub>+<b>w<sub>a</sub>E<sub>anchor</sub></b>+AL(planarity)","Paper ω ではない。","Implementation"),"3D optimization iterations":("K₃D solver","min E<sub>Assembled</sub>(v)","反復上限。","Solver"),"2D optimization iterations":("K₂D solver","min E<sub>Flat</sub>(v)","反復上限。","Solver"),"K2D strict solver time budget":("K₂D solver","min E<sub>Flat</sub>(v), time≤<b>T</b>","T = 時間上限。","Solver"),"hinge connection weight":("T₂D · EHinge","ω₁E<sub>Rigid</sub>+ω₂E<sub>Collision</sub>+<b>ω₃E<sub>Conn</sub></b>","connection 項の重み。"),"hinge collision weight":("T₂D · EHinge","ω₁E<sub>Rigid</sub>+<b>ω₂E<sub>Collision</sub></b>+ω₃E<sub>Conn</sub>","collision 項の重み。"),"hinge layout anchor weight":("T₂D stabilization","E<sub>impl</sub>=E<sub>Hinge</sub>+<b>w<sub>a</sub>E<sub>anchor</sub></b>","Paper Sec.4.4 の三項にはない。","Implementation")}
_LOCK=threading.Lock();_THREAD=None
def _paths():
 root=Path(__file__).resolve().parents[2];c=root/".paper_cache";c.mkdir(exist_ok=True);return c/"onestringpull_authors_version_compressed.pdf",c/"figure5.png"
def _load():
 pdf,png=_paths()
 try:
  if png.exists() and png.stat().st_size>10000:return
  import fitz
  if not pdf.exists() or pdf.stat().st_size<100000:
   req=urllib.request.Request(PAPER_PDF,headers={"User-Agent":"Mozilla/5.0"});last=None
   for ctx in (ssl.create_default_context(),ssl._create_unverified_context()):
    try:
     with urllib.request.urlopen(req,context=ctx,timeout=25) as s,pdf.open("wb") as d:d.write(s.read())
     last=None;break
    except Exception as e:last=e;pdf.unlink(missing_ok=True)
   if last:raise last
  doc=fitz.open(pdf);pix=doc[4].get_pixmap(matrix=fitz.Matrix(3,3),clip=fitz.Rect(45,72,570,258),alpha=False);png.write_bytes(pix.tobytes("png"))
 except Exception:pass
def _figure():
 global _THREAD
 _,p=_paths()
 if p.exists() and p.stat().st_size>10000:
  try:return p.read_bytes()
  except Exception:pass
 with _LOCK:
  if _THREAD is None or not _THREAD.is_alive():_THREAD=threading.Thread(target=_load,daemon=True);_THREAD.start()
 return None
BOX={"S":(2.0,5.0,14.5,40.0),"Omega":(5.0,50.0,10.0,38.0),"M2D":(20.0,51.0,18.0,36.0),"M3D":(20.0,5.0,17.0,40.0),"K3D":(40.5,5.0,16.5,40.0),"T3D":(61.0,5.0,16.0,40.0),"K2D":(40.0,50.0,18.0,38.0),"T2D Top":(60.0,49.0,19.0,40.0),"T2D Dual":(83.0,49.0,16.0,40.0),"Hinge":(83.0,49.0,16.0,40.0)}
VIEW={"S":"S","Omega":"Omega","M2D":"M2D","M3D":"M3D","K3D":"K3D","T3D":"T3D","K2D":"K2D","T2D Top":"T2D Top","T2D Dual":"T2D Dual","T2D":"T2D Dual","Split Map":"M2D"}
def _stage(v,text):
 try:p=float(v);p=p/100 if p>1 else p
 except Exception:p=0
 p=max(0,min(1,p));t=(text or "").lower()
 for keys,n in [(("hinge","dual","t2d"),"Hinge"),(("k2d","2d optim","edge"),"K2D"),(("t3d","extrusion"),"T3D"),(("k3d","3d optim","planar"),"K3D"),(("m3d","lift"),"M3D"),(("m2d","grid","crop"),"M2D"),(("omega","parameter","bff","optcuts"),"Omega")]:
  if any(k in t for k in keys):return n,p
 bands=[(.12,"S"),(.24,"Omega"),(.34,"M2D"),(.45,"M3D"),(.60,"K3D"),(.70,"T3D"),(.84,"K2D"),(1.01,"Hinge")];lo=0
 for hi,n in bands:
  if p<hi:return n,(p-lo)/max(hi-lo,1e-9)
  lo=hi
 return "Hinge",1
def _fig(data,a,q):
 uri="data:image/png;base64,"+base64.b64encode(data).decode();ov=""
 if a in BOX:
  x,y,w,h=BOX[a];deg=max(0,min(1,float(q or 0)))*360
  ov=f"<div class='os-hi' style='left:{x}%;top:{y}%;width:{w}%;height:{h}%;--p:{deg}deg'><i></i></div>"
 return f"<div class='os-f5'><img src='{uri}'>{ov}</div>"
def _render(st,caption,a="",q=0):
 d=_figure()
 if d is None:st.markdown("<div class='os-loading'>Figure 5 preparing… <small>calculation continues</small></div>",unsafe_allow_html=True)
 else:st.markdown(_fig(d,a,q),unsafe_allow_html=True);st.caption(caption)
 if a:st.markdown(f"<div class='os-stage'><b>{a}</b><span>{int(100*max(0,min(1,float(q or 0))))}%</span></div>",unsafe_allow_html=True)
def install_paper_ui_20260916_patch():
 try:import streamlit as st
 except Exception:return
 def unwrap(fn):return getattr(fn,"_onestring_base",fn)
 md=unwrap(st.markdown);hdr=unwrap(st.header);sub0=getattr(st,"subheader",None);sub=unwrap(sub0) if callable(sub0) else None;sel0=unwrap(st.selectbox);prog0=unwrap(st.progress)
 st.markdown=md;st.header=hdr;st.selectbox=sel0;st.progress=prog0
 if callable(sub):st.subheader=sub
 for n in ("number_input","slider","checkbox","toggle","text_input"):
  b=getattr(st,n,None)
  if callable(b):setattr(st,n,unwrap(b))
 md("""<style>
section[data-testid=stSidebar] .os-eq-card{padding:12px 13px;margin:6px 0 14px;border:1px solid rgba(70,110,160,.22);border-radius:14px;background:rgba(240,246,253,.68)}.os-eq-top{display:flex;justify-content:space-between;font-size:11px}.os-eq-top em{font-style:normal;font-size:9px;opacity:.65}.os-eq{font-family:Georgia,serif;font-size:16px;line-height:1.5;margin:8px 0}.os-eq b{color:#1672d4}.os-eq-map{font-size:11px;opacity:.7}
.os-f5{position:relative;width:100%;padding:8px;border:1px solid rgba(120,130,145,.18);border-radius:18px;background:rgba(245,248,252,.75);box-sizing:border-box}.os-f5 img{width:100%;display:block;border-radius:11px}
.os-hi{position:absolute;box-sizing:border-box;border-radius:14px;padding:4px;background:conic-gradient(from -90deg,rgba(31,137,247,.96) 0 var(--p),rgba(116,187,255,.52) var(--p),rgba(151,164,181,.28) calc(var(--p) + 18deg),rgba(151,164,181,.28) 360deg);box-shadow:0 0 18px rgba(35,145,255,.25);-webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);-webkit-mask-composite:xor;mask-composite:exclude;pointer-events:none;transition:background .65s cubic-bezier(.22,.61,.36,1),box-shadow .65s ease}
.os-hi i{position:absolute;inset:-4px;border-radius:16px;background:linear-gradient(110deg,rgba(52,150,255,.20),rgba(52,150,255,.88),rgba(185,224,255,.42),rgba(52,150,255,.92),rgba(52,150,255,.20));background-size:260% 100%;animation:os-wave 2.8s ease-in-out infinite;filter:blur(3px);opacity:.72;-webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);-webkit-mask-composite:xor;mask-composite:exclude;padding:4px}
@keyframes os-wave{0%{background-position:0% 50%;opacity:.48}50%{background-position:100% 50%;opacity:.9}100%{background-position:0% 50%;opacity:.48}}
.os-stage{display:flex;justify-content:space-between;padding:7px 11px;margin:6px 0;border-radius:11px;background:linear-gradient(100deg,rgba(225,239,255,.42),rgba(196,224,255,.72),rgba(230,242,255,.44));background-size:220% 100%;animation:os-stage-wave 3.4s ease-in-out infinite;font-size:12px;transition:all .5s ease}@keyframes os-stage-wave{0%,100%{background-position:0% 50%}50%{background-position:100% 50%}}
.os-loading{padding:45px;text-align:center;border-radius:16px;background:rgba(240,245,250,.7)}.os-loading small{display:block;opacity:.6}
</style>""",unsafe_allow_html=True)
 # One card per semantic parameter on each script run. Some legacy UI paths create the same
 # label twice; dedupe by label instead of relying on wrapper installation state.
 seen_sections=set();seen_controls=set()
 def heading(base):
  def f(body,*a,**k):
   key=str(body);out=base(body,*a,**k)
   if key in SECTION_CARDS and key not in seen_sections:
    seen_sections.add(key);md(_card(*SECTION_CARDS[key]),unsafe_allow_html=True)
   return out
  f._onestring_base=base;return f
 st.header=heading(hdr)
 if callable(sub):st.subheader=heading(sub)
 def wrap(base):
  def f(label,*a,**k):
   key=str(label)
   if key in CONTROL_CARDS and key not in seen_controls:
    seen_controls.add(key);md(_card(*CONTROL_CARDS[key]),unsafe_allow_html=True)
   return base(label,*a,**k)
  f._onestring_base=base;return f
 for n in ("number_input","slider","checkbox","toggle","text_input"):
  b=getattr(st,n,None)
  if callable(b):setattr(st,n,wrap(unwrap(b)))
 def select(label,options,*a,**k):
  val=sel0(label,options,*a,**k)
  if str(label)=="View stage":
   s=VIEW.get(str(val),"");_render(st,"Original Figure 5 · blue frame = displayed stage",s,1 if s else 0)
  return val
 select._onestring_base=sel0;st.selectbox=select
 owner={"taken":False}
 class P:
  def __init__(self,b,p):self.b=b;self.p=p
  def progress(self,v,*a,**k):
   n,q=_stage(v,str(k.get("text","")));self.p.empty()
   with self.p.container():_render(st,"Original Figure 5 · current process",n,q)
   self.b.progress(v,*a,**k);return self
  def empty(self):self.p.empty();return self.b.empty()
  def __getattr__(self,n):return getattr(self.b,n)
 def progress(v=0,*a,**k):
  bar=prog0(v,*a,**k)
  if owner["taken"]:return bar
  owner["taken"]=True;ph=st.empty();n,q=_stage(v,str(k.get("text","")))
  with ph.container():_render(st,"Original Figure 5 · current process",n,q)
  return P(bar,ph)
 progress._onestring_base=prog0;st.progress=progress
__all__=["install_paper_ui_20260916_patch"]