"""Compatibility app.py for the side-face contact patch plus stable camera animation UI.

This delegates to the backed-up original app.py so the old Copy-Item workflow can
still copy an app.py from the patch zip without requiring a full app copy.

Important fix in this version:
- Plotly's built-in animate() redraws 3D scenes and can reset scene.camera every
  frame.  For 3D figures with frames, this wrapper renders a custom browser-side
  player that updates only trace data via Plotly.restyle().  It never touches
  layout.scene.camera during playback, so rotate/zoom/pan are preserved.
"""

from __future__ import annotations

import runpy
from pathlib import Path


_ROOT = Path(__file__).resolve().parent
_CANDIDATES = [
    _ROOT / "app_backup_before_sideface_contact.py",
    _ROOT / "app_backup_before_mitered_t3d.py",
]


def _is_wrapper_app(path: Path) -> bool:
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:4000]
    except Exception:
        return False
    markers = [
        "Compatibility app.py for the side-face contact patch",
        "_install_plotly_view_patch",
        "_CANDIDATES = [",
        "runpy.run_path",
    ]
    return sum(marker in head for marker in markers) >= 2


def _find_original_app() -> Path:
    for path in _CANDIDATES:
        if not path.exists() or path.resolve() == Path(__file__).resolve():
            continue
        if _is_wrapper_app(path):
            # Re-applying the patch can accidentally back up the previous wrapper.
            # Do not run it recursively; keep looking for the original app.
            continue
        return path
    tried = "\n  - ".join(str(p) for p in _CANDIDATES)
    raise RuntimeError(
        "Could not find a backed-up original app.py.\n"
        "The backup file appears to be another patch wrapper, so running it would recurse.\n"
        "Restore app.py from GitHub or an older backup, then run:\n"
        "  Copy-Item .\\app.py .\\app_backup_before_sideface_contact.py -Force\n\n"
        f"Tried:\n  - {tried}"
    )


def _install_plotly_view_patch() -> None:
    """Make 3D charts pan-friendly and render animations without camera reset."""
    try:
        import copy
        import json
        import uuid

        import plotly.graph_objects as go
        import plotly.io as pio
        from plotly.utils import PlotlyJSONEncoder
        import streamlit as st
        import streamlit.components.v1 as components
    except Exception:
        return

    if getattr(st, "_onestring_stable_camera_patch_installed", False):
        return

    original_plotly_chart = st.plotly_chart

    def _figure_has_3d_scene(fig) -> bool:
        try:
            if any(getattr(trace, "type", "") in {"mesh3d", "scatter3d", "surface"} for trace in getattr(fig, "data", [])):
                return True
            layout = getattr(fig, "layout", None)
            if layout is not None and getattr(layout, "scene", None) is not None:
                return True
        except Exception:
            return False
        return False

    def _figure_has_frames(fig) -> bool:
        try:
            return bool(getattr(fig, "frames", None)) and len(getattr(fig, "frames", [])) > 0
        except Exception:
            return False

    def _patch_3d_figure(fig):
        if fig is None or not _figure_has_3d_scene(fig):
            return fig
        try:
            fig.update_layout(
                uirevision="onestring-camera-stable-v4",
                transition=dict(duration=0),
            )
            fig.update_scenes(
                uirevision="onestring-camera-stable-v4",
                dragmode="orbit",
            )
        except Exception:
            pass
        return fig

    def _render_camera_stable_animation(fig, *, config: dict | None = None):
        """Render a Plotly 3D animation without Plotly.animate().

        Plotly.animate(frame.redraw=True) can rebuild the WebGL 3D scene and reset
        scene.camera once per frame.  This player preloads the same frame payloads
        but advances by calling Plotly.restyle() on the animated traces only.
        The layout is not re-applied during playback, so the current camera is
        preserved even while the user drags/zooms/pans.
        """
        try:
            _patch_3d_figure(fig)
            fig_dict = fig.to_plotly_json()
            frames = fig_dict.get("frames", []) or []
            if not frames:
                return False

            base_dict = copy.deepcopy(fig_dict)
            base_dict.pop("frames", None)
            layout = base_dict.setdefault("layout", {})
            # Remove Plotly's built-in animation controls.  They use animate(),
            # which is the source of the per-frame camera reset.
            layout.pop("updatemenus", None)
            layout.pop("sliders", None)
            layout["uirevision"] = "onestring-camera-stable-v4"
            layout.setdefault("scene", {})["uirevision"] = "onestring-camera-stable-v4"
            layout.setdefault("scene", {})["dragmode"] = "orbit"
            layout.setdefault("transition", {})["duration"] = 0

            div_id = f"onestring_stable_anim_{uuid.uuid4().hex}"
            base_fig = go.Figure(base_dict)
            chart_config = dict(config or {})
            chart_config.setdefault("scrollZoom", True)
            chart_config.setdefault("displayModeBar", True)
            chart_config.setdefault("responsive", True)
            buttons = list(chart_config.get("modeBarButtonsToAdd", []) or [])
            for name in ["pan3d", "orbitRotation", "tableRotation", "resetCameraDefault3d", "zoom3d"]:
                if name not in buttons:
                    buttons.append(name)
            chart_config["modeBarButtonsToAdd"] = buttons

            chart_html = pio.to_html(
                base_fig,
                include_plotlyjs=True,
                full_html=False,
                config=chart_config,
                div_id=div_id,
            )
            frames_json = json.dumps(frames, cls=PlotlyJSONEncoder)
            height = int(getattr(getattr(fig, "layout", None), "height", None) or 720)
            html = f"""
<div class="onestring-player-wrap">
  <div class="onestring-player-controls">
    <button id="{div_id}_play" type="button">▶ Play</button>
    <button id="{div_id}_pause" type="button">⏸ Pause</button>
    <button id="{div_id}_reset" type="button">⏮ Reset</button>
    <label>frame <span id="{div_id}_label">1</span> / <span id="{div_id}_total">{len(frames)}</span></label>
    <input id="{div_id}_slider" type="range" min="0" max="{max(0, len(frames)-1)}" value="0" step="1" />
    <label>fps <input id="{div_id}_fps" type="number" min="1" max="60" value="10" step="1" /></label>
    <span class="onestring-note">camera-stable player: frame updates do not touch scene.camera</span>
  </div>
  {chart_html}
</div>
<style>
.onestring-player-wrap {{ width: 100%; }}
.onestring-player-controls {{
  display: flex; align-items: center; gap: 0.55rem; flex-wrap: wrap;
  font-family: sans-serif; font-size: 13px; padding: 0.35rem 0.1rem 0.45rem 0.1rem;
}}
.onestring-player-controls button {{
  border: 1px solid rgba(49,51,63,.22); border-radius: 6px; background: white;
  padding: 0.25rem 0.55rem; cursor: pointer;
}}
.onestring-player-controls input[type=range] {{ min-width: 220px; flex: 1; }}
.onestring-player-controls input[type=number] {{ width: 3.6rem; }}
.onestring-note {{ color: rgba(49,51,63,.62); }}
</style>
<script>
(function() {{
  const gd = document.getElementById({json.dumps(div_id)});
  const frames = {frames_json};
  const slider = document.getElementById({json.dumps(div_id + '_slider')});
  const label = document.getElementById({json.dumps(div_id + '_label')});
  const playButton = document.getElementById({json.dumps(div_id + '_play')});
  const pauseButton = document.getElementById({json.dumps(div_id + '_pause')});
  const resetButton = document.getElementById({json.dumps(div_id + '_reset')});
  const fpsInput = document.getElementById({json.dumps(div_id + '_fps')});
  let frameIndex = 0;
  let timer = null;
  let savedCamera = null;
  let previousDragMode = 'orbit';
  let middleDrag = null;

  function clone(obj) {{
    if (!obj) return obj;
    try {{ return JSON.parse(JSON.stringify(obj)); }} catch (err) {{ return obj; }}
  }}

  function sceneKeys() {{
    const layout = gd && (gd._fullLayout || gd.layout) || {{}};
    const keys = Object.keys(layout).filter(k => /^scene[0-9]*$/.test(k));
    return keys.length ? keys : ['scene'];
  }}

  function readCamera(sceneKey) {{
    const full = gd && gd._fullLayout && gd._fullLayout[sceneKey];
    const lay = gd && gd.layout && gd.layout[sceneKey];
    return (full && full.camera) || (lay && lay.camera) || null;
  }}

  function saveCamera() {{
    const out = {{}};
    sceneKeys().forEach(k => {{
      const cam = readCamera(k);
      if (cam) out[k] = clone(cam);
    }});
    if (Object.keys(out).length) savedCamera = out;
  }}

  function restoreCamera() {{
    if (!savedCamera || !window.Plotly || !window.Plotly.relayout) return Promise.resolve();
    const update = {{}};
    Object.keys(savedCamera).forEach(k => {{ update[k + '.camera'] = clone(savedCamera[k]); }});
    if (!Object.keys(update).length) return Promise.resolve();
    try {{ return window.Plotly.relayout(gd, update); }} catch (err) {{ return Promise.resolve(); }}
  }}

  function relayoutDragMode(mode) {{
    if (!window.Plotly || !window.Plotly.relayout) return;
    const update = {{}};
    sceneKeys().forEach(k => {{ update[k + '.dragmode'] = mode; }});
    try {{ window.Plotly.relayout(gd, update); }} catch (err) {{}}
  }}

  function cameraRelayout(ev) {{
    if (!ev) return false;
    return Object.keys(ev).some(k => /(^|\.)camera(\.|$)/.test(k) || /^scene[0-9]*\.camera/.test(k));
  }}

  function wrapForRestyle(traceData) {{
    const update = {{}};
    Object.keys(traceData || {{}}).forEach(k => {{
      if (k === 'type') return;
      update[k] = [traceData[k]];
    }});
    return update;
  }}

  function applyFrame(i) {{
    if (!frames.length) return;
    i = Math.max(0, Math.min(frames.length - 1, i));
    frameIndex = i;
    slider.value = String(i);
    label.textContent = String(i + 1);

    // Capture the camera immediately before changing data.  This handles the
    // case where the user is dragging/zooming while playback is active.
    saveCamera();
    const frame = frames[i] || {{}};
    const frameData = frame.data || [];
    const frameTraces = frame.traces || frameData.map((_, idx) => idx);
    const promises = [];
    frameData.forEach((traceData, idx) => {{
      const traceIndex = frameTraces[idx] == null ? idx : frameTraces[idx];
      const update = wrapForRestyle(traceData);
      if (Object.keys(update).length) {{
        try {{ promises.push(window.Plotly.restyle(gd, update, [traceIndex])); }} catch (err) {{}}
      }}
    }});
    Promise.all(promises).then(() => restoreCamera());
  }}

  function play() {{
    saveCamera();
    if (timer) clearInterval(timer);
    const fps = Math.max(1, Math.min(60, parseInt(fpsInput.value || '10', 10)));
    const interval = Math.max(16, Math.round(1000 / fps));
    timer = setInterval(() => {{
      const next = (frameIndex + 1) % Math.max(1, frames.length);
      applyFrame(next);
    }}, interval);
  }}

  function pause() {{
    if (timer) clearInterval(timer);
    timer = null;
    saveCamera();
  }}

  function cloneAsLeftMouseEvent(type, e, buttons) {{
    return new MouseEvent(type, {{
      bubbles: true, cancelable: true, view: window,
      screenX: e.screenX, screenY: e.screenY,
      clientX: e.clientX, clientY: e.clientY,
      ctrlKey: e.ctrlKey, shiftKey: e.shiftKey, altKey: e.altKey, metaKey: e.metaKey,
      button: 0, buttons: buttons
    }});
  }}

  gd.addEventListener('auxclick', function(e) {{
    if (e.button === 1) {{ e.preventDefault(); e.stopPropagation(); }}
  }}, true);

  gd.addEventListener('mousedown', function(e) {{
    if (e.button !== 1) return;
    pause();
    saveCamera();
    e.preventDefault();
    e.stopPropagation();
    previousDragMode = ((gd.layout || {{}}).scene || {{}}).dragmode || 'orbit';
    middleDrag = {{ target: e.target }};
    relayoutDragMode('pan');
    e.target.dispatchEvent(cloneAsLeftMouseEvent('mousedown', e, 1));
  }}, true);

  document.addEventListener('mousemove', function(e) {{
    if (!middleDrag) return;
    e.preventDefault();
    e.stopPropagation();
    middleDrag.target.dispatchEvent(cloneAsLeftMouseEvent('mousemove', e, 1));
  }}, true);

  document.addEventListener('mouseup', function(e) {{
    if (!middleDrag) return;
    e.preventDefault();
    e.stopPropagation();
    middleDrag.target.dispatchEvent(cloneAsLeftMouseEvent('mouseup', e, 0));
    saveCamera();
    relayoutDragMode(previousDragMode || 'orbit');
    middleDrag = null;
  }}, true);

  if (gd && gd.on) {{
    gd.on('plotly_relayout', function(ev) {{ if (cameraRelayout(ev)) setTimeout(saveCamera, 0); }});
    gd.on('plotly_afterplot', function() {{ if (!savedCamera) setTimeout(saveCamera, 0); }});
  }}

  slider.addEventListener('input', function() {{ pause(); applyFrame(parseInt(slider.value || '0', 10)); }});
  playButton.addEventListener('click', play);
  pauseButton.addEventListener('click', pause);
  resetButton.addEventListener('click', function() {{ pause(); applyFrame(0); }});
  fpsInput.addEventListener('change', function() {{ if (timer) play(); }});

  setTimeout(function() {{ saveCamera(); applyFrame(0); }}, 100);
}})();
</script>
"""
            components.html(html, height=height + 78, scrolling=False)
            return True
        except Exception as exc:
            try:
                st.warning(f"Camera-stable animation renderer failed; falling back to Streamlit Plotly chart: {exc}")
            except Exception:
                pass
            return False

    def patched_plotly_chart(fig, *args, **kwargs):
        _patch_3d_figure(fig)
        config = dict(kwargs.pop("config", {}) or {})
        config.setdefault("scrollZoom", True)
        config.setdefault("displayModeBar", True)
        config.setdefault("responsive", True)
        buttons = list(config.get("modeBarButtonsToAdd", []) or [])
        for name in ["pan3d", "orbitRotation", "tableRotation", "resetCameraDefault3d", "zoom3d"]:
            if name not in buttons:
                buttons.append(name)
        config["modeBarButtonsToAdd"] = buttons
        kwargs["config"] = config

        # Critical fix: never let Plotly.animate drive 3D animation frames.  It
        # resets scene.camera in this Streamlit setup.  Use our data-only player.
        if fig is not None and _figure_has_3d_scene(fig) and _figure_has_frames(fig):
            rendered = _render_camera_stable_animation(fig, config=config)
            if rendered:
                return None

        return original_plotly_chart(fig, *args, **kwargs)

    st.plotly_chart = patched_plotly_chart
    st._onestring_stable_camera_patch_installed = True

    # For non-animation Streamlit Plotly charts, keep middle-mouse pan available
    # in the parent document as well.
    try:
        components.html(
            r'''
<script>
(function () {
  const root = window.parent && window.parent.document ? window.parent.document : document;
  const win = window.parent || window;
  if (root.__onestringMiddlePanInstalledV4) return;
  root.__onestringMiddlePanInstalledV4 = true;

  function getPlotly() { return win.Plotly || window.Plotly || null; }
  function closestPlot(target) { return target && target.closest ? target.closest('.js-plotly-plot') : null; }
  function sceneKeys(gd) {
    const layout = (gd && (gd._fullLayout || gd.layout)) || {};
    const keys = Object.keys(layout).filter(k => /^scene[0-9]*$/.test(k));
    return keys.length ? keys : ['scene'];
  }
  function relayoutDragMode(gd, mode) {
    const Plotly = getPlotly();
    if (!Plotly || !Plotly.relayout || !gd) return;
    const updates = {};
    sceneKeys(gd).forEach(k => { updates[k + '.dragmode'] = mode; });
    try { Plotly.relayout(gd, updates); } catch (err) {}
  }
  function cloneAsLeft(type, e, buttons) {
    return new MouseEvent(type, {
      bubbles: true, cancelable: true, view: e.view || win,
      screenX: e.screenX, screenY: e.screenY,
      clientX: e.clientX, clientY: e.clientY,
      ctrlKey: e.ctrlKey, shiftKey: e.shiftKey, altKey: e.altKey, metaKey: e.metaKey,
      button: 0, buttons: buttons
    });
  }
  let active = null;
  root.addEventListener('auxclick', function (e) {
    if (e.button === 1 && closestPlot(e.target)) { e.preventDefault(); e.stopPropagation(); }
  }, true);
  root.addEventListener('mousedown', function (e) {
    if (e.button !== 1) return;
    const gd = closestPlot(e.target);
    if (!gd) return;
    e.preventDefault(); e.stopPropagation();
    active = { gd: gd, target: e.target, previousDragmode: ((gd.layout || {}).scene || {}).dragmode || 'orbit' };
    relayoutDragMode(gd, 'pan');
    e.target.dispatchEvent(cloneAsLeft('mousedown', e, 1));
  }, true);
  root.addEventListener('mousemove', function (e) {
    if (!active) return;
    e.preventDefault(); e.stopPropagation();
    active.target.dispatchEvent(cloneAsLeft('mousemove', e, 1));
  }, true);
  root.addEventListener('mouseup', function (e) {
    if (!active) return;
    e.preventDefault(); e.stopPropagation();
    active.target.dispatchEvent(cloneAsLeft('mouseup', e, 0));
    relayoutDragMode(active.gd, active.previousDragmode || 'orbit');
    active = null;
  }, true);
})();
</script>
            ''',
            height=0,
            width=0,
        )
    except Exception:
        return


_install_plotly_view_patch()
runpy.run_path(str(_find_original_app()), run_name="__main__")
