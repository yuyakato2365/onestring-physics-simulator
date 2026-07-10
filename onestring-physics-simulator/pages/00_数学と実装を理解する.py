from __future__ import annotations

import base64
import zlib
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="OneString 数学・実装ガイド", layout="wide")

_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "onestring_learning"
_payload = "".join(path.read_text(encoding="ascii") for path in sorted(_ASSET_DIR.glob("part_*.txt")))
html = zlib.decompress(base64.b85decode(_payload.encode("ascii"))).decode("utf-8")

st.markdown(
    """<style>
    [data-testid="stHeader"],[data-testid="stSidebar"]{display:none}
    .block-container{padding:0!important;max-width:none!important}
    iframe{border:0}
    </style>""",
    unsafe_allow_html=True,
)
components.html(html, height=920, scrolling=False)
