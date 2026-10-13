"""Launch the isolated 2026-09-23 feature-flag experiment."""
import os
from pathlib import Path
import runpy

# Plotly 5.x checks pandas.Series and pandas.Index while validating traces.
# Preload pandas before Streamlit/Plotly render threads can see a partially
# initialized pandas module in sys.modules.
import pandas as pd
_ = (pd.Series, pd.Index)
os.environ["ONESTRING_EXTRUSION_AWARE_20260923"] = "1"
runpy.run_path(str(Path(__file__).with_name("app_optcuts_20260920_paper_t3d.py")), run_name="__main__")
