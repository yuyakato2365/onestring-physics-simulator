from __future__ import annotations
import numpy as np

def test_paper_t3d_patch_forces_fixed_frustum(monkeypatch):
    from onestring_physics import onestring_pipeline as pipeline
    from onestring_physics.paper_t3d_20260920_patch import install_paper_t3d_20260920_patch

    old = pipeline._extrude_tiles
    try:
        install_paper_t3d_20260920_patch(pipeline)
        assert pipeline._extrude_tiles is not old
        assert getattr(pipeline, "_onestring_paper_t3d_20260920_installed", False)
    finally:
        pipeline._extrude_tiles = old
        pipeline._onestring_paper_t3d_20260920_installed = False
