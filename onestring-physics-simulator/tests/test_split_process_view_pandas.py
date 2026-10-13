"""Regression: Split Map can construct Plotly traces with Pandas fully initialized.

The test uses synthetic M2D output; no BFF/solver compute is involved.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
from onestring_physics.split_process_view import figure_split_process


def test_split_map_draws_before_and_after_split_without_pandas_partial_init():
    # Two adjacent quads, split into two recorded components.
    vertices = np.array([
        [0., 0.], [1., 0.], [2., 0.],
        [0., 1.], [1., 1.], [2., 1.],
    ])
    faces = np.array([[0, 1, 4, 3], [1, 2, 5, 4]])
    mesh = SimpleNamespace(
        vertices=vertices,
        faces=faces,
        metrics={
            "csf_split_initial_components": [
                {"face_ids": [0, 1], "sigma": 2.4},
            ],
            "csf_split_step_analysis": [
                {
                    "face_ids": [0, 1],
                    "child_face_ids": [[0], [1]],
                    "sigma_after": [1.2, 1.3],
                    "split_boundary_pairs": [(0, 1, 1, 3)],
                }
            ],
        },
    )
    assert isinstance(pd.Series([1.0]), pd.Series)
    for step, expected_groups in [(0, 1), (1, 2)]:
        fig, rows = figure_split_process(mesh, step)
        assert fig is not None
        assert len(rows) == expected_groups
        assert len(fig.data) == expected_groups + 2
        assert all("hovertemplate" in t.to_plotly_json()
                   for t in fig.data[:expected_groups])
