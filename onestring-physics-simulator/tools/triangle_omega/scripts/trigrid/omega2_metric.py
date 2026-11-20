"""Measure the actual local change of scale in the assembled Omega2 chart."""
import numpy as np


def relative_stretch(before, after, faces):
    def twice_area(uv):
        p, q, r = uv[faces].transpose(1, 0, 2)
        a, b = q - p, r - p
        return np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])

    a, b = twice_area(before), twice_area(after)
    # Grid spacing follows the area of the chart. A global zoom therefore
    # has no effect on the distribution of lattice panels.
    scale = np.sqrt(b / a)
    return scale / np.sqrt(b.sum() / a.sum())
