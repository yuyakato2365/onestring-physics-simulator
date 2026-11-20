import numpy as np
from scipy.sparse import csr_matrix

from tools.triangle_omega.controlled.rolling import converge_guide


def test_slow_face_converges_with_its_neighbors():
    # A face nearly equidistant from two strong, differently oriented guides
    # can take hundreds of updates to settle. A fourth face stays flat.
    tilt = .304
    c, s = np.cos(tilt), np.sin(tilt)
    normals = np.array([[c,s,0], [c,-s,0], [1.,0,0], [1.,0,0]])
    spatial = np.array([[100.,0,1.,0], [0,100.,.999,0], [1.,.999,.05,0], [0,0,0,1.]])
    guide, updates, reported = converge_guide(csr_matrix(spatial), normals)
    assert updates > 200
    delta = guide[:, None] - guide[None, :]
    weights = spatial * np.exp(-np.sum(delta**2, axis=2) / .18)
    expected = weights @ normals
    expected /= np.linalg.norm(expected, axis=1)[:, None]
    residual = np.max(np.linalg.norm(expected - guide, axis=1))
    assert residual < 1e-5
    np.testing.assert_allclose(residual, reported, atol=1e-14)
