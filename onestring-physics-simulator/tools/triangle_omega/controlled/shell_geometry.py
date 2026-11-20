"""Connectivity of a six-corner panel from a consistently triangulated shell."""
import numpy as np


def shell_topology(faces):
    panels = len(faces)
    triangles = np.empty((panels, 8, 3), int)
    triangles[:, 0] = [0, 1, 2]
    triangles[:, 1] = [5, 4, 3]
    for k in range(3):
        j = (k+1) % 3
        # Both incident panels use the diagonal from the lower global ID on
        # the outer surface to the higher global ID on the inner surface.
        forward = faces[:, k] < faces[:, j]
        triangles[:, 2+2*k] = np.where(forward[:,None], [k,j+3,j], [k,k+3,j])
        triangles[:, 3+2*k] = np.where(forward[:,None], [k,k+3,j+3], [j,k+3,j+3])
    a,b,c = np.argsort(faces, axis=1).T
    tets = np.stack([np.stack(t, axis=1) for t in
                     ((a,b,c,c+3), (a,b,b+3,c+3), (a,a+3,b+3,c+3))], axis=1)
    return triangles, tets
