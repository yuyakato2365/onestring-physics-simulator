import numpy as np

from tools.triangle_omega.controlled.seam_view import final_flat_faces, panel_paths


def test_welded_path_uses_updated_vertices_and_excludes_rim():
    grid = dict(VW=np.zeros((4,3)),Xg=np.array([[0,0],[1,0],[1,1],[2,0],[2,1],[3,1]]),
                Tg=np.array([[0,1,2],[3,5,4]]),
                wid=np.array([0,1,2,1,2,3]), pairs=[(0,1)], sec_of_run={0:0,1:1},
                sections=[dict(start=0,n=1),dict(start=2,n=1)], L=np.arange(4),
                pin_g={0:1,1:2,2:4,3:3}, sec_g={0:[],1:[]})
    # After relaxation the two UV banks share the same interior edge in 3D.
    panels = dict(X=np.array([[0,0,0],[1,0,.3],[1,1,.5],[2,1,0]]),
                  T=np.array([[0,1,2],[1,3,2]]),kept_vertices=np.arange(4),removed_panels=[])
    [(branch, path)] = panel_paths(grid,panels)
    assert branch == 0
    np.testing.assert_equal(path[:2], panels['X'][[1,2]])
    assert len(list(panel_paths(grid,panels,flat=True))) == 2
    grid['pairs'] = []
    assert list(panel_paths(grid,panels)) == []


def test_flat_faces_remove_collapsed_welds_before_indexing_removed_fins():
    grid = dict(wid=np.array([0,0,1,2,3,4]), Tg=np.array([[0,1,2],[0,2,3],[2,3,4],[3,4,5]]))
    np.testing.assert_equal(final_flat_faces(grid,dict(removed_panels=[1])),grid['Tg'][[1,3]])
