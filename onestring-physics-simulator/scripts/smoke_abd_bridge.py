from pathlib import Path
from types import SimpleNamespace as NS
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from onestring_physics.abd_backend import ABDBackendConfig, prepare_abd_job


tile = np.asarray(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, -0.1], [1, 0, -0.1], [1, 1, -0.1], [0, 1, -0.1]],
    dtype=float,
)
assembly = NS(vertices=np.asarray([tile, tile + [1.2, 0.0, 0.0]]), metrics={"tile_thickness": 0.1})
hinge = NS(
    tile_a=0, tile_b=1, surface="top", local_vertex_a=1, local_vertex_b=0,
    target_position_3d=np.asarray([1.1, 0.0, 0.0]),
)
gaps = [
    NS(id=0, surrounding_tiles=[0], centroid_2d=np.asarray([0.0, 0.0]), centroid_3d=np.asarray([0.0, 0.0, 0.0])),
    NS(id=1, surrounding_tiles=[1], centroid_2d=np.asarray([1.2, 0.0]), centroid_3d=np.asarray([1.2, 0.0, 0.0])),
]
state = NS(
    tiles_3d=assembly,
    tiles_2d_dual_hinge=NS(vertices=assembly.vertices.copy()),
    hinge_graph=NS(hinges=[hinge]),
    gap_graph=NS(gaps=gaps),
    string_path=NS(gap_ids=[0, 1]),
)
job = prepare_abd_job(state, ABDBackendConfig(steps=12), ROOT / "output" / "abd_bridge_smoke")
scene = json.loads(job.scene_path.read_text(encoding="utf-8"))
manifest = json.loads(job.manifest_path.read_text(encoding="utf-8"))
assert job.guide_count == 2
assert len(scene["rigid_body_problem"]["rigid_bodies"]) == 2
assert len(scene["rigid_body_problem"]["linear_constraints"]) == 1
assert manifest["string"]["inequality"] == "L(q) <= L_command(t)"
print(json.dumps({"scene": str(job.scene_path), "guides": job.guide_count, "bodies": 2}, indent=2))
