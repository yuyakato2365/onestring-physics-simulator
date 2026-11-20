"""Redistribute mesh vertices with LargeSteps, projecting onto the supplied surface."""
import json
from pathlib import Path
import sys

import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src' / 'onestring_physics'))
from large_steps_mesh_conditioning import (
    LargeStepsMeshConditioningConfig,
    condition_mesh_with_large_steps,
)


def condition(source, output):
    mesh = trimesh.load(source, process=False)
    torch.set_num_threads(1)
    vertices, metrics = condition_mesh_with_large_steps(
        mesh.vertices.copy(), mesh.faces.copy(),
        LargeStepsMeshConditioningConfig(device='cpu', dtype='float64'),
        progress_callback=lambda event: print(json.dumps(event), flush=True),
    )
    trimesh.Trimesh(vertices, mesh.faces, process=False).export(output, digits=17)
    output.with_suffix('.json').write_text(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    condition(Path(sys.argv[1]), Path(sys.argv[2]))
