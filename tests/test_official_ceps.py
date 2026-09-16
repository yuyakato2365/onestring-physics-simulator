from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from onestring_physics.official_ceps import _parse_ceps_obj, official_ceps_rectangle


def _grid_mesh(nx: int = 6, ny: int = 5) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    for y in np.linspace(-0.8, 0.8, ny):
        for x in np.linspace(-1.0, 1.0, nx):
            vertices.append([x, y, 0.15 * np.sin(x) * np.cos(y)])
    faces = []
    for row in range(ny - 1):
        for column in range(nx - 1):
            a = row * nx + column
            b = a + 1
            c = a + nx
            d = c + 1
            faces.extend([[a, b, d], [a, d, c]])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def test_parse_ceps_obj_welds_per_corner_texture_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "refined.obj"
    path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 1 1 0",
                "v 0 1 0",
                "vt 0 0",
                "vt 1 0",
                "vt 1 1",
                "vt 0 0",
                "vt 1 1",
                "vt 0 1",
                "f 1/1 2/2 3/3",
                "f 1/4 3/5 4/6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = _parse_ceps_obj(path)
    assert result.surface_vertices.shape == (4, 3)
    assert result.surface_faces.shape == (2, 3)
    assert result.uv_vertices.shape == (4, 2)
    assert result.uv_faces.shape == (2, 3)
    assert result.raw_texture_coordinate_count == 6


def test_official_ceps_bridge_runs_fake_reference_cli_and_imports_common_refinement(tmp_path: Path) -> None:
    fake = tmp_path / "fake_ceps.py"
    fake.write_text(
        r'''
from pathlib import Path
import sys

input_path = Path(sys.argv[1])
options = {}
for argument in sys.argv[2:]:
    if argument.startswith("--") and "=" in argument:
        key, value = argument[2:].split("=", 1)
        options[key] = Path(value)

vertices = []
faces = []
for line in input_path.read_text(encoding="utf-8").splitlines():
    fields = line.split()
    if not fields:
        continue
    if fields[0] == "v":
        vertices.append(tuple(map(float, fields[1:4])))
    elif fields[0] == "f":
        faces.append([int(token.split("/")[0]) - 1 for token in fields[1:]])

output = options["outputLinearTextureFilename"]
with output.open("w", encoding="utf-8", newline="\n") as stream:
    for x, y, z in vertices:
        stream.write(f"v {x} {y} {z}\n")
    for face in faces:
        for vertex in face:
            x, y, _z = vertices[vertex]
            stream.write(f"vt {x} {y}\n")
    texture = 0
    for face in faces:
        refs = []
        for vertex in face:
            texture += 1
            refs.append(f"{vertex + 1}/{texture}")
        stream.write("f " + " ".join(refs) + "\n")

options["outputVertexMapFilename"].write_text(
    "\n".join(str(index) for index in range(len(vertices))) + "\n",
    encoding="utf-8",
)
options["outputLogFilename"].write_text("fake CEPS success\n", encoding="utf-8")
''',
        encoding="utf-8",
    )

    class Params:
        ceps_command = [sys.executable, str(fake)]
        ceps_timeout_seconds = 30.0
        boundary_target_aspect_mode = "fixed"
        boundary_target_aspect_ratio = 1.25
        boundary_target_aspect_min = 0.2
        boundary_target_aspect_max = 5.0

    vertices, faces = _grid_mesh()
    result, boundary, metrics = official_ceps_rectangle(
        vertices,
        faces,
        Params(),
        project_root=tmp_path,
    )

    assert result.surface_vertices.shape == vertices.shape
    assert result.surface_faces.shape == faces.shape
    assert result.uv_faces.shape == faces.shape
    assert boundary.shape[1] == 2
    assert np.allclose(boundary[0], boundary[-1])
    assert metrics["ceps_backend_used"] == "official_ceps_cli"
    assert metrics["ceps_common_refinement_used"] is True
    assert metrics["ceps_projective_interpolation_used"] is False
    assert metrics["ceps_prescribed_boundary_curvature"] is True
    assert metrics["uv_triangle_flip_count"] == 0
    assert metrics["uv_degenerate_triangle_count"] == 0
    assert "--noFreeBoundary" in metrics["ceps_command"]
