"""Exact assembled-coordinate sharing; all exported Split vertex IDs survive."""
import copy
import numpy as np


def paired_vertices(vertices, faces, seam_edges):
    """Resolve endpoint order only on explicitly recorded matching seam edges."""
    vertices, faces = np.asarray(vertices), np.asarray(faces, int)
    pairs = set()
    tol = max(float(np.linalg.norm(np.ptp(vertices, axis=0))) * 1e-9, 1e-12)
    for a, ea, b, eb in seam_edges:
        left = faces[a, [ea, (ea+1) % 4]]
        right = faces[b, [eb, (eb+1) % 4]]
        if np.linalg.norm(vertices[left]-vertices[right[::-1]]) < np.linalg.norm(vertices[left]-vertices[right]):
            right = right[::-1]
        if np.max(np.linalg.norm(vertices[left]-vertices[right], axis=1)) > tol:
            raise ValueError('Recorded Split seam endpoints do not coincide in canonical coordinates')
        pairs.update(tuple(sorted((int(i), int(j)))) for i, j in zip(left, right) if i != j)
    return [list(pair) for pair in sorted(pairs)]


def propagate(source, target):
    pairs = source.metrics.get('split_vertex_pairs')
    if pairs is None:
        pairs = paired_vertices(getattr(source, '_split_panel_source_vertices', source.vertices),
                                source.faces, source.metrics.get('split_boundary_pairs', []))
    source.metrics['split_vertex_pairs'] = copy.deepcopy(pairs)
    target.metrics['split_vertex_pairs'] = copy.deepcopy(pairs)
    target.metrics['split_boundary_pairs'] = copy.deepcopy(source.metrics.get('split_boundary_pairs', []))


def pair_errors(vertices, pairs):
    ids = np.asarray(pairs, int).reshape(-1, 2)
    error = np.linalg.norm(np.asarray(vertices)[ids[:, 0]]-np.asarray(vertices)[ids[:, 1]], axis=1)
    return dict(split_pair_count=len(ids), split_pair_error_max=float(error.max()) if len(error) else 0.,
                split_pair_error_mean=float(error.mean()) if len(error) else 0.,
                split_pair_error_rms=float(np.sqrt(np.mean(error**2))) if len(error) else 0.)


def optimize_shared(base, target, mesh, parameterization, params):
    pairs = mesh.metrics.get('split_vertex_pairs', [])
    if not pairs:
        result, report = base(target, mesh, parameterization, params)
        result.metrics.update(pair_errors(result.vertices, pairs))
        return result, report
    parent = np.arange(len(mesh.vertices))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for a, b in pairs:
        parent[root(int(b))] = root(int(a))
    roots = np.array([root(i) for i in range(len(parent))])
    representatives, mapping = np.unique(roots, return_inverse=True)
    reduced = copy.copy(mesh)
    reduced.vertices = np.asarray(mesh.vertices)[representatives].copy()
    reduced.faces = mapping[np.asarray(mesh.faces)]
    reduced.metrics = dict(mesh.metrics, split_vertex_pairs=[], split_boundary_pairs=[])
    result, report = base(target, reduced, parameterization, params)
    if result.vertices.shape != reduced.vertices.shape or not np.array_equal(result.faces, reduced.faces):
        raise RuntimeError('K3D solver changed reduced mesh topology; Split correspondence cannot be restored')
    result.vertices = result.vertices[mapping].copy()
    result.faces = np.asarray(mesh.faces).copy()
    result.metrics.update(split_vertex_pairs=copy.deepcopy(pairs),
                          split_boundary_pairs=copy.deepcopy(mesh.metrics.get('split_boundary_pairs', [])),
                          assembled_geometry_map=mapping.tolist(),
                          assembled_geometry_reduced_vertex_count=len(representatives),
                          split_geometry_constraint='shared optimization variables; exported topology unchanged',
                          split_geometry_classification='project-specific exact equality constraint')
    result.metrics.update(pair_errors(result.vertices, pairs))
    return result, report
