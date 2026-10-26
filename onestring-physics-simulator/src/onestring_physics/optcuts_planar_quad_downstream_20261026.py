"""Narrow compatibility fixes for unstructured all-quad OneString grids.

All three adapters are no-ops for original rectangular QuadGrid. For irregular
quads collision pairs MUST be conservative: never silently examine zero pairs
because nx=ny=0. Boundary edges are derived from actual face incidence.
"""
from __future__ import annotations
import numpy as np
from .optcuts_planar_quad_20261026 import UnstructuredQuadGrid


def install_unstructured_quad_downstream_20261026():
    from . import legacy_pipeline_base as old
    if getattr(old, "_onestring_unstructured_quad_20261026_installed", False):
        return
    original_pairs = old._collision_candidate_pairs
    original_boundary = old._boundary_tile_edges
    original_initial = old._initial_auxetic_vertex_hinge_layout

    def actual_pairs(tile_count, grid, all_pairs):
        if not isinstance(grid, UnstructuredQuadGrid):
            return original_pairs(tile_count, grid, all_pairs)
        # Explicitly conservative: no missed collisions based on fictitious
        # rectangular row/column adjacency. Future acceleration can use BVH.
        return [(i, j) for i in range(int(tile_count))
                for j in range(i+1, int(tile_count))]

    def actual_boundary(grid):
        if not isinstance(grid, UnstructuredQuadGrid):
            return original_boundary(grid)
        owner: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
        for tile in grid.tiles:
            for k in range(4):
                a, b = tile.vertex_ids[k], tile.vertex_ids[(k+1) % 4]
                owner.setdefault(tuple(sorted((int(a), int(b)))), []).append(
                    (tile.id, k, (k+1) % 4))
        boundary = [items[0] for items in owner.values() if len(items) == 1]
        outgoing = {}
        for tile, a, b in boundary:
            face = grid.tiles[int(tile)].vertex_ids
            initial = face[int(a)]
            end = face[int(b)]
            if initial in outgoing:
                raise ValueError("PLANAR_QUAD_BOUNDARY_BRANCHING")
            outgoing[initial] = (end, int(tile), (int(a), int(b)))
        first = min(outgoing)
        here = first
        ordered = []
        seen = set()
        while here not in seen:
            seen.add(here)
            if here not in outgoing:
                raise ValueError("PLANAR_QUAD_BOUNDARY_DISCONNECTED")
            end, tile, corners = outgoing[here]
            ordered.append((tile, corners))
            here = end
        if here != first or len(ordered) != len(boundary):
            raise ValueError("PLANAR_QUAD_BOUNDARY_NOT_SINGLE_CYCLE")
        return ordered

    def initial_layout(raw_tiles, grid):
        if not isinstance(grid, UnstructuredQuadGrid):
            return original_initial(raw_tiles, grid)
        tiles = np.asarray(raw_tiles, float)
        if len(tiles) == 0:
            return tiles.copy()
        centers = tiles.mean(axis=1)
        global_center = centers.mean(axis=0)
        base_len = max(float(grid.tile_size), 1e-8)
        opening = min(.65, max(.12, float(grid.gap_size) / base_len * 2.5))
        adj: list[list[int]] = [[] for _ in tiles]
        for h in grid.hinges:
            adj[h.tile_a].append(h.tile_b)
            adj[h.tile_b].append(h.tile_a)
        signs = np.zeros(len(tiles), int)
        for seed in range(len(tiles)):
            if signs[seed]:
                continue
            signs[seed] = 1
            stack = [seed]
            while stack:
                i = stack.pop()
                for j in adj[i]:
                    if not signs[j]:
                        signs[j] = -signs[i]
                        stack.append(j)
        out = tiles.copy()
        for i in range(len(tiles)):
            angle = float(signs[i]) * opening
            c, sn = np.cos(angle), np.sin(angle)
            rot = np.array([[c, -sn], [sn, c]])
            out[i] = (out[i] - centers[i]) @ rot.T + centers[i]
            # Radially expand the layout without assigning fictitious (row,col).
            out[i] += (centers[i] - global_center) * grid.gap_size / base_len
        return out

    old._collision_candidate_pairs = actual_pairs
    old._boundary_tile_edges = actual_boundary
    old._initial_auxetic_vertex_hinge_layout = initial_layout
    old._onestring_unstructured_quad_20261026_installed = True
