"""Deterministic local stars from accepted correspondence edges."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class StarTopology:
    stars: list[tuple[int, ...]]  # center first, then score-sorted neighbors
    edge_scores: dict[tuple[int, int], float]
    components: list[list[int]]
    restored_edges: list[tuple[int, int]]


def graph_components(num_images, edges):
    parent = list(range(num_images))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in edges:
        parent[root(i)] = root(j)
    groups = {}
    for i in range(num_images):
        groups.setdefault(root(i), []).append(i)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def build_stars(num_images, correspondences, *, max_neighbors=25, min_matches=15,
                sequential_edges=()):
    """Top-K FlexGlue neighbors; restore bridges pruned by the per-star cap.

    Restoration may exceed max_neighbors, as in GlueMap. Never introduce an
    edge rejected by the frontend or invent connectivity between components.
    Scores are FlexGlue covisibility (not Doppelgangers++ probabilities).
    """
    if num_images < 1 or max_neighbors < 1 or min_matches < 1:
        raise ValueError('Image count, neighbor count and minimum matches must be positive')
    edges = {}
    for (i, j), matches in correspondences.items():
        i, j = int(i), int(j)
        if not (0 <= i < num_images and 0 <= j < num_images) or i == j:
            raise ValueError('Invalid correspondence image indices')
        if matches.metadata.get('rejected') or len(matches.indices) < min_matches:
            continue
        score = float(matches.metadata['covisibility_score'])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid FlexGlue covisibility score')
        key = (min(i, j), max(i, j))
        edges[key] = max(edges.get(key, 0.), score)
    adjacency = [[] for _ in range(num_images)]
    for (i, j), score in edges.items():
        adjacency[i].append((j, score))
        adjacency[j].append((i, score))
    sequential = {tuple(sorted(edge)) for edge in sequential_edges}
    neighbors = []
    for i, row in enumerate(adjacency):
        ranked = sorted(row, key=lambda v: (-v[1], v[0]))
        if sequential and len(row) > max_neighbors:
            temporal = sorted(j for j, _ in row if (min(i, j), max(i, j)) in sequential)
            # Reserve up to five non-temporal slots, also supporting small stars.
            limit = min(20, max(0, max_neighbors-5))
            if len(temporal) > limit:
                if limit >= 2:
                    middle = np.unique(np.round(np.linspace(0, len(temporal)-3, limit-2)).astype(int))
                    temporal = [temporal[0], *(temporal[1:-1][j] for j in middle), temporal[-1]]
                else:
                    temporal = temporal[:limit]
            nonseq = [j for j, _ in ranked if (min(i, j), max(i, j)) not in sequential]
            neighbors.append(set(temporal + nonseq[:max_neighbors-len(temporal)]))
        else:
            neighbors.append(set(j for j, _ in ranked[:max_neighbors]))
    kept = {(min(i, j), max(i, j)) for i, js in enumerate(neighbors) for j in js}
    # Union-find restores only bridges and retains the original components.
    parent = list(range(num_images))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in kept:
        parent[root(i)] = root(j)
    restored = []
    for (i, j), _ in sorted(edges.items(), key=lambda v: (-v[1], v[0])):
        if root(i) != root(j):
            parent[root(i)] = root(j)
            neighbors[i].add(j)
            neighbors[j].add(i)
            restored.append((i, j))
    stars = [(i, *sorted(js, key=lambda j: (-edges[min(i,j), max(i,j)], j)))
             for i, js in enumerate(neighbors) if js]
    return StarTopology(stars, edges, graph_components(num_images, edges), restored)
