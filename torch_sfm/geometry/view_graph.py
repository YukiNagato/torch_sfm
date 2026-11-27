"""Directed DMap pair selection from symmetric frustum overlap."""
import numpy as np


def frustum_overlap(poses, depth, k):
    """Use camera-to-world poses and intrinsics scaled to the depth grid."""
    n, h, w = depth.shape
    y, x = np.mgrid[2:h:4, 2:w:4] + .5
    pixels = np.stack((x.ravel(), y.ravel()), -1)
    z = depth[:, 2::4, 2::4].reshape(n, -1, 1)
    rays = (pixels - k[:, None, 2:]) / k[:, None, :2]
    rotations, centers = poses[:, :3, :3], poses[:, :3, 3]
    world = np.concatenate((rays * z, z), axis=-1) @ rotations.transpose(0, 2, 1) + centers[:, None]
    coverage = np.zeros((n, n))
    for source, points in enumerate(world):
        valid = np.isfinite(z[source, :, 0]) & (z[source, :, 0] > 0)
        if not valid.any():
            continue
        target_xyz = (points[None, valid] - centers[:, None]) @ rotations
        uv = target_xyz[:, :, :2] / np.maximum(target_xyz[:, :, 2:], 1e-12) * k[:, None, :2] + k[:, None, 2:]
        inside = (target_xyz[:, :, 2] > 0) & ((uv >= 0) & (uv < [w, h])).all(-1)
        coverage[source] = inside.mean(axis=1)
    np.fill_diagonal(coverage, 0.)
    return np.sqrt(coverage * coverage.T)


def select_pairs(overlap, budget=16, seed=0):
    """Sample targets independently per source using weighted effective resistance."""
    if budget < 1:
        raise ValueError('The per-source edge budget must be positive')
    laplacian = np.diag(overlap.sum(axis=1)) - overlap
    inverse = np.linalg.pinv(laplacian, hermitian=True)
    resistance = inverse.diagonal()[:, None] + inverse.diagonal() - 2 * inverse
    leverage = overlap * np.maximum(resistance, 0.)
    rng = np.random.default_rng(seed)
    edges = []
    for source, row in enumerate(leverage):
        candidates = np.flatnonzero(row > 0)
        if not len(candidates):
            continue
        weights = row[candidates]
        targets = rng.choice(candidates, min(budget, len(candidates)), replace=False, p=weights / weights.sum())
        edges.extend((source, int(target)) for target in targets)
    return sorted(edges)
