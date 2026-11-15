"""Triangulation angle filtering shared by global and legacy BA."""

import math

import torch
import torch.nn.functional as F


def triangulation_mask(centers, points, observations, minimum):
    if not len(points) or minimum == 0:
        return torch.ones(len(points), dtype=torch.bool, device=points.device)
    pid, fid, _ = observations.T
    directions = F.normalize(points[pid] - centers[fid], dim=-1)
    counts = torch.bincount(pid, minlength=len(points))
    order = pid.argsort(stable=True)
    offsets = counts.cumsum(0) - counts
    threshold = math.cos(math.radians(minimum))
    if points.is_cuda and points.dtype in (torch.float32, torch.float64):
        from .warp.track_angles import track_angle_mask
        return track_angle_mask(directions, order, offsets, counts, threshold)
    slots = torch.empty_like(pid)
    slots[order] = torch.arange(len(order), device=points.device) - offsets[pid[order]]
    padded = points.new_zeros((len(points), int(counts.max()), 3))
    valid = torch.zeros(padded.shape[:2], dtype=torch.bool, device=points.device)
    padded[pid, slots], valid[pid, slots] = directions, True
    minimum_cosine = points.new_ones(len(points))
    for slot in range(padded.shape[1]):
        cosine = (padded * padded[:, slot : slot + 1]).sum(-1).abs()
        pairs = valid & valid[:, slot : slot + 1]
        minimum_cosine = torch.minimum(
            minimum_cosine, cosine.masked_fill(~pairs, 1).amin(-1))
    return minimum_cosine <= threshold
