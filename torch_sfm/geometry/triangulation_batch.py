"""Bounded look-ahead of geometric evidence for incremental triangulation.

The cache never stores assignments, inlier sets or chosen tracks. A caller can
reuse a pair's geometry after other observations have been claimed, then score
only the observations still present in its current track, in the original order.
"""

import math
from itertools import combinations

import torch


class CandidateCache:
    def __init__(self, owner):
        self.owner = owner
        device = owner.device
        self.poses = owner.poses.to(device)
        self.rotations = owner.rotations.to(device)
        self.centers = owner.centers.to(device)
        self.frame_ids = owner.frame_ids.to(device)
        self.rays = owner.rays.to(device)
        self.xy = owner.xy.to(device)
        self.calibration = owner.k[:, [0, 1, 0, 1], [0, 1, 2, 2]].to(device)
        self.entries = {}
        self.stats = {"batches": 0, "pairs": 0, "errors": 0, "hits": 0, "misses": 0}

    def prepare(self, tracks):
        self.entries.clear()
        pair_nodes = {}
        work = 0
        for nodes in tracks:
            # This upper bound deliberately counts duplicate work; it bounds
            # both temporary tensors and Python metadata before deduplication.
            size = len(nodes) * (len(nodes) - 1) // 2 * len(nodes)
            if work + size > self.owner.config.batch_max_observations:
                break
            work += size
            for a, b in combinations(nodes, 2):
                pair = (min(a, b), max(a, b))
                pair_nodes.setdefault(pair, set()).update(nodes)
        if not pair_nodes:
            return
        self.stats["batches"] += 1
        pairs = list(pair_nodes)
        idx = torch.tensor(pairs, device=self.poses.device)
        fid = self.frame_ids[idx]
        poses = self.poses[fid]
        rays = self.rays[idx]
        xy = rays[..., :2] / rays[..., 2:]
        a = (xy[..., None] * poses[:, :, 2:3] - poses[:, :, :2]).reshape(-1, 4, 4)
        _, singular, vectors = torch.linalg.svd(a)
        h = vectors[:, -1]
        xyz = h[:, :3] / h[:, 3:]
        offset = xyz[:, None] - self.centers[fid]
        local = (self.rotations[fid] @ offset[..., None]).squeeze(-1)
        length = offset.norm(dim=-1)
        direction = offset / length[..., None]
        cosine = (direction[:, 0] * direction[:, 1]).sum(-1).abs()
        k_pair = self.calibration[fid]
        projection = local[..., :2] / local[..., 2:] * k_pair[..., :2] + k_pair[..., 2:]
        pixel_error = (projection - self.xy[idx]).norm(dim=-1)
        angle_limit = math.cos(math.radians(self.owner.config.min_angle))
        stable = (
            torch.isfinite(xyz).all(-1)
            & torch.isfinite(local).all(dim=(-1, -2))
            & torch.isfinite(pixel_error).all(-1)
            & (local[..., 2] >= 1e-12).all(-1)
            & (length > 0).all(-1)
            & (cosine <= angle_limit)
        )
        # Ambiguous null spaces or threshold-adjacent decisions use the scalar
        # path, including its original pair order, instead of guessing parity.
        safe = (
            (singular[:, -2] > singular[:, 0] * 1e-10)
            & ((cosine - angle_limit).abs() > 1e-12)
            & ((local[..., 2] - 1e-12).abs() > 1e-12).all(-1)
        )
        targets = [sorted(pair_nodes[p]) for p in pairs]
        candidate_ids = [i for i, nodes in enumerate(targets) for _ in nodes]
        nodes = [n for group in targets for n in group]
        ci = torch.tensor(candidate_ids, device=xyz.device)
        ni = torch.tensor(nodes, device=xyz.device)
        fid = self.frame_ids[ni]
        local = (
            self.rotations[fid] @ (xyz[ci] - self.centers[fid])[..., None]
        ).squeeze(-1)
        k = self.calibration[fid]
        ray = torch.cat(
            ((self.xy[ni] - k[:, 2:]) / k[:, :2], torch.ones_like(local[:, :1])), -1
        )
        denominator = (ray.square().sum(-1) * local.square().sum(-1)).sqrt()
        error = ((ray * local).sum(-1) / denominator).clamp(-1, 1).acos()
        error = error.masked_fill(
            ~torch.isfinite(local).all(-1) | (local[:, 2] < 1e-12), math.inf
        )
        errors = error.cpu().tolist()
        models, stable, safe = (
            xyz.cpu().tolist(),
            stable.cpu().tolist(),
            safe.cpu().tolist(),
        )
        start = 0
        threshold = math.radians(self.owner.config.create_max_angle_error)
        for pair, model, valid, reusable, targets_for_pair in zip(
            pairs, models, stable, safe, targets, strict=True
        ):
            end = start + len(targets_for_pair)
            values = errors[start:end]
            if reusable and not any(abs(e - threshold) < 5e-8 for e in values):
                self.entries[pair] = (
                    tuple(model) if valid else None,
                    dict(zip(targets_for_pair, values, strict=True)),
                )
            start = end
        self.stats["pairs"] += len(pairs)
        self.stats["errors"] += len(nodes)

    def get(self, nodes, pairs):
        candidates = []
        for a, b in pairs:
            entry = self.entries.get((min(a, b), max(a, b)))
            if entry is None or any(n not in entry[1] for n in nodes):
                self.stats["misses"] += 1
                return None
            model, errors = entry
            candidates.append((model, [errors[n] for n in nodes]))
        self.stats["hits"] += 1
        return candidates
