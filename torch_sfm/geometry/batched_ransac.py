"""Independent LO-RANSAC states sharing GPU hypothesis generation and scoring."""

import math

import torch
import torch.nn.functional as F

from .two_view import (essential_five_point, fundamental_seven_point, homography_dlt,
                       homogeneous, _epipolar_design, _project_essential)


def model_errors(models, owners, points0, points1, valid, *, homography=False, backend='auto'):
    if models.is_cuda and backend != 'torch':
        try:
            from .warp_two_view import score_models
        except ImportError:
            if backend == 'warp':
                raise
        else:
            return score_models(models, owners, points0, points1, valid, homography)
    x0, x1 = homogeneous(points0[owners]), homogeneous(points1[owners])
    line1 = x0 @ models.transpose(-1, -2)
    if homography:
        z = line1[..., 2]
        usable = z.abs() > 1e-12
        z = torch.where(usable, z, 1)
        errors = (line1[..., :2] / z[..., None] - x1[..., :2]).square().sum(-1)
        errors = errors.masked_fill(~usable, torch.inf)
    else:
        line0 = x1 @ models
        errors = (x1 * line1).sum(-1).square() / (
            line0[..., :2].square().sum(-1) + line1[..., :2].square().sum(-1)).clamp_min(1e-30)
    errors = errors.nan_to_num(nan=torch.inf, posinf=torch.inf, neginf=torch.inf)
    return torch.where(valid[owners], errors, 0)


def _normalize_masked(points, mask):
    weights = mask.to(points)
    count = weights.sum(-1).clamp_min(1)
    center = (points * weights[..., None]).sum(-2) / count[:, None]
    centered = points - center[:, None]
    scale = math.sqrt(2) * count / (centered.norm(dim=-1) * weights).sum(-1).clamp_min(1e-12)
    transform = torch.eye(3, device=points.device, dtype=points.dtype).repeat(len(points), 1, 1)
    transform[:, 0, 0] = transform[:, 1, 1] = scale
    transform[:, :2, 2] = -center * scale[:, None]
    return centered * scale[:, None, None], transform


def local_refit(points0, points1, mask, kind):
    """Padded, masked linear refit; no ragged indexing or per-pair SVD calls."""
    a, transform0 = _normalize_masked(points0, mask)
    b, transform1 = _normalize_masked(points1, mask)
    if kind == 'H':
        xy = homogeneous(a)
        zero = torch.zeros_like(xy)
        row0 = torch.cat((-xy, zero, b[..., :1] * xy), -1)
        row1 = torch.cat((zero, -xy, b[..., 1:] * xy), -1)
        design = torch.stack((row0, row1), -2) * mask[..., None, None]
        design = design.flatten(-3, -2)
    else:
        design = _epipolar_design(a, b) * mask[..., None]
    # Householder QR preserves the right singular vectors, but reduces the
    # expensive SVD from a tall correspondence matrix to at most 9x9.
    reduced = torch.linalg.qr(design, mode='r').R
    _, singular, vh = torch.linalg.svd(reduced, full_matrices=reduced.shape[-2] < 9)
    matrix = vh[:, -1].reshape(-1, 3, 3)
    if kind == 'H':
        matrix = torch.linalg.solve(transform1, matrix @ transform0)
        good = (singular[:, 7] > 1e-10 * singular[:, 0]) & (torch.linalg.det(matrix).abs() > 1e-12)
    else:
        u, s, vh = torch.linalg.svd(matrix)
        s[:, -1] = 0
        matrix = transform1.transpose(-1, -2) @ ((u * s[:, None]) @ vh) @ transform0
        if kind == 'E':
            matrix = _project_essential(matrix)
        rank8 = singular[:, 7]
        good = rank8 > 1e-10 * singular[:, 0]
    return matrix, good & (mask.sum(-1) >= 8) & torch.isfinite(matrix).all(dim=(-2, -1))


def _samples(counts, valid, trial, batch_size, sample_size, seed, backend='auto'):
    if valid.is_cuda and backend != 'torch':
        try:
            from .warp_ransac import sample_indices
        except ImportError:
            if backend == 'warp':
                raise
        else:
            return sample_indices(counts, valid, trial, batch_size, sample_size, seed)
    # Stateless 32-bit hash: pair batching/padding cannot change a pair's samples.
    observations = torch.arange(valid.shape[1], device=valid.device, dtype=torch.int64)
    trials = torch.arange(trial, trial + batch_size, device=valid.device, dtype=torch.int64)
    counter = trials[None, :, None] * counts[:, None, None] + observations[None, None]
    value = (counter + (seed % 2**32) * 104729 + 374761393) & 0xffffffff
    value = ((value ^ (value >> 16)) * 2246822519) & 0xffffffff
    value = ((value ^ (value >> 13)) * 3266489917) & 0xffffffff
    value = (value ^ (value >> 16)).to(torch.float64)
    return value.masked_fill(~valid[:, None], -1).topk(sample_size, -1).indices


@torch.no_grad()
def ransac_batch(points0, points1, valid, pixel0, pixel1, inverse0, inverse1, kind, cfg,
                 min_inlier_ratio=None, *, fuse_state=True):
    count, capacity = valid.shape
    models = torch.eye(3, device=points0.device, dtype=points0.dtype).repeat(count, 1, 1)
    masks = torch.zeros_like(valid)
    best_score = points0.new_full((count,), torch.inf)
    required = torch.full((count,), cfg.max_trials, dtype=torch.long, device=points0.device)
    found = torch.zeros(count, dtype=torch.bool, device=points0.device)
    sizes = valid.sum(-1)
    estimator, sample_size, salt = {'E': (essential_five_point, 5, 0),
        'F': (fundamental_seven_point, 7, 1), 'H': (homography_dlt, 4, 2)}[kind]
    if capacity < max(sample_size, 8):
        return models, masks, found
    threshold = cfg.max_error ** 2
    if min_inlier_ratio is not None:
        probability = min_inlier_ratio.pow(sample_size)
        budget = torch.ceil(math.log1p(-cfg.confidence) / torch.log1p(-probability.clamp_max(1 - 1e-15)))
        required = budget.clamp(cfg.min_trials, cfg.max_trials).long()
    limits = required.clone()
    fused = None
    if fuse_state and points0.is_cuda and points0.dtype == torch.float64 and cfg.backend != 'torch':
        try:
            from .warp_ransac import State
        except ImportError:
            if cfg.backend == 'warp':
                raise
        else:
            fused = State(models, masks, best_score, found, valid, sizes, limits, required, cfg)

    def score(candidates, owners, pair_ids):
        if kind == 'E':
            candidates = inverse1[pair_ids[owners]].transpose(-1, -2) @ candidates @ inverse0[pair_ids[owners]]
        return model_errors(candidates, owners, pixel0[pair_ids], pixel1[pair_ids], valid[pair_ids],
                            homography=kind == 'H', backend=cfg.backend)

    trial = 0
    while trial < cfg.max_trials:
        active = torch.where((sizes >= max(sample_size, cfg.min_num_inliers)) & (trial < required))[0]
        if len(active) == 0:
            break
        batch = min(cfg.batch_size, cfg.max_trials - trial)
        samples = _samples(sizes[active], valid[active], trial, batch, sample_size,
                           cfg.seed + salt * 1009, backend=cfg.backend if fuse_state else 'torch')
        pair_index = active[:, None, None]
        options = dict(return_owners=True)
        if kind == 'E':
            options['backend'] = cfg.backend
        candidates, owners = estimator(points0[pair_index, samples].reshape(-1, sample_size, 2),
                                       points1[pair_index, samples].reshape(-1, sample_size, 2), **options)
        trial += batch
        if len(candidates) == 0:
            continue
        owners = owners // batch
        errors = score(candidates, owners, active)
        costs = errors.clamp_max(threshold).sum(-1)
        if fused is not None:
            improved = fused.select(candidates, owners, costs, active, errors)
        else:
            per_pair = costs.new_full((len(active),), torch.inf).scatter_reduce_(0, owners, costs, reduce='amin')
            indices = torch.arange(len(candidates), device=points0.device)
            selected = torch.full((len(active),), len(candidates), dtype=torch.long, device=points0.device)
            selected.scatter_reduce_(0, owners, torch.where(costs == per_pair[owners], indices, len(candidates)), reduce='amin')
            improved = (selected < len(candidates)) & (per_pair < best_score[active])
            selected = selected.clamp_max(len(candidates) - 1)
            models[active] = torch.where(improved[:, None, None], candidates[selected], models[active])
            masks[active] = torch.where(improved[:, None], (errors[selected] <= threshold) & valid[active], masks[active])
            best_score[active] = torch.minimum(per_pair, best_score[active])
            found[active] |= improved
        # Refine survivors together. Independent pairs never share an inlier set.
        for _ in range(5):
            survivors = torch.where(improved)[0]
            if len(survivors) == 0:
                break
            pair_ids = active[survivors]
            refined, good = local_refit(points0[pair_ids], points1[pair_ids], masks[pair_ids], kind)
            local_errors = score(refined, torch.arange(len(pair_ids), device=points0.device), pair_ids)
            local_score = local_errors.clamp_max(threshold).sum(-1)
            if fused is not None:
                fused.local(refined, good, local_score, pair_ids, survivors, improved, local_errors)
            else:
                accepted = good & (local_score < best_score[pair_ids])
                improved[survivors] = accepted
                models[pair_ids] = torch.where(accepted[:, None, None], refined, models[pair_ids])
                masks[pair_ids] = torch.where(accepted[:, None], (local_errors <= threshold) & valid[pair_ids], masks[pair_ids])
                best_score[pair_ids] = torch.where(accepted, local_score, best_score[pair_ids])
        if fused is not None and torch.get_default_dtype() == torch.float32:
            fused.update_budget(active, sample_size)
        else:
            probability = (masks[active].sum(-1) / sizes[active]).pow(sample_size)
            estimate = torch.ceil(math.log1p(-cfg.confidence) / torch.log1p(-probability.clamp_max(1 - 1e-15)))
            estimate = torch.where(probability >= 1, cfg.min_trials, estimate)
            required[active] = torch.minimum(estimate.clamp(cfg.min_trials, cfg.max_trials).long(), limits[active])
    return models, masks, found
