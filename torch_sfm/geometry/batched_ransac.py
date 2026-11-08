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


def _prepare_model_scorer(points0, points1, valid, kind, cfg):
    """Optional full-error plus MSAC reduction with per-call observation views."""
    if (points0.is_cuda and points0.dtype == torch.float64 and cfg.backend != 'torch'
            and not (kind == 'E' and cfg.essential_cheirality)):
        try:
            from .warp_two_view import ModelScorer
        except ImportError:
            if cfg.backend == 'warp':
                raise
        else:
            return ModelScorer(points0, points1, valid, cfg.max_error ** 2, kind == 'H')
    return None


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


def _prepare_local_refit(points0, points1, mask, kind, backend='auto'):
    if (backend != 'torch' and points0.is_cuda and points0.dtype == torch.float64
            and not (torch.is_grad_enabled() and (points0.requires_grad or points1.requires_grad))):
        try:
            from .warp_local_refit import prepare
        except ImportError:
            if backend == 'warp':
                raise
        else:
            return prepare(points0, points1, mask, kind)
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
    return design, transform0, transform1


def _local_refit_qr(design, backend):
    if (backend != 'torch' and design.is_cuda and design.dtype == torch.float64
            and not (torch.is_grad_enabled() and design.requires_grad)):
        try:
            from .warp_refit_qr import reduced_qr
        except ImportError:
            if backend == 'warp':
                raise
        else:
            return reduced_qr(design)
    return torch.linalg.qr(design, mode='r').R


def local_refit(points0, points1, mask, kind, backend='auto'):
    """Padded, masked linear refit; no ragged indexing or per-pair SVD calls."""
    design, transform0, transform1 = _prepare_local_refit(points0, points1, mask, kind, backend)
    # Householder QR preserves the right singular vectors, but reduces the
    # expensive SVD from a tall correspondence matrix to at most 9x9.
    reduced = _local_refit_qr(design, backend)
    _, singular, vh = torch.linalg.svd(reduced, full_matrices=reduced.shape[-2] < 9)
    if (backend != 'torch' and vh.is_cuda and vh.dtype == torch.float64
            and singular.shape[-1] == 9
            and not (torch.is_grad_enabled() and (vh.requires_grad or transform0.requires_grad or transform1.requires_grad))):
        try:
            from .warp_refit_finish import finish
        except ImportError:
            if backend == 'warp':
                raise
        else:
            return finish(vh, singular, transform0, transform1, mask, kind)
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


def _shared_focal_local_candidates(candidates, focals, owners, errors, costs,
                                    points0, points1, valid, principal0, principal1,
                                    previous_cost, cfg):
    """Speculate on all record improvements before choosing a batch winner.

    Sequential LO-RANSAC refines early record hypotheses too. Refining only the
    final raw winner can lose a different, better local minimum. Owners arrive
    sorted by pair, allowing one prefix minimum with separated cost ranges.
    """
    from .fundamental_refinement import refine_shared_focal
    threshold=cfg.max_error**2
    refinement_backend=cfg.shared_focal_refinement_backend
    if refinement_backend=='auto' and cfg.backend=='torch':
        refinement_backend='torch'
    update=None
    if cfg.backend!='torch' and candidates.is_cuda and candidates.dtype==torch.float64:
        try:
            from .warp_shared_focal_local import update
        except ImportError:
            if cfg.backend=='warp':raise
    adjusted=costs-owners*(points0.shape[1]*threshold+1)
    prefix=adjusted.cummin(0).values
    previous=torch.cat((prefix.new_full((1,),torch.inf),prefix[:-1]))
    selected=torch.where((adjusted<previous)&(costs<previous_cost[owners]))[0]
    chunk=cfg.shared_focal_local_batch_size
    for start in range(0,len(selected),chunk):
        ids=selected[start:start+chunk];pair=owners[ids]
        model=candidates[ids].clone();focal=focals[ids].clone()
        mask=(errors[ids]<=threshold)&valid[pair]
        score=costs[ids].clone();active=torch.ones(len(ids),device=ids.device,dtype=torch.bool)
        for _ in range(10):
            sub=torch.where(active&(mask.sum(-1)>=8))[0]
            if len(sub)==0:break
            p=pair[sub]
            fitted,new_focal=refine_shared_focal(model[sub],points0[p],points1[p],mask[sub],
                focal[sub],principal0[p],principal1[p],max_iterations=25,focal_ratio_bounds=(1e-6,1e6),
                use_cuda_graph=cfg.shared_focal_cuda_graph,backend=refinement_backend)
            if update is not None:
                update(fitted,new_focal,sub,pair,ids,points0,points1,valid,threshold,
                       model,focal,score,mask,active,errors)
                continue
            residual=model_errors(fitted,p,points0,points1,valid,backend=cfg.backend)
            new_score=residual.clamp_max(threshold).sum(-1)
            accept=torch.isfinite(new_score)&(new_score<score[sub])
            active[sub]=accept
            take=sub[accept]
            model[take]=fitted[accept];focal[take]=new_focal[accept]
            score[take]=new_score[accept];mask[take]=(residual[accept]<=threshold)&valid[pair[take]]
            errors[ids[take]]=residual[accept]
        candidates[ids]=model;focals[ids]=focal;costs[ids]=score


@torch.no_grad()
def ransac_batch(points0, points1, valid, pixel0, pixel1, inverse0, inverse1, kind, cfg,
                 min_inlier_ratio=None, *, fuse_state=True):
    if cfg.calibration=='calibrated' and cfg.calibrated_frontend!='legacy' and kind in ('E','F','H'):
        from ..warp_frontend.calibrated_ransac import CalibratedRansac, enabled
        if enabled(cfg,points0):
            result=CalibratedRansac(points0,points1,valid,pixel0,pixel1,inverse0,inverse1,kind,cfg,min_inlier_ratio).run()
            return result.models,result.masks,result.found
    if kind=='SF' and cfg.shared_focal_frontend!='legacy':
        use_warp=(points0.is_cuda and points0.dtype==torch.float64 and cfg.backend!='torch'
                  and cfg.shared_focal_root_backend=='warp_poselib'
                  and cfg.shared_focal_refinement_backend!='torch')
        if cfg.shared_focal_frontend=='warp' and not use_warp:
            raise ValueError('Warp shared-focal frontend requires CUDA float64 inputs')
        if use_warp:
            from ..warp_frontend.ransac import ransac_batch as solve
            return solve(points0,points1,valid,inverse0,inverse1,cfg,min_inlier_ratio)
    count, capacity = valid.shape
    models = torch.eye(3, device=points0.device, dtype=points0.dtype).repeat(count, 1, 1)
    masks = torch.zeros_like(valid)
    best_score = points0.new_full((count,), torch.inf)
    required = torch.full((count,), cfg.max_trials, dtype=torch.long, device=points0.device)
    found = torch.zeros(count, dtype=torch.bool, device=points0.device)
    sizes = valid.sum(-1)
    from .shared_focal import shared_focal_six_point
    estimator, sample_size, salt = {'E': (essential_five_point, 5, 0),
        'F': (fundamental_seven_point, 7, 1), 'H': (homography_dlt, 4, 2),
        'SF': (shared_focal_six_point, 6, 3)}[kind]
    if kind == 'SF':
        principal0 = -inverse0[:,:2,2] / inverse0.diagonal(dim1=-2,dim2=-1)[:,:2]
        principal1 = -inverse1[:,:2,2] / inverse1.diagonal(dim1=-2,dim2=-1)[:,:2]
        focals = points0.new_ones(count)
    if capacity < max(sample_size, 8):
        return models, masks, found
    threshold = cfg.max_error ** 2
    if min_inlier_ratio is not None:
        probability = min_inlier_ratio.pow(sample_size)
        budget = torch.ceil(math.log1p(-cfg.confidence) / torch.log1p(-probability.clamp_max(1 - 1e-15)))
        required = budget.clamp(cfg.min_trials, cfg.max_trials).long()
    limits = required.clone()
    fused = None
    if fuse_state and kind != 'SF' and points0.is_cuda and points0.dtype == torch.float64 and cfg.backend != 'torch':
        try:
            from .warp_ransac import State
        except ImportError:
            if cfg.backend == 'warp':
                raise
        else:
            fused = State(models, masks, best_score, found, valid, sizes, limits, required, cfg)

    if kind == 'E' and cfg.essential_cheirality:
        from .essential_scoring import prepare_essential_observations, essential_errors_prepared
        observations = prepare_essential_observations(pixel0, pixel1, inverse0, inverse1)

    model_scorer = _prepare_model_scorer(pixel0, pixel1, valid, kind, cfg)
    sampler = None
    if fuse_state and points0.is_cuda and points0.dtype == torch.float64 and cfg.backend != 'torch':
        try:
            from .warp_ransac_sampling import Sampler
        except ImportError:
            if cfg.backend == 'warp':
                raise
        else:
            sampler = Sampler(points0, points1, valid, sizes, sample_size, cfg.seed + salt * 1009)

    def score(candidates, owners, pair_ids):
        if kind == 'E':
            if cfg.essential_cheirality:
                errors = essential_errors_prepared(candidates, pair_ids[owners], *observations,
                    valid, cfg.backend)
                return errors, errors.clamp_max(threshold).sum(-1)
            candidates = inverse1[pair_ids[owners]].transpose(-1, -2) @ candidates @ inverse0[pair_ids[owners]]
        if model_scorer is not None:
            return model_scorer(candidates, pair_ids[owners])
        errors = model_errors(candidates, owners, pixel0[pair_ids], pixel1[pair_ids], valid[pair_ids],
                              homography=kind == 'H', backend=cfg.backend)
        return errors, errors.clamp_max(threshold).sum(-1)

    trial = 0
    while trial < cfg.max_trials:
        active = torch.where((sizes >= max(sample_size, cfg.min_num_inliers)) & (trial < required))[0]
        if len(active) == 0:
            break
        batch = min(cfg.batch_size, cfg.max_trials - trial)
        if sampler is not None:
            sampled0, sampled1 = sampler.sample(active, trial, batch)
        else:
            samples = _samples(sizes[active], valid[active], trial, batch, sample_size,
                               cfg.seed + salt * 1009, backend=cfg.backend if fuse_state else 'torch')
            pair_index = active[:, None, None]
            sampled0 = points0[pair_index, samples].reshape(-1, sample_size, 2)
            sampled1 = points1[pair_index, samples].reshape(-1, sample_size, 2)
        options = dict(return_owners=True)
        if kind in ('E', 'F', 'H'):
            options['backend'] = cfg.backend
        if kind == 'E' and cfg.five_point_backend != 'auto':
            options['solver_backend'] = cfg.five_point_backend
        if kind == 'SF':
            candidates, candidate_focals, owners = estimator(sampled0, sampled1,
                principal0[active].repeat_interleave(batch,dim=0),
                principal1[active].repeat_interleave(batch,dim=0),backend=cfg.backend,
                cpu_workers=cfg.shared_focal_cpu_workers,root_backend=cfg.shared_focal_root_backend)
        else:
            candidates, owners = estimator(sampled0, sampled1, **options)
        trial += batch
        if len(candidates) == 0:
            continue
        owners = owners // batch
        errors, costs = score(candidates, owners, active)
        if kind == 'SF':
            _shared_focal_local_candidates(candidates,candidate_focals,owners,errors,costs,
                points0[active],points1[active],valid[active],principal0[active],principal1[active],
                best_score[active],cfg)
        if fused is not None:
            # Packed F/H and PoseLib five-point roots preserve sample order.
            # The legacy five-point backend can append unsorted fallback roots.
            improved = fused.select(candidates, owners, costs, active, errors,
                sorted_owners=kind != 'E' or cfg.five_point_backend == 'warp_poselib')
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
            if kind == 'SF':
                focals[active] = torch.where(improved, candidate_focals[selected], focals[active])
        # Refine survivors together. Independent pairs never share an inlier set.
        for _ in range(0 if kind == 'SF' else 5):
            survivors = torch.where(improved)[0]
            if len(survivors) == 0:
                break
            pair_ids = active[survivors]
            refined, good = local_refit(points0[pair_ids], points1[pair_ids], masks[pair_ids], kind, backend=cfg.backend)
            local_errors, local_score = score(refined, torch.arange(len(pair_ids), device=points0.device), pair_ids)
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
            multiplier=1.
            if kind == 'SF':
                # COLMAP's finite-population sampling probability and its
                # conservative dynamic trial multiplier.
                inliers=masks[active].sum(-1).double();total=sizes[active].double()
                probability=torch.ones_like(inliers)
                for sample in range(sample_size):
                    probability *= ((inliers-sample)/(total-sample)).clamp(0,1)
                multiplier=3.
            estimate = torch.ceil(multiplier*math.log1p(-cfg.confidence) / torch.log1p(-probability.clamp_max(1 - 1e-15)))
            estimate = torch.where(probability >= 1, cfg.min_trials, estimate)
            required[active] = torch.minimum(estimate.clamp(cfg.min_trials, cfg.max_trials).long(), limits[active])
    return models, masks, found
