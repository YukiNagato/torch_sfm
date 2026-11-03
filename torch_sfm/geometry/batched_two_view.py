"""Pair-batched E/F/H verification, retaining the scalar verifier as reference."""

import math

import torch
from torchls.geometry import skew

from ..correspondence import Matches
from .two_view import (TwoViewConfig, TwoViewGeometry, camera_matrix, homogeneous,
    homography_poses, _choose_pose, _watermark, triangulate_relative, sampson_error)
from .batched_ransac import ransac_batch, model_errors
from .relative_pose import refine_relative_pose


def _essential_pose_batch(essential, rays0, rays1, mask):
    u, _, vh = torch.linalg.svd(essential)
    u, vh = u.clone(), vh.clone()
    u[:, :, -1] *= torch.linalg.det(u)[:, None]
    vh[:, -1] *= torch.linalg.det(vh)[:, None]
    w = essential.new_tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    rotations = torch.stack((u @ w @ vh, u @ w @ vh, u @ w.T @ vh, u @ w.T @ vh), 1)
    translations = u[:, None, :, 2] * essential.new_tensor([1, -1, 1, -1])[None, :, None]
    d0 = torch.nn.functional.normalize(rays0, dim=-1)[:, None]
    d1 = torch.nn.functional.normalize(rays1, dim=-1)[:, None] @ rotations
    center = -(rotations.transpose(-1, -2) @ translations[..., None]).squeeze(-1)
    cosine = (d0 * d1).sum(-1)
    denominator = (1 - cosine.square()).clamp_min(1e-15)
    a, b = (d0 * center[:, :, None]).sum(-1), (d1 * center[:, :, None]).sum(-1)
    depth0, depth1 = (a - cosine * b) / denominator, (cosine * a - b) / denominator
    points = (depth0[..., None] * d0 + center[:, :, None] + depth1[..., None] * d1) / 2
    positive = (depth0 > 1e-8) & (depth1 > 1e-8) & mask[:, None] & torch.isfinite(points).all(-1)
    errors = (torch.nn.functional.normalize(points, dim=-1) - d0).square().sum(-1)
    counts = positive.sum(-1)
    costs = (errors * positive).sum(-1).masked_fill(counts != counts.amax(-1, keepdim=True), torch.inf)
    best = costs.argmin(-1)
    ids = torch.arange(len(essential), device=essential.device)
    return rotations[ids, best], translations[ids, best], positive[ids, best]


@torch.inference_mode(False)
@torch.no_grad()
def verify_two_view_batch(keypoints0, keypoints1, matches, intrinsics0, intrinsics1,
                          cfg=None, image_sizes=None):
    """Verify a bounded group of independent pairs; returned indices are original.

    Keypoints/matches are sequences of tensors/Matches; intrinsics are (B,4) or
    (B,3,3). Padding and nonfinite correspondences never enter sampling or costs.
    """
    cfg = TwoViewConfig() if cfg is None else cfg
    batch = len(matches)
    if batch == 0:
        return []
    if len(keypoints0) != batch or len(keypoints1) != batch or (image_sizes is not None and len(image_sizes) != batch):
        raise ValueError('Provide keypoints and optional image sizes for every image pair')
    if batch > cfg.pair_batch_size:
        raise ValueError('Split verification into chunks of pair_batch_size')
    device = keypoints0[0].device
    matrix0 = camera_matrix(intrinsics0.detach().to(device=device, dtype=torch.float64).clone())
    matrix1 = camera_matrix(intrinsics1.detach().to(device=device, dtype=torch.float64).clone())
    if matrix0.ndim == 2:
        matrix0 = matrix0.expand(batch, 3, 3)
    if matrix1.ndim == 2:
        matrix1 = matrix1.expand(batch, 3, 3)
    if matrix0.shape != (batch, 3, 3) or matrix1.shape != (batch, 3, 3):
        raise ValueError('Provide shared intrinsics or one matrix per image pair')
    capacity = max(max(len(m.indices) for m in matches), 8)
    points0 = matrix0.new_zeros((batch, capacity, 2))
    points1 = torch.zeros_like(points0)
    valid = torch.zeros((batch, capacity), dtype=torch.bool, device=device)
    indices, scores = [], []
    for i, match in enumerate(matches):
        index = match.indices.to(device=device).clone()
        indices.append(index)
        scores.append(match.scores.to(device=device).clone())
        count = len(index)
        points0[i, :count] = keypoints0[i].to(device=device, dtype=torch.float64)[index[:, 0]]
        points1[i, :count] = keypoints1[i].to(device=device, dtype=torch.float64)[index[:, 1]]
        valid[i, :count] = True
    valid &= torch.isfinite(points0).all(-1) & torch.isfinite(points1).all(-1)
    points0 = torch.where(valid[..., None], points0, 0)
    points1 = torch.where(valid[..., None], points1, 0)
    inverse0, inverse1 = torch.linalg.inv(matrix0), torch.linalg.inv(matrix1)
    rays0, rays1 = homogeneous(points0) @ inverse0.transpose(-1, -2), homogeneous(points1) @ inverse1.transpose(-1, -2)
    normalized0, normalized1 = rays0[..., :2] / rays0[..., 2:], rays1[..., :2] / rays1[..., 2:]
    essential, mask_e, found_e = ransac_batch(normalized0, normalized1, valid,
        points0, points1, inverse0, inverse1, 'E', cfg)
    fundamental, mask_f, found_f = ransac_batch(points0, points1, valid,
        points0, points1, inverse0, inverse1, 'F', cfg)
    homography, mask_h, found_h = ransac_batch(points0, points1, valid,
        points0, points1, inverse0, inverse1, 'H', cfg,
        min_inlier_ratio=cfg.max_H_inlier_ratio * torch.minimum(mask_e.sum(-1), mask_f.sum(-1))
                         / valid.sum(-1).clamp_min(1))
    ids = torch.arange(batch, device=device)
    if cfg.refine_relative_pose:
        rotation, translation, positive = _essential_pose_batch(essential, rays0, rays1, mask_e)
        refine_mask = positive & (positive.sum(-1) >= cfg.min_num_inliers)[:, None] & found_e[:, None]
        if bool(refine_mask.any()):
            rotation, translation = refine_relative_pose(rotation, translation, points0, points1, matrix0, matrix1,
                refine_mask, use_cuda_graph=cfg.pose_cuda_graph, backend=cfg.backend)
        candidate = skew(translation) @ rotation
        old_error = model_errors(inverse1.transpose(-1, -2) @ essential @ inverse0,
            ids, points0, points1, valid, backend=cfg.backend)
        new_error = model_errors(inverse1.transpose(-1, -2) @ candidate @ inverse0,
            ids, points0, points1, valid, backend=cfg.backend)
        improved = (refine_mask.sum(-1) >= cfg.min_num_inliers) & (
            new_error.clamp_max(cfg.max_error ** 2).sum(-1) < old_error.clamp_max(cfg.max_error ** 2).sum(-1))
        essential = torch.where(improved[:, None, None], candidate, essential)
        mask_e = torch.where(improved[:, None], (new_error <= cfg.max_error ** 2) & valid, mask_e)

    counts = torch.stack((mask_e.sum(-1), mask_f.sum(-1), mask_h.sum(-1), valid.sum(-1),
                          found_e.long(), found_f.long(), found_h.long()), -1).cpu().tolist()
    results, selected_masks = [], torch.zeros_like(valid)
    calibrated = []
    for i, (ne, nf, nh, count, has_e, has_f, has_h) in enumerate(counts):
        result = TwoViewGeometry(Matches(indices[i][:0], scores[i][:0]),
            E=essential[i] if has_e else None, F=fundamental[i] if has_f else None,
            H=homography[i] if has_h else None)
        results.append(result)
        minimum = max(cfg.min_num_inliers, math.ceil(cfg.min_inlier_ratio * count))
        if max(ne, nf, nh) < minimum:
            continue
        if has_e and ne >= minimum and ne >= cfg.min_E_F_inlier_ratio * nf:
            mask, result.config = mask_e[i] if ne >= nf else mask_f[i], 'CALIBRATED'
            if has_h and nh > cfg.max_H_inlier_ratio * ne:
                result.config = 'PLANAR'
                if nh > max(ne, nf):
                    mask = mask_h[i]
        elif has_f and nf >= minimum:
            mask, result.config = mask_f[i], 'UNCALIBRATED'
            if has_h and nh > cfg.max_H_inlier_ratio * nf:
                result.config, mask = 'PLANAR', mask_h[i]
        elif has_h and nh >= minimum:
            mask, result.config = mask_h[i], 'PLANAR'
        else:
            continue
        mask = mask[:len(indices[i])]
        selected_masks[i, :len(mask)] = mask
        result.matches = Matches(indices[i][mask], scores[i][mask])
        sizes = None if image_sizes is None else image_sizes[i]
        if cfg.detect_watermark and _watermark(points0[i, :len(mask)][mask], points1[i, :len(mask)][mask], sizes, cfg):
            result.config = 'WATERMARK'
        elif result.config == 'CALIBRATED':
            calibrated.append(i)
        elif result.config == 'PLANAR':
            rotation, translation, positive, angle = _choose_pose(
                homography_poses(homography[i], matrix0[i], matrix1[i]), rays0[i, selected_masks[i]], rays1[i, selected_masks[i]])
            _store_pose(result, rotation, translation, positive, angle, minimum, cfg)
    if calibrated:
        group = torch.tensor(calibrated, device=device)
        masks = selected_masks[group]
        rotation, translation, positive = _essential_pose_batch(essential[group], rays0[group], rays1[group], masks)
        if cfg.refine_relative_pose:
            refine_mask = positive & (positive.sum(-1) >= torch.tensor([
                max(cfg.min_num_inliers, math.ceil(cfg.min_inlier_ratio * counts[i][3])) for i in calibrated],
                device=device))[:, None]
            if bool(refine_mask.any()):
                rotation, translation = refine_relative_pose(rotation, translation, points0[group], points1[group],
                    matrix0[group], matrix1[group], refine_mask, use_cuda_graph=cfg.pose_cuda_graph, backend=cfg.backend)
        for local, i in enumerate(calibrated):
            result = results[i]
            result.E = skew(translation[local]) @ rotation[local]
            rays_a, rays_b = rays0[i, selected_masks[i]], rays1[i, selected_masks[i]]
            _, positive, angles = triangulate_relative(rotation[local], translation[local], rays_a, rays_b)
            error = sampson_error((inverse1[i].T @ result.E @ inverse0[i])[None],
                points0[i, selected_masks[i]], points1[i, selected_masks[i]])[0]
            positive &= error <= cfg.max_error ** 2
            angle = float(angles[positive].median()) if bool(positive.any()) else 0.0
            minimum = max(cfg.min_num_inliers, math.ceil(cfg.min_inlier_ratio * counts[i][3]))
            _store_pose(result, rotation[local], translation[local], positive, angle, minimum, cfg)
    return results


def _store_pose(result, rotation, translation, positive, angle, minimum, cfg):
    if float(translation.norm()) < 1e-8:
        result.config = 'PANORAMIC'
    elif int(positive.sum()) < minimum:
        result.config = 'DEGENERATE'
        result.matches = Matches(result.matches.indices[:0], result.matches.scores[:0])
        return
    result.R, result.t, result.tri_angle = rotation, translation, angle
    result.valid_translation = bool(translation.norm() > 1e-8 and angle >= cfg.min_tri_angle)
    result.matches = Matches(result.matches.indices[positive], result.matches.scores[positive])
