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


def _watermark_candidates(points0, points1, masks, image_sizes, cfg):
    """One batch readback for the cheap border test; retain exact scalar checks.

    A conservative margin admits boundary cases to the existing test, so this
    prefilter cannot change the final watermark decision through float rounding.
    """
    if not cfg.detect_watermark or image_sizes is None:
        return [False]*len(points0)
    dimensions=[s if s is not None else ((0,0),(0,0)) for s in image_sizes]
    sizes=points0.new_tensor(dimensions)
    margins=points0.new_tensor([[math.hypot(*hw)*cfg.watermark_border_size for hw in s]
                               for s in dimensions])
    present=torch.tensor([s is not None for s in image_sizes],device=points0.device)
    border=masks.clone()
    for side,points in enumerate((points0,points1)):
        hw=sizes[:,side];wh=hw.flip(-1)
        margin=margins[:,side]
        border &= torch.minimum(points,wh[:,None]-points).amin(-1)<margin[:,None]
    count=masks.sum(-1)
    possible=present & (count>0) & (border.sum(-1).double()/count.clamp_min(1)
                                   >=cfg.watermark_min_inlier_ratio-1e-7)
    return possible.cpu().tolist()


def _store_pose_batch(results, members, rotation, translation, rays0, rays1,
                      selected_masks, indices, scores, minimums, cfg,
                      *, fundamental=None, points0=None, points1=None):
    """Filter refined poses in a batch; read metadata and compact indices once."""
    group=torch.tensor(members,device=rotation.device)
    a=torch.nn.functional.normalize(rays0[group],dim=-1)
    b=torch.nn.functional.normalize(rays1[group],dim=-1)@rotation
    center=-(rotation.transpose(-1,-2)@translation[...,None]).squeeze(-1)
    cosine=(a*b).sum(-1)
    denominator=(1-cosine.square()).clamp_min(1e-15)
    ac=(a*center[:,None]).sum(-1);bc=(b*center[:,None]).sum(-1)
    d0=(ac-cosine*bc)/denominator;d1=(cosine*ac-bc)/denominator
    points=(d0[...,None]*a+center[:,None]+d1[...,None]*b)/2
    positive=selected_masks[group] & (d0>1e-8) & (d1>1e-8) & torch.isfinite(points).all(-1)
    if fundamental is not None:
        x,y=homogeneous(points0[group]),homogeneous(points1[group])
        line1=x@fundamental.transpose(-1,-2);line0=y@fundamental
        error=(y*line1).sum(-1).square()/(line0[...,:2].square().sum(-1)
                                               +line1[...,:2].square().sum(-1)).clamp_min(1e-30)
        positive &= error<=cfg.max_error**2
    counts=positive.sum(-1)
    angles=(torch.acos(cosine.clamp(-1,1))*(180/math.pi)).masked_fill(~positive,torch.inf)
    ordered=angles.sort(-1).values
    median=ordered.gather(1,((counts-1).clamp_min(0)//2)[:,None]).squeeze(-1)
    median=torch.where(counts>0,median,0.)
    norms=translation.norm(dim=-1)
    metadata=torch.stack((counts.to(norms),median,norms),-1).cpu().tolist()
    _,columns=torch.where(positive)
    selected=columns.split([int(row[0]) for row in metadata])
    for local,i in enumerate(members):
        count,angle,norm=metadata[local];result=results[i]
        if norm<1e-8:
            result.config='PANORAMIC'
        elif count<minimums[local]:
            result.config='DEGENERATE'
            result.matches=Matches(indices[i][:0],scores[i][:0])
            continue
        result.R,result.t,result.tri_angle=rotation[local],translation[local],angle
        result.valid_translation=norm>1e-8 and angle>=cfg.min_tri_angle
        result.matches=Matches(indices[i][selected[local]],scores[i][selected[local]])


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
    if batch > cfg.verification_pair_batch_size:
        raise ValueError('Split verification into chunks of verification_pair_batch_size')
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
    unknown = cfg.calibration == 'uncalibrated'
    if unknown:
        essential = torch.eye(3, device=device, dtype=points0.dtype).repeat(batch, 1, 1)
        mask_e, found_e = torch.zeros_like(valid), torch.zeros(batch, device=device, dtype=torch.bool)
    else:
        essential, mask_e, found_e = ransac_batch(normalized0, normalized1, valid,
            points0, points1, inverse0, inverse1, 'E', cfg)
    f_kind = 'SF' if unknown and cfg.uncalibrated_model == 'shared_focal' else 'F'
    fundamental, mask_f, found_f = ransac_batch(points0, points1, valid,
        points0, points1, inverse0, inverse1, f_kind, cfg)
    homography, mask_h, found_h = ransac_batch(points0, points1, valid,
        points0, points1, inverse0, inverse1, 'H', cfg,
        min_inlier_ratio=cfg.max_H_inlier_ratio * (mask_f.sum(-1) if unknown else torch.minimum(mask_e.sum(-1), mask_f.sum(-1)))
                         / valid.sum(-1).clamp_min(1))
    ids = torch.arange(batch, device=device)
    if cfg.refine_relative_pose and not unknown:
        rotation, translation, positive = _essential_pose_batch(essential, rays0, rays1, mask_e)
        refine_mask = positive & (positive.sum(-1) >= cfg.min_num_inliers)[:, None] & found_e[:, None]
        if bool(refine_mask.any()):
            rotation, translation = refine_relative_pose(rotation, translation, points0, points1, matrix0, matrix1,
                refine_mask, use_cuda_graph=cfg.pose_cuda_graph, backend=cfg.backend)
        candidate = skew(translation) @ rotation
        if cfg.essential_cheirality:
            from .essential_scoring import essential_errors
            old_error = essential_errors(essential, ids, points0, points1, inverse0, inverse1, valid, cfg.backend)
            new_error = essential_errors(candidate, ids, points0, points1, inverse0, inverse1, valid, cfg.backend)
        else:
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
    results, choices, selected_counts = [], [], []
    calibrated, uncalibrated = [], []
    for i, (ne, nf, nh, count, has_e, has_f, has_h) in enumerate(counts):
        result = TwoViewGeometry(Matches(indices[i][:0], scores[i][:0]),
            E=essential[i] if has_e else None, F=fundamental[i] if has_f else None,
            H=homography[i] if has_h else None)
        results.append(result)
        choice, chosen_count = 0, 0
        minimum = max(cfg.min_num_inliers, math.ceil(cfg.min_inlier_ratio * count))
        if max(ne, nf, nh) < minimum:
            choices.append(choice);selected_counts.append(chosen_count)
            continue
        if has_e and ne >= minimum and ne >= cfg.min_E_F_inlier_ratio * nf:
            choice, chosen_count = (0,ne) if ne>=nf else (1,nf)
            result.config = 'CALIBRATED'
            if has_h and nh > cfg.max_H_inlier_ratio * ne:
                result.config = 'PLANAR'
                if nh > max(ne, nf):
                    choice, chosen_count = 2,nh
        elif has_f and nf >= minimum:
            choice, chosen_count, result.config = 1,nf,'UNCALIBRATED'
            if has_h and nh > cfg.max_H_inlier_ratio * nf:
                result.config, choice, chosen_count = 'PLANAR',2,nh
        elif has_h and nh >= minimum:
            choice, chosen_count, result.config = 2,nh,'PLANAR'
        choices.append(choice);selected_counts.append(chosen_count)
    selection=torch.tensor(choices,device=device)
    selected_masks=torch.stack((mask_e,mask_f,mask_h),1)[torch.arange(batch,device=device),selection]
    selected_masks &= torch.tensor(selected_counts,device=device)[:,None]>0
    # Reuse one ordered compaction for both match tensors and watermark input.
    _, columns=torch.where(selected_masks)
    selected_ids=columns.split(selected_counts)
    watermark_candidates=_watermark_candidates(points0,points1,selected_masks,image_sizes,cfg)
    for i,result in enumerate(results):
        if not selected_counts[i]:continue
        selected=selected_ids[i]
        result.matches = Matches(indices[i][selected], scores[i][selected])
        sizes = None if image_sizes is None else image_sizes[i]
        if watermark_candidates[i] and _watermark(points0[i,selected],points1[i,selected],sizes,cfg):
            result.config = 'WATERMARK'
        elif result.config == 'CALIBRATED':
            calibrated.append(i)
        elif result.config == 'UNCALIBRATED' and not unknown:
            uncalibrated.append(i)
        elif result.config == 'PLANAR' and not unknown:
            minimum=max(cfg.min_num_inliers,math.ceil(cfg.min_inlier_ratio*counts[i][3]))
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
        fitted=skew(translation)@rotation
        for local,i in enumerate(calibrated):results[i].E=fitted[local]
        _store_pose_batch(results,calibrated,rotation,translation,rays0,rays1,selected_masks,
            indices,scores,[max(cfg.min_num_inliers,math.ceil(cfg.min_inlier_ratio*counts[i][3]))
                            for i in calibrated],cfg,
            fundamental=inverse1[group].transpose(-1,-2)@fitted@inverse0[group],points0=points0,points1=points1)
    if uncalibrated:
        group = torch.tensor(uncalibrated, device=device)
        # A failed E/F support test does not invalidate F. As in COLMAP,
        # decompose K1.T F K0 using the current calibration. Keep this model's
        # classification and its F inliers; E-RANSAC may have selected another
        # (e.g. repeated-texture) consensus.
        from_f = matrix1[group].transpose(-1, -2) @ fundamental[group] @ matrix0[group]
        rotation, translation, _ = _essential_pose_batch(
            from_f, rays0[group], rays1[group], selected_masks[group])
        _store_pose_batch(results,uncalibrated,rotation,translation,rays0,rays1,selected_masks,
                         indices,scores,[cfg.min_num_inliers]*len(uncalibrated),cfg)
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


@torch.no_grad()
def complete_fundamental_poses(geometries, frames, intrinsics, cfg):
    """Recover missing F poses, including caches produced before F decomposition.

    F inliers have already passed the configured support test. No new RANSAC
    is run and no pose is recovered during the uncalibrated first pass.
    """
    if cfg.calibration == 'uncalibrated':
        return
    pending = [(pair, g) for pair, g in geometries.items()
               if g.config == 'UNCALIBRATED' and g.R is None and g.F is not None]
    if not pending:
        return
    matrices = camera_matrix(intrinsics)
    if matrices.ndim == 2:
        matrices = matrices.expand(len(frames), -1, -1)
    for start in range(0, len(pending), cfg.pair_batch_size):
        chunk = pending[start:start+cfg.pair_batch_size]
        device = chunk[0][1].F.device
        k = matrices.to(device=device, dtype=torch.float64)
        size = max(len(g.matches.indices) for _, g in chunk)
        a = k.new_zeros((len(chunk), size, 3)); b = torch.zeros_like(a)
        mask = torch.zeros(a.shape[:2], device=device, dtype=torch.bool)
        essentials = []
        for row, ((i, j), g) in enumerate(chunk):
            indices = g.matches.indices
            p = frames[i].keypoints_2d[indices[:, 0].to(frames[i].keypoints_2d.device)].to(k)
            q = frames[j].keypoints_2d[indices[:, 1].to(frames[j].keypoints_2d.device)].to(k)
            n = len(p)
            a[row, :n] = homogeneous(p) @ torch.linalg.inv(k[i]).T
            b[row, :n] = homogeneous(q) @ torch.linalg.inv(k[j]).T
            mask[row, :n] = True
            essentials.append(k[j].T @ g.F.to(k) @ k[i])
        rotations, translations, _ = _essential_pose_batch(torch.stack(essentials), a, b, mask)
        for row, (_, g) in enumerate(chunk):
            _, positive, angles = triangulate_relative(rotations[row], translations[row],
                a[row, mask[row]], b[row, mask[row]])
            angle = float(angles[positive].median()) if bool(positive.any()) else 0.0
            _store_pose(g, rotations[row], translations[row], positive, angle, cfg.min_num_inliers, cfg)


@torch.no_grad()
def complete_homography_poses(geometries, frames, intrinsics, cfg):
    """Decompose retained first-pass planar models with the calibrated K.

    As in COLMAP's view-graph calibration, planar/panoramic models do not go
    through the stricter epipolar re-estimation. Preserve their verified
    matches; positive depth chooses the decomposition, not a new consensus.
    """
    matrices = camera_matrix(intrinsics)
    if matrices.ndim == 2:
        matrices = matrices.expand(len(frames), -1, -1)
    for (i, j), g in geometries.items():
        if g.config not in ('PLANAR', 'PANORAMIC') or g.R is not None or g.H is None:
            continue
        k = matrices.to(device=g.H.device, dtype=g.H.dtype)
        indices = g.matches.indices
        a = frames[i].keypoints_2d[indices[:, 0].to(frames[i].keypoints_2d.device)].to(k)
        b = frames[j].keypoints_2d[indices[:, 1].to(frames[j].keypoints_2d.device)].to(k)
        a = homogeneous(a) @ torch.linalg.inv(k[i]).T
        b = homogeneous(b) @ torch.linalg.inv(k[j]).T
        rotation, translation, positive, angle = _choose_pose(homography_poses(g.H, k[i], k[j]), a, b)
        matches = g.matches
        _store_pose(g, rotation, translation, positive, angle, 1, cfg)
        if g.R is not None:
            g.matches = matches
