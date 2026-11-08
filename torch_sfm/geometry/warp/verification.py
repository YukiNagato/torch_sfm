"""Pair-batched geometric verification using the fixed-slot Warp RANSAC controllers."""
import math
import torch
from ...correspondence import Matches
from ..two_view import TwoViewConfig, TwoViewGeometry, camera_matrix, homogeneous, homography_poses, _choose_pose, _watermark
from ..two_view_poses import _store_pose
from .calibrated_ransac import CalibratedRansac
from .ransac import ransac_batch as shared_focal_ransac
from .pose import pre_refine_essential, finish_poses


def _ransac(p0,p1,valid,pixel0,pixel1,inverse0,inverse1,kind,cfg,min_inlier_ratio=None):
    if kind=='SF':
        return shared_focal_ransac(p0,p1,valid,inverse0,inverse1,cfg,min_inlier_ratio)
    result=CalibratedRansac(p0,p1,valid,pixel0,pixel1,inverse0,inverse1,kind,cfg,min_inlier_ratio).run()
    return result.models,result.masks,result.found

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
        essential, mask_e, found_e = _ransac(normalized0, normalized1, valid,
            points0, points1, inverse0, inverse1, 'E', cfg)
    f_kind = 'SF' if unknown and cfg.uncalibrated_model == 'shared_focal' else 'F'
    fundamental, mask_f, found_f = _ransac(points0, points1, valid,
        points0, points1, inverse0, inverse1, f_kind, cfg)
    homography, mask_h, found_h = _ransac(points0, points1, valid,
        points0, points1, inverse0, inverse1, 'H', cfg,
        min_inlier_ratio=cfg.max_H_inlier_ratio * (mask_f.sum(-1) if unknown else torch.minimum(mask_e.sum(-1), mask_f.sum(-1)))
                         / valid.sum(-1).clamp_min(1))
    if cfg.refine_relative_pose and not unknown:
        pre_refine_essential(essential,mask_e,found_e,points0,points1,valid,inverse0,inverse1,cfg)

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
    if calibrated or uncalibrated:
        finish_poses(results,calibrated+uncalibrated,essential,fundamental,points0,points1,
            rays0,rays1,selected_masks,inverse0,inverse1,matrix0,matrix1,indices,scores,counts,cfg)
    return results
