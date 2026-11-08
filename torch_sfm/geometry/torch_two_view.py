"""Pure PyTorch single-pair reference for calibrated and uncalibrated verification."""
import math
import torch
from ..correspondence import Matches
from .two_view import (TwoViewConfig, TwoViewGeometry, camera_matrix, homogeneous,
    essential_five_point, fundamental_seven_point, epipolar_eight_point, homography_dlt,
    sampson_error, homography_error, essential_poses, homography_poses,
    _choose_pose, _watermark, _refine_pose, triangulate_relative)

@torch.no_grad()
def _ransac(points0, points1, estimator, local_estimator, residual, sample_size, cfg, generator):
    count = len(points0)
    empty = torch.zeros(count, dtype=torch.bool, device=points0.device)
    if count < sample_size:
        return None, empty
    best_model, best_mask, best_score = None, empty, math.inf
    trial, required = 0, cfg.max_trials
    threshold = cfg.max_error ** 2
    while trial < max(cfg.min_trials, required):
        batch = min(cfg.batch_size, cfg.max_trials - trial)
        if batch <= 0:
            break
        samples = torch.rand((batch, count), device=points0.device, generator=generator).topk(sample_size, -1).indices
        models = estimator(points0[samples], points1[samples])
        trial += batch
        if len(models) == 0:
            continue
        errors = residual(models).nan_to_num(nan=torch.inf, posinf=torch.inf, neginf=torch.inf)
        scores = errors.clamp_max(threshold).sum(-1)
        selected = int(scores.argmin())
        if float(scores[selected]) >= best_score:
            continue
        best_model, best_score = models[selected], float(scores[selected])
        best_mask = errors[selected] <= threshold
        for _ in range(5):
            if int(best_mask.sum()) < 8:
                break
            refined = local_estimator(points0[best_mask][None], points1[best_mask][None])
            if len(refined) == 0:
                break
            refined_errors = residual(refined)
            refined_scores = refined_errors.clamp_max(threshold).sum(-1)
            index = int(refined_scores.argmin())
            if not float(refined_scores[index]) < best_score:
                break
            best_model, best_score = refined[index], float(refined_scores[index])
            best_mask = refined_errors[index] <= threshold
        probability = (int(best_mask.sum()) / count) ** sample_size
        if probability >= 1:
            required = cfg.min_trials
        elif probability > 0:
            required = min(cfg.max_trials, math.ceil(math.log1p(-cfg.confidence) / math.log1p(-probability)))
    return best_model, best_mask


@torch.inference_mode(False)
def verify_two_view(keypoints0, keypoints1, matches, intrinsics0, intrinsics1,
                    cfg=None, image_sizes=None):
    """Verify a calibrated pinhole pair, retaining model classification on failure.

    Numeric inputs must share a float64/float32 device. Returned match indices
    index the original keypoint arrays. F-supported pairs remain UNCALIBRATED;
    in calibrated mode their pose is recovered from F and the supplied K.
    """
    cfg = TwoViewConfig() if cfg is None else cfg
    unknown = cfg.calibration == 'uncalibrated'
    device = keypoints0.device
    dtype = torch.float64
    keypoints0 = keypoints0.detach().to(dtype=dtype).clone()
    keypoints1 = keypoints1.detach().to(device=device, dtype=dtype).clone()
    matrix0 = camera_matrix(intrinsics0.to(device=device, dtype=dtype).clone())
    matrix1 = camera_matrix(intrinsics1.to(device=device, dtype=dtype).clone())
    indices = matches.indices.to(device=device).clone()
    scores = matches.scores.to(device=device).clone()
    points0, points1 = keypoints0[indices[:, 0]], keypoints1[indices[:, 1]]
    finite = torch.isfinite(points0).all(-1) & torch.isfinite(points1).all(-1)
    indices, scores, points0, points1 = indices[finite], scores[finite], points0[finite], points1[finite]
    empty_matches = Matches(indices[:0], scores[:0])
    result = TwoViewGeometry(empty_matches)
    if len(points0) < cfg.min_num_inliers:
        return result
    rays0 = homogeneous(points0) @ torch.linalg.inv(matrix0).T
    rays1 = homogeneous(points1) @ torch.linalg.inv(matrix1).T
    normalized0, normalized1 = rays0[:, :2] / rays0[:, 2:], rays1[:, :2] / rays1[:, 2:]
    inverse0, inverse1 = torch.linalg.inv(matrix0), torch.linalg.inv(matrix1)
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    def essential_error(models):
        if cfg.essential_cheirality:
            from .essential_scoring import essential_errors
            return essential_errors(models, torch.zeros(len(models), device=device, dtype=torch.long),
                points0[None], points1[None], inverse0[None], inverse1[None],
                torch.ones((1,len(points0)),device=device,dtype=torch.bool),'torch')
        return sampson_error(inverse1.T @ models @ inverse0, points0, points1)
    if unknown:
        essential, mask_e = None, torch.zeros(len(points0),device=device,dtype=torch.bool)
    else:
        essential, mask_e = _ransac(normalized0, normalized1, essential_five_point,
            lambda a, b: epipolar_eight_point(a, b, essential=True),
            essential_error, 5, cfg, generator)
    if unknown and cfg.uncalibrated_model == 'shared_focal':
        fundamental, mask_f = _shared_focal_ransac(points0,points1,matrix0[:2,2],matrix1[:2,2],cfg,generator)
    else:
        fundamental, mask_f = _ransac(points0, points1, fundamental_seven_point, epipolar_eight_point,
            lambda models: sampson_error(models, points0, points1), 7, cfg, generator)
    homography, mask_h = _ransac(points0, points1, homography_dlt, homography_dlt,
        lambda models: homography_error(models, points0, points1), 4, cfg, generator)
    # A projected linear refit can be less accurate than an essential-manifold
    # fit. Refine before the E/F calibration test to avoid rejecting correct K
    # merely because the unconstrained F model was optimized more accurately.
    if essential is not None and cfg.refine_relative_pose and int(mask_e.sum()) >= cfg.min_num_inliers:
        r, t, positive, _ = _choose_pose(essential_poses(essential), rays0[mask_e], rays1[mask_e])
        if int(positive.sum()) >= cfg.min_num_inliers:
            r, t = _refine_pose(r, t, points0[mask_e][positive], points1[mask_e][positive],
                matrix0, matrix1, cfg.pose_cuda_graph, 'torch')
            from torchls.geometry import skew
            candidate = skew(t) @ r
            old_error = essential_error(essential[None])[0]
            new_error = essential_error(candidate[None])[0]
            if new_error.clamp_max(cfg.max_error ** 2).sum() < old_error.clamp_max(cfg.max_error ** 2).sum():
                essential = candidate
                mask_e = new_error <= cfg.max_error ** 2
    result.E, result.F, result.H = essential, fundamental, homography
    ne, nf, nh = int(mask_e.sum()), int(mask_f.sum()), int(mask_h.sum())
    minimum = max(cfg.min_num_inliers, math.ceil(cfg.min_inlier_ratio * len(points0)))
    if max(ne, nf, nh) < minimum:
        return result
    if essential is not None and ne >= minimum and ne >= cfg.min_E_F_inlier_ratio * nf:
        mask = mask_e if ne >= nf else mask_f
        result.config = 'CALIBRATED'
        if homography is not None and nh > cfg.max_H_inlier_ratio * ne:
            result.config = 'PLANAR'
            if nh > int(mask.sum()):
                mask = mask_h
    elif fundamental is not None and nf >= minimum:
        mask = mask_f
        result.config = 'UNCALIBRATED'
        if homography is not None and nh > cfg.max_H_inlier_ratio * nf:
            result.config, mask = 'PLANAR', mask_h
    elif homography is not None and nh >= minimum:
        result.config, mask = 'PLANAR', mask_h
    else:
        return result
    result.matches = Matches(indices[mask], scores[mask])
    if cfg.detect_watermark and _watermark(points0[mask], points1[mask], image_sizes, cfg):
        result.config = 'WATERMARK'
        return result
    if unknown:
        return result
    if result.config == 'UNCALIBRATED':
        essential = matrix1.T @ fundamental @ matrix0
    poses = (homography_poses(homography, matrix0, matrix1) if result.config == 'PLANAR'
             else essential_poses(essential))
    rotation, translation, positive, angle = _choose_pose(poses, rays0[mask], rays1[mask])
    if result.config == 'CALIBRATED' and cfg.refine_relative_pose and int(positive.sum()) >= minimum:
        rotation, translation = _refine_pose(rotation, translation, points0[mask][positive],
            points1[mask][positive], matrix0, matrix1, cfg.pose_cuda_graph, 'torch')
    if result.config == 'CALIBRATED':
        from torchls.geometry import skew
        result.E = skew(translation) @ rotation
        _, positive, angles = triangulate_relative(rotation, translation, rays0[mask], rays1[mask])
        error = sampson_error((inverse1.T @ result.E @ inverse0)[None], points0[mask], points1[mask])[0]
        positive &= error <= cfg.max_error ** 2
        angle = float(angles[positive].median()) if positive.any() else 0.0
    if translation.norm() < 1e-8:
        result.config = 'PANORAMIC'
    elif int(positive.sum()) < minimum:
        result.config = 'DEGENERATE'
        result.matches = empty_matches
        return result
    result.R, result.t, result.tri_angle = rotation.detach(), translation.detach(), angle
    result.valid_translation = bool(translation.norm() > 1e-8 and angle >= cfg.min_tri_angle)
    result.matches = Matches(indices[mask][positive], scores[mask][positive])
    return result


@torch.no_grad()
def _shared_focal_ransac(points0, points1, principal0, principal1, cfg, generator):
    """Six-point reference with record-candidate LO and adaptive trial budgets.

    This intentionally favors a simple per-pair implementation over GPU
    throughput. Minimal solves, scoring, and LM all use PyTorch.
    """
    from .shared_focal import shared_focal_six_point
    from .fundamental_refinement import refine_shared_focal

    count = len(points0)
    best_model = None
    best_mask = torch.zeros(count, device=points0.device, dtype=torch.bool)
    if count < 6:
        return best_model, best_mask
    best_score = math.inf
    trial, required = 0, cfg.max_trials
    threshold = cfg.max_error ** 2
    while trial < max(cfg.min_trials, required):
        batch = min(cfg.batch_size, cfg.max_trials - trial)
        if batch <= 0:
            break
        samples = torch.rand((batch, count), device=points0.device, generator=generator).topk(6, -1).indices
        models, focals, _ = shared_focal_six_point(
            points0[samples], points1[samples], principal0.expand(batch, 2), principal1.expand(batch, 2))
        trial += batch
        if not len(models):
            continue
        errors = sampson_error(models, points0, points1).nan_to_num(
            nan=torch.inf, posinf=torch.inf, neginf=torch.inf)
        costs = errors.clamp_max(threshold).sum(-1)
        prefix = costs.cummin(0).values
        previous = torch.cat((costs.new_full((1,), math.inf), prefix[:-1]))
        records = torch.where((costs < previous) & (costs < best_score))[0]
        for index in records.tolist():
            model, focal = models[index:index+1].clone(), focals[index:index+1].clone()
            cost = float(costs[index])
            mask = errors[index] <= threshold
            for _ in range(10):
                if int(mask.sum()) < 8:
                    break
                fitted, new_focal = refine_shared_focal(
                    model, points0[None], points1[None], mask[None], focal,
                    principal0[None], principal1[None], max_iterations=25,
                    focal_ratio_bounds=(1e-6, 1e6), backend='torch')
                residual = sampson_error(fitted, points0, points1)[0]
                proposed = float(residual.clamp_max(threshold).sum())
                if not math.isfinite(proposed) or proposed >= cost:
                    break
                model, focal, cost = fitted, new_focal, proposed
                mask = residual <= threshold
            if cost < best_score:
                best_model, best_mask, best_score = model[0], mask, cost
        inliers = int(best_mask.sum())
        probability = math.prod(max(0., min(1., (inliers-i)/(count-i))) for i in range(6))
        if probability >= 1:
            required = cfg.min_trials
        elif probability > 0:
            required = min(cfg.max_trials, math.ceil(3 * math.log1p(-cfg.confidence) / math.log1p(-probability)))
    return best_model, best_mask
