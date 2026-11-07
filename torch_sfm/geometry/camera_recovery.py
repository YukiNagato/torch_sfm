"""Recover excluded cameras from existing 2D--3D correspondences, in eager Torch.

RANSAC combines calibrated six-point DLT and two-point translation hypotheses
when a rotation estimate is available. Full SE(3) refinement and pixel/depth
validation decide acceptance; no camera is registered from rotation alone.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
import torchls as tls

from torch_sfm.core.storage import ObservationIndex
from .two_view import camera_matrix


@dataclass
class CameraRecoveryConfig:
    enabled: bool = True
    max_rounds: int = 2
    max_trials: int = 1024
    hypothesis_batch_size: int = 128
    refinement_candidates: int = 4
    refinement_iterations: int = 30
    refinement_solver: str = 'analytic'
    refinement_device: str = 'cpu'
    min_inliers: int = 30
    min_inlier_ratio: float = 0.2
    max_reprojection_error: float = 4.0
    seed_error_multiplier: float = 2.0
    min_image_coverage: float = 0.005
    min_correspondence_votes: int = 1
    random_seed: int = 0

    def __post_init__(self):
        if self.refinement_solver not in ('analytic', 'torchls'):
            raise ValueError('refinement_solver must be analytic or torchls')
        if self.refinement_device not in ('cpu', 'input'):
            raise ValueError('refinement_device must be cpu or input')
        for key in ('max_rounds', 'max_trials', 'hypothesis_batch_size',
                    'refinement_candidates', 'refinement_iterations', 'min_correspondence_votes'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if isinstance(self.min_inliers, bool) or not isinstance(self.min_inliers, int) or self.min_inliers < 6:
            raise ValueError('Camera recovery needs at least six inliers')
        if not 0 < self.min_inlier_ratio <= 1 or not 0 <= self.min_image_coverage <= 1:
            raise ValueError('Invalid camera recovery support thresholds')
        if not math.isfinite(self.max_reprojection_error) or self.max_reprojection_error <= 0:
            raise ValueError('Recovery pixel threshold must be finite and positive')
        if not math.isfinite(self.seed_error_multiplier) or self.seed_error_multiplier < 1:
            raise ValueError('Seed threshold multiplier must be finite and at least one')


def _errors(poses, points, pixels, k):
    local = torch.einsum('bij,nj->bni', poses[:, :3, :3], points) + poses[:, None, :3, 3]
    good = (local[..., 2] > torch.finfo(points.dtype).eps) & torch.isfinite(local).all(-1)
    depth = torch.where(good, local[..., 2], torch.ones_like(local[..., 2]))
    projected = local[..., :2] / depth[..., None] * k[:2] + k[2:]
    return (projected - pixels).square().sum(-1).masked_fill(~good, torch.inf)


def _dlt(points, normalized, samples):
    xyz = points[samples]
    mean = xyz.mean(1)
    scale = (xyz - mean[:, None]).square().sum(-1).mean(1).sqrt().clamp_min(1e-12)
    x = (xyz - mean[:, None]) / scale[:, None, None]
    spread = torch.linalg.svdvals(x)
    valid = (spread[:, 2] > spread[:, 0] * 1e-4) & (samples.sort(1).values.diff(dim=1) != 0).all(1)
    xh = torch.cat((x, torch.ones_like(x[..., :1])), -1)
    uv = normalized[samples]
    z = torch.zeros_like(xh)
    a = torch.stack((torch.cat((xh, z, -uv[..., :1] * xh), -1),
                     torch.cat((z, xh, -uv[..., 1:] * xh), -1)), 2).flatten(1, 2)
    _, _, vh = torch.linalg.svd(a, full_matrices=False)
    projection = vh[:, -1].reshape(-1, 3, 4)
    sign = torch.where(torch.linalg.det(projection[:, :, :3]) < 0, -1., 1.)
    projection = projection * sign[:, None, None]
    u, singular, vh = torch.linalg.svd(projection[:, :, :3])
    rotation = u @ vh
    valid &= torch.linalg.det(rotation) > 0
    translation = projection[:, :, 3] / singular.mean(1).clamp_min(1e-12)[:, None]
    translation = scale[:, None] * translation - (rotation @ mean[..., None]).squeeze(-1)
    pose = torch.eye(4, device=points.device, dtype=points.dtype).repeat(len(samples), 1, 1)
    pose[:, :3, :3], pose[:, :3, 3] = rotation, translation
    return pose[valid & torch.isfinite(pose).all(dim=(1, 2))]


def _rotation_seeds(points, normalized, rotation, samples):
    rays = F.normalize(torch.cat((normalized, torch.ones_like(normalized[:, :1])), -1), dim=-1) @ rotation
    projectors = torch.eye(3, device=points.device, dtype=points.dtype) - rays[:, :, None] * rays[:, None, :]
    normal = projectors[samples].sum(1)
    rhs = (projectors[samples] @ points[samples][..., None]).sum(1).squeeze(-1)
    centers, info = torch.linalg.solve_ex(normal, rhs)
    valid = (info == 0) & (torch.linalg.eigvalsh(normal)[:, 0] > 1e-6)
    pose = torch.eye(4, device=points.device, dtype=points.dtype).repeat(len(samples), 1, 1)
    pose[:, :3, :3] = rotation
    pose[:, :3, 3] = -(rotation @ centers[..., None]).squeeze(-1)
    return pose[valid & torch.isfinite(pose).all(dim=(1, 2))]


def _residual(pose, point, pixel, k):
    local = pose[:3, :3] @ point + pose[:3, 3]
    good = local[2] > torch.finfo(local.dtype).eps
    depth = torch.where(good, local[2], torch.ones_like(local[2]))
    return torch.where(good, local[:2] / depth * k[:2] + k[2:] - pixel, torch.zeros_like(pixel))


def _refine_torchls(pose, points, pixels, k, config):
    with torch.inference_mode(False), torch.enable_grad():
        problem = tls.Problem({'pose': pose[None].clone()}, manifolds={'pose': tls.SE3Var})
        problem.add_residual(_residual, ('pose', torch.zeros(len(points), device=points.device, dtype=torch.long)),
                             points, pixels, k.expand(len(points), 4),
                             loss=tls.HuberLoss(delta=config.max_reprojection_error), batch_size=None)
        values = problem.solve(linear_solver='dense_qr', jacobi_scaling=True,
            termination=tls.TerminationConfig(max_iterations=config.refinement_iterations,
                cost_tolerance=1e-8, gradient_tolerance=1e-8, parameter_tolerance=1e-8),
            trust_region=tls.TrustRegionConfig(lambda_initial=1e-4, strategy='adaptive', damping_mode='diagonal'))
    return values['pose'][0].detach()


def _refine(pose, points, pixels, k, config):
    device = 'cpu' if config.refinement_device == 'cpu' else pose.device
    inputs = [x.to(device=device) for x in (pose, points, pixels, k)]
    if config.refinement_solver == 'analytic':
        from .pose_refinement import refine_pose
        result = refine_pose(*inputs, loss_scale=config.max_reprojection_error,
                             max_iterations=config.refinement_iterations)
    else:
        result = _refine_torchls(*inputs, config)
    return result.to(pose)


@torch.no_grad()
def estimate_pose(points, pixels, k, rotation_prior, image_size, config, *, seed=0):
    """Return a world-to-camera pose and strict inlier mask, or a rejection report."""
    n = len(points)
    report = {'correspondences': n, 'accepted': False}
    if n < config.min_inliers:
        return None, None, report | {'reason': 'insufficient_correspondences'}
    if not torch.isfinite(points).all() or not torch.isfinite(pixels).all():
        raise ValueError('PnP inputs must be finite')
    generator = torch.Generator(device=points.device).manual_seed(config.random_seed + seed)
    normalized = (pixels - k[2:]) / k[:2]
    prior_valid = (rotation_prior is not None and bool(torch.isfinite(rotation_prior).all()))
    candidates, scores = [], []
    seed_threshold = (config.max_reprojection_error * config.seed_error_multiplier) ** 2
    for start in range(0, config.max_trials, config.hypothesis_batch_size):
        count = min(config.hypothesis_batch_size, config.max_trials - start)
        samples = torch.randint(n, (count, 6), generator=generator, device=points.device)
        poses = [_dlt(points, normalized, samples)]
        if prior_valid:
            poses.append(_rotation_seeds(points, normalized, rotation_prior, samples[:, :2]))
        poses = torch.cat(poses)
        if not len(poses):
            continue
        errors = _errors(poses, points, pixels, k)
        # Count first, then bounded error; the secondary term is strictly < 1.
        score = (errors <= seed_threshold).sum(1) - errors.clamp(max=seed_threshold).mean(1) / (seed_threshold * 2)
        selected = score.topk(min(config.refinement_candidates, len(score))).indices
        candidates.append(poses[selected]); scores.append(score[selected])
    if not candidates:
        return None, None, report | {'reason': 'degenerate_hypotheses'}
    candidates, scores = torch.cat(candidates), torch.cat(scores)
    best, best_mask, best_score = None, None, -math.inf
    for i in scores.topk(min(config.refinement_candidates, len(scores))).indices.tolist():
        pose = candidates[i]
        mask = _errors(pose[None], points, pixels, k)[0] <= seed_threshold
        if int(mask.sum()) < config.min_inliers:
            continue
        for _ in range(2):
            pose = _refine(pose, points[mask], pixels[mask], k, config)
            error = _errors(pose[None], points, pixels, k)[0]
            mask = error <= config.max_reprojection_error ** 2
            if int(mask.sum()) < config.min_inliers:
                break
        count = int(mask.sum())
        score = count - float(error.clamp(max=config.max_reprojection_error ** 2).mean()) / (2 * config.max_reprojection_error ** 2)
        if count >= config.min_inliers and score > best_score:
            best, best_mask, best_score = pose, mask, score
    if best is None:
        return None, None, report | {'reason': 'insufficient_inliers'}
    count = int(best_mask.sum())
    xy = pixels[best_mask]
    span = xy.amax(0) - xy.amin(0)
    coverage = float(span.prod()) / (image_size[0] * image_size[1])
    spread = torch.linalg.svdvals(points[best_mask] - points[best_mask].mean(0))
    error = _errors(best[None], points, pixels, k)[0][best_mask]
    report.update(inliers=count, inlier_ratio=count/n, image_coverage=coverage,
                  rms_px=float(error.mean().sqrt()))
    if count/n < config.min_inlier_ratio:
        return None, None, report | {'reason': 'low_inlier_ratio'}
    if coverage < config.min_image_coverage or float(spread[1]) < float(spread[0]) * 1e-3:
        return None, None, report | {'reason': 'insufficient_spatial_support'}
    return best, best_mask, report | {'accepted': True, 'reason': 'accepted'}


@torch.no_grad()
def recover_cameras(frames, geometries, rotations, centers, points, observations, intrinsics, registered, config):
    """Mutate accepted camera poses/mask; append only validated 2D--3D inliers."""
    pending = (~registered).nonzero().flatten().tolist()
    report = {'attempted': [], 'recovered': []}
    if not config.enabled or not pending or not len(points):
        return observations, report
    index = ObservationIndex.from_frames(frames, device=points.device, dtype=points.dtype)
    owner = torch.full((len(index.xy),), -1, device=points.device, dtype=torch.long)
    offsets = torch.tensor(index.frame_offsets, device=points.device)
    ids = offsets[observations[:, 1]] + observations[:, 2]
    if len(torch.unique(ids)) != len(ids):
        raise ValueError('Camera recovery requires unique feature ownership')
    owner[ids] = observations[:, 0]
    calibration = camera_matrix(intrinsics.to(points))
    added = []
    for image in pending:
        pairs = []
        for (i, j), geometry in geometries.items():
            if i == image and bool(registered[j]):
                local = geometry.matches.indices.to(points.device)
                pid = owner[index.frame_offsets[j] + local[:, 1]]
                pairs.append(torch.stack((local[:, 0], pid), -1))
            elif j == image and bool(registered[i]):
                local = geometry.matches.indices.to(points.device)
                pid = owner[index.frame_offsets[i] + local[:, 0]]
                pairs.append(torch.stack((local[:, 1], pid), -1))
        if not pairs:
            report['attempted'].append({'image': image, 'accepted': False, 'reason': 'no_registered_neighbors'})
            continue
        pairs = torch.cat(pairs); pairs = pairs[pairs[:, 1] >= 0]
        pairs, votes = torch.unique(pairs, dim=0, return_counts=True)
        good = votes >= config.min_correspondence_votes
        pairs, votes = pairs[good], votes[good]
        # Choose strongest 3D assignment per feature, then one feature per point.
        for column in (0, 1):
            order = votes.argsort(descending=True, stable=True)
            order = order[pairs[order, column].argsort(stable=True)]
            pairs, votes = pairs[order], votes[order]
            keep = torch.ones(len(pairs), device=points.device, dtype=torch.bool)
            keep[1:] = pairs[1:, column] != pairs[:-1, column]
            pairs, votes = pairs[keep], votes[keep]
        xy = frames[image].keypoints_2d.to(points)[pairs[:, 0]]
        xyz = points[pairs[:, 1]]
        matrix = calibration if calibration.ndim == 2 else calibration[image]
        k = matrix[[0, 1, 0, 1], [0, 1, 2, 2]]
        size = getattr(frames[image], 'image_size', None)
        if size is None and getattr(frames[image], 'frame_path', None) is not None:
            from PIL import Image
            with Image.open(frames[image].frame_path) as source:
                size = (source.height, source.width)
        if size is None:
            raise ValueError('Camera recovery needs image dimensions for spatial support validation')
        pose, mask, status = estimate_pose(xyz, xy, k, rotations[image], size, config, seed=image)
        report['attempted'].append({'image': image, **status})
        if pose is None:
            continue
        rotations[image] = pose[:3, :3]
        centers[image] = -(pose[:3, :3].T @ pose[:3, 3])
        registered[image] = True
        chosen = pairs[mask]
        rows = torch.stack((chosen[:, 1], torch.full_like(chosen[:, 0], image), chosen[:, 0]), -1)
        added.append(rows)
        owner[index.frame_offsets[image] + chosen[:, 0]] = chosen[:, 1]
        report['recovered'].append(image)
    return torch.cat([observations, *added]) if added else observations, report
