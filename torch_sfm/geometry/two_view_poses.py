"""Pose decomposition and completion shared by geometric verification backends."""
import math
import torch
from ..correspondence import Matches
from .two_view import camera_matrix, homogeneous, homography_poses, _choose_pose, triangulate_relative

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
