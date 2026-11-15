"""Dense round-trip covisibility for local pose/depth priors.

Following GlueMap's N^2 check: bilinearly sample target world points, project
back, divide inlier counts by the full source image area, symmetrize by max,
then propagate maximum-product path scores. Coordinates use pixel centers
(x+.5,y+.5), as in Solv3R training. This is a consistency test, not ground truth;
pure rotation cannot constrain depth with round-trip pixel error alone.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DepthConsistency:
    direct_scores: torch.Tensor  # (N,N), directed source -> target
    center_scores: torch.Tensor  # (N,), best path from view 0
    valid_views: torch.Tensor  # (N,); reference is retained
    center_masks: torch.Tensor  # (N,H,W), direct checks in reference pixels


def propagate_covisibility(direct_scores):
    """Max-product paths on the undirected max of the directed overlap graph."""
    scores = torch.maximum(direct_scores, direct_scores.T).clone()
    scores.fill_diagonal_(1)
    for k in range(len(scores)):
        scores = torch.maximum(scores, scores[:, k:k+1] * scores[k:k+1, :])
    return scores[0]


@torch.inference_mode()
def check_depth_consistency(camera_to_world, intrinsics, depth, *,
                            reprojection_threshold=4., min_overlap=.05,
                            pair_batch_size=16) -> DepthConsistency:
    """Check a single star; tensors share a device, depths are camera z-depth.

    Thresholds are in the depth map's pixel coordinates. Confidence is not a
    calibrated probability and is deliberately not used as an inlier weight.
    Scratch memory is bounded by pair_batch_size, not N^2 dense maps.
    """
    if depth.ndim != 3 or min(depth.shape) < 1:
        raise ValueError('depth must have shape (N,H,W), with positive dimensions')
    n, h, w = depth.shape
    if camera_to_world.shape != (n, 4, 4) or intrinsics.shape != (n, 3, 3):
        raise ValueError('Camera/depth dimensions disagree')
    if pair_batch_size < 1 or reprojection_threshold <= 0 or not 0 <= min_overlap <= 1:
        raise ValueError('Invalid consistency thresholds or batch size')
    if not torch.isfinite(camera_to_world).all() or not torch.isfinite(intrinsics).all():
        raise ValueError('Camera matrices must be finite')
    depth, c2w, K = depth.float(), camera_to_world.float(), intrinsics.float()
    device = depth.device
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    pixels = torch.stack((x + .5, y + .5, torch.ones_like(x)), -1).float().reshape(-1, 3)
    invK = torch.linalg.inv(K)
    valid_depth = torch.isfinite(depth) & (depth > 0)
    clean_depth = torch.where(valid_depth, depth, 0.)
    camera_points = (pixels[None] @ invK.transpose(-1, -2)) * clean_depth.reshape(n, -1, 1)
    R, C = c2w[:, :3, :3], c2w[:, :3, 3]
    world = camera_points @ R.transpose(-1, -2) + C[:, None]
    world_maps = world.transpose(1, 2).reshape(n, 3, h, w)
    # Append validity so interpolation crossing a missing-depth pixel is rejected.
    maps = torch.cat((world_maps, valid_depth[:, None].float()), dim=1)
    scores = depth.new_zeros(n, n)
    masks = torch.zeros((n, h, w), dtype=torch.bool, device=device)
    masks[0] = valid_depth[0]
    scores.diagonal().copy_(valid_depth.float().mean((-1, -2)))
    source, target = torch.where(~torch.eye(n, device=device, dtype=torch.bool))
    bounds = depth.new_tensor([w, h])
    for start in range(0, len(source), pair_batch_size):
        i, j = source[start:start+pair_batch_size], target[start:start+pair_batch_size]
        xyz = (world[i] - C[j, None]) @ R[j]
        projected = xyz @ K[j].transpose(-1, -2)
        z = projected[..., 2:3]
        uv = projected[..., :2] / torch.where(z > 0, z, 1.)
        inside = ((uv >= .5 - 1e-4) & (uv <= bounds - .5 + 1e-4)).all(-1)
        grid = torch.nan_to_num(2 * uv / bounds - 1, nan=2., posinf=2., neginf=-2.)
        sampled = F.grid_sample(maps[j], grid[:, None], mode='bilinear',
                                padding_mode='border', align_corners=False).squeeze(2).transpose(1, 2)
        back = (sampled[..., :3] - C[i, None]) @ R[i]
        back_pixel = back @ K[i].transpose(-1, -2)
        back_z = back_pixel[..., 2:3]
        back_uv = back_pixel[..., :2] / torch.where(back_z > 0, back_z, 1.)
        error_sq = (back_uv - pixels[None, :, :2]).square().sum(-1)
        valid = (valid_depth[i].flatten(1) & inside & (xyz[..., 2] > 0)
                 & (sampled[..., 3] >= 1 - 1e-6) & (back[..., 2] > 0)
                 & torch.isfinite(error_sq) & (error_sq < reprojection_threshold ** 2))
        scores[i, j] = valid.float().mean(-1)
        # The ordered-pair list starts with the reference's N-1 neighbors.
        if start < n - 1:
            count = min(len(i), n - 1 - start)
            masks[j[:count]] = valid[:count].reshape(count, h, w)
    center_scores = propagate_covisibility(scores)
    valid_views = (center_scores > 0) & (center_scores >= min_overlap)
    valid_views[0] = True
    return DepthConsistency(scores, center_scores, valid_views, masks)
