"""Pose and ray-anchored log-depth bundle adjustment."""
from dataclasses import dataclass

import torch
import torchls as tls
from torchls.losses import Loss

from .bundle_adjustment import _PoseVar


@dataclass(frozen=True)
class CauchyLoss(Loss):
    scale: float
    weight: float = 1.

    def value(self, squared_norm):
        scale2 = self.scale**2
        return self.weight * scale2 * torch.log1p(squared_norm / scale2)

    def evaluate(self, squared_norm):
        scale2 = self.scale**2
        inverse = (1 + squared_norm / scale2).reciprocal()
        return self.value(squared_norm), self.weight * inverse, -self.weight * inverse.square() / scale2

    def _linearize(self, residual, blocks, squared_norm, evaluation):
        weight = evaluation[1].sqrt().to(residual.dtype)
        return residual * weight[:, None], tuple(block * weight[:, None, None] for block in blocks)


def target_coordinates(source, target, log_depth, ray):
    # R is world-to-camera and C is the world-space camera center.
    offset = log_depth.exp() * ray
    world = (source[..., :3, :].mT @ offset[..., None]).squeeze(-1) + source[..., 3, :]
    return (target[..., :3, :] @ (world - target[..., 3, :])[..., None]).squeeze(-1)


def reprojection(source, target, log_depth, ray, intrinsics, observed):
    local = target_coordinates(source, target, log_depth, ray)
    z = local[..., 2:3].clamp_min(.01)
    return local[..., :2] / z * intrinsics[..., :2] + intrinsics[..., 2:] - observed


def pixel_rays(xy, intrinsics):
    ray_xy = (xy - intrinsics[..., 2:]) / intrinsics[..., :2]
    return torch.cat((ray_xy, torch.ones_like(ray_xy[..., :1])), dim=-1)


def depth_prior(log_depth, initial_log_depth):
    return log_depth - initial_log_depth


def is_connected(sources, targets, image_count):
    keys = (sources * image_count + targets).unique().tolist()
    edges = [divmod(key, image_count) for key in keys]
    connected = {0}
    for _ in range(image_count):
        expanded = connected | {j for i, j in edges if i in connected} | {i for i, j in edges if j in connected}
        if expanded == connected:
            break
        connected = expanded
    return len(connected) == image_count


def refine_poses(camera_to_world, depth, intrinsics, sizes, matches, *, match_size=512,
                 iterations=100, thresholds=(2., 1.)):
    """Optimize on CUDA. Input intrinsics use original pixels, matches use match_size pixels."""
    poses = camera_to_world.cuda().double()
    grid = depth.shape[-1]
    initial_depth = depth.cuda().double().flatten()
    K = intrinsics.cuda().double() * (match_size / sizes.cuda().double()).repeat(1, 2)
    anchor = matches['anchor'].cuda()
    active = (initial_depth[anchor].isfinite() & (initial_depth[anchor] > 0)).nonzero().flatten()
    target = matches['target'].cuda()
    if not is_connected(anchor[active] // grid**2, target[active], len(poses)):
        raise ValueError('Valid depth and matches must connect all input images')

    rays_used = anchor[active].unique()
    scale = initial_depth[rays_used].median()
    origin = poses[0, :3, 3]
    masks = poses.new_ones(len(poses), 2, 3)
    masks[0] = 0
    state = torch.cat((poses[:, :3, :3].mT, ((poses[:, :3, 3] - origin) / scale)[:, None], masks), 1)
    prior = (initial_depth / scale).log()[:, None]
    log_z = prior.clone()
    pixels = matches['pixels'].cuda().double()
    confidence = matches['confidence'].cuda().double()
    rounds = []
    for threshold in (None, *thresholds):
        if threshold is not None:
            selected = active[errors <= threshold]
            if len(selected) == len(active):
                continue
            if not is_connected(anchor[selected] // grid**2, target[selected], len(poses)):
                break
            active = selected
        rays_used, depth_idx = anchor[active].unique(return_inverse=True)
        source, dest = anchor[active] // grid**2, target[active]
        offset = rays_used % grid**2
        xy = (torch.stack((offset % grid, offset // grid), -1).double() + .5) * (match_size / grid)
        rays = pixel_rays(xy, K[rays_used // grid**2])[depth_idx]
        problem = tls.Problem(dict(poses=state, log_depth=log_z[rays_used]), manifolds=dict(poses=_PoseVar))
        problem.add_residual(reprojection, ('poses', source), ('poses', dest), ('log_depth', depth_idx),
                             rays, K[dest], pixels[active], loss=CauchyLoss(3.), weights=confidence[active],
                             jac_mode='reverse', batch_size=65536)
        problem.add_residual(depth_prior, ('log_depth', torch.arange(len(rays_used))), prior[rays_used],
                             loss=CauchyLoss(.3), batch_size=65536)
        solution = problem.solve(linear_solver='schur_dense', eliminate='log_depth',
                                 trust_region=tls.TrustRegionConfig(strategy='adaptive', damping_mode='diagonal'),
                                 termination=tls.TerminationConfig(max_iterations=iterations, cost_tolerance=1e-6))
        state, log_z[rays_used] = solution['poses'], solution['log_depth']
        chunks = []
        for idx in torch.arange(len(active), device=active.device).split(65536):
            local = target_coordinates(state[source[idx]], state[dest[idx]], log_z[anchor[active[idx]]], rays[idx])
            uv = local[:, :2] / local[:, 2:].clamp_min(.01) * K[dest[idx], :2] + K[dest[idx], 2:]
            error = (uv - pixels[active[idx]]).norm(dim=-1)
            chunks.append(torch.where((local[:, 2] > .01) & error.isfinite(), error, torch.inf))
        errors = torch.cat(chunks)
        rounds.append(dict(threshold_px=threshold, observations=len(active)))
    refined = poses.clone()
    refined[:, :3, :3] = state[:, :3].mT
    refined[:, :3, 3] = state[:, 3] * scale + origin
    final_depth = initial_depth.clone()
    final_depth[rays_used] = log_z[rays_used, 0].exp() * scale
    # The dense raster also contains untouched priors. Keep final inlier measurements
    # so exports can identify optimized anchors and their actual target observations.
    supported = active[errors <= 1.]
    views = target[supported]
    measured_xy = pixels[supported] * (sizes.to(pixels)[views] / match_size)
    support = dict(anchor=anchor[supported].cpu(), target=views.cpu(), xy=measured_xy.cpu())
    return dict(camera_to_world=refined.cpu(), depth=final_depth.reshape_as(depth).cpu(), ba_rounds=rounds,
                depth_support=support)
