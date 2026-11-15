"""Seed a sparse map from verified tracks while keeping cameras fixed.

Ragged observations are reduced into independent 3x3 point systems. No camera
variables, global PCG, or learned depths enter this solve.
"""
from dataclasses import dataclass, replace
import math
import time

import torch
import torch.nn.functional as F

from .averaging import build_tracks
from .bundle_adjustment import _compact, _measurements, _active_component
from .two_view import camera_matrix, homogeneous


@dataclass
class TrackInitializationConfig:
    min_views: int = 3
    intra_image_threshold: float = 10.
    track_backend: str = 'auto'
    max_reprojection_error: float = 8.  # initial cameras are still approximate
    min_tri_angle: float = 1.
    lm_iterations: int = 15
    loss_scale: float = 2.
    lambda_initial: float = 1e-3

    def __post_init__(self):
        if self.min_views < 3 or self.lm_iterations < 0:
            raise ValueError('Seed tracks need at least three views and nonnegative LM iterations')
        if self.track_backend not in ('auto', 'cpu', 'warp'):
            raise ValueError('Unknown seed track backend')
        if not 0 <= self.min_tri_angle < 180 or self.intra_image_threshold < 0:
            raise ValueError('Invalid seed geometry thresholds')
        if not all(math.isfinite(x) and x > 0 for x in (
                self.max_reprojection_error, self.loss_scale, self.lambda_initial)):
            raise ValueError('Seed pixel and damping thresholds must be finite and positive')


def _sum_tracks(values, ids, count):
    return values.new_zeros((count, *values.shape[1:])).index_add_(0, ids, values)


def _project(points, ids, rotations, centers, calibration, pixels):
    local = (rotations @ (points[ids]-centers)[..., None]).squeeze(-1)
    valid = torch.isfinite(local).all(-1) & (local[:, 2] > 1e-12)
    z = local[:, 2].clamp_min(1e-12)
    residual = local[:, :2]/z[:, None]*calibration[:, :2]+calibration[:, 2:]-pixels
    return local, z, residual, valid


def refine_points(points, observations, rotations, centers, intrinsics, pixels, cfg):
    """Batched robust point-only LM; return points and per-track Huber costs."""
    ids, fid = observations[:, 0], observations[:, 1]
    R, C = rotations[fid], centers[fid]
    K = intrinsics.expand(len(rotations), 3, 3) if intrinsics.ndim == 2 else intrinsics
    k = K[fid]
    calibration = torch.stack((k[:, 0, 0], k[:, 1, 1], k[:, 0, 2], k[:, 1, 2]), -1)
    n = len(points)
    counts = torch.bincount(ids, minlength=n)
    def evaluate(x):
        local, z, residual, positive = _project(x, ids, R, C, calibration, pixels)
        error = residual.norm(dim=-1)
        cost = torch.where(error <= cfg.loss_scale, .5*error.square(),
                           cfg.loss_scale*(error-.5*cfg.loss_scale))
        cost = _sum_tracks(cost, ids, n)
        valid = (_sum_tracks(positive.long(), ids, n) == counts) & torch.isfinite(cost)
        return local, z, residual, error, torch.where(valid, cost, torch.inf), valid
    points = points.clone()
    local, z, residual, error, cost, valid = evaluate(points)
    initial_cost = cost.clone()
    damping = points.new_full((n,), cfg.lambda_initial)
    identity = torch.eye(3, dtype=points.dtype, device=points.device)
    accepted = torch.zeros(n, dtype=torch.long, device=points.device)
    for _ in range(cfg.lm_iterations):
        ju = calibration[:, 0, None]/z[:, None]*(R[:, 0]-local[:, 0, None]/z[:, None]*R[:, 2])
        jv = calibration[:, 1, None]/z[:, None]*(R[:, 1]-local[:, 1, None]/z[:, None]*R[:, 2])
        weight = (cfg.loss_scale/error.clamp_min(1e-30)).clamp_max(1)
        H = _sum_tracks(weight[:, None, None]*(ju[:, :, None]*ju[:, None, :]
                           +jv[:, :, None]*jv[:, None, :]), ids, n)
        g = _sum_tracks(weight[:, None]*(ju*residual[:, :1]+jv*residual[:, 1:]), ids, n)
        diagonal = H.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12)
        matrix = H+torch.diag_embed(diagonal*damping[:, None])
        matrix = torch.where(valid[:, None, None], matrix, identity)
        g = torch.where(valid[:, None], g, 0.)
        step, info = torch.linalg.solve_ex(matrix, -g)
        candidate = points+step
        new = evaluate(candidate)
        keep = valid & new[-1] & (info == 0) & torch.isfinite(step).all(-1) & (new[-2] < cost)
        points = torch.where(keep[:, None], candidate, points)
        accepted += keep
        damping = torch.where(keep, damping/3., damping*10.).clamp(1e-12, 1e12)
        local, z, residual, error, cost, valid = evaluate(points)
    return points, initial_cost, cost, accepted


@torch.no_grad()
def initialize_tracks(frames, geometries, rotations, centers, intrinsics, registered, ba, cfg=None):
    cfg = TrackInitializationConfig() if cfg is None else cfg
    def sync():
        if rotations.is_cuda:
            torch.cuda.synchronize(rotations.device)
    sync(); begin = time.perf_counter()
    observations = build_tracks(geometries, torch.where(registered)[0].tolist(),
        min_views=cfg.min_views, strategy='colmap', frames=frames,
        intra_image_threshold=cfg.intra_image_threshold, backend=cfg.track_backend).to(rotations.device)
    if not len(observations):
        raise RuntimeError('No multi-view tracks available for map initialization')
    # The COLMAP builder admits nearby same-image features. Keep a deterministic
    # single feature per camera: multiple features cannot count as extra views.
    keys = observations[:, 0]*len(frames)+observations[:, 1]
    order = keys.argsort(stable=True)
    first = torch.ones(len(order), dtype=torch.bool, device=order.device)
    first[1:] = keys[order[1:]] != keys[order[:-1]]
    duplicate_count = len(observations)-int(first.sum())
    observations = observations[order[first]].clone()
    n = int(observations[:, 0].max())+1
    pixels = _measurements(frames, observations, rotations)
    dummy = rotations.new_zeros((n, 3))
    _, observations = _compact(dummy, observations, torch.isfinite(pixels).all(-1), cfg.min_views)
    if not len(observations):
        raise RuntimeError('No finite multi-view tracks available for map initialization')
    pixels = _measurements(frames, observations, rotations)
    ids, fid = observations[:, 0], observations[:, 1]
    n = int(ids.max())+1
    sync(); build_seconds = time.perf_counter()-begin
    start = time.perf_counter()
    K = camera_matrix(intrinsics)
    matrices = K.expand(len(frames), 3, 3) if K.ndim == 2 else K
    rays = F.normalize(torch.linalg.solve(matrices[fid], homogeneous(pixels)[..., None]).squeeze(-1), dim=-1)
    directions = (rotations[fid].transpose(-1,-2) @ rays[..., None]).squeeze(-1)
    identity = torch.eye(3, device=rotations.device, dtype=rotations.dtype)
    A = identity-directions[..., :, None]*directions[..., None, :]
    counts = torch.bincount(ids, minlength=n)
    starts = counts.cumsum(0)-counts
    origin = centers[fid[starts]]
    normal = _sum_tracks(A, ids, n)
    rhs = _sum_tracks((A @ (centers[fid]-origin[ids])[..., None]).squeeze(-1), ids, n)
    conditioned = torch.linalg.det(normal) > 1e-12*normal.diagonal(dim1=-2,dim2=-1).sum(-1).pow(3)
    delta, info = torch.linalg.solve_ex(torch.where(conditioned[:, None, None], normal, identity), rhs)
    points = origin+delta
    local = (rotations[fid] @ (points[ids]-centers[fid])[..., None]).squeeze(-1)
    positive = (local[:, 2] > 1e-12) & torch.isfinite(local).all(-1)
    usable = conditioned & (info == 0) & (_sum_tracks(positive.long(), ids, n) == counts)
    points, observations = _compact(points, observations, usable[ids], cfg.min_views)
    if not len(points):
        raise RuntimeError('No stable positive-depth seed tracks')
    pixels = _measurements(frames, observations, rotations)
    sync(); linear_seconds = time.perf_counter()-start
    start = time.perf_counter()
    points, before, after, accepted = refine_points(points, observations, rotations, centers, K, pixels, cfg)
    sync(); lm_seconds = time.perf_counter()-start
    start = time.perf_counter()
    from .bundle_adjustment import reprojection_errors
    error, positive = reprojection_errors(frames, rotations, centers, points, observations, K)
    points, observations = _compact(points, observations,
        positive & torch.isfinite(error) & (error <= cfg.max_reprojection_error), cfg.min_views)
    # At least one well-separated camera pair; compare ragged tracks in bounded
    # chunks to avoid a global N_tracks x max_track_length padded allocation.
    if len(points) and cfg.min_tri_angle:
        from .global_bundle_adjustment import triangulation_mask
        good = torch.zeros(len(points), dtype=torch.bool, device=points.device)
        counts = torch.bincount(observations[:, 0], minlength=len(points))
        ends = counts.cumsum(0).cpu().tolist()
        for lo in range(0, len(points), 2048):
            hi = min(lo+2048, len(points))
            obs = observations[ends[lo-1] if lo else 0:ends[hi-1]].clone()
            obs[:, 0] -= lo
            good[lo:hi] = triangulation_mask(centers, points[lo:hi], obs, cfg.min_tri_angle)
        points, observations = _compact(points, observations, good[observations[:, 0]], cfg.min_views)
    points, observations, supported = _active_component(points, observations, len(frames),
        replace(ba, min_views_per_track=cfg.min_views))
    if len(points) == 0 or int(supported.sum()) < 2:
        raise RuntimeError('No supported seed map survives geometry filtering')
    sync(); filter_seconds = time.perf_counter()-start
    support = torch.bincount(observations[:, 1], minlength=len(frames))
    report = dict(candidate_tracks=n, duplicate_observations_removed=duplicate_count,
        linearly_triangulated_tracks=len(before), points=len(points), observations=len(observations),
        registered=int(supported.sum()), excluded_images=torch.where(registered & ~supported)[0].tolist(),
        observations_per_image=support.cpu().tolist(),
        lm_initial_cost=float(before.sum()), lm_final_cost=float(after.sum()),
        lm_accepted_steps=int(accepted.sum()),
        timings_seconds=dict(build_tracks=build_seconds, linear_triangulation=linear_seconds,
            point_lm=lm_seconds, filtering=filter_seconds, total=time.perf_counter()-begin))
    return points, observations, supported, report
