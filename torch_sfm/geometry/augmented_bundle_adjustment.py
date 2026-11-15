# Adapted algorithms from colmap/gluemap (BSD-3-Clause).
# Copyright (c), ETH Zurich.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# * Neither the name of ETH Zurich nor the names of its contributors
# may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""GlueMap's two-round triangulation / augmented BA controller in Torch.

Tracks are kept in separate real/virtual stores. Virtual measurements share
camera and intrinsic variables with real measurements, and optimize free 3D
points. Reference policy: colmap/gluemap adc9e4b (global_refinement.py,
controllers/augmented_bundle_adjustment.py, pybind/track_selection.cc).
"""
from collections import Counter
from dataclasses import dataclass, asdict, replace
import math
import time

import numpy as np
import torch
import torchls as tls
from torchls.losses import Loss

from .averaging import _check_solution, _pcg_backend
from .bundle_adjustment import (_PoseVar, _IntrinsicsVar, _pinhole_parameters,
    _restore_intrinsics, _measurements, _pose_reprojection_residual, _project_residual,
    _ba_cg_config, _compact)
from .two_view import camera_matrix
from .virtual_tracks import VirtualTracks, build_virtual_tracks


@dataclass
class AugmentedBAConfig:
    max_filter_iterations: int = 3
    normalized_reprojection_threshold: float = .01
    convergence_threshold: float = .01
    angular_error_degrees: float = .5
    virtual_loss_scale: float = 5.  # Ceres ArctanLoss(a): a*atan(s/a), NOT a^2

    def __post_init__(self):
        if self.max_filter_iterations < 1:
            raise ValueError('Need at least one augmented BA/filter round')
        if not all(math.isfinite(v) and v > 0 for v in (
                self.normalized_reprojection_threshold, self.angular_error_degrees, self.virtual_loss_scale)):
            raise ValueError('Augmented BA thresholds must be finite and positive')
        if not math.isfinite(self.convergence_threshold) or self.convergence_threshold < 0:
            raise ValueError('Invalid augmented BA convergence threshold')


@dataclass(frozen=True)
class ArctanLoss(Loss):
    """Ceres ArctanLoss(a), whose parameter a has squared-residual units."""
    scale: float = 5.

    def __post_init__(self):
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('Arctan scale must be finite and positive')

    def evaluate(self, s):
        x = s / self.scale
        first = 1 / (1 + x.square())
        return self.scale * x.atan(), first, -2 / self.scale * x * first.square()

    def value(self, s):
        return self.scale * (s / self.scale).atan()

    def _linearize(self, residual, blocks, squared_norm, evaluation):
        weight = evaluation[1].sqrt().to(residual.dtype)
        return residual * weight[:, None], tuple(b * weight[:, None, None] for b in blocks)


def virtual_residual(pose, point, intrinsics, observed, sign):
    # Negate the complete camera-space point for expected negative depths.
    # A sign crossing / failed projection contributes zero, as upstream COLMAP.
    local = (pose[:3] @ (point - pose[3])) * sign
    return _project_residual(local, intrinsics, observed)


def observation_errors(R, C, K, points, obs, pixels, negative=None, angular=False):
    fid = obs[:, 1]
    matrices = camera_matrix(K)
    matrices = matrices.expand(len(R), -1, -1) if matrices.ndim == 2 else matrices
    local = (R[fid] @ (points[obs[:, 0]] - C[fid])[..., None]).squeeze(-1)
    if negative is not None:
        local = local * torch.where(negative, -1., 1.)[:, None]
    valid = torch.isfinite(local).all(-1) & (local[:, 2] >= torch.finfo(local.dtype).eps)
    if angular:
        ray = torch.cat((pixels, torch.ones_like(pixels[:, :1])), -1)
        ray = torch.linalg.solve(matrices[fid], ray[..., None]).squeeze(-1)
        cos = (torch.nn.functional.normalize(local, dim=-1) * torch.nn.functional.normalize(ray, dim=-1)).sum(-1)
        error = cos.clamp(-1, 1).acos().rad2deg()
    else:
        projected = (matrices[fid] @ local[..., None]).squeeze(-1)
        uv = projected[:, :2] / torch.where(valid, projected[:, 2], 1.)[:, None]
        error = (uv-pixels).norm(dim=-1) / matrices[fid][:, (0, 1), (0, 1)].mean(-1)
    return torch.where(valid & torch.isfinite(error), error, torch.inf)


def _track_views(obs, num_points):
    array = obs[:, :2].detach().cpu().numpy()
    array = array[np.argsort(array[:, 0], kind='stable')]
    offsets = np.r_[0, np.bincount(array[:, 0], minlength=num_points).cumsum()]
    return [array[offsets[i]:offsets[i+1], 1] for i in range(num_points)]


def select_virtual_tracks(real_obs, num_real, virtual, num_images, threshold=512, seed=42):
    """Keep all real tracks, then greedily cover deficient image pairs.

    Like upstream SelectVirtualTracksToDelete, the boundary is <= threshold.
    Seeded selection is deterministic, but RNG order is not C++ std::shuffle.
    Uses sparse pair counts, so memory does not grow quadratically with images.
    """
    counts = Counter()
    def pairs(views):
        a, b = np.triu_indices(len(views), 1)
        return (np.minimum(views[a], views[b])*num_images + np.maximum(views[a], views[b])).tolist()
    for views in _track_views(real_obs, num_real):
        counts.update(pairs(views))
    views = _track_views(virtual.observations, len(virtual.points))
    keep = np.zeros(len(views), dtype=bool)
    for i in np.random.default_rng(seed).permutation(len(views)):
        keys = pairs(views[i])
        if any(counts[k] <= threshold for k in keys):
            keep[i] = True
            counts.update(keys)
    keep = torch.from_numpy(keep).to(virtual.points.device)
    return virtual.subset(keep[virtual.observations[:, 0]])


def select_tracked_points(points, obs, sift_counts, threshold=512, seed=42):
    """GlueMap SelectTrack: retain all-SIFT tracks, cover sparse pairs with others.

    A track is all-SIFT only if every observation references the SIFT prefix of
    its image. Pair-count boundary is <= threshold, like upstream. The RNG uses
    NumPy rather than C++ std::shuffle; within-image multiplicities are retained.
    """
    counts = sift_counts.to(device=obs.device, dtype=torch.long)
    non_sift = obs[:, 2] >= counts[obs[:, 1]]
    protected = torch.bincount(obs[non_sift, 0], minlength=len(points)) == 0
    keep = protected.cpu().numpy().copy()
    views = _track_views(obs, len(points))
    coverage = Counter()
    n = len(counts)
    def pairs(images):
        a, b = np.triu_indices(len(images), 1)
        return (np.minimum(images[a], images[b])*n + np.maximum(images[a], images[b])).tolist()
    for i in np.flatnonzero(keep):
        coverage.update(pairs(views[i]))
    candidates = np.flatnonzero(~keep)
    for i in np.random.default_rng(seed).permutation(candidates):
        keys = pairs(views[i])
        if any(coverage[k] <= threshold for k in keys):
            keep[i] = True
            coverage.update(keys)
    mask = torch.from_numpy(keep).to(obs.device)
    result = dict(all_sift_tracks=int(protected.sum()), tracked_candidates=len(candidates),
        tracked_selected=int(keep.sum())-int(protected.sum()), removed_tracks=int((~keep).sum()),
        pair_coverage_threshold=threshold)
    mapping = mask.long().cumsum(0)-1
    selected_obs = obs[mask[obs[:, 0]]].clone()
    selected_obs[:, 0] = mapping[selected_obs[:, 0]]
    return points[mask], selected_obs, result


def solve_augmented_ba(frames, R, C, points, obs, K, virtual, cfg, virtual_loss_scale=5.):
    """One joint torchls solve, with common cameras and two robust cost groups."""
    all_points = torch.cat((points, virtual.points))
    vobs = virtual.observations.clone()
    vobs[:, 0] += len(points)
    all_obs = torch.cat((obs[:, :2], vobs))
    if not len(all_obs):
        raise ValueError('No observations for augmented BA')
    nodes = torch.unique(all_obs[:, 1], sorted=True)
    # Do not silently solve disconnected components with only one gauge anchor.
    parents = list(range(len(frames)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for views in _track_views(all_obs, len(all_points)):
        for i in views[1:]:
            parents[root(int(i))] = root(int(views[0]))
    if len({root(i) for i in nodes.tolist()}) != 1:
        raise ValueError('Augmented BA track graph is disconnected; each component needs its own gauge')
    mapping = torch.full((len(frames),), -1, dtype=torch.long, device=R.device)
    mapping[nodes] = torch.arange(len(nodes), device=R.device)
    cmask = torch.ones_like(C[nodes]); cmask[0] = 0
    distance = (C[nodes] - C[nodes[0]]).norm(dim=-1)
    anchor = int(distance.argmax())
    axis = int((C[nodes[anchor]]-C[nodes[0]]).abs().argmax())
    if float(distance[anchor]) < 1e-8:
        raise ValueError('Augmented BA needs a nonzero baseline to fix scale')
    cmask[anchor, axis] = 0
    rmask = torch.ones((len(nodes), 1, 3), dtype=R.dtype, device=R.device); rmask[0] = 0
    parameters = _pinhole_parameters(K)
    shared = parameters.ndim == 1
    values = parameters[None].clone() if shared else parameters[nodes].clone()
    values[:, :2] = values[:, :2].log()
    kmask = torch.zeros_like(values)
    kmask[:, :2] = float(cfg.refine_focal_length)
    kmask[:, 2:] = float(cfg.refine_principal_point)
    problem = tls.Problem(dict(poses=torch.cat((R[nodes], C[nodes, None], rmask, cmask[:, None]), 1),
        points=all_points.clone(), intrinsics=torch.stack((values, kmask), 1)),
        manifolds=dict(poses=_PoseVar, intrinsics=_IntrinsicsVar))
    batch = tls.AutoBatchSize(memory_budget_bytes=int(cfg.jacobian_batch_memory_mb*1024**2)) if cfg.jacobian_batch_size == 'auto' else cfg.jacobian_batch_size
    for observations, pixels, loss, sign in (
        (obs, _measurements(frames, obs, points), tls.HuberLoss(delta=cfg.loss_scale), None),
        (vobs, virtual.pixels, ArctanLoss(virtual_loss_scale), torch.where(virtual.negative, -1., 1.).to(R))):
        if not len(observations):
            continue
        camera_ids = mapping[observations[:, 1]]
        intrinsic_ids = torch.zeros_like(camera_ids) if shared else camera_ids
        args = [('poses', camera_ids), ('points', observations[:, 0]), ('intrinsics', intrinsic_ids), pixels]
        if sign is not None:
            args.append(sign)
        problem.add_residual(_pose_reprojection_residual if sign is None else virtual_residual,
                             *args, loss=loss, batch_size=batch)
    solver = cfg.linear_solver
    if solver == 'auto':
        solver = 'conjugate_gradient'
    cg = _ba_cg_config(cfg, solver)
    backend = _pcg_backend(cg.backend, R)
    cg = replace(cg, backend=backend, normal_matvec='warp' if backend=='warp' else 'torch')
    options = dict(linear_solver=solver, jacobi_scaling=True, return_summary=True,
        normal_blocks_backend=cfg.normal_blocks_backend, conjugate_gradient=cg,
        termination=tls.TerminationConfig(max_iterations=cfg.max_iterations, cost_tolerance=cfg.cost_tolerance,
            gradient_tolerance=cfg.gradient_tolerance, parameter_tolerance=cfg.parameter_tolerance),
        trust_region=tls.TrustRegionConfig(lambda_initial=cfg.lambda_initial, strategy='adaptive',
            damping_mode='diagonal', step_quality_min=cfg.step_quality_min))
    if solver.startswith('schur_'):
        options['eliminate'] = 'points'
    solved, summary = problem.solve(**options)
    _check_solution(summary, 'Augmented bundle adjustment')
    R, C = R.clone(), C.clone()
    R[nodes], C[nodes] = solved['poses'][:, :3], solved['poses'][:, 3]
    values = solved['intrinsics'][:, 0].clone(); values[:, :2] = values[:, :2].exp()
    if shared:
        parameters = values[0]
    else:
        parameters = parameters.clone(); parameters[nodes] = values
    virtual = replace(virtual, points=solved['points'][len(points):])
    return R, C, solved['points'][:len(points)], _restore_intrinsics(parameters, K), virtual, summary


def refine_augmented(frames, geometries, initial, cfg, device, manifest, directory):
    from .track_refinement import refinement_intrinsics
    from .triangulation_tensor import TensorTriangulator
    from .triangulation import IncrementalTriangulator
    from .triangulation_speculative import SpeculativeTriangulator
    R, C = initial['rotations'].to(device).double().clone(), initial['centers'].to(device).double().clone()
    K = refinement_intrinsics(initial, cfg.bundle_adjustment).to(R)
    registered = initial['registered'].to(device).clone()
    ba = replace(cfg.bundle_adjustment, conjugate_gradient=replace(cfg.bundle_adjustment.conjugate_gradient,
        backend=_pcg_backend(cfg.pcg_backend, R)))
    policy = cfg.augmented
    def sync():
        if R.is_cuda: torch.cuda.synchronize(R.device)
    sync(); begin = time.perf_counter()
    virtual = build_virtual_tracks(manifest, directory, initial, cfg.virtual_tracks,
        {tuple(sorted(pair)): len(g.matches.indices) for pair,g in geometries.items()}, device)
    generated = len(virtual.points)
    generated_negative = int(virtual.negative.sum())
    verr = observation_errors(R, C, K, virtual.points, virtual.observations, virtual.pixels, virtual.negative, angular=True)
    virtual = virtual.subset(verr < policy.angular_error_degrees)
    sync(); virtual_seconds = time.perf_counter()-begin
    initialized = len(virtual.points)
    summaries, stages = [], []
    triangulator_class = dict(tensor=TensorTriangulator, sequential=IncrementalTriangulator,
                             speculative=SpeculativeTriangulator)[cfg.triangulation.implementation]
    for round_id in range(cfg.num_rounds):
        sync(); start = time.perf_counter()
        tri = triangulator_class(frames, geometries, registered, cfg.triangulation)
        tri.update_geometry(R, C, K)
        tri.triangulate(); tri.complete_and_merge(); tri.retriangulate()
        points, obs = tri.export(); del tri
        sync(); tri_time = time.perf_counter()-start
        start = time.perf_counter()
        real_selection = None
        if cfg.track_source == 'sift_vggsfm':
            points,obs,real_selection = select_tracked_points(points,obs,initial['sift_keypoint_counts'],
                cfg.sift.track_pair_coverage,cfg.tracker.seed)
        if real_selection is None or len(virtual.points):
            virtual = select_virtual_tracks(obs, len(points), virtual, len(frames), cfg.virtual_tracks.pair_coverage, cfg.virtual_tracks.seed)
        selected = len(virtual.points)
        def filter_current(threshold, angular=False):
            nonlocal points, obs, virtual
            before = len(obs)+len(virtual.observations)
            err = observation_errors(R,C,K,points,obs,_measurements(frames,obs,points),angular=angular)
            points, obs = _compact(points,obs,err < threshold,2)
            err = observation_errors(R,C,K,virtual.points,virtual.observations,virtual.pixels,virtual.negative,angular=angular)
            virtual = virtual.subset(err < threshold)
            return before-len(obs)-len(virtual.observations)
        filter_current(policy.angular_error_degrees, angular=True)
        sync(); selection_time = time.perf_counter()-start
        start = time.perf_counter()
        iteration, solve_count, filters = 0, 0, []
        while iteration < policy.max_filter_iterations:
            R,C,points,K,virtual,summary = solve_augmented_ba(frames,R,C,points,obs,K,virtual,ba,policy.virtual_loss_scale)
            summaries.append(asdict(summary)); solve_count += 1
            total_filtered = 0
            while iteration < policy.max_filter_iterations:
                threshold = max(3-iteration,1)*policy.normalized_reprojection_threshold
                removed = filter_current(threshold)
                total_filtered += removed
                filters.append(dict(threshold=threshold, removed_observations=removed))
                iteration += 1
                if len(points)+len(virtual.points)>0 and total_filtered > policy.convergence_threshold*(len(points)+len(virtual.points)):
                    break
        sync(); ba_time = time.perf_counter()-start
        stages.append(dict(round=round_id+1, triangulation_seconds=tri_time, selection_seconds=selection_time,
            ba_seconds=ba_time, ba_solves=solve_count, real_points=len(points), real_observations=len(obs),
            real_track_selection=real_selection, selected_virtual_points=selected,
            virtual_points=len(virtual.points), virtual_observations=len(virtual.observations), filters=filters))
    all_obs = torch.cat((obs[:, :2], virtual.observations))
    supported = torch.zeros_like(registered)
    supported[torch.unique(all_obs[:, 1])] = True
    registered &= supported
    errors = observation_errors(R,C,K,points,obs,_measurements(frames,obs,points))
    matrices = camera_matrix(K)
    focal = (matrices[(0,1),(0,1)].mean() if matrices.ndim==2 else matrices[obs[:,1]][:,(0,1),(0,1)].mean(-1))
    errors = errors*focal
    R, C = R.clone(), C.clone(); R[~registered] = torch.nan; C[~registered] = torch.nan
    sync()
    report = dict(track_source=cfg.track_source, strategy='augmented', virtual_tracks=len(virtual.points),
        virtual_generated=generated, virtual_initialized=initialized, virtual_generated_negative_observations=generated_negative,
        virtual_observations=len(virtual.observations), virtual_negative_observations=int(virtual.negative.sum()),
        registered=int(registered.sum()), points=len(points), observations=len(obs),
        reprojection_rms_px=float(errors.square().mean().sqrt()) if len(errors) else None,
        reprojection_median_px=float(errors.median()) if len(errors) else None,
        ba_solves=len(summaries), rounds=stages, config=asdict(cfg),
        timings_seconds=dict(virtual_tracks=virtual_seconds, triangulation=sum(s['triangulation_seconds'] for s in stages),
            ba=sum(s['ba_seconds'] for s in stages),selection=sum(s['selection_seconds'] for s in stages),total=time.perf_counter()-begin))
    return dict(rotations=R.cpu(), centers=C.cpu(), registered=registered.cpu(),
        translations=(-(R@C[...,None]).squeeze(-1)).cpu(), intrinsics=K.cpu(),
        points3d=points.cpu(), observations=obs.cpu(), reprojection_errors=errors.cpu(),
        virtual_tracks=asdict(virtual.to('cpu')), camera_groups=initial.get('camera_groups'),
        image_paths=initial['image_paths'], initial_intrinsics=initial['intrinsics'].cpu(),
        solver_summaries=summaries, refinement_report=report, stage='augmented_bundle_adjustment')
