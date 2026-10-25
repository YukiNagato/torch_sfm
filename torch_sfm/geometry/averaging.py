"""Robust rotation averaging and GLOMAP-style bearing global positioning.

All numerical work uses PyTorch and torchls. Python dictionaries and union-find
only handle view-graph and feature-track connectivity.
"""

from dataclasses import dataclass, field
import math

import torch
import torch.nn.functional as F
import torchls as tls

from .two_view import camera_matrix, homogeneous


class SO3Var(tls.Var, default_factory=lambda: torch.eye(3), tangent_dim=3,
             retract_fn=lambda rotation, delta: rotation @ tls.so3_exp(delta)):
    pass


class UnitDirectionVar(tls.Var, default_factory=lambda: torch.tensor([0., 0., 1.]),
                       tangent_dim=3, retract_fn=lambda direction, delta: F.normalize(direction + delta, dim=-1)):
    """Unit direction; normalization removes the redundant radial component."""


class PositiveScaleVar(tls.Var, default_factory=lambda: torch.ones(1), tangent_dim=1,
                      retract_fn=lambda scale, delta: (scale * delta.exp()).clamp_min(1e-5)):
    pass


@dataclass
class RotationConfig:
    l1_iterations: int = 5
    irls_iterations: int = 20
    inner_iterations: int = 10
    loss_scale: float = 5.0  # degrees, Geman-McClure
    max_error: float = 10.0  # degrees, post-solve edge filtering
    convergence: float = 1e-3  # radians, average update

    def __post_init__(self):
        if min(self.l1_iterations, self.irls_iterations) < 0 or self.inner_iterations < 1:
            raise ValueError('Invalid rotation iteration limits')
        if not 0 < self.loss_scale < 180 or not 0 < self.max_error <= 180 or self.convergence <= 0:
            raise ValueError('Invalid rotation angular thresholds')


@dataclass
class PositionConfig:
    max_iterations: int = 100
    loss_scale: float = 0.1  # Huber threshold for unit-bearing residuals
    min_views_per_track: int = 3

    def __post_init__(self):
        if self.max_iterations < 1 or self.loss_scale <= 0 or self.min_views_per_track < 2:
            raise ValueError('Invalid global positioning options')


@dataclass
class RotationResult:
    rotations: torch.Tensor  # (N, 3, 3), world-to-camera; NaN outside component
    registered: torch.Tensor  # (N,), bool
    edges: dict
    summaries: list = field(default_factory=list)


@dataclass
class PositionResult:
    centers: torch.Tensor  # (N, 3), world coordinates; NaN if unregistered
    registered: torch.Tensor
    points3d: torch.Tensor
    observations: torch.Tensor  # (M, 3), [point index, frame-list index, keypoint index]
    method: str
    summaries: list = field(default_factory=list)


def so3_log(rotation):
    """Stable log including rotations close to pi; compatible with torch.func."""
    r00, r01, r02 = rotation[..., 0, :].unbind(-1)
    r10, r11, r12 = rotation[..., 1, :].unbind(-1)
    r20, r21, r22 = rotation[..., 2, :].unbind(-1)
    absolute = torch.stack((1 + r00 + r11 + r22, 1 + r00 - r11 - r22,
                            1 - r00 + r11 - r22, 1 - r00 - r11 + r22), -1).clamp_min(1e-12).sqrt()
    candidates = torch.stack((
        torch.stack((absolute[..., 0].square(), r21-r12, r02-r20, r10-r01), -1),
        torch.stack((r21-r12, absolute[..., 1].square(), r10+r01, r02+r20), -1),
        torch.stack((r02-r20, r10+r01, absolute[..., 2].square(), r12+r21), -1),
        torch.stack((r10-r01, r20+r02, r21+r12, absolute[..., 3].square()), -1)), -2)
    candidates = candidates / (2 * absolute[..., :, None])
    index = absolute.argmax(-1)[..., None, None].expand(*absolute.shape[:-1], 1, 4)
    quaternion = candidates.gather(-2, index).squeeze(-2)
    quaternion = torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)
    length = quaternion[..., 1:].square().sum(-1, keepdim=True).clamp_min(1e-24).sqrt()
    angle = 2 * torch.atan2(length, quaternion[..., :1])
    return quaternion[..., 1:] * (angle / length)


def largest_component(num_frames, pairs):
    neighbors = [[] for _ in range(num_frames)]
    for i, j in pairs:
        neighbors[i].append(j)
        neighbors[j].append(i)
    visited, best = set(), []
    for root in range(num_frames):
        if root in visited or not neighbors[root]:
            continue
        component, queue = [], [root]
        visited.add(root)
        while queue:
            i = queue.pop()
            component.append(i)
            for j in neighbors[i]:
                if j not in visited:
                    visited.add(j)
                    queue.append(j)
        if len(component) > len(best):
            best = component
    return sorted(best)


def _spanning_initialization(nodes, edges, reference):
    mapping = {node: index for index, node in enumerate(nodes)}
    neighbors = [[] for _ in nodes]
    parent = list(range(len(nodes)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for (i, j), geometry in sorted(edges.items(), key=lambda item: len(item[1].matches.indices), reverse=True):
        a, b = mapping[i], mapping[j]
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        parent[ra] = rb
        neighbors[a].append((b, geometry.R))
        neighbors[b].append((a, geometry.R.T))
    rotations = torch.eye(3, device=reference.device, dtype=reference.dtype).repeat(len(nodes), 1, 1)
    visited, queue = {0}, [0]
    while queue:
        i = queue.pop()
        for j, relative in neighbors[i]:
            if j not in visited:
                rotations[j] = relative @ rotations[i]
                visited.add(j)
                queue.append(j)
    return rotations


def _rotation_residual(rotation0, rotation1, relative, weight):
    return weight * so3_log(rotation1 @ rotation0.T @ relative.T)


def _rotation_prior(rotation):
    return so3_log(rotation)


def _solve_options(max_iterations):
    return dict(
        termination=tls.TerminationConfig(max_iterations=max_iterations, gradient_tolerance=1e-9),
        conjugate_gradient=tls.ConjugateGradientConfig(rtol=1e-8, max_iterations=300),
        trust_region=tls.TrustRegionConfig(lambda_initial=1e-4, strategy='adaptive', damping_mode='diagonal'),
        jacobi_scaling=True, return_summary=True,
    )


def _check_solution(summary, stage):
    if not math.isfinite(summary.final_cost) or summary.termination_reason == 'nonfinite_linearization':
        raise RuntimeError(f'{stage} failed: {summary.termination_reason}')


@torch.inference_mode(False)
def rotation_averaging(num_frames, geometries, cfg=None):
    cfg = RotationConfig() if cfg is None else cfg
    edges = {pair: geometry for pair, geometry in geometries.items() if geometry.R is not None}
    if not edges:
        raise ValueError('No geometrically verified rotation edges')
    reference = next(iter(edges.values())).R
    rotations = reference.new_full((num_frames, 3, 3), torch.nan)
    registered = torch.zeros(num_frames, dtype=torch.bool, device=reference.device)
    summaries = []
    # Filter gross rotation outliers, then solve the surviving largest component.
    for filtering_pass in range(3):
        nodes = largest_component(num_frames, edges)
        if len(nodes) < 2:
            raise ValueError('No connected rotation component after filtering')
        node_set = set(nodes)
        edges = {pair: g for pair, g in edges.items() if pair[0] in node_set and pair[1] in node_set}
        mapping = {node: index for index, node in enumerate(nodes)}
        pairs = torch.tensor([(mapping[i], mapping[j]) for i, j in edges], device=reference.device)
        relative = torch.stack([g.R for g in edges.values()]).detach().clone()
        initial = _spanning_initialization(nodes, edges, reference)
        if filtering_pass and torch.isfinite(rotations[nodes]).all():
            initial = rotations[nodes].clone()
        current = initial
        sigma = math.radians(cfg.loss_scale)
        phases = ['l1'] * cfg.l1_iterations + ['gm'] * cfg.irls_iterations
        for iteration, phase in enumerate(phases):
            error = so3_log(current[pairs[:, 1]] @ current[pairs[:, 0]].transpose(-1, -2)
                            @ relative.transpose(-1, -2)).norm(dim=-1)
            weights = (error.clamp_min(1e-3).rsqrt() if phase == 'l1'
                       else sigma ** 2 / (sigma ** 2 + error.square()))
            problem = tls.Problem({'rotations': current}, manifolds={'rotations': SO3Var})
            problem.add_residual(_rotation_residual, ('rotations', pairs[:, 0]),
                                 ('rotations', pairs[:, 1]), relative, weights[:, None])
            problem.add_residual(_rotation_prior, ('rotations', 0))
            solved, summary = problem.solve(**_solve_options(cfg.inner_iterations))
            _check_solution(summary, 'Rotation averaging')
            summaries.append(summary)
            updated = solved['rotations']
            step = float(so3_log(updated @ current.transpose(-1, -2)).norm(dim=-1).mean())
            current = updated.detach()
            next_phase = phases[iteration + 1] if iteration + 1 < len(phases) else None
            if step < cfg.convergence and phase == 'gm' and next_phase == 'gm':
                break
        current = current @ current[0].T
        rotations.fill_(torch.nan)
        registered.zero_()
        rotations[nodes] = current
        registered[nodes] = True
        error = so3_log(current[pairs[:, 1]] @ current[pairs[:, 0]].transpose(-1, -2)
                        @ relative.transpose(-1, -2)).norm(dim=-1)
        keep = error <= math.radians(cfg.max_error)
        filtered = {pair: geometry for (pair, geometry), valid in zip(edges.items(), keep.tolist()) if valid}
        if len(filtered) == len(edges):
            return RotationResult(rotations, registered, edges, summaries)
        if filtering_pass == 2:
            # Never retain registrations that became disconnected on the last pass.
            nodes = largest_component(num_frames, filtered)
            registered.zero_()
            registered[nodes] = True
            rotations[~registered] = torch.nan
            filtered = {pair: g for pair, g in filtered.items() if pair[0] in nodes and pair[1] in nodes}
        edges = filtered
    return RotationResult(rotations, registered, edges, summaries)


def build_tracks(geometries, active_frames, min_views=3):
    """Merge verified matches, refusing unions with two features in one frame."""
    active = set(active_frames)
    parent, observations, visible, lookup = [], [], [], {}

    def node(frame, keypoint):
        key = frame, keypoint
        if key not in lookup:
            lookup[key] = len(parent)
            parent.append(len(parent))
            observations.append(key)
            visible.append({frame})
        return lookup[key]

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for (i, j), geometry in sorted(geometries.items(), key=lambda item: len(item[1].matches.indices), reverse=True):
        if i not in active or j not in active:
            continue
        order = geometry.matches.scores.argsort(descending=True)
        for kp0, kp1 in geometry.matches.indices[order].tolist():
            a, b = find(node(i, kp0)), find(node(j, kp1))
            if a == b or visible[a] & visible[b]:
                continue
            if len(visible[a]) < len(visible[b]):
                a, b = b, a
            parent[b] = a
            visible[a].update(visible[b])
            visible[b].clear()
    tracks = {}
    for index, observation in enumerate(observations):
        tracks.setdefault(find(index), []).append(observation)
    rows, count = [], 0
    for track in tracks.values():
        if len(track) < min_views:
            continue
        rows.extend((count, frame, keypoint) for frame, keypoint in sorted(track))
        count += 1
    return torch.tensor(rows, dtype=torch.long).reshape(-1, 3)


def _huber(error, delta):
    squared = error.square().sum()
    length = squared.clamp_min(1e-24).sqrt()
    rho = torch.where(length <= delta, squared, 2 * delta * length - delta ** 2)
    return error * (rho.clamp_min(1e-24) / squared.clamp_min(1e-24)).sqrt()


def _direction_residual(center0, center1, scale, direction, delta):
    return _huber(direction - scale[0] * (center1 - center0), delta)


def _center_prior(center):
    return center * 10


def _baseline_prior(center0, center1):
    return ((center1 - center0).norm() - 1).reshape(1) * 10


def _add_gauge(problem, anchor):
    problem.add_residual(_center_prior, ('centers', 0))
    problem.add_residual(_baseline_prior, ('centers', anchor[0]), ('centers', anchor[1]))


def _normalize_gauge(centers, points, anchor):
    scale = (centers[anchor[1]] - centers[anchor[0]]).norm().clamp_min(1e-12)
    origin = centers[0].clone()
    return (centers - origin) / scale, (points - origin) / scale


def _initial_centers(nodes, geometries, rotations):
    mapping = {node: index for index, node in enumerate(nodes)}
    directions, pairs = [], []
    neighbors = [[] for _ in nodes]
    for (i, j), geometry in geometries.items():
        a, b = mapping[i], mapping[j]
        direction = F.normalize(-rotations[j].T @ geometry.t, dim=0)
        pairs.append((a, b))
        directions.append(direction)
        neighbors[a].append((b, direction))
        neighbors[b].append((a, -direction))
    centers = rotations.new_zeros((len(nodes), 3))
    visited, queue = {0}, [0]
    while queue:
        i = queue.pop()
        for j, direction in neighbors[i]:
            if j not in visited:
                centers[j] = centers[i] + direction
                visited.add(j)
                queue.append(j)
    pairs = torch.tensor(pairs, device=rotations.device)
    directions = torch.stack(directions)
    return centers, pairs, directions


@torch.inference_mode(False)
def translation_averaging(frames, geometries, rotations, intrinsics, cfg=None):
    """BATA camera initialization, then point-to-camera bearing optimization.

    Return the largest usable component. Three-view tracks resolve relative
    baseline scales that a chain of two-view directions cannot determine.
    When no such tracks exist, report a pairwise-only estimate explicitly.
    """
    cfg = PositionConfig() if cfg is None else cfg
    rotations = rotations.detach().clone()
    device, dtype = rotations.device, rotations.dtype
    centers_all = rotations.new_full((len(frames), 3), torch.nan)
    registered = torch.zeros(len(frames), dtype=torch.bool, device=device)
    empty = PositionResult(centers_all, registered, rotations.new_empty((0, 3)),
                           torch.empty((0, 3), dtype=torch.long, device=device), 'unobservable')
    usable = {pair: g for pair, g in geometries.items() if g.valid_translation
              and torch.isfinite(rotations[list(pair)]).all()}
    nodes = largest_component(len(frames), usable)
    if len(nodes) < 2:
        return empty
    node_set = set(nodes)
    usable = {pair: g for pair, g in usable.items() if pair[0] in node_set and pair[1] in node_set}
    centers, pairs, directions = _initial_centers(nodes, usable, rotations)
    supports = [len(geometry.matches.indices) for geometry in usable.values()]
    strongest = max(range(len(supports)), key=supports.__getitem__)
    anchor = pairs[strongest].tolist()
    centers, _ = _normalize_gauge(centers, centers[:0], anchor)
    baseline = (centers[pairs[:, 1]] - centers[pairs[:, 0]]).norm(dim=-1).clamp_min(1e-3)
    problem = tls.Problem({'centers': centers, 'scales': baseline.reciprocal()[:, None]},
                          manifolds={'scales': PositiveScaleVar})
    problem.add_residual(_direction_residual, ('centers', pairs[:, 0]), ('centers', pairs[:, 1]),
                         ('scales', torch.arange(len(pairs), device=device)), directions, cfg.loss_scale)
    _add_gauge(problem, anchor)
    solution, summary = problem.solve(linear_solver='schur_cg', eliminate='scales',
                                      **_solve_options(cfg.max_iterations))
    _check_solution(summary, 'Pairwise positioning')
    summaries = [summary]
    centers, _ = _normalize_gauge(solution['centers'], centers[:0], anchor)
    observations = build_tracks(geometries, nodes, cfg.min_views_per_track).to(device)
    if len(observations) == 0:
        centers_all[nodes] = centers
        registered[nodes] = True
        return PositionResult(centers_all, registered, centers[:0], observations, 'pairwise', summaries)

    # A disconnected track graph carries an independent scale in each component.
    track_pairs = []
    for track in torch.split(observations[:, 1].cpu(), torch.bincount(observations[:, 0]).tolist()):
        values = track.tolist()
        track_pairs.extend((values[0], frame) for frame in values[1:])
    track_nodes = largest_component(len(frames), track_pairs)
    selected = torch.isin(observations[:, 1], torch.tensor(track_nodes, device=device))
    observations = observations[selected]
    _, observations[:, 0] = torch.unique(observations[:, 0], sorted=True, return_inverse=True)
    old_mapping = {node: index for index, node in enumerate(nodes)}
    centers = centers[[old_mapping[node] for node in track_nodes]]
    nodes = track_nodes
    mapping = torch.full((len(frames),), -1, dtype=torch.long, device=device)
    mapping[nodes] = torch.arange(len(nodes), device=device)
    camera_ids = mapping[observations[:, 1]]
    point_ids = observations[:, 0]
    num_points = int(point_ids.max()) + 1
    matrices = camera_matrix(intrinsics.to(device=device, dtype=dtype).clone())
    if matrices.ndim == 2:
        matrices = matrices.expand(len(frames), 3, 3)
    inverse = torch.linalg.inv(matrices)
    rays = rotations.new_empty((len(observations), 3))
    for frame_index in nodes:
        mask = observations[:, 1] == frame_index
        keypoints = frames[frame_index].keypoints_2d.detach().to(device=device, dtype=dtype).clone()
        local = homogeneous(keypoints[observations[mask, 2]]) @ inverse[frame_index].T
        rays[mask] = F.normalize(local @ rotations[frame_index], dim=-1)
    # Least-squares intersection provides a deterministic track initialization.
    projectors = torch.eye(3, device=device, dtype=dtype) - rays[:, :, None] * rays[:, None, :]
    normal = rotations.new_zeros((num_points, 3, 3)).index_add_(0, point_ids, projectors)
    rhs = rotations.new_zeros((num_points, 3)).index_add_(
        0, point_ids, (projectors @ centers[camera_ids, :, None]).squeeze(-1))
    points = (torch.linalg.pinv(normal, rtol=1e-10) @ rhs[..., None]).squeeze(-1)
    anchor = [0, int((centers - centers[0]).norm(dim=-1).argmax())]
    if (centers[anchor[1]] - centers[0]).norm() < 1e-6:
        raise ValueError('Global positioning has no nonzero baseline')
    centers, points = _normalize_gauge(centers, points, anchor)
    offsets = points[point_ids] - centers[camera_ids]
    # Initialize inverse depth by projecting the measured unit bearing.
    scales = ((rays * offsets).sum(-1) / offsets.square().sum(-1).clamp_min(1e-12)).clamp_min(1e-5)
    problem = tls.Problem({'centers': centers, 'points': points, 'scales': scales[:, None]},
                          manifolds={'scales': PositiveScaleVar})
    problem.add_residual(_direction_residual, ('centers', camera_ids), ('points', point_ids),
                         ('scales', torch.arange(len(rays), device=device)), rays, cfg.loss_scale,
                         batch_size=4096)
    _add_gauge(problem, anchor)
    solution, summary = problem.solve(linear_solver='schur_cg', eliminate='scales',
                                      **_solve_options(cfg.max_iterations))
    _check_solution(summary, 'Track positioning')
    summaries.append(summary)
    centers, points = _normalize_gauge(solution['centers'], solution['points'], anchor)
    if not torch.isfinite(centers).all() or not torch.isfinite(points).all():
        raise RuntimeError('Global positioning produced nonfinite values')
    centers_all[nodes] = centers
    registered[nodes] = True
    return PositionResult(centers_all, registered, points, observations, 'tracks', summaries)
