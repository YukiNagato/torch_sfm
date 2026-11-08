"""Robust rotation averaging and GLOMAP-style bearing global positioning.

Numerical work uses PyTorch/torchls with optional Warp PCG on CUDA. Dictionaries and union-find
only handle view-graph and feature-track connectivity.
"""

from dataclasses import dataclass, field, replace
import math

import torch
import torch.nn.functional as F
import torchls as tls

from .two_view import camera_matrix, homogeneous


class SO3Var(tls.Var, default_factory=lambda: torch.eye(3), tangent_dim=3,
             retract_fn=lambda rotation, delta: rotation @ tls.so3_exp(delta),
             retract_jvp_fn=tls.so3_retract_jvp):
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
    pcg_backend: str = 'auto'  # Warp for CUDA float32/64 when installed; Torch otherwise
    method: str = 'auto'  # fixed-system L1/IRLS below the dense size limit; otherwise LM
    linear_max_cameras: int = 2048
    linear_filter_passes: int = 2
    ridge_regularization: float = 1e-9
    admm_initial_iterations: int = 10
    admm_max_iterations: int = 100
    admm_absolute_tolerance: float = 1e-4
    admm_relative_tolerance: float = 1e-2

    def __post_init__(self):
        if min(self.l1_iterations, self.irls_iterations) < 0 or self.inner_iterations < 1:
            raise ValueError('Invalid rotation iteration limits')
        if not 0 < self.loss_scale < 180 or not 0 < self.max_error <= 180 or self.convergence <= 0:
            raise ValueError('Invalid rotation angular thresholds')
        if self.pcg_backend not in ('auto', 'torch', 'warp'):
            raise ValueError('PCG backend must be auto, torch or warp')
        if self.method not in ('auto', 'lm', 'l1_irls'):
            raise ValueError('Rotation method must be auto, lm or l1_irls')
        if self.linear_max_cameras < 2 or self.linear_filter_passes < 1:
            raise ValueError('Invalid fixed-system size/filtering limits')
        if not 1 <= self.admm_initial_iterations <= self.admm_max_iterations:
            raise ValueError('Invalid ADMM iteration limits')
        if not all(math.isfinite(x) and x >= 0 for x in (
                self.ridge_regularization, self.admm_absolute_tolerance, self.admm_relative_tolerance)):
            raise ValueError('Invalid ADMM tolerances or ridge')


@dataclass
class PositionConfig:
    max_iterations: int = 100
    loss_scale: float = 0.1  # Huber threshold for unit-bearing residuals
    min_views_per_track: int = 3
    scale_parameterization: str = 'bounded'  # Euclidean scales with solver-side lower bounds
    scale_initialization: str = 'unit'  # 'geometry' retains the previous initialization
    pairwise_linear_solver: str = 'conjugate_gradient'
    joint_linear_solver: str = 'conjugate_gradient'
    jacobian_batch_size: int | None = None
    normal_blocks_backend: str = 'auto'
    normal_matvec: str = 'auto'  # full PCG: Warp when PCG backend is Warp
    cache_retained_matrix: bool = False
    pcg_rtol: float = 1e-3
    pairwise_max_iterations: int = 20  # warm start when tracks permit joint refinement
    track_backend: str = 'auto'  # colmap policy: Warp on CUDA when installed, CPU otherwise
    track_strategy: str = 'conflict_aware'  # or 'colmap': union first, reject inconsistent tracks
    track_intra_image_consistency_threshold: float = 10.0  # pixels, used by 'colmap'
    pcg_backend: str = 'auto'  # Warp on supported CUDA when installed; Torch otherwise

    def __post_init__(self):
        if self.max_iterations < 1 or self.loss_scale <= 0 or self.min_views_per_track < 2:
            raise ValueError('Invalid global positioning options')
        if self.scale_parameterization not in ('bounded', 'log'):
            raise ValueError('Position scales must use bounded or log parameterization')
        if self.scale_initialization not in ('unit', 'geometry'):
            raise ValueError('Position scale initialization must be unit or geometry')
        if self.joint_linear_solver not in ('schur_cg', 'conjugate_gradient'):
            raise ValueError('Unknown joint positioning solver')
        if (self.jacobian_batch_size is not None and
                (isinstance(self.jacobian_batch_size, bool) or
                 not isinstance(self.jacobian_batch_size, int) or self.jacobian_batch_size < 1)):
            raise ValueError('Jacobian batch size must be a positive integer or None')
        if self.normal_blocks_backend not in ('auto', 'torch', 'warp'):
            raise ValueError('Unknown normal blocks backend')
        if self.normal_matvec not in ('auto', 'torch', 'warp'):
            raise ValueError('Unknown normal matvec backend')
        if self.pairwise_linear_solver not in ('auto', 'schur_dense', 'schur_cg', 'conjugate_gradient'):
            raise ValueError('Unknown pairwise positioning solver')
        if not math.isfinite(self.pcg_rtol) or not 0 < self.pcg_rtol < 1:
            raise ValueError('Position PCG tolerance must lie between zero and one')
        if self.pairwise_max_iterations < 1:
            raise ValueError('Pairwise initialization budget must be positive')
        if self.track_backend not in ('auto', 'cpu', 'warp'):
            raise ValueError('Track backend must be auto, cpu or warp')
        if self.track_backend == 'warp' and self.track_strategy != 'colmap':
            raise ValueError('Warp tracks require the colmap strategy')
        if self.track_strategy not in ('conflict_aware', 'colmap'):
            raise ValueError('Unknown positioning track strategy')
        if (not math.isfinite(self.track_intra_image_consistency_threshold)
                or self.track_intra_image_consistency_threshold < 0):
            raise ValueError('Track consistency threshold must be finite and nonnegative')
        if self.pcg_backend not in ('auto', 'torch', 'warp'):
            raise ValueError('PCG backend must be auto, torch or warp')


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


def _pcg_backend(backend, reference):
    if backend not in ('auto', 'torch', 'warp'):
        raise ValueError('PCG backend must be auto, torch or warp')
    supported = (reference is not None and reference.device.type == 'cuda'
                 and reference.dtype in (torch.float32, torch.float64))
    if backend == 'warp' and not supported:
        raise ValueError('Warp PCG requires CUDA float32 or float64 tensors')
    if backend == 'auto':
        from importlib.util import find_spec
        return 'warp' if supported and find_spec('warp') is not None else 'torch'
    return backend


def _solve_options(max_iterations, *, pcg_backend='torch', reference=None):
    return dict(
        termination=tls.TerminationConfig(max_iterations=max_iterations, gradient_tolerance=1e-9),
        conjugate_gradient=tls.ConjugateGradientConfig(
            rtol=1e-8, max_iterations=300, backend=_pcg_backend(pcg_backend, reference)),
        trust_region=tls.TrustRegionConfig(lambda_initial=1e-4, strategy='adaptive', damping_mode='diagonal'),
        # Change only PCG execution. Keep the existing Torch normal operator and
        # block assembly for a controlled comparison; Warp kernels are optional
        # follow-up optimizations independent of the PCG backend.
        normal_blocks_backend='torch', jacobi_scaling=True, return_summary=True,
    )


def _position_options(cfg, num_cameras, *, pairwise=False, initializer=False, reference=None):
    iterations = min(cfg.max_iterations, cfg.pairwise_max_iterations) if initializer else cfg.max_iterations
    options = _solve_options(iterations, pcg_backend=cfg.pcg_backend, reference=reference)
    solver = cfg.pairwise_linear_solver if pairwise else cfg.joint_linear_solver
    if solver == 'auto':
        solver = 'schur_dense' if 3*num_cameras <= 512 else 'schur_cg'
    options.update(linear_solver=solver, normal_blocks_backend=cfg.normal_blocks_backend)
    cg = options['conjugate_gradient']
    matvec = 'torch'
    if solver == 'conjugate_gradient':
        matvec = ('warp' if cg.backend == 'warp' else 'torch') if cfg.normal_matvec == 'auto' else cfg.normal_matvec
    else:
        options['eliminate'] = 'scales'
    options['conjugate_gradient'] = replace(cg, rtol=cfg.pcg_rtol, normal_matvec=matvec,
        cache_retained_matrix=cfg.cache_retained_matrix and solver == 'schur_cg')
    if cfg.scale_parameterization == 'bounded':
        options['bounds'] = {'scales': (1e-5, None)}
    return options


def _check_solution(summary, stage):
    if not math.isfinite(summary.final_cost) or summary.termination_reason == 'nonfinite_linearization':
        raise RuntimeError(f'{stage} failed: {summary.termination_reason}')


@torch.inference_mode(False)
def rotation_averaging(num_frames, geometries, cfg=None):
    cfg = RotationConfig() if cfg is None else cfg
    if cfg.method != 'lm':
        edges = {pair: g for pair, g in geometries.items() if g.R is not None}
        count = len(largest_component(num_frames, edges))
        if cfg.method == 'l1_irls' or count <= cfg.linear_max_cameras:
            from .rotation_averaging_linear import rotation_averaging_linear
            return rotation_averaging_linear(num_frames, geometries, cfg)
    return _rotation_averaging_lm(num_frames, geometries, cfg)


def _rotation_averaging_lm(num_frames, geometries, cfg):
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
            solved, summary = problem.solve(**_solve_options(
                cfg.inner_iterations, pcg_backend=cfg.pcg_backend, reference=reference))
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


def build_tracks(geometries, active_frames, min_views=3, *, strategy='conflict_aware',
                 frames=None, intra_image_threshold=10.0, backend='auto'):
    """Build positioning tracks from the caller's rotation-filtered image pairs.

    conflict_aware rejects unions with multiple features from the same image.
    colmap unions all matches first, rejects a whole component if any two of its
    features in one image are farther apart than intra_image_threshold pixels,
    and requires min_views distinct images. Nearby same-image features remain.
    The colmap strategy requires frames for the keypoint coordinates.
    backend='auto' selects Warp for CUDA colmap inputs when installed; otherwise
    the CPU reference is used. Warp returns CUDA indices, CPU returns CPU indices.
    Both return identical canonical track IDs and row order. Explicit Warp is
    supported only for the colmap policy. No runtime failures are hidden.
    """
    if backend not in ('auto', 'cpu', 'warp'):
        raise ValueError('Track backend must be auto, cpu or warp')
    if strategy == 'colmap':
        if frames is None:
            raise ValueError('colmap track strategy requires frames with keypoints')
        if min_views < 2 or not math.isfinite(intra_image_threshold) or intra_image_threshold < 0:
            raise ValueError('Invalid track view count or consistency threshold')
        if backend != 'cpu' and frames:
            from importlib.util import find_spec
            supported = frames[0].keypoints_2d.is_cuda
            if backend == 'warp' and not supported:
                raise ValueError('Warp track construction requires CUDA keypoints')
            if backend == 'warp' or (supported and find_spec('warp') is not None):
                from .warp.tracks import build_colmap_tracks
                return build_colmap_tracks(geometries, active_frames, frames, min_views,
                                           intra_image_threshold)
        return _build_colmap_tracks(geometries, active_frames, frames, min_views,
                                    intra_image_threshold)
    if backend == 'warp':
        raise ValueError('Warp tracks require the colmap strategy')
    if strategy != 'conflict_aware':
        raise ValueError('Unknown positioning track strategy')
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


def _build_colmap_tracks(geometries, active_frames, frames, min_views, threshold):
    """GlobalMapper::EstablishTracks with the default unlimited track budget.

    Track IDs/order are canonical here, rather than C++ hash-table order. The
    observation partitions and consistency rule match the native implementation.
    """
    if frames is None:
        raise ValueError('colmap track strategy requires frames with keypoints')
    if min_views < 2 or not math.isfinite(threshold) or threshold < 0:
        raise ValueError('Invalid track view count or consistency threshold')
    active = set(active_frames)
    parent, sizes, observations, lookup = [], [], [], {}

    def node(frame, keypoint):
        key = frame, keypoint
        if key not in lookup:
            lookup[key] = len(parent)
            parent.append(len(parent))
            sizes.append(1)
            observations.append(key)
        return lookup[key]

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for (i, j), geometry in geometries.items():
        if i not in active or j not in active:
            continue
        for kp0, kp1 in geometry.matches.indices.detach().cpu().tolist():
            a, b = find(node(i, kp0)), find(node(j, kp1))
            if a == b:
                continue
            if sizes[a] < sizes[b]:
                a, b = b, a
            parent[b] = a
            sizes[a] += sizes[b]
    tracks = {}
    for index, observation in enumerate(observations):
        tracks.setdefault(find(index), []).append(observation)
    xy = {i: frames[i].keypoints_2d.detach().cpu().tolist() for i in active}
    retained = []
    squared_threshold = threshold * threshold
    for track in tracks.values():
        by_image = {}
        consistent = True
        for image, keypoint in track:
            point = xy[image][keypoint]
            previous = by_image.setdefault(image, [])
            if any((point[0] - other[0]) ** 2 + (point[1] - other[1]) ** 2
                   > squared_threshold for other in previous):
                consistent = False
                break
            previous.append(point)
        if consistent and len(by_image) >= min_views:
            retained.append(sorted(track))
    retained.sort(key=lambda track: (-len(track), track))
    rows = [(pid, frame, keypoint) for pid, track in enumerate(retained)
            for frame, keypoint in track]
    return torch.tensor(rows, dtype=torch.long).reshape(-1, 3)


def _direction_residual(center0, center1, scale, direction):
    return direction - scale[0] * (center1 - center0)


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
    # Track connectivity is independent of the current camera centers. Build it
    # once so pairwise-only outputs retain the full optimization budget.
    observations = build_tracks(
        geometries, nodes, cfg.min_views_per_track, strategy=cfg.track_strategy,
        frames=frames,
        intra_image_threshold=cfg.track_intra_image_consistency_threshold,
        backend=cfg.track_backend).to(device)
    centers, pairs, directions = _initial_centers(nodes, usable, rotations)
    supports = [len(geometry.matches.indices) for geometry in usable.values()]
    strongest = max(range(len(supports)), key=supports.__getitem__)
    anchor = pairs[strongest].tolist()
    centers, _ = _normalize_gauge(centers, centers[:0], anchor)
    baseline = (centers[pairs[:, 1]] - centers[pairs[:, 0]]).norm(dim=-1).clamp_min(1e-3)
    scales = torch.ones_like(baseline) if cfg.scale_initialization == 'unit' else baseline.reciprocal()
    manifolds = {'scales': PositiveScaleVar} if cfg.scale_parameterization == 'log' else {}
    problem = tls.Problem({'centers': centers, 'scales': scales[:, None]}, manifolds=manifolds)
    problem.add_residual(_direction_residual, ('centers', pairs[:, 0]), ('centers', pairs[:, 1]),
                         ('scales', torch.arange(len(pairs), device=device)), directions,
                         loss=tls.HuberLoss(delta=cfg.loss_scale))
    _add_gauge(problem, anchor)
    solution, summary = problem.solve(**_position_options(
        cfg, len(nodes), pairwise=True, initializer=len(observations) > 0, reference=rotations))
    _check_solution(summary, 'Pairwise positioning')
    summaries = [summary]
    centers, _ = _normalize_gauge(solution['centers'], centers[:0], anchor)
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
    # Unit scales keep initially reversed/poorly triangulated bearings active.
    scales = (torch.ones(len(rays), device=device, dtype=dtype) if cfg.scale_initialization == 'unit'
              else ((rays * offsets).sum(-1) / offsets.square().sum(-1).clamp_min(1e-12)).clamp_min(1e-5))
    problem = tls.Problem({'centers': centers, 'points': points, 'scales': scales[:, None]},
                          manifolds=manifolds)
    problem.add_residual(_direction_residual, ('centers', camera_ids), ('points', point_ids),
                         ('scales', torch.arange(len(rays), device=device)), rays,
                         loss=tls.HuberLoss(delta=cfg.loss_scale), batch_size=cfg.jacobian_batch_size)
    _add_gauge(problem, anchor)
    solution, summary = problem.solve(**_position_options(cfg, len(nodes), reference=rotations))
    _check_solution(summary, 'Track positioning')
    summaries.append(summary)
    centers, points = _normalize_gauge(solution['centers'], solution['points'], anchor)
    if not torch.isfinite(centers).all() or not torch.isfinite(points).all():
        raise RuntimeError('Global positioning produced nonfinite values')
    centers_all[nodes] = centers
    registered[nodes] = True
    return PositionResult(centers_all, registered, points, observations, 'tracks', summaries)
