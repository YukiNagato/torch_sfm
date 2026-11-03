"""Pinhole global BA, track filtering and retriangulation using Torch/torchls."""

from dataclasses import dataclass, field, asdict, replace
import math

from PIL import Image
import torch
import torch.nn.functional as F
import torchls as tls

from .averaging import _check_solution, build_tracks, largest_component
from .two_view import camera_matrix, homogeneous
from .triangulation import TriangulationConfig


@dataclass
class BundleAdjustmentConfig:
    strategy: str = 'glomap'
    enabled: bool = True
    optimizer_backend: str = 'torchls'  # optional 'bae' LM/PCG adapter
    num_rounds: int = 3
    fixed_rotation_iterations: int = 200
    max_iterations: int = 200
    loss_scale: float = 1.0  # Huber, pixels
    max_reprojection_error: float = 4.0  # final pixel threshold
    initial_max_reprojection_error: float = 40.0
    initial_max_angular_error: float = 2.0  # degrees
    min_tri_angle: float = 1.0  # degrees
    min_views_per_track: int = 2
    min_observations_per_image: int = 3
    refine_focal_length: bool = True
    refine_principal_point: bool = False  # native GLOMAP keeps this fixed
    principal_point_prior_sigma: float | None = None  # fraction of width/height; None disables prior
    retriangulate: bool = True
    linear_solver: str = 'auto'
    normal_blocks_backend: str = 'auto'  # auto enables fused blocks with Warp PCG
    pose_parameterization: str = 'joint'  # same updates in one 6D variable; 'split' for comparison
    fixed_camera_handling: str = 'masked'  # 'grouped': pass fixed cameras as residual data
    conjugate_gradient: tls.ConjugateGradientConfig = field(default_factory=lambda:
        tls.ConjugateGradientConfig(rtol=1e-6, max_iterations=300,
                                    preconditioner_groups=(("rotations", "centers"),)))
    jacobian_batch_size: int | str | None = 'auto'  # None: one vmap; int: explicit chunks
    jacobian_batch_memory_mb: float = 256.0  # estimated temporary AD working set
    min_track_length: int = 3  # BA admission; filtering can still retain two observations
    cost_tolerance: float = 1e-5
    gradient_tolerance: float = 1e-4
    parameter_tolerance: float = 0.0
    lambda_initial: float = 1e-4
    step_quality_min: float = 1e-3
    max_normalized_reprojection_error: float = .01
    filter_min_change: float = .001
    normalize: bool = True
    triangulation: TriangulationConfig = field(default_factory=TriangulationConfig)
    retriangulation_max_refinements: int = 5
    retriangulation_max_iterations: int = 50
    retriangulation_min_change: float = .0005
    refinement_max_reprojection_error: float = 4.0
    refinement_min_tri_angle: float = 1.5

    @classmethod
    def legacy(cls, **overrides):
        values = dict(strategy='legacy', fixed_rotation_iterations=20, max_iterations=50,
                      refine_principal_point=True, principal_point_prior_sigma=.02,
                      min_track_length=2, cost_tolerance=1e-7,
                      gradient_tolerance=1e-7, parameter_tolerance=1e-10,
                      lambda_initial=1e-3, step_quality_min=1e-4)
        values.update(overrides)
        return cls(**values)

    def __post_init__(self):
        if self.optimizer_backend not in ('torchls', 'bae'):
            raise ValueError('optimizer_backend must be torchls or bae')
        if self.strategy not in ('glomap', 'legacy'):
            raise ValueError('BA strategy must be glomap or legacy')
        if self.num_rounds < 1 or self.max_iterations < 1 or self.fixed_rotation_iterations < 0:
            raise ValueError('Invalid bundle adjustment iteration limits')
        if min(self.loss_scale, self.max_reprojection_error, self.initial_max_reprojection_error) <= 0:
            raise ValueError('BA loss and pixel thresholds must be positive')
        if not 0 <= self.min_tri_angle < 180 or not 0 < self.initial_max_angular_error <= 180:
            raise ValueError('Invalid BA angular thresholds')
        if self.min_views_per_track < 2 or self.min_observations_per_image < 3:
            raise ValueError('BA needs at least two views per track and three observations per image')
        if self.linear_solver not in ('auto', 'schur_dense', 'schur_cg', 'conjugate_gradient', 'dense_qr'):
            raise ValueError('Unknown BA linear solver')
        if self.normal_blocks_backend not in ('auto', 'torch', 'warp'):
            raise ValueError('normal_blocks_backend must be auto, torch or warp')
        if self.pose_parameterization not in ('split', 'joint'):
            raise ValueError('pose_parameterization must be split or joint')
        if self.fixed_camera_handling not in ('masked', 'grouped'):
            raise ValueError('fixed_camera_handling must be masked or grouped')
        if self.fixed_camera_handling == 'grouped' and self.pose_parameterization != 'joint':
            raise ValueError('Grouped fixed cameras currently require joint pose_parameterization')
        if not isinstance(self.conjugate_gradient, tls.ConjugateGradientConfig):
            raise TypeError('conjugate_gradient must be a ConjugateGradientConfig')
        if self.jacobian_batch_size is not None and self.jacobian_batch_size != 'auto':
            if (isinstance(self.jacobian_batch_size, bool)
                    or not isinstance(self.jacobian_batch_size, int) or self.jacobian_batch_size < 1):
                raise ValueError('jacobian_batch_size must be auto, a positive integer, or None')
        if not math.isfinite(self.jacobian_batch_memory_mb) or self.jacobian_batch_memory_mb < 1 / 1024**2:
            raise ValueError('jacobian_batch_memory_mb must be finite and at least one byte')
        if self.min_track_length < 2 or self.retriangulation_max_refinements < 0 or self.retriangulation_max_iterations < 1:
            raise ValueError('Invalid BA track length or refinement budgets')
        if not isinstance(self.triangulation, TriangulationConfig):
            raise TypeError('triangulation must be a TriangulationConfig')
        if not all(math.isfinite(v) and v >= 0 for v in (
                self.cost_tolerance, self.gradient_tolerance, self.parameter_tolerance,
                self.filter_min_change, self.retriangulation_min_change)):
            raise ValueError('BA tolerances must be finite and nonnegative')
        if not all(math.isfinite(v) and v > 0 for v in (
                self.max_normalized_reprojection_error, self.refinement_max_reprojection_error,
                self.lambda_initial)) or not 0 <= self.step_quality_min < 1:
            raise ValueError('Invalid BA thresholds or damping')
        if not 0 <= self.refinement_min_tri_angle < 90:
            raise ValueError('Invalid refinement triangulation angle')
        if self.principal_point_prior_sigma is not None and (
                not math.isfinite(self.principal_point_prior_sigma) or self.principal_point_prior_sigma <= 0):
            raise ValueError('Principal point prior sigma must be positive and finite, or None')


@dataclass
class BundleAdjustmentResult:
    rotations: torch.Tensor
    centers: torch.Tensor
    points3d: torch.Tensor
    observations: torch.Tensor
    registered: torch.Tensor
    intrinsics: torch.Tensor
    reprojection_errors: torch.Tensor
    summaries: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# torchls accepts custom retractions. Each value stores its update mask in a
# second row: zeros enforce exact constants without adding soft gauge priors.
class _CenterVar(tls.Var, default_factory=lambda: torch.stack((torch.zeros(3), torch.ones(3))),
                 tangent_dim=3,
                 retract_fn=lambda value, delta: torch.stack((value[0] + value[1] * delta, value[1]))):
    pass


class _IntrinsicsVar(tls.Var, default_factory=lambda: torch.stack((torch.zeros(4), torch.ones(4))),
                     tangent_dim=4,
                     retract_fn=lambda value, delta: torch.stack((value[0] + value[1] * delta, value[1]))):
    pass


class _RotationVar(tls.Var, default_factory=lambda: torch.cat((torch.eye(3), torch.ones(1, 3))),
                   tangent_dim=3,
                   retract_fn=lambda value, delta: torch.cat((
                       value[:3] @ tls.so3_exp(value[3] * delta), value[3:])),
                   retract_jvp_fn=lambda value, direction: torch.cat((
                       tls.so3_retract_jvp(value[:3], value[3] * direction),
                       torch.zeros_like(value[3:])))):
    pass


def _pose_retract(value, delta):
    # Rows: R(3), world camera center(1), rotation mask(1), center mask(1).
    # Product retraction, exactly matching _RotationVar and _CenterVar.
    return torch.cat((value[:3] @ tls.so3_exp(value[4] * delta[:3]),
                      (value[3] + value[5] * delta[3:])[None], value[4:]))


def _pose_retract_jvp(value, direction):
    # Exact derivative of the existing rotation + world-center product update.
    # Keep its masks and convention; this is not an SE(3) change of coordinates.
    return torch.cat((
        tls.so3_retract_jvp(value[:3], value[4] * direction[:3]),
        (value[5] * direction[3:])[None], torch.zeros_like(value[4:]),
    ))


class _PoseVar(tls.Var,
               default_factory=lambda: torch.cat((torch.eye(3), torch.zeros(1, 3), torch.ones(2, 3))),
               tangent_dim=6, retract_fn=_pose_retract, retract_jvp_fn=_pose_retract_jvp):
    pass


def _ba_cg_config(cfg, solver):
    cg = cfg.conjugate_gradient
    if solver not in ('conjugate_gradient', 'schur_cg'):
        return replace(cg, preconditioner_groups=())
    if cfg.pose_parameterization == 'split':
        return cg
    groups = []
    for group in cg.preconditioner_groups:
        members = set(group) & {'rotations', 'centers'}
        if members and members != {'rotations', 'centers'}:
            raise ValueError('Joint poses require rotations and centers in the same '
                             'preconditioner group; use poses to name the combined variable')
        mapped = tuple(dict.fromkeys('poses' if key in members else key for key in group))
        if len(mapped) > 1:
            groups.append(mapped)
    return replace(cg, preconditioner_groups=tuple(groups))


def _pinhole_parameters(intrinsics):
    matrix = camera_matrix(intrinsics)
    if ((matrix[..., 0, 1].abs() > 1e-12).any() or (matrix[..., 1, 0].abs() > 1e-12).any()
            or not torch.allclose(matrix[..., 2, :], matrix.new_tensor([0, 0, 1]).expand_as(matrix[..., 2, :]))):
        raise ValueError('BA supports undistorted, zero-skew PINHOLE cameras')
    return torch.stack((matrix[..., 0, 0], matrix[..., 1, 1], matrix[..., 0, 2], matrix[..., 1, 2]), -1)


def _restore_intrinsics(parameters, template):
    return camera_matrix(parameters) if template.shape[-2:] == (3, 3) else parameters


def _measurements(frames, observations, reference):
    pixels = reference.new_empty((len(observations), 2))
    for index in torch.unique(observations[:, 1]).tolist():
        mask = observations[:, 1] == index
        keypoints = frames[index].keypoints_2d.detach().to(reference).clone()
        pixels[mask] = keypoints[observations[mask, 2]]
    return pixels


def _observation_matrices(intrinsics, frame_ids):
    matrix = camera_matrix(intrinsics)
    return matrix.expand(len(frame_ids), 3, 3) if matrix.ndim == 2 else matrix[frame_ids]


@torch.no_grad()
def reprojection_errors(frames, rotations, centers, points3d, observations, intrinsics):
    """Per-observation pixel error; nonpositive depth has infinite error."""
    if len(observations) == 0:
        return points3d.new_empty(0), torch.empty(0, dtype=torch.bool, device=points3d.device)
    point_ids, frame_ids, _ = observations.T
    local = (rotations[frame_ids] @ (points3d[point_ids] - centers[frame_ids])[..., None]).squeeze(-1)
    projected = (_observation_matrices(intrinsics, frame_ids) @ local[..., None]).squeeze(-1)
    positive = (local[:, 2] > 1e-8) & torch.isfinite(local).all(-1)
    denominator = torch.where(positive, projected[:, 2], torch.ones_like(projected[:, 2]))
    pixels = _measurements(frames, observations, points3d)
    errors = (projected[:, :2] / denominator[:, None] - pixels).norm(dim=-1)
    positive &= torch.isfinite(errors)
    return errors.masked_fill(~positive, torch.inf), positive


def _error_stats(errors):
    finite = errors[torch.isfinite(errors)]
    if len(finite) == 0:
        return {'observations': len(errors), 'finite': 0, 'rms': None, 'median': None, 'p90': None}
    return {'observations': len(errors), 'finite': len(finite),
            'rms': float(finite.square().mean().sqrt()), 'median': float(finite.median()),
            'p90': float(torch.quantile(finite, .9))}


def _compact(points, observations, keep, min_views):
    observations = observations[keep]
    if len(observations) == 0:
        return points[:0], observations.reshape(0, 3)
    counts = torch.bincount(observations[:, 0], minlength=len(points))
    observations = observations[counts[observations[:, 0]] >= min_views].clone()
    point_ids, inverse = torch.unique(observations[:, 0], sorted=True, return_inverse=True)
    observations[:, 0] = inverse
    return points[point_ids], observations


def _active_component(points, observations, num_frames, cfg):
    """Remove weak cameras/orphan tracks, then retain one connected component."""
    registered = torch.zeros(num_frames, dtype=torch.bool, device=points.device)
    while len(observations):
        counts = torch.bincount(observations[:, 1], minlength=num_frames)
        keep = counts[observations[:, 1]] >= cfg.min_observations_per_image
        if keep.all():
            break
        points, observations = _compact(points, observations, keep, cfg.min_views_per_track)
    if len(observations) == 0:
        return points[:0], observations, registered
    ordered = observations[observations[:, 0].argsort()].cpu()
    neighbors = []
    for track in ordered.split(torch.bincount(ordered[:, 0]).tolist()):
        frame_ids = track[:, 1].tolist()
        neighbors.extend((frame_ids[0], frame) for frame in frame_ids[1:])
    nodes = largest_component(num_frames, neighbors)
    registered[nodes] = True
    points, observations = _compact(points, observations, registered[observations[:, 1]], cfg.min_views_per_track)
    return points, observations, registered


@torch.no_grad()
def filter_tracks(frames, rotations, centers, points3d, observations, intrinsics, cfg,
                  max_error, max_angular_error=None):
    errors, keep = reprojection_errors(frames, rotations, centers, points3d, observations, intrinsics)
    keep &= errors <= max_error
    if len(observations) and max_angular_error is not None:
        point_ids, frame_ids, _ = observations.T
        pixels = _measurements(frames, observations, points3d)
        matrices = _observation_matrices(intrinsics, frame_ids)
        measured = F.normalize(torch.linalg.solve(matrices, homogeneous(pixels)[..., None]).squeeze(-1), dim=-1)
        local = F.normalize((rotations[frame_ids] @ (points3d[point_ids] - centers[frame_ids])[..., None]).squeeze(-1), dim=-1)
        cosine = (measured * local).sum(-1)
        keep &= cosine >= math.cos(math.radians(max_angular_error))
    points3d, observations = _compact(points3d, observations, keep, cfg.min_views_per_track)
    if len(observations) and cfg.min_tri_angle:
        # Triangulation angle uses point-to-camera rays, independent of noisy pixels.
        # Compare each camera to all observations of the same track in batches.
        point_ids, frame_ids, _ = observations.T
        directions = F.normalize(points3d[point_ids] - centers[frame_ids], dim=-1)
        counts = torch.bincount(point_ids, minlength=len(points3d))
        order = point_ids.argsort(stable=True)
        offsets = counts.cumsum(0) - counts
        slots = torch.empty_like(point_ids)
        slots[order] = torch.arange(len(order), device=order.device) - offsets[point_ids[order]]
        padded = directions.new_zeros((len(points3d), int(counts.max()), 3))
        valid = torch.zeros(padded.shape[:2], dtype=torch.bool, device=padded.device)
        padded[point_ids, slots], valid[point_ids, slots] = directions, True
        minimum = points3d.new_ones(len(points3d))
        for slot in range(padded.shape[1]):
            cosine = (padded * padded[:, slot:slot + 1]).sum(-1).abs()
            pair_valid = valid & valid[:, slot:slot + 1]
            minimum = torch.minimum(minimum, cosine.masked_fill(~pair_valid, 1).amin(-1))
        good_tracks = minimum <= math.cos(math.radians(cfg.min_tri_angle))
        points3d, observations = _compact(points3d, observations, good_tracks[point_ids], cfg.min_views_per_track)
    return _active_component(points3d, observations, len(frames), cfg)


def _reprojection_residual(rotation, center, point, intrinsics, observed):
    local = rotation[:3] @ (point - center[0])
    return _project_residual(local, intrinsics, observed)


def _pose_reprojection_residual(pose, point, intrinsics, observed):
    return _project_residual(pose[:3] @ (point - pose[3]), intrinsics, observed)


def _project_residual(local, intrinsics, observed):
    # Match COLMAP PINHOLE's HasProjectableDepth + ReprojErrorCostFunctor:
    # a failed projection contributes zero residual and zero Jacobian.
    projectable = local[2] >= torch.finfo(local.dtype).eps
    # torch.where evaluates both branches; do not divide by an invalid depth
    # even though its residual will be masked out below.
    depth = torch.where(projectable, local[2], torch.ones_like(local[2]))
    predicted = intrinsics[0, :2].exp() * local[:2] / depth + intrinsics[0, 2:]
    return torch.where(projectable, predicted - observed, torch.zeros_like(observed))


def _grouped_camera_problem(rotations, centers, rotation_mask, center_mask, points,
                            intrinsic_values, camera_ids, point_ids, intrinsic_ids,
                            pixels, cfg, batch_size, fixed_rotations):
    """Keep every observation; fixed arguments are ordinary batched data."""
    active = center_mask.bool().any(-1) | rotation_mask.bool().any(dim=(-2, -1))
    active_ids = torch.where(active)[0]
    remap = torch.full_like(active, -1, dtype=torch.long)
    remap[active_ids] = torch.arange(len(active_ids), device=active.device)
    pose_values = torch.cat((rotations, centers[:, None], rotation_mask, center_mask[:, None]), 1)
    camera_key = 'centers' if fixed_rotations else 'poses'
    camera_values = (torch.stack((centers, center_mask), 1) if fixed_rotations else pose_values)
    parameters, manifolds = {}, {}
    if len(active_ids):
        parameters[camera_key] = camera_values[active_ids]
        manifolds[camera_key] = _CenterVar if fixed_rotations else _PoseVar
    parameters['points'] = points.clone()
    active_intrinsics = bool(intrinsic_values[:, 1].bool().any())
    if active_intrinsics:
        parameters['intrinsics'] = intrinsic_values
        manifolds['intrinsics'] = _IntrinsicsVar
    problem = tls.Problem(parameters, manifolds=manifolds)
    moving = active[camera_ids]
    for is_moving in (True, False):
        rows = torch.where(moving == is_moving)[0]
        if not len(rows):
            continue
        ci, ii = camera_ids[rows], intrinsic_ids[rows]
        intrinsic_arg = ('intrinsics', ii) if active_intrinsics else intrinsic_values[ii]
        if is_moving and fixed_rotations:
            # No rotation parameter/Jacobian columns in the fixed-rotation stage.
            residual = _reprojection_residual
            args = (rotations[ci], ('centers', remap[ci]))
        else:
            residual = _pose_reprojection_residual
            args = ((('poses', remap[ci]) if is_moving else pose_values[ci]),)
        problem.add_residual(residual, *args, ('points', point_ids[rows]), intrinsic_arg,
                             pixels[rows], loss=tls.HuberLoss(delta=cfg.loss_scale),
                             batch_size=batch_size)
    return problem, camera_key, active_ids


def _principal_point_prior(intrinsics, reference, sigma):
    return (intrinsics[0, 2:] - reference) / sigma


def _principal_point_sigma(frames, nodes, intrinsics, relative_sigma):
    """Pixel standard deviations, once per shared camera or per active frame."""
    sizes = []
    for index in nodes.tolist():
        frame = frames[index]
        if frame.image is not None:
            height, width = frame.image.shape[-2:]
        elif frame.frame_path is not None:
            with Image.open(frame.frame_path) as image:
                width, height = image.size
        else:
            raise ValueError('Principal point prior requires frame images or image paths')
        sizes.append((width, height))
    sizes = intrinsics.new_tensor(sizes)
    if _pinhole_parameters(intrinsics).ndim == 1:
        sizes = sizes.mean(0, keepdim=True)
    return sizes * relative_sigma


@torch.inference_mode(False)
def _solve_ba(frames, rotations, centers, points, observations, intrinsics, cfg,
              fixed_rotations, max_iterations, principal_point_reference=None):
    original_points = points
    counts = torch.bincount(observations[:, 0], minlength=len(points))
    eligible = counts >= cfg.min_track_length
    keep = eligible[observations[:, 0]]
    if not bool(keep.any()):
        summary = tls.SolveSummary(0., 0., termination_reason='no_eligible_tracks')
        return rotations.clone(), centers.clone(), points.clone(), intrinsics.clone(), summary
    points, observations = _compact(points, observations, keep, cfg.min_track_length)
    optimized_point_ids = torch.where(eligible)[0]
    nodes = torch.unique(observations[:, 1], sorted=True)
    mapping = torch.full((len(frames),), -1, dtype=torch.long, device=points.device)
    mapping[nodes] = torch.arange(len(nodes), device=points.device)
    camera_ids = mapping[observations[:, 1]]
    current_centers = centers[nodes].clone()
    center_mask = torch.ones_like(current_centers)
    center_mask[0] = 0
    # Fix one nonzero baseline coordinate in addition to the first camera pose.
    distance = (current_centers - current_centers[0]).norm(dim=-1)
    anchor = int(distance.argmax())
    axis = int((current_centers[anchor] - current_centers[0]).abs().argmax())
    if float(distance[anchor]) < 1e-8:
        raise ValueError('BA needs a nonzero baseline to fix scale')
    center_mask[anchor, axis] = 0
    rotation_mask = torch.zeros((len(nodes), 1, 3), dtype=points.dtype, device=points.device)
    if not fixed_rotations:
        rotation_mask[1:] = 1
    params = _pinhole_parameters(intrinsics)
    shared = params.ndim == 1
    current_intrinsics = params[None].clone() if shared else params[nodes].clone()
    current_intrinsics[:, :2] = current_intrinsics[:, :2].log()
    intrinsic_mask = torch.zeros_like(current_intrinsics)
    intrinsic_mask[:, :2] = float(cfg.refine_focal_length)
    intrinsic_mask[:, 2:] = float(cfg.refine_principal_point)
    intrinsic_values = torch.stack((current_intrinsics, intrinsic_mask), 1)
    intrinsic_ids = torch.zeros_like(camera_ids) if shared else camera_ids
    batch_size = (tls.AutoBatchSize(memory_budget_bytes=int(cfg.jacobian_batch_memory_mb * 1024**2))
                  if cfg.jacobian_batch_size == 'auto' else cfg.jacobian_batch_size)
    pixels = _measurements(frames, observations, points)
    if cfg.optimizer_backend == 'bae':
        # Import only when selected: BAE registers process-wide sparse operators.
        from .bae_bundle_adjustment import solve_bae
        solved_r, solved_c, solved_p, solved_k, summary = solve_bae(
            rotations[nodes], current_centers, rotation_mask, center_mask, points,
            intrinsic_values, camera_ids, observations[:, 0], intrinsic_ids, pixels,
            cfg, max_iterations,
        )
        updated_r, updated_c = rotations.clone(), centers.clone()
        updated_r[nodes], updated_c[nodes] = solved_r, solved_c
        updated_p = original_points.clone()
        updated_p[optimized_point_ids] = solved_p
        solved_k = solved_k.clone()
        solved_k[:, :2] = solved_k[:, :2].exp()
        if shared:
            updated_k = solved_k[0]
        else:
            updated_k = params.clone()
            updated_k[nodes] = solved_k
        _check_solution(summary, 'BAE bundle adjustment')
        return updated_r, updated_c, updated_p, _restore_intrinsics(updated_k, intrinsics), summary
    grouped = cfg.fixed_camera_handling == 'grouped'
    if grouped:
        problem, _, active_camera_ids = _grouped_camera_problem(
            rotations[nodes], current_centers, rotation_mask, center_mask, points,
            intrinsic_values, camera_ids, observations[:, 0], intrinsic_ids, pixels,
            cfg, batch_size, fixed_rotations)
    else:
        if cfg.pose_parameterization == 'joint':
            parameters = {'poses': torch.cat((rotations[nodes], current_centers[:, None],
                                             rotation_mask, center_mask[:, None]), 1)}
            manifolds = {'poses': _PoseVar}
            residual_fn = _pose_reprojection_residual
            pose_args = (('poses', camera_ids),)
        else:
            parameters = {'rotations': torch.cat((rotations[nodes], rotation_mask), 1),
                          'centers': torch.stack((current_centers, center_mask), 1)}
            manifolds = {'rotations': _RotationVar, 'centers': _CenterVar}
            residual_fn = _reprojection_residual
            pose_args = (('rotations', camera_ids), ('centers', camera_ids))
        parameters.update(points=points.clone(), intrinsics=intrinsic_values)
        manifolds['intrinsics'] = _IntrinsicsVar
        problem = tls.Problem(parameters, manifolds=manifolds)
        problem.add_residual(residual_fn, *pose_args,
            ('points', observations[:, 0]), ('intrinsics', intrinsic_ids), pixels,
            loss=tls.HuberLoss(delta=cfg.loss_scale), batch_size=batch_size)
    if cfg.refine_principal_point and cfg.principal_point_prior_sigma is not None:
        reference = intrinsics if principal_point_reference is None else principal_point_reference.to(intrinsics)
        reference = _pinhole_parameters(reference)
        reference = reference[None, 2:] if shared else reference[nodes, 2:]
        sigma = _principal_point_sigma(frames, nodes, intrinsics, cfg.principal_point_prior_sigma)
        # One Gaussian prior per intrinsic block, independent of observation count.
        problem.add_residual(_principal_point_prior,
            ('intrinsics', torch.arange(len(current_intrinsics), device=points.device)), reference, sigma)
    retained_dim = problem.tangent_dim - points.numel()
    solver = cfg.linear_solver
    if solver == 'auto':
        solver = 'schur_dense' if retained_dim <= 1024 else 'schur_cg'
    cg = _ba_cg_config(cfg, solver)
    if grouped and fixed_rotations:
        cg = replace(cg, preconditioner_groups=tuple(
            tuple('centers' if key == 'poses' else key for key in group)
            for group in cg.preconditioner_groups))
    options = dict(
        linear_solver=solver, jacobi_scaling=True, return_summary=True,
        normal_blocks_backend=cfg.normal_blocks_backend,
        termination=tls.TerminationConfig(max_iterations=max_iterations, cost_tolerance=cfg.cost_tolerance,
                                         gradient_tolerance=cfg.gradient_tolerance,
                                         parameter_tolerance=cfg.parameter_tolerance),
        conjugate_gradient=cg,
        trust_region=tls.TrustRegionConfig(lambda_initial=cfg.lambda_initial, strategy='adaptive',
                                          damping_mode='diagonal', step_quality_min=cfg.step_quality_min),
    )
    if solver.startswith('schur_'):
        options['eliminate'] = 'points'
    solved, summary = problem.solve(**options)
    _check_solution(summary, 'Bundle adjustment')
    updated_rotations, updated_centers = rotations.clone(), centers.clone()
    if grouped:
        active_nodes = nodes[active_camera_ids]
        if fixed_rotations:
            updated_centers[active_nodes] = solved['centers'][:, 0]
        else:
            updated_rotations[active_nodes] = solved['poses'][:, :3]
            updated_centers[active_nodes] = solved['poses'][:, 3]
    elif cfg.pose_parameterization == 'joint':
        updated_rotations[nodes] = solved['poses'][:, :3]
        updated_centers[nodes] = solved['poses'][:, 3]
    else:
        updated_rotations[nodes] = solved['rotations'][:, :3]
        updated_centers[nodes] = solved['centers'][:, 0]
    updated_parameters = (solved['intrinsics'][:, 0].clone()
                          if bool(intrinsic_mask.bool().any()) or not grouped
                          else current_intrinsics.clone())
    updated_parameters[:, :2] = updated_parameters[:, :2].exp()
    if shared:
        output_parameters = updated_parameters[0]
    else:
        output_parameters = params.clone()
        output_parameters[nodes] = updated_parameters
    output_intrinsics = _restore_intrinsics(output_parameters, intrinsics)
    updated_points = original_points.clone()
    updated_points[optimized_point_ids] = solved['points']
    return updated_rotations, updated_centers, updated_points, output_intrinsics, summary


@torch.no_grad()
def retriangulate_tracks(frames, rotations, centers, points, observations, intrinsics, geometries, registered, cfg):
    """Robustly triangulate verified tracks, including missing two-view tracks.

    Candidates are a multi-ray fit, intersections from three anchor views and
    an existing optimized point where available. Score all candidates in pixels.
    """
    candidates_obs = build_tracks(geometries, torch.where(registered)[0].tolist(),
                                  cfg.min_views_per_track).to(points.device)
    if len(candidates_obs) == 0:
        return points, observations
    point_ids, frame_ids, keypoint_ids = candidates_obs.T
    count = int(point_ids.max()) + 1
    matrices = _observation_matrices(intrinsics, frame_ids)
    pixels = _measurements(frames, candidates_obs, centers)
    local = torch.linalg.solve(matrices, homogeneous(pixels)[..., None]).squeeze(-1)
    rays = F.normalize((rotations[frame_ids].transpose(-1, -2) @ local[..., None]).squeeze(-1), dim=-1)
    camera_centers = centers[frame_ids]
    projectors = torch.eye(3, dtype=centers.dtype, device=centers.device) - rays[:, :, None] * rays[:, None, :]
    normal = centers.new_zeros((count, 3, 3)).index_add_(0, point_ids, projectors)
    rhs = centers.new_zeros((count, 3)).index_add_(
        0, point_ids, (projectors @ camera_centers[..., None]).squeeze(-1))
    all_ray_fit = (torch.linalg.pinv(normal, rtol=1e-10) @ rhs[..., None]).squeeze(-1)
    counts = torch.bincount(point_ids, minlength=count)
    offsets = counts.cumsum(0) - counts
    candidates = [all_ray_fit]
    for anchor_ids in (offsets, offsets + counts // 2, offsets + counts - 1):
        direction0 = rays[anchor_ids]
        cosine = (rays * direction0[point_ids]).sum(-1)
        angle = 1 - cosine.abs()
        best_angle = centers.new_full((count,), -1).scatter_reduce_(0, point_ids, angle, reduce='amax')
        indices = torch.arange(len(point_ids), device=points.device)
        selected = torch.full_like(indices, len(indices))
        selected = torch.where(angle == best_angle[point_ids], indices, selected)
        partner = torch.full((count,), len(indices), dtype=torch.long, device=points.device)
        partner.scatter_reduce_(0, point_ids, selected, reduce='amin')
        direction1 = rays[partner]
        baseline = camera_centers[partner] - camera_centers[anchor_ids]
        dot = (direction0 * direction1).sum(-1)
        depth = ((baseline * direction0).sum(-1) - dot * (baseline * direction1).sum(-1))
        depth /= (1 - dot.square()).clamp_min(1e-15)
        candidates.append(camera_centers[anchor_ids] + depth[:, None] * direction0)
    # Reuse optimized points as hypotheses, preserving tracks that are already good.
    existing = centers.new_full((count, 3), torch.nan)
    if len(observations):
        stride = max(len(frame.keypoints_2d) for frame in frames) + 1
        old_keys = observations[:, 1] * stride + observations[:, 2]
        keys, order = old_keys.sort()
        new_keys = frame_ids * stride + keypoint_ids
        locations = torch.searchsorted(keys, new_keys).clamp_max(len(keys) - 1)
        matched = keys[locations] == new_keys
        source = torch.where(matched, torch.arange(len(point_ids), device=points.device), len(point_ids))
        first = torch.full((count,), len(point_ids), dtype=torch.long, device=points.device)
        first.scatter_reduce_(0, point_ids, source, reduce='amin')
        valid = first < len(point_ids)
        existing[valid] = points[observations[order[locations[first[valid]]], 0]]
    candidates.append(existing)
    hypotheses = torch.stack(candidates, 1)
    local = (rotations[frame_ids, None] @ (hypotheses[point_ids] - camera_centers[:, None])[..., None]).squeeze(-1)
    projected = (matrices[:, None] @ local[..., None]).squeeze(-1)
    errors = (projected[..., :2] / projected[..., 2:].clamp_min(1e-8) - pixels[:, None]).square().sum(-1)
    errors = errors.masked_fill((local[..., 2] <= 1e-8) | ~torch.isfinite(errors), torch.inf)
    threshold = cfg.max_reprojection_error ** 2
    good = errors <= threshold
    support = centers.new_zeros((count, len(candidates))).index_add_(0, point_ids, good.to(centers.dtype))
    costs = centers.new_zeros((count, len(candidates))).index_add_(0, point_ids, errors.clamp_max(threshold))
    costs = costs.masked_fill(support != support.amax(-1, keepdim=True), torch.inf)
    selected = costs.argmin(-1)
    keep = good.gather(1, selected[point_ids, None]).squeeze(1)
    weighted = projectors * keep[:, None, None]
    normal = centers.new_zeros((count, 3, 3)).index_add_(0, point_ids, weighted)
    rhs = centers.new_zeros((count, 3)).index_add_(
        0, point_ids, (weighted @ camera_centers[..., None]).squeeze(-1))
    refitted = (torch.linalg.pinv(normal, rtol=1e-10) @ rhs[..., None]).squeeze(-1)
    # A refit is adopted only if it improves the selected inlier set.
    chosen = hypotheses[torch.arange(count, device=points.device), selected]
    for_fit = candidates_obs[keep]
    old_error, _ = reprojection_errors(frames, rotations, centers, chosen, for_fit, intrinsics)
    new_error, _ = reprojection_errors(frames, rotations, centers, refitted, for_fit, intrinsics)
    old_cost = centers.new_zeros(count).index_add_(0, for_fit[:, 0], old_error.square())
    new_cost = centers.new_zeros(count).index_add_(0, for_fit[:, 0], new_error.square())
    improved = torch.isfinite(refitted).all(-1) & (new_cost <= old_cost)
    chosen = torch.where(improved[:, None], refitted, chosen)
    return _compact(chosen, candidates_obs, keep, cfg.min_views_per_track)


@torch.inference_mode(False)
def legacy_bundle_adjustment(frames, rotations, centers, points3d, observations, intrinsics,
                      registered, cfg=None, geometries=None):
    cfg = BundleAdjustmentConfig.legacy() if cfg is None else cfg
    rotations = rotations.detach().clone()
    centers = centers.detach().to(rotations).clone()
    points = points3d.detach().to(rotations).clone()
    observations = observations.detach().to(device=rotations.device, dtype=torch.long).clone()
    intrinsics = intrinsics.detach().to(rotations).clone()
    initial_intrinsics = intrinsics.clone()
    registered = registered.to(device=rotations.device).clone()
    original_errors, _ = reprojection_errors(frames, rotations, centers, points, observations, intrinsics)
    stats = {'enabled': cfg.enabled, 'initial': _error_stats(original_errors), 'stages': []}
    if not cfg.enabled or len(observations) == 0:
        stats['skipped'] = 'disabled' if not cfg.enabled else 'no_tracks'
        stats['final'] = stats['initial']
        return BundleAdjustmentResult(rotations, centers, points, observations, registered,
                                       intrinsics, original_errors, stats=stats)
    if rotations.shape != (len(frames), 3, 3) or centers.shape != (len(frames), 3):
        raise ValueError('BA expects one rotation and center per frame')
    _pinhole_parameters(intrinsics)
    keep = registered[observations[:, 1]]
    points, observations = _compact(points, observations, keep, cfg.min_views_per_track)
    points, observations, registered = filter_tracks(frames, rotations, centers, points, observations,
        intrinsics, cfg, cfg.initial_max_reprojection_error, cfg.initial_max_angular_error)
    summaries = []
    if len(observations) == 0:
        raise ValueError('No tracks remain after pre-BA geometry filtering')

    def solve_stage(name, max_iterations, fixed, refine_principal_point=False):
        nonlocal rotations, centers, points, intrinsics
        before, _ = reprojection_errors(frames, rotations, centers, points, observations, intrinsics)
        stage_cfg = replace(cfg, refine_principal_point=refine_principal_point)
        initial_parameters = _pinhole_parameters(intrinsics).tolist()
        rotations, centers, points, intrinsics, summary = _solve_ba(
            frames, rotations, centers, points, observations, intrinsics, stage_cfg, fixed, max_iterations,
            principal_point_reference=initial_intrinsics)
        after, _ = reprojection_errors(frames, rotations, centers, points, observations, intrinsics)
        summaries.append(summary)
        stats['stages'].append({'name': name, 'points': len(points), 'before': _error_stats(before),
                                'after': _error_stats(after), 'solver': asdict(summary),
                                'refine_principal_point': refine_principal_point,
                                'intrinsics_before': initial_parameters,
                                'intrinsics_after': _pinhole_parameters(intrinsics).tolist()})

    for iteration in range(cfg.num_rounds):
        if cfg.fixed_rotation_iterations:
            solve_stage(f'round_{iteration + 1}_fixed_rotation', cfg.fixed_rotation_iterations, True)
        solve_stage(f'round_{iteration + 1}_joint', cfg.max_iterations, False)
        threshold = max(cfg.num_rounds - iteration, 1) * cfg.max_reprojection_error
        points, observations, registered = filter_tracks(frames, rotations, centers, points, observations,
            intrinsics, cfg, threshold)
        if len(observations) == 0:
            raise RuntimeError('No tracks remain after bundle adjustment')

    if cfg.retriangulate and geometries:
        before_count = len(points)
        points, observations = retriangulate_tracks(frames, rotations, centers, points, observations,
            intrinsics, geometries, registered, cfg)
        points, observations, registered = filter_tracks(frames, rotations, centers, points, observations,
            intrinsics, cfg, cfg.max_reprojection_error)
        stats['retriangulation'] = {'points_before': before_count, 'points_after': len(points)}
        if len(observations):
            solve_stage('retriangulated_joint', cfg.max_iterations, False)

    if cfg.refine_principal_point and len(observations):
        solve_stage('final_principal_point_joint', cfg.max_iterations, False, True)
        nodes = torch.unique(observations[:, 1], sorted=True)
        reference = _pinhole_parameters(initial_intrinsics)
        final = _pinhole_parameters(intrinsics)
        if reference.ndim > 1:
            reference, final = reference[nodes], final[nodes]
        sigma = None if cfg.principal_point_prior_sigma is None else _principal_point_sigma(
            frames, nodes, intrinsics, cfg.principal_point_prior_sigma).tolist()
        stats['principal_point'] = {
            'reference': reference[..., 2:].tolist(), 'final': final[..., 2:].tolist(),
            'shift_pixels': (final[..., 2:] - reference[..., 2:]).tolist(),
            'prior_relative_sigma': cfg.principal_point_prior_sigma, 'prior_sigma_pixels': sigma,
        }

    points, observations, registered = filter_tracks(frames, rotations, centers, points, observations,
        intrinsics, cfg, cfg.max_reprojection_error)
    if len(observations) == 0:
        raise RuntimeError('No tracks remain after final BA filtering')
    centers[~registered] = torch.nan
    errors, positive = reprojection_errors(frames, rotations, centers, points, observations, intrinsics)
    stats['final'] = _error_stats(errors)
    stats['registered'] = int(registered.sum())
    stats['positive_depth'] = int(positive.sum())
    return BundleAdjustmentResult(rotations, centers, points, observations, registered,
                                  intrinsics, errors, summaries, stats)


@torch.inference_mode(False)
def bundle_adjustment(frames, rotations, centers, points3d, observations, intrinsics,
                      registered, cfg=None, geometries=None):
    cfg = BundleAdjustmentConfig() if cfg is None else cfg
    if cfg.strategy == 'legacy':
        return legacy_bundle_adjustment(frames, rotations, centers, points3d, observations,
                                        intrinsics, registered, cfg, geometries)
    from .global_bundle_adjustment import global_bundle_adjustment
    return global_bundle_adjustment(frames, rotations, centers, points3d, observations,
                                     intrinsics, registered, cfg, geometries)
