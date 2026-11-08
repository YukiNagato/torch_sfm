"""Pinhole two-view verification using Torch and optional Warp LO-RANSAC.

Conventions follow COLMAP: x2.T E x1 = 0 and X2 = R X1 + t.
Five-point estimation uses the essential variety's cubic constraints. The
reference uses a 10x10 action matrix; CUDA can use polynomial elimination and
Sturm real roots. No external geometry solver is called.
"""

from dataclasses import dataclass
from itertools import product
import math

import torch
import torch.nn.functional as F

from ..correspondence import Matches


@dataclass
class TwoViewConfig:
    max_error: float = 4.0  # pixels, for all three competing models
    confidence: float = 0.999
    min_trials: int = 100
    max_trials: int = 10000
    batch_size: int = 240  # hypotheses per pair; measured on RTX 4090 D (issues/009)
    pair_batch_size: int = 64
    backend: str = 'auto'  # Warp CUDA geometry kernels, otherwise Torch
    pose_cuda_graph: bool = False  # opt in for repeatedly reused shapes; capture has startup cost
    min_num_inliers: int = 15
    min_inlier_ratio: float = 0.0
    min_E_F_inlier_ratio: float = 0.95
    max_H_inlier_ratio: float = 0.8
    min_tri_angle: float = 1.0  # degrees; translation edges need parallax
    refine_relative_pose: bool = True
    detect_watermark: bool = True
    watermark_min_inlier_ratio: float = 0.7
    watermark_border_size: float = 0.1
    seed: int = 0
    calibration: str = 'calibrated'  # uncalibrated estimates F/H without using K
    uncalibrated_model: str = 'fundamental'  # or shared_focal: six-point RANSAC
    shared_focal_cuda_graph: bool = True
    essential_cheirality: bool = True  # reject behind-camera points during E hypothesis scoring
    shared_focal_cpu_workers: int = 0  # 0: up to 8 allocated CPUs; 1: serial LAPACK batch
    shared_focal_refinement_backend: str = 'auto'  # fused Warp LM on CUDA float64, Torch otherwise
    shared_focal_local_batch_size: int = 512  # selected LO hypotheses per refinement chunk
    shared_focal_pair_batch_size: int = 256  # 0 uses pair_batch_size; independent of trials per pair
    shared_focal_max_padded_matches: int = 524288  # pipeline batching memory bound, pairs * max matches
    shared_focal_root_backend: str = 'auto'  # auto/warp: guarded QR + LAPACK; warp_poselib: full Warp solver, no fallback
    shared_focal_frontend: str = 'auto'  # new Warp controller for CUDA warp_poselib; legacy keeps the old controller
    five_point_backend: str = 'auto'  # warp_poselib: full Warp minimal solver + all-five cheirality
    calibrated_frontend: str = 'auto'  # fixed-slot Warp E/F/H with warp_poselib; legacy keeps the old controller

    @property
    def verification_pair_batch_size(self):
        if self.calibration=='uncalibrated' and self.uncalibrated_model=='shared_focal':
            return self.shared_focal_pair_batch_size or self.pair_batch_size
        return self.pair_batch_size

    def __post_init__(self):
        if self.calibrated_frontend not in ('auto','legacy','warp'):
            raise ValueError('calibrated_frontend must be auto, legacy or warp')
        if self.calibrated_frontend=='warp' and (self.backend=='torch' or not self.essential_cheirality):
            raise ValueError('Warp calibrated frontend requires Warp kernels and essential cheirality')
        if self.shared_focal_frontend not in ('auto','legacy','warp'):
            raise ValueError('shared_focal_frontend must be auto, legacy or warp')
        if self.shared_focal_frontend=='warp' and (self.backend=='torch' or self.shared_focal_root_backend!='warp_poselib'
                                                   or self.shared_focal_refinement_backend=='torch'):
            raise ValueError('Warp frontend requires warp_poselib, a CUDA-capable backend and Warp refinement')
        if self.five_point_backend not in ('auto','warp_poselib'):
            raise ValueError('five_point_backend must be auto or warp_poselib')
        if self.five_point_backend == 'warp_poselib' and self.backend == 'torch':
            raise ValueError('warp_poselib cannot be combined with backend="torch"')
        if self.shared_focal_root_backend not in ('auto','cpu','warp','warp_poselib'):
            raise ValueError('shared_focal_root_backend must be auto, cpu, warp or warp_poselib')
        if self.shared_focal_root_backend == 'warp_poselib' and self.backend == 'torch':
            raise ValueError('warp_poselib cannot be combined with backend="torch"')
        if self.calibration not in ('calibrated', 'uncalibrated'):
            raise ValueError('Two-view calibration must be calibrated or uncalibrated')
        if self.uncalibrated_model not in ('fundamental','shared_focal'):
            raise ValueError('Unknown uncalibrated epipolar model')
        if not isinstance(self.shared_focal_cpu_workers,int) or self.shared_focal_cpu_workers<0:
            raise ValueError('shared_focal_cpu_workers must be a nonnegative integer')
        if self.shared_focal_refinement_backend not in ('auto','torch','warp','warp_fused'):
            raise ValueError('shared_focal_refinement_backend must be auto, torch, warp or warp_fused')
        if type(self.shared_focal_local_batch_size) is not int or self.shared_focal_local_batch_size<1:
            raise ValueError('shared_focal_local_batch_size must be a positive integer')
        if not isinstance(self.shared_focal_pair_batch_size,int) or self.shared_focal_pair_batch_size<0:
            raise ValueError('shared_focal_pair_batch_size must be a nonnegative integer')
        if not isinstance(self.shared_focal_max_padded_matches,int) or self.shared_focal_max_padded_matches<1:
            raise ValueError('shared_focal_max_padded_matches must be a positive integer')
        if self.max_error <= 0 or not math.isfinite(self.max_error):
            raise ValueError('max_error must be finite and positive')
        if not 0 < self.confidence < 1 or not 0 < self.min_trials <= self.max_trials:
            raise ValueError('Invalid RANSAC confidence or trial limits')
        if self.batch_size < 1 or self.pair_batch_size < 1 or self.min_num_inliers < 5:
            raise ValueError('batch_size must be positive and min_num_inliers >= 5')
        if self.backend not in ('auto', 'torch', 'warp'):
            raise ValueError('Two-view backend must be auto, torch or warp')
        for ratio in (self.min_inlier_ratio, self.min_E_F_inlier_ratio,
                      self.max_H_inlier_ratio, self.watermark_min_inlier_ratio):
            if not 0 <= ratio <= 1:
                raise ValueError('Inlier ratios must lie in [0, 1]')
        if not 0 <= self.min_tri_angle < 180:
            raise ValueError('min_tri_angle must lie in [0, 180)')


@dataclass
class TwoViewGeometry:
    matches: Matches
    config: str = 'DEGENERATE'
    R: torch.Tensor | None = None
    t: torch.Tensor | None = None
    E: torch.Tensor | None = None
    F: torch.Tensor | None = None
    H: torch.Tensor | None = None
    tri_angle: float = 0.0
    valid_translation: bool = False


def camera_matrix(intrinsics: torch.Tensor) -> torch.Tensor:
    """Accept (..., 4) pinhole parameters or (..., 3, 3) camera matrices."""
    if intrinsics.shape[-2:] == (3, 3):
        matrix = intrinsics
    elif intrinsics.shape[-1:] == (4,):
        fx, fy, cx, cy = intrinsics.unbind(-1)
        zero, one = torch.zeros_like(fx), torch.ones_like(fx)
        matrix = torch.stack((fx, zero, cx, zero, fy, cy, zero, zero, one), -1)
        matrix = matrix.reshape(*intrinsics.shape[:-1], 3, 3)
    else:
        raise ValueError('Use pinhole intrinsics (..., 4) or (..., 3, 3)')
    if (not torch.isfinite(matrix).all() or (matrix[..., 0, 0] <= 0).any()
            or (matrix[..., 1, 1] <= 0).any()
            or (torch.linalg.det(matrix).abs() < 1e-12).any()):
        raise ValueError('Camera intrinsics must be finite and nonsingular, with positive focal lengths')
    return matrix


def homogeneous(points):
    return torch.cat((points, torch.ones_like(points[..., :1])), -1)


def normalize_points(points):
    center = points.mean(-2, keepdim=True)
    scale = math.sqrt(2) / (points - center).norm(dim=-1).mean(-1).clamp_min(1e-12)
    transform = torch.eye(3, dtype=points.dtype, device=points.device).expand(
        *points.shape[:-2], 3, 3).clone()
    transform[..., 0, 0] = transform[..., 1, 1] = scale
    transform[..., :2, 2] = -center.squeeze(-2) * scale[..., None]
    return (points - center) * scale[..., None, None], transform


def _epipolar_design(points0, points1):
    return (homogeneous(points1)[..., :, None] * homogeneous(points0)[..., None, :]).flatten(-2)


def _project_essential(matrix):
    u, s, vh = torch.linalg.svd(matrix)
    mean = (s[..., 0] + s[..., 1]) / 2
    values = torch.stack((mean, mean, torch.zeros_like(mean)), -1)
    return (u * values[..., None, :]) @ vh


# Cubic monomials first, followed by a basis for the quotient ring.
_MONOMIALS = [(3, 0, 0), (2, 1, 0), (1, 2, 0), (0, 3, 0),
              (2, 0, 1), (1, 1, 1), (0, 2, 1), (1, 0, 2), (0, 1, 2), (0, 0, 3),
              (2, 0, 0), (1, 1, 0), (0, 2, 0), (1, 0, 1), (0, 1, 1), (0, 0, 2),
              (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)]
_LINEAR = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)]
_CUBIC_INDICES = [_MONOMIALS.index(tuple(sum(v[d] for v in terms) for d in range(3)))
                  for terms in product(_LINEAR, repeat=3)]


def essential_five_point(points0, points1, return_owners=False, backend='torch', *, solver_backend='auto'):
    """Batched minimal solver; return all finite, real essential matrices."""
    if solver_backend == 'warp_poselib':
        if backend == 'torch':
            raise ValueError('warp_poselib cannot be combined with backend="torch"')
        from .warp_five_point_poselib import essential_five_point as solve
        return solve(points0,points1,return_owners=return_owners)
    if solver_backend != 'auto':
        raise ValueError('solver_backend must be auto or warp_poselib')
    design = _epipolar_design(points0, points1)
    _, singular, vh = torch.linalg.svd(design, full_matrices=True)
    basis = vh[:, -4:].transpose(-1, -2).reshape(-1, 3, 3, 4)
    # 2 E E^T E - trace(E E^T) E = 0, plus det(E) = 0.
    eet = torch.einsum('bikp,blkq->bilpq', basis, basis)
    trace = torch.einsum('bikp,bikq->bpq', basis, basis)
    cubics = (2 * torch.einsum('bilpq,bljr->bijpqr', eet, basis)
              - torch.einsum('bpq,bijr->bijpqr', trace, basis))
    determinant = basis.new_zeros((len(basis), 4, 4, 4))
    for cols, sign in (((0, 1, 2), 1), ((1, 2, 0), 1), ((2, 0, 1), 1),
                       ((0, 2, 1), -1), ((1, 0, 2), -1), ((2, 1, 0), -1)):
        determinant += sign * torch.einsum('bp,bq,br->bpqr',
            basis[:, 0, cols[0]], basis[:, 1, cols[1]], basis[:, 2, cols[2]])
    raw = torch.cat((cubics.reshape(-1, 9, 64), determinant.reshape(-1, 1, 64)), 1)
    indices = torch.tensor(_CUBIC_INDICES, device=raw.device).expand_as(raw)
    coefficients = raw.new_zeros((len(raw), 10, 20)).scatter_add_(2, indices, raw)
    if points0.is_cuda and points0.dtype == torch.float64 and backend != 'torch':
        try:
            from .five_point import essential_from_constraints
            from . import warp_roots
        except ImportError:
            if backend == 'warp':
                raise
        else:
            return essential_from_constraints(points0, points1, coefficients, basis, singular, return_owners)
    reduction, info = torch.linalg.solve_ex(coefficients[:, :, :10], -coefficients[:, :, 10:])
    valid = ((info == 0) & (singular[:, -1] > 1e-10 * singular[:, 0])
             & torch.isfinite(reduction).all(dim=(-2, -1)))
    reduction = reduction[valid]
    basis = basis[valid]
    owners = torch.arange(len(points0), device=points0.device)[valid]
    if len(basis) == 0:
        models = points0.new_empty((0, 3, 3))
        return (models, owners) if return_owners else models
    action = reduction.new_zeros((len(reduction), 10, 10))
    action[:, :6] = reduction[:, 4:10]
    action[:, 6, 3] = action[:, 7, 4] = action[:, 8, 5] = action[:, 9, 8] = 1
    eigenvalues = torch.linalg.eigvals(action)
    valid = eigenvalues.imag.abs() < 1e-7 * (1 + eigenvalues.real.abs())
    sample_ids = torch.arange(len(action), device=action.device)[:, None].expand_as(valid)[valid]
    roots = eigenvalues.real[valid]
    # Only the last four eigenvector entries are needed. Fix v[9]=1 and
    # reconstruct via a batched 9x9 solve, avoiding general eigenvectors.
    selected = action[sample_ids]
    eye = torch.eye(9, dtype=action.dtype, device=action.device)
    vectors, info = torch.linalg.solve_ex(selected[:, :9, :9] - roots[:, None, None] * eye,
                                         -selected[:, :9, 9])
    coordinates = torch.cat((vectors[:, 6:9], torch.ones_like(vectors[:, :1])), -1)
    models = torch.einsum('bijp,bp->bij', basis[sample_ids], coordinates)
    owners = owners[sample_ids]
    models = F.normalize(models.flatten(-2), dim=-1).reshape(-1, 3, 3)
    eet = models @ models.transpose(-1, -2)
    error = 2 * eet @ models - eet.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None] * models
    valid = (info == 0) & torch.isfinite(models).all(dim=(-2, -1)) & (error.norm(dim=(-2, -1)) < 1e-5)
    return (models[valid], owners[valid]) if return_owners else models[valid]


def _cubic_real_roots(coefficients):
    """Three root slots with validity, without a general 3x3 eigensolver."""
    c, b, a = (coefficients[:, :3] / coefficients[:, 3:]).unbind(-1)
    p, q = b - a.square() / 3, 2 * a.pow(3) / 27 - a * b / 3 + c
    discriminant = (q / 2).square() + (p / 3).pow(3)
    multiple = discriminant.abs() <= 1e-14 * (
        (q / 2).square() + (p / 3).pow(3).abs()).clamp_min(1e-30)
    value = -q / 2 - torch.copysign(discriminant.clamp_min(0).sqrt(), q)
    u = value.sign() * value.abs().pow(1 / 3)
    v = torch.where(u.abs() > 1e-15, -p / (3 * u), torch.zeros_like(u))
    one_root = u + v - a / 3
    radius = (-p / 3).clamp_min(1e-30).sqrt()
    angle = torch.acos((-q / (2 * radius.pow(3))).clamp(-1, 1)) / 3
    offsets = coefficients.new_tensor([0, 2 * math.pi / 3, 4 * math.pi / 3])
    roots = 2 * radius[:, None] * torch.cos(angle[:, None] - offsets) - a[:, None] / 3
    roots = torch.where((discriminant >= 0)[:, None], one_root[:, None], roots)
    repeated = (-u - a / 3)[:, None].expand_as(roots)
    slots = torch.arange(3, device=roots.device)[None]
    roots = torch.where(multiple[:, None] & (slots > 0), repeated, roots)
    valid = (discriminant < 0)[:, None] | (slots == 0) | multiple[:, None]
    for _ in range(2):
        value = ((roots + a[:, None]) * roots + b[:, None]) * roots + c[:, None]
        derivative = (3 * roots + 2 * a[:, None]) * roots + b[:, None]
        roots = roots - torch.where(derivative.abs() > 1e-15, value / derivative, 0)
    return roots, valid & torch.isfinite(roots)


def _seven_point_coefficients(basis):
    a0, a1, a2 = basis[:, 0].unbind(-2)
    b0, b1, b2 = basis[:, 1].unbind(-2)
    cross_aa, cross_bb = torch.linalg.cross(a1, a2), torch.linalg.cross(b1, b2)
    cross_ab = torch.linalg.cross(a1, b2) + torch.linalg.cross(b1, a2)
    return torch.stack(((a0 * cross_aa).sum(-1),
        (b0 * cross_aa + a0 * cross_ab).sum(-1),
        (a0 * cross_bb + b0 * cross_ab).sum(-1), (b0 * cross_bb).sum(-1)), -1)


def fundamental_seven_point(points0, points1, return_owners=False, backend='torch'):
    if points0.is_cuda and points0.dtype == torch.float64 and points0.shape[-2] == 7 and backend != 'torch':
        try:
            from .warp_seven_point import fundamental_seven_point as fused_seven_point
        except ImportError:
            if backend == 'warp':
                raise
        else:
            return fused_seven_point(points0, points1, return_owners)
    normalized0, transform0 = normalize_points(points0)
    normalized1, transform1 = normalize_points(points1)
    design = _epipolar_design(normalized0, normalized1)
    _, singular, vh = torch.linalg.svd(design, full_matrices=design.shape[-2] < 9)
    basis = vh[:, -2:].reshape(-1, 2, 3, 3)
    rank = singular[:, -1] > 1e-10 * singular[:, 0]
    coefficients = _seven_point_coefficients(basis)
    valid = (coefficients[:, 3].abs() > 1e-12) & rank
    coefficients, basis = coefficients[valid], basis[valid]
    owners = torch.arange(len(points0), device=points0.device)[valid]
    if len(basis) == 0:
        models = points0.new_empty((0, 3, 3))
        return (models, owners) if return_owners else models
    roots, real = _cubic_real_roots(coefficients)
    models = basis[:, None, 0] + roots[..., None, None] * basis[:, None, 1]
    models = transform1[valid, None].transpose(-1, -2) @ models @ transform0[valid, None]
    models = F.normalize(models[real].flatten(-2), dim=-1).reshape(-1, 3, 3)
    return (models, owners[:, None].expand_as(real)[real]) if return_owners else models


def epipolar_eight_point(points0, points1, essential=False):
    normalized0, transform0 = normalize_points(points0)
    normalized1, transform1 = normalize_points(points1)
    design = _epipolar_design(normalized0, normalized1)
    _, singular, vh = torch.linalg.svd(design, full_matrices=design.shape[-2] < 9)
    matrix = vh[..., -1, :].reshape(-1, 3, 3)
    u, s, vh = torch.linalg.svd(matrix)
    s[..., -1] = 0
    matrix = transform1.transpose(-1, -2) @ ((u * s[..., None, :]) @ vh) @ transform0
    if essential:
        matrix = _project_essential(matrix)
    return matrix[singular[..., -2] > 1e-10 * singular[..., 0]]


def homography_dlt(points0, points1, return_owners=False, backend='torch'):
    if points0.is_cuda and points0.dtype == torch.float64 and points0.shape[-2] == 4 and backend != 'torch':
        try:
            from .warp_homography import homography_dlt as fused_homography
        except ImportError:
            if backend == 'warp':
                raise
        else:
            return fused_homography(points0, points1, return_owners)
    normalized0, transform0 = normalize_points(points0)
    normalized1, transform1 = normalize_points(points1)
    xy1 = homogeneous(normalized0)
    zero = torch.zeros_like(xy1)
    row0 = torch.cat((-xy1, zero, normalized1[..., :1] * xy1), -1)
    row1 = torch.cat((zero, -xy1, normalized1[..., 1:] * xy1), -1)
    design = torch.stack((row0, row1), -2).flatten(-3, -2)
    _, singular, vh = torch.linalg.svd(design, full_matrices=design.shape[-2] < 9)
    vector = vh[..., -1, :]
    rank = singular[..., 7] > 1e-10 * singular[..., 0]
    matrix = torch.linalg.solve(transform1, vector.reshape(-1, 3, 3) @ transform0)
    valid = rank & (torch.linalg.det(matrix).abs() > 1e-12)
    return (matrix[valid], torch.arange(len(points0), device=points0.device)[valid]) if return_owners else matrix[valid]


def sampson_error(models, points0, points1):
    x0, x1 = homogeneous(points0), homogeneous(points1)
    lines1 = torch.einsum('bij,nj->bni', models, x0)
    lines0 = torch.einsum('bji,nj->bni', models, x1)
    numerator = (x1 * lines1).sum(-1).square()
    denominator = lines0[..., :2].square().sum(-1) + lines1[..., :2].square().sum(-1)
    return numerator / denominator.clamp_min(1e-30)


def homography_error(models, points0, points1):
    projected = torch.einsum('bij,nj->bni', models, homogeneous(points0))
    valid = projected[..., 2].abs() > 1e-12
    denominator = torch.where(valid, projected[..., 2], torch.ones_like(projected[..., 2]))
    error = (projected[..., :2] / denominator[..., None] - points1).square().sum(-1)
    return error.masked_fill(~valid, torch.inf)


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


def triangulate_relative(rotation, translation, rays0, rays1):
    """Closest intersection of two unit rays; positive depths in both views."""
    direction0 = F.normalize(rays0, dim=-1)
    direction1 = F.normalize(rays1, dim=-1) @ rotation
    center1 = -rotation.T @ translation
    cosine = (direction0 * direction1).sum(-1)
    denominator = (1 - cosine.square()).clamp_min(1e-15)
    d0 = (direction0 * center1).sum(-1)
    d1 = (direction1 * center1).sum(-1)
    depth0 = (d0 - cosine * d1) / denominator
    depth1 = (cosine * d0 - d1) / denominator
    points = (depth0[:, None] * direction0 + center1 + depth1[:, None] * direction1) / 2
    positive = (depth0 > 1e-8) & (depth1 > 1e-8) & torch.isfinite(points).all(-1)
    angles = torch.acos(cosine.clamp(-1, 1)) * (180 / math.pi)
    return points, positive, angles


def essential_poses(essential):
    u, _, vh = torch.linalg.svd(essential)
    u = u.clone()
    vh = vh.clone()
    u[:, -1] *= torch.linalg.det(u)
    vh[-1] *= torch.linalg.det(vh)
    w = essential.new_tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    return [(u @ middle @ vh, sign * u[:, 2]) for middle in (w, w.T) for sign in (1, -1)]


def homography_poses(homography, matrix0, matrix1):
    normalized = torch.linalg.solve(matrix1, homography @ matrix0)
    if torch.linalg.det(normalized) < 0:
        normalized = -normalized
    u, s, vh = torch.linalg.svd(normalized)
    normalized, s = normalized / s[1], s / s[1]
    if (s - 1).abs().max() < 1e-3:
        return [(u @ vh, s.new_zeros(3))]
    a, b = s[0], s[2]
    cosine = (1 + a * b) / (a + b)
    sine = ((a.square() - 1).clamp_min(0) * (1 - b.square()).clamp_min(0)).sqrt() / (a + b)
    poses = []
    for sign in (1, -1):
        middle = torch.eye(3, dtype=s.dtype, device=s.device)
        middle[0, 0] = middle[2, 2] = cosine
        middle[0, 2], middle[2, 0] = sign * sine, -sign * sine
        delta = torch.diag(s) - middle
        tu, ts, _ = torch.linalg.svd(delta)
        translation = u @ (tu[:, 0] * ts[0])
        rotation = u @ middle @ vh
        poses.extend(((rotation, F.normalize(translation, dim=0)),
                      (rotation, -F.normalize(translation, dim=0))))
    return poses


def _choose_pose(poses, rays0, rays1):
    best, best_count, best_error = None, -1, math.inf
    for rotation, translation in poses:
        if translation.norm() < 1e-8:
            predicted = F.normalize(rays0 @ rotation.T, dim=-1)
            error = (predicted - F.normalize(rays1, dim=-1)).norm(dim=-1)
            return rotation, translation, torch.ones_like(error, dtype=torch.bool), 0.0
        points, positive, angles = triangulate_relative(rotation, translation, rays0, rays1)
        error = (F.normalize(points, dim=-1) - F.normalize(rays0, dim=-1)).square().sum(-1)
        count, score = int(positive.sum()), float(error[positive].sum())
        if count > best_count or (count == best_count and score < best_error):
            angle = float(angles[positive].median()) if count else 0.0
            best = rotation, translation, positive, angle
            best_count, best_error = count, score
    return best


@torch.inference_mode(False)
def _refine_pose(rotation, translation, pixels0, pixels1, matrix0, matrix1, use_cuda_graph=True, backend='auto'):
    from .relative_pose import refine_relative_pose
    rotations, translations = refine_relative_pose(rotation[None], translation[None],
        pixels0[None], pixels1[None], matrix0[None], matrix1[None], use_cuda_graph=use_cuda_graph, backend=backend)
    return rotations[0], translations[0]


def _watermark(points0, points1, image_sizes, cfg):
    if image_sizes is None or len(points0) == 0:
        return False
    border = torch.ones(len(points0), dtype=torch.bool, device=points0.device)
    for points, (height, width) in zip((points0, points1), image_sizes):
        margin = math.hypot(height, width) * cfg.watermark_border_size
        edge = torch.minimum(points, points.new_tensor([width, height]) - points).amin(-1)
        border &= edge < margin
    if float(border.float().mean()) < cfg.watermark_min_inlier_ratio:
        return False
    displacement = points1 - points0
    # Translation-only hypotheses sampled from the inlier border region.
    candidates = displacement[border][::max(1, int(border.sum()) // 128)]
    errors = (displacement[None] - candidates[:, None]).square().sum(-1)
    support = ((errors <= cfg.max_error ** 2) & border[None]).sum(-1).max()
    return int(support) >= cfg.watermark_min_inlier_ratio * len(points0)


@torch.inference_mode(False)
def verify_two_view(keypoints0, keypoints1, matches, intrinsics0, intrinsics1,
                    cfg=None, image_sizes=None):
    """Verify a calibrated pinhole pair, retaining model classification on failure.

    Numeric inputs must share a float64/float32 device. Returned match indices
    index the original keypoint arrays. F-supported pairs remain UNCALIBRATED;
    in calibrated mode their pose is recovered from F and the supplied K.
    """
    cfg = TwoViewConfig() if cfg is None else cfg
    if cfg.calibration == 'uncalibrated':
        from .batched_two_view import verify_two_view_batch
        return verify_two_view_batch([keypoints0], [keypoints1], [matches],
            intrinsics0, intrinsics1, cfg, None if image_sizes is None else [image_sizes])[0]
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
                torch.ones((1,len(points0)),device=device,dtype=torch.bool),cfg.backend)
        return sampson_error(inverse1.T @ models @ inverse0, points0, points1)
    essential_estimator = essential_five_point
    if cfg.five_point_backend != 'auto':
        def essential_estimator(a,b):
            return essential_five_point(a,b,backend=cfg.backend,solver_backend=cfg.five_point_backend)
    essential, mask_e = _ransac(normalized0, normalized1, essential_estimator,
        lambda a, b: epipolar_eight_point(a, b, essential=True),
        essential_error, 5, cfg, generator)
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
                matrix0, matrix1, cfg.pose_cuda_graph, cfg.backend)
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
    if result.config == 'UNCALIBRATED':
        essential = matrix1.T @ fundamental @ matrix0
    poses = (homography_poses(homography, matrix0, matrix1) if result.config == 'PLANAR'
             else essential_poses(essential))
    rotation, translation, positive, angle = _choose_pose(poses, rays0[mask], rays1[mask])
    if result.config == 'CALIBRATED' and cfg.refine_relative_pose and int(positive.sum()) >= minimum:
        rotation, translation = _refine_pose(rotation, translation, points0[mask][positive],
            points1[mask][positive], matrix0, matrix1, cfg.pose_cuda_graph, cfg.backend)
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
