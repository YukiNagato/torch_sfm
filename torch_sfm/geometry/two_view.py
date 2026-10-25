"""Pinhole two-view verification using tensor-only LO-RANSAC.

Conventions follow COLMAP: x2.T E x1 = 0 and X2 = R X1 + t.
Five-point estimation uses the essential variety's cubic constraints and a
10-by-10 action matrix; no external geometry solver is used.
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
    batch_size: int = 32
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

    def __post_init__(self):
        if self.max_error <= 0 or not math.isfinite(self.max_error):
            raise ValueError('max_error must be finite and positive')
        if not 0 < self.confidence < 1 or not 0 < self.min_trials <= self.max_trials:
            raise ValueError('Invalid RANSAC confidence or trial limits')
        if self.batch_size < 1 or self.min_num_inliers < 5:
            raise ValueError('batch_size must be positive and min_num_inliers >= 5')
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


def essential_five_point(points0, points1):
    """Batched minimal solver; return all finite, real essential matrices."""
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
    reduction, info = torch.linalg.solve_ex(coefficients[:, :, :10], -coefficients[:, :, 10:])
    valid = ((info == 0) & (singular[:, -1] > 1e-10 * singular[:, 0])
             & torch.isfinite(reduction).all(dim=(-2, -1)))
    reduction = reduction[valid]
    basis = basis[valid]
    if len(basis) == 0:
        return points0.new_empty((0, 3, 3))
    action = reduction.new_zeros((len(reduction), 10, 10))
    action[:, :6] = reduction[:, 4:10]
    action[:, 6, 3] = action[:, 7, 4] = action[:, 8, 5] = action[:, 9, 8] = 1
    eigenvalues, eigenvectors = torch.linalg.eig(action)
    vectors = eigenvectors.transpose(-1, -2)
    valid = ((eigenvalues.imag.abs() < 1e-7 * (1 + eigenvalues.real.abs()))
             & (vectors[..., 9].abs() > 1e-10))
    xyz = (vectors[..., 6:9] / vectors[..., 9:10]).real
    coordinates = torch.cat((xyz, torch.ones_like(xyz[..., :1])), -1)
    models = torch.einsum('bijp,bkp->bkij', basis, coordinates)[valid]
    models = F.normalize(models.flatten(-2), dim=-1).reshape(-1, 3, 3)
    eet = models @ models.transpose(-1, -2)
    error = 2 * eet @ models - eet.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None] * models
    return models[torch.isfinite(models).all(dim=(-2, -1)) & (error.norm(dim=(-2, -1)) < 1e-5)]


def fundamental_seven_point(points0, points1):
    normalized0, transform0 = normalize_points(points0)
    normalized1, transform1 = normalize_points(points1)
    design = _epipolar_design(normalized0, normalized1)
    _, singular, vh = torch.linalg.svd(design, full_matrices=design.shape[-2] < 9)
    basis = vh[:, -2:].reshape(-1, 2, 3, 3)
    coefficients = points0.new_zeros((len(points0), 4))
    # det(A + lambda B), in ascending powers of lambda.
    for cols, sign in (((0, 1, 2), 1), ((1, 2, 0), 1), ((2, 0, 1), 1),
                       ((0, 2, 1), -1), ((1, 0, 2), -1), ((2, 1, 0), -1)):
        for powers in product(range(2), repeat=3):
            coefficients[:, sum(powers)] += sign * (
                basis[:, powers[0], 0, cols[0]] * basis[:, powers[1], 1, cols[1]]
                * basis[:, powers[2], 2, cols[2]])
    valid = ((coefficients[:, 3].abs() > 1e-12)
             & (singular[:, -1] > 1e-10 * singular[:, 0]))
    coefficients, basis = coefficients[valid], basis[valid]
    if len(basis) == 0:
        return points0.new_empty((0, 3, 3))
    companion = coefficients.new_zeros((len(coefficients), 3, 3))
    companion[:, 0, 2] = -coefficients[:, 0] / coefficients[:, 3]
    companion[:, 1, 2] = -coefficients[:, 1] / coefficients[:, 3]
    companion[:, 2, 2] = -coefficients[:, 2] / coefficients[:, 3]
    companion[:, 1, 0] = companion[:, 2, 1] = 1
    roots = torch.linalg.eigvals(companion)
    models = basis[:, None, 0] + roots.real[..., None, None] * basis[:, None, 1]
    models = transform1[valid, None].transpose(-1, -2) @ models @ transform0[valid, None]
    models = models[roots.imag.abs() < 1e-7 * (1 + roots.real.abs())]
    return F.normalize(models.flatten(-2), dim=-1).reshape(-1, 3, 3)


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


def homography_dlt(points0, points1):
    normalized0, transform0 = normalize_points(points0)
    normalized1, transform1 = normalize_points(points1)
    xy1 = homogeneous(normalized0)
    zero = torch.zeros_like(xy1)
    row0 = torch.cat((-xy1, zero, normalized1[..., :1] * xy1), -1)
    row1 = torch.cat((zero, -xy1, normalized1[..., 1:] * xy1), -1)
    design = torch.stack((row0, row1), -2).flatten(-3, -2)
    _, singular, vh = torch.linalg.svd(design, full_matrices=design.shape[-2] < 9)
    matrix = torch.linalg.solve(transform1, vh[..., -1, :].reshape(-1, 3, 3) @ transform0)
    rank = singular[..., 7] > 1e-10 * singular[..., 0]
    return matrix[rank & (torch.linalg.det(matrix).abs() > 1e-12)]


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
def _refine_pose(rotation, translation, pixels0, pixels1, matrix0, matrix1):
    from .averaging import SO3Var, UnitDirectionVar
    from torchls import Problem, TerminationConfig
    from torchls.geometry import skew

    def residual(r, t, x0, x1, inverse0, inverse1):
        fundamental = inverse1.T @ skew(t) @ r @ inverse0
        line1, line0 = fundamental @ x0, fundamental.T @ x1
        denominator = (line0[:2].square().sum() + line1[:2].square().sum()).clamp_min(1e-20).sqrt()
        return ((x1 @ line1) / denominator).reshape(1)

    problem = Problem({'rotation': rotation[None].clone(), 'direction': translation[None].clone()},
                      manifolds={'rotation': SO3Var, 'direction': UnitDirectionVar})
    observation_ids = torch.zeros(len(pixels0), dtype=torch.long, device=pixels0.device)
    problem.add_residual(residual, ('rotation', observation_ids), ('direction', observation_ids),
        homogeneous(pixels0).clone(), homogeneous(pixels1).clone(),
        torch.linalg.inv(matrix0)[None].clone(), torch.linalg.inv(matrix1)[None].clone(), batch_size=1024)
    solved, summary = problem.solve(termination=TerminationConfig(max_iterations=15), return_summary=True)
    if not math.isfinite(summary.final_cost):
        return rotation, translation
    return solved['rotation'][0], solved['direction'][0]


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
    index the original keypoint arrays. Unsupported calibration is classified
    UNCALIBRATED rather than silently producing a pose with incorrect K.
    """
    cfg = TwoViewConfig() if cfg is None else cfg
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
    essential, mask_e = _ransac(normalized0, normalized1, essential_five_point,
        lambda a, b: epipolar_eight_point(a, b, essential=True),
        lambda models: sampson_error(inverse1.T @ models @ inverse0, points0, points1), 5, cfg, generator)
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
            r, t = _refine_pose(r, t, points0[mask_e][positive], points1[mask_e][positive], matrix0, matrix1)
            from torchls.geometry import skew
            candidate = skew(t) @ r
            old_error = sampson_error((inverse1.T @ essential @ inverse0)[None], points0, points1)[0]
            new_error = sampson_error((inverse1.T @ candidate @ inverse0)[None], points0, points1)[0]
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
        return result
    poses = (homography_poses(homography, matrix0, matrix1) if result.config == 'PLANAR'
             else essential_poses(essential))
    rotation, translation, positive, angle = _choose_pose(poses, rays0[mask], rays1[mask])
    if result.config == 'CALIBRATED' and cfg.refine_relative_pose and int(positive.sum()) >= minimum:
        rotation, translation = _refine_pose(rotation, translation, points0[mask][positive],
            points1[mask][positive], matrix0, matrix1)
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
