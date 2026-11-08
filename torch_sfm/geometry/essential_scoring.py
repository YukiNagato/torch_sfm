"""Pixel tangent Sampson scores with cheirality, for calibrated RANSAC.

The four essential decompositions vote over all valid bearings, as in COLMAP.
This rejects behind-camera correspondences during model selection, rather than
only after selecting a winning essential matrix. No native solver is called.
"""
import torch
from .two_view import homogeneous


def _rays_and_jacobian(points, inverse):
    raw = homogeneous(points) @ inverse.transpose(-1, -2)
    norm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    rays = raw / norm
    derivative = inverse[:, None, :, :2].expand(-1, points.shape[1], -1, -1)
    projection = (rays[..., :, None] * derivative).sum(-2)
    jacobian = (derivative - rays[..., :, None] * projection[..., None, :]) / norm[..., None]
    return rays, jacobian


def _poses(models):
    u, _, vh = torch.linalg.svd(models)
    u = u * torch.linalg.det(u)[:, None, None]
    vh = vh * torch.linalg.det(vh)[:, None, None]
    w = models.new_tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    rotations = torch.stack((u @ w @ vh, u @ w.T @ vh), 1)
    return rotations, u[:, :, 2]


def _torch_errors(models, owners, rays0, rays1, jac0, jac1, valid, rotations, translation):
    output = models.new_empty((len(models), valid.shape[1]))
    for start in range(0, len(models), 256):
        stop = start + 256
        pair = owners[start:stop]
        a, b = rays0[pair], rays1[pair]
        r, t = rotations[start:stop], translation[start:stop]
        rotated = a[:, None] @ r.transpose(-1, -2)
        cosine = -(rotated * b[:, None]).sum(-1)
        first = -(rotated * t[:, None, None]).sum(-1)
        second = (b * t[:, None]).sum(-1)[:, None]
        depth0, depth1 = first - cosine * second, second - cosine * first
        positive = torch.cat(((depth0 > 0) & (depth1 > 0),
                              (depth0 < 0) & (depth1 < 0)), 1) & valid[pair, None]
        counts = positive.sum(-1)
        # COLMAP picks the last candidate in a tie.
        best = 3 - counts.flip(-1).argmax(-1)
        selected = positive[torch.arange(len(pair), device=pair.device), best]
        matrix = models[start:stop]
        line1, line0 = a @ matrix.transpose(-1, -2), b @ matrix
        numerator = (b * line1).sum(-1).square()
        denominator = ((jac0[pair] * line0[..., None]).sum(-2).square().sum(-1)
                       + (jac1[pair] * line1[..., None]).sum(-2).square().sum(-1))
        error = numerator / denominator
        error = torch.where(selected & (denominator > 0) & torch.isfinite(error), error, torch.inf)
        output[start:stop] = torch.where(valid[pair], error, 0)
    return output


def prepare_essential_observations(points0, points1, inverse0, inverse1):
    """Prepare immutable per-pair bearings and pixel Jacobians for one batch.

    Reuse only while the pixels and intrinsics remain unchanged. The caller
    owns these tensors; there is no persistent/global observation cache.
    """
    rays0, jac0 = _rays_and_jacobian(points0, inverse0)
    rays1, jac1 = _rays_and_jacobian(points1, inverse1)
    return rays0, rays1, jac0, jac1


def essential_errors(models, owners, points0, points1, inverse0, inverse1, valid, backend='auto'):
    """Return (hypotheses, padded points) squared pixel errors; padding is zero."""
    if len(models) == 0:
        return models.new_empty((0, valid.shape[1]))
    prepared = prepare_essential_observations(points0, points1, inverse0, inverse1)
    return essential_errors_prepared(models, owners, *prepared, valid, backend)


def essential_errors_prepared(models, owners, rays0, rays1, jac0, jac1, valid, backend='auto'):
    """Score candidates whose owners index the original prepared pair batch."""
    if len(models) == 0:
        return models.new_empty((0, valid.shape[1]))
    if models.is_cuda and models.dtype == torch.float64 and backend != 'torch':
        try:
            from .warp.essential_scoring import score
            from .warp.essential_pose import decompose
        except ImportError:
            if backend == 'warp':
                raise
        else:
            rotations, translation, safe = decompose(models)
            errors, ambiguous = score(models, owners, rays0, rays1, jac0, jac1, valid,
                                      rotations, translation, return_ambiguous=True)
            # Jacobi can reorder the four poses. Unique winners are invariant;
            # ties must use the original SVD ordering (last candidate wins).
            fallback = torch.where(~safe | ambiguous)[0]
            if len(fallback):
                rotations, translation = _poses(models[fallback])
                errors[fallback] = score(models[fallback], owners[fallback], rays0, rays1,
                                         jac0, jac1, valid, rotations, translation)
            return errors
    rotations, translation = _poses(models)
    return _torch_errors(models, owners, rays0, rays1, jac0, jac1, valid, rotations, translation)
