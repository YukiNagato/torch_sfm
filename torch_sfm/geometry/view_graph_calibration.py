# Fetzer coefficient/residual equations adapted from COLMAP calibration.h.
# Copyright (c), ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
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

"""Pinhole view-graph focal calibration in Torch, following Fetzer/COLMAP.

Only the principal point is treated as known. A shared input K represents one
camera; an array of K represents independent cameras. No COLMAP runtime calls.
The pipeline uses shared-focal six-point initialization for shared K and
seven-point F initialization for independent cameras.
"""
from dataclasses import dataclass
import math

import torch
import torchls as tls
from torchls.losses import Loss

from .two_view import camera_matrix


@dataclass
class ViewGraphCalibrationConfig:
    mode: str = 'estimate'  # estimate an approximate focal, or trust supplied K
    loss_scale: float = .01
    max_calibration_error: float = 2.
    min_focal_ratio: float = .1
    max_focal_ratio: float = 10.
    min_pairs: int = 3
    max_iterations: int = 100
    relpose_max_error: float = 1.
    relpose_min_num_inliers: int = 30
    relpose_min_inlier_ratio: float = .25
    refine_fundamental: bool = True
    fundamental_refine_iterations: int = 20
    fundamental_refine_model: str = 'auto'  # shared-focal model for shared K, otherwise rank-two F
    shared_focal_ransac: bool = True
    shared_analytic_jacobian: bool = True
    shared_cpu_max_pairs: int = 2048  # 0 keeps the solve on the input device

    def __post_init__(self):
        if self.mode not in ('estimate', 'trusted'):
            raise ValueError('Calibration mode must be estimate or trusted')
        if self.fundamental_refine_model not in ('auto','fundamental'):
            raise ValueError('fundamental_refine_model must be auto or fundamental')
        for name in ('loss_scale', 'max_calibration_error', 'relpose_max_error'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not 0 < self.min_focal_ratio < 1 < self.max_focal_ratio < float('inf'):
            raise ValueError('Focal ratio bounds must contain 1 strictly')
        if (self.min_pairs < 1 or self.max_iterations < 1 or self.relpose_min_num_inliers < 5
                or self.fundamental_refine_iterations < 1):
            raise ValueError('Invalid calibration iteration/inlier limits')
        if not 0 <= self.relpose_min_inlier_ratio <= 1:
            raise ValueError('relpose_min_inlier_ratio must be in [0,1]')
        if not isinstance(self.shared_cpu_max_pairs, int) or self.shared_cpu_max_pairs < 0:
            raise ValueError('shared_cpu_max_pairs must be a nonnegative integer')


def fetzer_coefficients(fundamental, principal0, principal1):
    """Batched principal-point-centered SVD; coefficients have shape (...,2,4).

    Equations 11--14 of Fetzer et al. (WACV 2020), with the sign correction
    documented in COLMAP estimators/cost_functions/calibration.h.
    """
    k0 = torch.eye(3, dtype=fundamental.dtype, device=fundamental.device).expand_as(fundamental).clone()
    k1 = k0.clone()
    k0[..., :2, 2], k1[..., :2, 2] = principal0, principal1
    u, s, vh = torch.linalg.svd(k1.transpose(-1, -2) @ fundamental @ k0)
    v0, v1 = vh[..., 0, :], vh[..., 1, :]
    u0, u1 = u[..., :, 0], u[..., :, 1]
    a = torch.stack((s[..., 0]**2 * v0[..., :2].square().sum(-1),
                     s[..., 0] * s[..., 1] * (v0[..., :2] * v1[..., :2]).sum(-1),
                     s[..., 1]**2 * v1[..., :2].square().sum(-1)), -1)
    b = torch.stack((s[..., 0]**2 * v0[..., 2]**2,
                     s[..., 0] * s[..., 1] * v0[..., 2] * v1[..., 2],
                     s[..., 1]**2 * v1[..., 2]**2), -1)
    c = torch.stack((u1[..., :2].square().sum(-1),
                     -(u0[..., :2] * u1[..., :2]).sum(-1), u0[..., :2].square().sum(-1)), -1)
    d = torch.stack((u1[..., 2]**2, -u0[..., 2] * u1[..., 2], u0[..., 2]**2), -1)
    def coefficients(i, j):
        return torch.stack((a[..., i]*c[..., j]-a[..., j]*c[..., i],
                            a[..., i]*d[..., j]-a[..., j]*d[..., i],
                            b[..., i]*c[..., j]-b[..., j]*c[..., i],
                            b[..., i]*d[..., j]-b[..., j]*d[..., i]), -1)
    result = torch.stack((coefficients(1, 0), coefficients(2, 1)), -2)
    return result / result.abs().amax(-1, keepdim=True).clamp_min(1e-300)


def fetzer_residual(focal0, focal1, coefficients):
    x, y = focal0.square(), focal1.square()
    a, b = coefficients[..., 0, :], coefficients[..., 1, :]
    den0, den1 = y*a[..., 0]+a[..., 1], x*b[..., 0]+b[..., 2]
    # Preserve the denominator's sign; singular equations are rejected before
    # solving, while this finite guard also handles singular trial parameters.
    def safe(den):
        return torch.where(den.abs() < 1e-18, torch.where(den < 0, -1e-18, 1e-18), den)
    return torch.stack((1 + (y*a[..., 2]+a[..., 3])/safe(den0)/x,
                        1 + (x*b[..., 1]+b[..., 3])/safe(den1)/y), -1)


class _Cauchy(Loss):
    def __init__(self, scale):
        self.scale2 = scale**2

    def value(self, s):
        return self.scale2 * torch.log1p(s/self.scale2)

    def evaluate(self, s):
        derivative = 1/(1+s/self.scale2)
        return self.value(s), derivative, -derivative.square()/self.scale2

    def _linearize(self, residual, blocks, squared_norm, evaluation):
        # Cauchy has nonpositive rho'', so the existing clamped curvature
        # correction is identically zero; only the square-root weight remains.
        weight = evaluation[1].sqrt().to(residual.dtype)
        return residual*weight[:,None], tuple(block*weight[:,None,None] for block in blocks)


def _shared(log_f, coefficients):
    return fetzer_residual(log_f[0].exp(), log_f[0].exp(), coefficients)


def _shared_jacobian(log_f, coefficients):
    """Raw 2x1 Jacobian in the additive log-focal chart, including safe guards."""
    x = log_f[0].exp().square()
    a, b = coefficients[0], coefficients[1]
    den = torch.stack((x*a[0]+a[1], x*b[0]+b[2]))
    numerator = torch.stack((x*a[2]+a[3], x*b[1]+b[3]))
    den_derivative = 2*x*torch.stack((a[0], b[0]))
    small = den.abs() < 1e-18
    den_derivative = torch.where(small, 0., den_derivative)
    den = torch.where(small, torch.where(den < 0, -1e-18, 1e-18), den)
    num_derivative = 2*x*torch.stack((a[2], b[1]))
    quotient = numerator/den
    derivative = (num_derivative/den-quotient*(den_derivative/den))/x-2*quotient/x
    return (derivative[:, None],)


def _independent(log_f0, log_f1, coefficients):
    return fetzer_residual(log_f0[0].exp(), log_f1[0].exp(), coefficients)


@torch.inference_mode(False)
@torch.no_grad()
def calibrate_view_graph(geometries, intrinsics, num_images, cfg=None):
    """Return updated K, surviving pairs and diagnostics; fail on unobservable K.

    Inputs must contain independently estimated F (not K^-T E K^-1). Planar
    edges do not constrain focal lengths; the pipeline retains H and recovers
    their poses with the new K.
    Anisotropic initial fx/fy are seeded by their mean, then calibrated to one
    focal per camera; subsequent BA can refine fx and fy independently.
    """
    cfg = ViewGraphCalibrationConfig() if cfg is None else cfg
    matrices = camera_matrix(intrinsics).detach().double().clone()
    shared = matrices.ndim == 2
    if not shared and matrices.shape != (num_images, 3, 3):
        raise ValueError('Expected shared K or one K per image')
    if cfg.mode == 'trusted':
        return intrinsics.clone(), set(geometries), {'mode': 'trusted', 'optimized': False}
    bank = matrices[None] if shared else matrices
    initial = (bank[:, 0, 0] + bank[:, 1, 1])/2
    pairs = [pair for pair, g in geometries.items()
             if g.config in ('UNCALIBRATED', 'CALIBRATED') and g.F is not None
             and bool(torch.isfinite(g.F).all()) and float(g.F.norm()) > 0]
    if not pairs:
        raise ValueError('No nonplanar fundamental matrices for focal calibration; supply trusted intrinsics if known')
    device = matrices.device
    indices = torch.tensor(pairs, device=device, dtype=torch.long)
    if (indices < 0).any() or (indices >= num_images).any():
        raise ValueError('View-graph image index out of range')
    camera_ids = torch.zeros_like(indices) if shared else indices
    f = torch.stack([geometries[pair].F.to(device=device, dtype=torch.float64) for pair in pairs])
    coefficients = fetzer_coefficients(f, bank[camera_ids[:, 0], :2, 2], bank[camera_ids[:, 1], :2, 2])
    f0, f1 = initial[camera_ids[:, 0]], initial[camera_ids[:, 1]]
    residual = fetzer_residual(f0, f1, coefficients)
    # Exclude scale-free/degenerate equations, including pure translation with
    # parallel optical axes. Check sensitivity before accepting an optimized K.
    step = 1e-4
    derivative = (fetzer_residual(f0*math.exp(step), f1*math.exp(step), coefficients)
                  - fetzer_residual(f0*math.exp(-step), f1*math.exp(-step), coefficients))/(2*step)
    usable = torch.isfinite(coefficients).all(dim=(-2, -1)) & torch.isfinite(residual).all(-1)
    usable &= torch.isfinite(derivative).all(-1) & (derivative.norm(dim=-1) > 1e-8)
    usable &= coefficients.abs().amax(dim=(-2, -1)) > 1e-12
    pairs = [pair for pair, keep in zip(pairs, usable.cpu().tolist()) if keep]
    coefficients, camera_ids = coefficients[usable], camera_ids[usable]
    counts = torch.bincount(camera_ids.flatten(), minlength=len(bank))
    if shared:
        counts //= 2
    if (counts < cfg.min_pairs).any():
        raise ValueError('Insufficient informative pairs for one or more cameras; focal calibration is unobservable')
    # This changes execution placement only. Keep the same bounded torchls LM
    # problem, robust objective, scaling, and termination settings.
    solve_device = (torch.device('cpu') if shared and len(pairs) <= cfg.shared_cpu_max_pairs
                    else device)
    solve_initial = initial.to(solve_device)
    solve_coefficients = coefficients.to(solve_device)
    solve_ids = camera_ids.to(solve_device)
    problem = tls.Problem({'log_focal': solve_initial.log()[:, None]})
    loss = _Cauchy(cfg.loss_scale)
    if shared:
        problem.add_residual(_shared, ('log_focal', solve_ids[:, 0]), solve_coefficients, loss=loss,
                             jacobian_fn=_shared_jacobian if cfg.shared_analytic_jacobian else None)
    else:
        problem.add_residual(_independent, ('log_focal', solve_ids[:, 0]),
                             ('log_focal', solve_ids[:, 1]), solve_coefficients, loss=loss)
    bounds = {'log_focal': ((solve_initial*cfg.min_focal_ratio).log()[:, None],
                            (solve_initial*cfg.max_focal_ratio).log()[:, None])}
    values, summary = problem.solve(return_summary=True,
        linear_solver='dense_qr' if len(bank) <= 32 else 'conjugate_gradient',
        bounds=bounds, jacobi_scaling=True,
        termination=tls.TerminationConfig(max_iterations=cfg.max_iterations,
            cost_tolerance=1e-8, gradient_tolerance=1e-10, parameter_tolerance=1e-9))
    focal = values['log_focal'][:, 0].exp().to(device)
    if (not math.isfinite(summary.final_cost) or not math.isfinite(summary.gradient_norm)
            or summary.final_cost > summary.initial_cost + 1e-10):
        raise ValueError('Nonfinite or unsuccessful view-graph calibration solve')
    ratio = focal/initial
    if (not torch.isfinite(focal).all() or (ratio <= cfg.min_focal_ratio*1.001).any()
            or (ratio >= cfg.max_focal_ratio/1.001).any()):
        raise ValueError('Focal calibration failed or reached its bounds; inspect the view graph')
    errors = fetzer_residual(focal[camera_ids[:, 0]], focal[camera_ids[:, 1]], coefficients).norm(dim=-1)
    good = torch.isfinite(errors) & (errors <= cfg.max_calibration_error)
    support = torch.bincount(camera_ids[good].flatten(), minlength=len(bank))
    if shared:
        support //= 2
    if (support < cfg.min_pairs).any():
        raise ValueError('Insufficient consistent pairs after focal calibration')
    accepted = {pair for pair, keep in zip(pairs, good.cpu().tolist()) if keep}
    accepted.update(pair for pair, g in geometries.items() if g.config in ('PLANAR', 'PANORAMIC'))
    bank[:, 0, 0] = bank[:, 1, 1] = focal
    result = bank[0] if shared else bank
    if intrinsics.shape[-1:] == (4,):
        result = result[..., [0, 1, 0, 1], [0, 1, 2, 2]]
    stats = dict(mode='estimate', optimized=True, initial_focal=initial.cpu().tolist(),
        calibrated_focal=focal.cpu().tolist(), informative_pairs=len(pairs),
        consistent_pairs=int(good.sum()), rejected_pairs=len(geometries)-len(accepted),
        iterations=summary.iterations, termination_reason=summary.termination_reason,
        initial_cost=summary.initial_cost, final_cost=summary.final_cost)
    stats.update(solver_device=str(solve_device),
                 analytic_jacobian=shared and cfg.shared_analytic_jacobian)
    return result.to(intrinsics), accepted, stats
