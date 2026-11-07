"""Pose-only PINHOLE refinement with an analytic Jacobian and a 6x6 LM solve.

All observations are fixed; only a world-to-camera SE(3) pose is optimized.
Updates multiply on the left, so tangent coordinates are in the camera frame:
T_new = Exp([translation, rotation]) @ T. No generic problem layout or autodiff
is needed. This is eager Torch, on CPU or CUDA, with double-precision arithmetic.
"""
import math

import torch
from torchls.geometry import se3_exp


def _project(pose, points, pixels, k):
    local = points @ pose[:3, :3].T + pose[:3, 3]
    valid = (local[:, 2] > torch.finfo(local.dtype).eps) & torch.isfinite(local).all(-1)
    depth = torch.where(valid, local[:, 2], 1.)
    xy = local[:, :2] / depth[:, None]
    residual = xy * k[:2] + k[2:] - pixels
    valid = valid & torch.isfinite(residual).all(-1)
    return (torch.where(valid[:, None], residual, 0.),
            torch.where(valid[:, None], xy, 0.),
            torch.where(valid, 1 / depth, 0.), valid)


def projection_jacobian(pose, points, pixels, k):
    """Raw residual (N,2) and left-SE3 Jacobian (N,2,6), zero at invalid depth."""
    residual, xy, invz, valid = _project(pose, points, pixels, k)
    x, y = xy.unbind(-1)
    fx, fy = k[:2].unbind()
    ax, ay = fx * invz, fy * invz
    zero = torch.zeros_like(x)
    jx = torch.stack((ax, zero, -ax*x, -fx*x*y, fx*(1+x*x), -fx*y), -1)
    jy = torch.stack((zero, ay, -ay*y, -fy*(1+y*y), fy*x*y, fy*x), -1)
    jacobian = torch.stack((jx, jy), 1)
    return residual, torch.where(valid[:, None, None], jacobian, 0.), valid


def _cost(residual, loss_scale):
    squared = residual.square().sum(-1)
    return .5 * torch.where(squared <= loss_scale**2, squared,
                            loss_scale*(2*squared.sqrt()-loss_scale)).sum()


def _linearize(pose, points, pixels, k, loss_scale):
    residual, jacobian, valid = projection_jacobian(pose, points, pixels, k)
    norm = residual.norm(dim=-1).clamp_min(torch.finfo(residual.dtype).tiny)
    weight = (loss_scale / norm).clamp_max(1).sqrt()
    matrix = (jacobian * weight[:, None, None]).reshape(-1, 6)
    rhs = (residual * weight[:, None]).reshape(-1)
    normal = matrix.T @ matrix
    return normal, matrix.T @ rhs, float(_cost(residual, loss_scale)), valid


@torch.no_grad()
def refine_pose(pose, points, pixels, intrinsics, *, loss_scale=4., max_iterations=30,
                cost_tolerance=1e-8, gradient_tolerance=1e-8,
                parameter_tolerance=1e-8, lambda_initial=1e-4, max_trials=20,
                return_summary=False):
    """Refine a single pose, with fixed world points/pixels/(fx,fy,cx,cy).

    Keeps the Huber objective and adaptive diagonal-damped LM acceptance policy.
    Invalid initial projections have zero residual/Jacobian, but a trial cannot
    improve its cost by losing a previously valid projection. A failed factor or
    rejected step increases damping; exhaustion returns the last accepted pose.
    Output keeps the input pose's device/dtype. Rank-deficient data can yield a
    stationary solution; callers must validate inlier count and spatial support.
    """
    if pose.shape != (4, 4) or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Expected pose (4,4) and points (N,3)')
    if pixels.shape != (len(points), 2) or intrinsics.shape != (4,):
        raise ValueError('Expected pixels (N,2) and PINHOLE intrinsics (4,)')
    inputs = (pose, points, pixels, intrinsics)
    if any(x.device != pose.device or not x.is_floating_point() for x in inputs):
        raise ValueError('Pose refinement inputs must be floating tensors on the same device')
    if any(not bool(torch.isfinite(x).all()) for x in inputs) or bool((intrinsics[:2] <= 0).any()):
        raise ValueError('Pose refinement needs finite inputs and positive focal lengths')
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in (max_iterations, max_trials)):
        raise ValueError('Iteration and trial limits must be positive integers')
    if not math.isfinite(loss_scale) or loss_scale <= 0 or not 1e-12 <= lambda_initial <= 1e12:
        raise ValueError('Invalid loss scale or initial damping')
    if any(not math.isfinite(x) or x < 0 for x in (cost_tolerance, gradient_tolerance, parameter_tolerance)):
        raise ValueError('Tolerances must be finite and nonnegative')
    output_dtype = pose.dtype
    pose, points, pixels, k = [x.detach().double() for x in inputs]
    pose = pose.clone()
    normal, gradient, cost, valid = _linearize(pose, points, pixels, k, loss_scale)
    summary = dict(initial_cost=cost, final_cost=cost, iterations=0, accepted_steps=0,
                   rejected_steps=0, cost_history=[cost], termination_reason='max_iterations')
    damping, rejection_factor = lambda_initial, 2.
    identity = torch.eye(6, dtype=pose.dtype, device=pose.device)
    column_scale = (1 + normal.diagonal().clamp_min(0).sqrt()).reciprocal()
    if int(valid.sum()) < 6:
        summary['termination_reason'] = 'insufficient_valid_observations'
    else:
        for iteration in range(max_iterations):
            summary['iterations'] = iteration + 1
            if not bool(torch.isfinite(normal).all() & torch.isfinite(gradient).all()) or not math.isfinite(cost):
                summary['termination_reason'] = 'nonfinite_linearization'
                break
            if float(gradient.abs().max()) <= gradient_tolerance:
                summary['termination_reason'] = 'gradient_tolerance'
                break
            # Same damping metric as torchls diagonal LM with fixed initial
            # Jacobi scaling; the solve itself is a direct 6x6 Cholesky solve.
            diagonal = (normal.diagonal() * column_scale.square()).clamp(1e-6, 1e32)
            scale = column_scale * diagonal.rsqrt()
            system = normal * scale[:, None] * scale[None, :]
            system = .5 * (system + system.T)
            rhs = -gradient * scale
            accepted = False
            for _ in range(max_trials):
                factor, info = torch.linalg.cholesky_ex(system + damping*identity)
                if int(info) == 0:
                    delta = scale * torch.cholesky_solve(rhs[:, None], factor).squeeze(-1)
                    if bool(torch.isfinite(delta).all()):
                        trial = se3_exp(delta) @ pose
                        trial_residual, _, _, trial_valid = _project(trial, points, pixels, k)
                        trial_cost = float(_cost(trial_residual, loss_scale))
                        actual = cost - trial_cost
                        predicted = float(-gradient @ delta - .5 * delta @ normal @ delta)
                        quality = actual / predicted if predicted > 0 else -math.inf
                        if (math.isfinite(trial_cost) and actual > 0 and quality >= 1e-4
                                and not bool((valid & ~trial_valid).any())):
                            old_cost = cost
                            pose = trial
                            normal, gradient, cost, valid = _linearize(pose, points, pixels, k, loss_scale)
                            accepted = True
                            summary['accepted_steps'] += 1
                            summary['cost_history'].append(cost)
                            summary['final_cost'] = cost
                            damping = max(1e-12, min(1e12, damping * max(1/3, 1-(2*min(quality, 1.)-1)**3)))
                            rejection_factor = 2.
                            if float(gradient.abs().max()) <= gradient_tolerance:
                                summary['termination_reason'] = 'gradient_tolerance'
                            elif actual <= cost_tolerance * max(old_cost, torch.finfo(pose.dtype).tiny):
                                summary['termination_reason'] = 'cost_tolerance'
                            elif float(delta.norm()) <= parameter_tolerance:
                                summary['termination_reason'] = 'parameter_tolerance'
                            break
                summary['rejected_steps'] += 1
                if damping >= 1e12:
                    break
                damping = min(1e12, damping * rejection_factor)
                rejection_factor = min(1e24, 2*rejection_factor)
            if not accepted:
                summary['termination_reason'] = 'no_acceptable_step'
                break
            if summary['termination_reason'] != 'max_iterations':
                break
    summary['gradient_norm'] = float(gradient.abs().max())
    result = pose.to(dtype=output_dtype)
    return (result, summary) if return_summary else result
