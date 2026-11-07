"""Fixed-size, five-DOF relative-pose LM with an analytic Sampson Jacobian.

Rotation uses right SO(3) updates, translation uses a two-dimensional tangent
plane on S2. CUDA float64 runs the whole LM loop, including the 5x5 solve,
inside one Warp kernel. The full reference path remains in Torch.
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torchls.geometry import skew, so3_exp


def tangent_basis(direction):
    axis = F.one_hot(direction.abs().argmin(-1), 3).to(direction)
    first = F.normalize(torch.linalg.cross(direction, axis), dim=-1)
    second = torch.linalg.cross(first, direction)
    return torch.stack((first, second), -1)


def sampson_residual_jacobian(rotation, translation, x0, x1, inverse0, inverse1, jacobian=True):
    """Batched homogeneous pixels; returns (B,N) residuals and (B,N,5) J."""
    essential = skew(translation) @ rotation
    fundamental = inverse1.transpose(-1, -2) @ essential @ inverse0
    line1 = x0 @ fundamental.transpose(-1, -2)
    line0 = x1 @ fundamental
    numerator = (x1 * line1).sum(-1)
    denominator2 = line0[..., :2].square().sum(-1) + line1[..., :2].square().sum(-1)
    inverse_norm = denominator2.clamp_min(1e-20).rsqrt()
    residual = numerator * inverse_norm
    if not jacobian:
        return residual
    basis = tangent_basis(translation)
    axes = skew(torch.eye(3, device=rotation.device, dtype=rotation.dtype))
    derivative = torch.cat((essential[:, None] @ axes[None],
        skew(basis.transpose(-1, -2)) @ rotation[:, None]), 1)
    derivative = inverse1[:, None].transpose(-1, -2) @ derivative @ inverse0[:, None]
    dline1 = torch.einsum('bkij,bnj->bnki', derivative, x0)
    dline0 = torch.einsum('bkji,bnj->bnki', derivative, x1)
    dnumerator = (x1[:, :, None] * dline1).sum(-1)
    half_dnorm = ((line0[..., None, :2] * dline0[..., :2]).sum(-1)
                 + (line1[..., None, :2] * dline1[..., :2]).sum(-1))
    half_dnorm = torch.where(denominator2[..., None] > 1e-20, half_dnorm, 0)
    derivative = dnumerator * inverse_norm[..., None] - (
        numerator * inverse_norm.pow(3))[..., None] * half_dnorm
    return residual, derivative


@torch.no_grad()
def _refine_relative_pose(rotations, translations, pixels0, pixels1, matrices0, matrices1,
                          valid=None, max_iterations=15, backend='auto'):
    """Solve independent 5x5 damped systems in parallel, with no scalar GPU reads.

    Rejected/nonfinite updates never replace the current pose. Converged pairs
    remain frozen while the other pairs finish the fixed iteration budget.
    """
    rotations, translations = rotations.clone(), F.normalize(translations, dim=-1)
    inverse0 = torch.linalg.inv_ex(matrices0, check_errors=False)[0]
    inverse1 = torch.linalg.inv_ex(matrices1, check_errors=False)[0]
    x0 = torch.cat((pixels0, torch.ones_like(pixels0[..., :1])), -1)
    x1 = torch.cat((pixels1, torch.ones_like(pixels1[..., :1])), -1)
    if valid is None:
        valid = torch.ones(pixels0.shape[:2], device=pixels0.device, dtype=torch.bool)
    if rotations.is_cuda and rotations.dtype == torch.float64 and backend != 'torch':
        try:
            from .warp_relative_pose_lm import refine
        except ImportError:
            if backend == 'warp':
                raise
        else:
            return refine(rotations, translations, x0, x1, inverse0, inverse1, valid, max_iterations)
    weight = valid.to(pixels0)
    damping = pixels0.new_full((len(rotations),), 1e-3)
    active = valid.sum(-1) >= 5
    eye = torch.eye(5, dtype=rotations.dtype, device=rotations.device)
    for _ in range(max_iterations):
        residual, jacobian = sampson_residual_jacobian(
            rotations, translations, x0, x1, inverse0, inverse1)
        residual = residual * weight
        jacobian = jacobian * weight[..., None]
        normal = jacobian.transpose(-1, -2) @ jacobian
        gradient = (jacobian.transpose(-1, -2) @ residual[..., None]).squeeze(-1)
        diagonal = normal.diagonal(dim1=-2, dim2=-1).clamp_min(1e-8)
        system = normal + torch.diag_embed(damping[:, None] * diagonal) + eye * 1e-12
        delta, info = torch.linalg.solve_ex(system, -gradient)
        usable = active & (info == 0) & torch.isfinite(delta).all(-1)
        delta = torch.where(usable[:, None], delta, 0)
        proposed_rotation = rotations @ so3_exp(delta[:, :3])
        proposed_translation = F.normalize(translations + (
            tangent_basis(translations) @ delta[:, 3:, None]).squeeze(-1), dim=-1)
        proposed_residual = sampson_residual_jacobian(proposed_rotation, proposed_translation,
            x0, x1, inverse0, inverse1, jacobian=False) * weight
        cost, proposed_cost = residual.square().sum(-1), proposed_residual.square().sum(-1)
        accept = usable & torch.isfinite(proposed_cost) & (proposed_cost < cost)
        rotations = torch.where(accept[:, None, None], proposed_rotation, rotations)
        translations = torch.where(accept[:, None], proposed_translation, translations)
        damping = (damping * torch.where(accept, 1 / 3, 10)).clamp(1e-12, 1e12)
        converged = (delta.norm(dim=-1) < 1e-9) | (gradient.abs().amax(-1) < 1e-8)
        converged |= accept & ((cost - proposed_cost) < 1e-9 * cost.clamp_min(1))
        active &= ~(usable & converged)
    return rotations, translations


class _PoseGraph:
    def __init__(self, reference, pairs, points, iterations, backend):
        self.stream = torch.cuda.current_stream(reference.device)
        eye = torch.eye(3, device=reference.device, dtype=reference.dtype).repeat(pairs, 1, 1)
        direction = reference.new_zeros((pairs, 3))
        direction[:, 2] = 1
        self.inputs = [eye, direction, reference.new_zeros((pairs, points, 2)),
                       reference.new_zeros((pairs, points, 2)), eye.clone(), eye.clone()]
        self.valid = torch.zeros((pairs, points), device=reference.device, dtype=torch.bool)
        side = torch.cuda.Stream(device=reference.device)
        side.wait_stream(self.stream)
        with torch.cuda.stream(side):
            for _ in range(2):
                _refine_relative_pose(*self.inputs, self.valid, iterations, backend)
        self.stream.wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=side):
            self.outputs = _refine_relative_pose(*self.inputs, self.valid, iterations, backend)

    def run(self, values, valid):
        pairs, points = values[2].shape[:2]
        self.valid.zero_()
        self.valid[:pairs, :points].copy_(valid)
        for destination, source in zip(self.inputs, values):
            if source.ndim == 3 and source.shape[-1] == 2:
                destination[:pairs, :points].copy_(source)
            else:
                destination[:pairs].copy_(source)
        self.graph.replay()
        # Outputs must survive a later replay using the same static buffers.
        return tuple(value[:pairs].clone() for value in self.outputs)


_POSE_GRAPHS = OrderedDict()
_POSE_GRAPH_CALLS = OrderedDict()


def clear_pose_graph_cache():
    for value in _POSE_GRAPHS.values():
        value.stream.synchronize()
    _POSE_GRAPHS.clear()
    _POSE_GRAPH_CALLS.clear()


@torch.no_grad()
def refine_relative_pose(rotations, translations, pixels0, pixels1, matrices0, matrices1,
                         valid=None, max_iterations=15, use_cuda_graph=True, backend='auto'):
    if backend not in ('auto', 'torch', 'warp'):
        raise ValueError('Relative pose backend must be auto, torch or warp')
    if (not rotations.is_cuda or not use_cuda_graph or len(rotations) == 0
            or torch.cuda.is_current_stream_capturing()):
        return _refine_relative_pose(rotations, translations, pixels0, pixels1, matrices0, matrices1,
                                     valid, max_iterations, backend)
    if valid is None:
        valid = torch.ones(pixels0.shape[:2], device=pixels0.device, dtype=torch.bool)
    pairs = 1 << (len(rotations) - 1).bit_length()
    points = 1 << max(0, (pixels0.shape[1] - 1).bit_length())
    stream = torch.cuda.current_stream(rotations.device)
    key = (rotations.device, rotations.dtype, stream.cuda_stream, pairs, points, max_iterations, backend)
    if key not in _POSE_GRAPHS:
        # Capture has a startup cost: one-off ragged shapes should stay eager.
        calls = _POSE_GRAPH_CALLS.get(key, 0) + 1
        _POSE_GRAPH_CALLS[key] = calls
        _POSE_GRAPH_CALLS.move_to_end(key)
        if len(_POSE_GRAPH_CALLS) > 32:
            _POSE_GRAPH_CALLS.popitem(last=False)
        if calls < 3:
            return _refine_relative_pose(rotations, translations, pixels0, pixels1, matrices0, matrices1,
                                         valid, max_iterations, backend)
        if len(_POSE_GRAPHS) >= 4:
            _, expired = _POSE_GRAPHS.popitem(last=False)
            expired.stream.synchronize()
        _POSE_GRAPHS[key] = _PoseGraph(rotations, pairs, points, max_iterations, backend)
    _POSE_GRAPHS.move_to_end(key)
    return _POSE_GRAPHS[key].run((rotations, translations, pixels0, pixels1, matrices0, matrices1), valid)
