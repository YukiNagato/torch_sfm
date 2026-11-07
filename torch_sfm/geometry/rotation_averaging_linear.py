"""GLOMAP-style fixed-incidence L1-ADMM and Geman-McClure IRLS.

Ordinary 3-DOF image rotations only. This uses a tangent-space approximation,
not an exact Jacobian of an arbitrary SO3 residual. The three coordinates share
one scalar graph matrix, with a gauge row and optional Tikhonov ridge.
"""

from dataclasses import dataclass, field
import math

import torch
import torchls as tls

from .averaging import RotationResult, _spanning_initialization, largest_component, so3_log


@dataclass
class RotationLinearSummary:
    phase: str
    initial_cost: float
    final_cost: float
    iterations: int = 0
    linear_iterations: int = 0  # direct solves, including ADMM; not PCG iterations
    factorizations: int = 0
    termination_reason: str = 'max_iterations'
    linear_solver: str = 'dense_cholesky_multi_rhs'
    cost_history: list = field(default_factory=list)
    step_history: list = field(default_factory=list)
    admm_iterations: list = field(default_factory=list)


def _system(pairs, n, reference, limit):
    m = len(pairs)
    rows = torch.arange(m, device=pairs.device).repeat_interleave(2)
    indices = torch.stack((torch.cat((rows, rows.new_tensor([m]))),
                           torch.cat((pairs.flatten(), rows.new_tensor([0])))))
    values = torch.cat((reference.new_tensor([-1., 1.]).repeat(m), reference.new_ones(1)))
    matrix = torch.sparse_coo_tensor(indices, values, (m+1, n)).coalesce()
    return tls.FixedLinearSystem(matrix, max_dense_columns=limit)


@torch.no_grad()
def solve_rotation_graph(initial, pairs, relative, cfg):
    """Solve one connected graph from explicit initial rotations (also useful for audits)."""
    current = initial.clone()
    anchor = initial[0].clone()
    system = _system(pairs, len(current), current, cfg.linear_max_cameras)
    sigma2 = math.radians(cfg.loss_scale)**2

    def residual():
        error = so3_log(current[pairs[:, 1]].transpose(-1, -2) @ relative @ current[pairs[:, 0]])
        gauge = -so3_log(anchor.T @ current[0])
        return torch.cat((error, gauge[None]), 0)

    def objective(r, phase):
        if phase == 'l1':
            return r.abs().sum()
        squared = r[:-1].square().sum(-1)
        return .5 * (squared / (sigma2 + squared)).sum() + .5 * r[-1].square().sum()

    summaries = []
    for phase, budget in [('l1', cfg.l1_iterations), ('gm', cfg.irls_iterations)]:
        if not budget:
            continue
        r = residual()
        initial_cost = float(objective(r, phase))
        summary = RotationLinearSummary(phase, initial_cost, initial_cost, cost_history=[initial_cost])
        factor = system.factor(ridge=cfg.ridge_regularization) if phase == 'l1' else None
        summary.factorizations = int(factor is not None)
        previous_norm = 0.
        for iteration in range(budget):
            if phase == 'l1':
                steps = min(cfg.admm_initial_iterations * 2**iteration, cfg.admm_max_iterations)
                result = tls.least_absolute_deviations(system, r, factor=factor,
                    max_iterations=steps, absolute_tolerance=cfg.admm_absolute_tolerance,
                    relative_tolerance=cfg.admm_relative_tolerance)
                delta = result.solution
                summary.admm_iterations.append(result.iterations)
                summary.linear_iterations += result.iterations
            else:
                weights = sigma2 / (sigma2 + r[:-1].square().sum(-1)).square()
                factor = system.factor(torch.cat((weights, weights.new_ones(1))),
                                       ridge=cfg.ridge_regularization)
                delta = factor.least_squares(r)
                summary.factorizations += 1
                summary.linear_iterations += 1
            if not bool(torch.isfinite(delta).all()):
                raise RuntimeError('Nonfinite rotation averaging step')
            current = current @ tls.so3_exp(delta)
            average_step = float(delta.norm(dim=-1).mean())
            norm = float(delta.norm())
            r = residual()
            summary.iterations += 1
            summary.step_history.append(average_step)
            summary.final_cost = float(objective(r, phase))
            summary.cost_history.append(summary.final_cost)
            if average_step < cfg.convergence:
                summary.termination_reason = 'step_tolerance'
                break
            if phase == 'l1' and abs(previous_norm - norm) < 1e-12:
                summary.termination_reason = 'step_stagnation'
                break
            previous_norm = norm
        summaries.append(summary)
    return current, summaries


@torch.no_grad()
def rotation_averaging_linear(num_frames, geometries, cfg):
    edges = {pair: g for pair, g in geometries.items() if g.R is not None}
    if not edges:
        raise ValueError('No geometrically verified rotation edges')
    reference = next(iter(edges.values())).R
    rotations = reference.new_full((num_frames, 3, 3), torch.nan)
    registered = torch.zeros(num_frames, dtype=torch.bool, device=reference.device)
    summaries = []
    for _ in range(cfg.linear_filter_passes):
        nodes = largest_component(num_frames, edges)
        if len(nodes) < 2:
            raise ValueError('No connected rotation component after filtering')
        if len(nodes) > cfg.linear_max_cameras:
            raise ValueError('Rotation graph exceeds linear_max_cameras; use method=lm or auto')
        active = set(nodes)
        edges = {p: g for p, g in edges.items() if p[0] in active and p[1] in active}
        mapping = {node: i for i, node in enumerate(nodes)}
        pairs = torch.tensor([(mapping[i], mapping[j]) for i, j in edges], device=reference.device)
        relative = torch.stack([g.R for g in edges.values()]).detach()
        initial = _spanning_initialization(nodes, edges, reference)
        current, stages = solve_rotation_graph(initial, pairs, relative, cfg)
        summaries.extend(stages)
        current = current @ current[0].T
        rotations.fill_(torch.nan)
        rotations[nodes] = current
        error = so3_log(current[pairs[:, 1]] @ current[pairs[:, 0]].transpose(-1, -2)
                        @ relative.transpose(-1, -2)).norm(dim=-1)
        keep = (error <= math.radians(cfg.max_error)).tolist()
        edges = {p: g for (p, g), valid in zip(edges.items(), keep) if valid}
        nodes = largest_component(num_frames, edges)
        if len(nodes) < 2:
            raise ValueError('No connected rotation component after filtering')
        active = set(nodes)
        edges = {p: g for p, g in edges.items() if p[0] in active and p[1] in active}
        registered.zero_()
        registered[nodes] = True
        rotations[~registered] = torch.nan
    return RotationResult(rotations, registered, edges, summaries)
