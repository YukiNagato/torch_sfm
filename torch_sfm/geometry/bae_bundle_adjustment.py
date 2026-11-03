"""Optional BAE LM adapter with a true Huber objective and IRLS linearization.

BAE's native LM currently does not apply its supplied loss kernel to J/R. We
provide a weighted factor with stop-gradient weights, and a separate true loss
to its optimizer. Sparse AD, full normal assembly, PCG, and LM remain BAE's.
BAE registers global sparse operators: benchmark backends in separate processes.
"""

import math
from unittest.mock import patch

import bae.optim.optimizer as optimizer_module
import pypose as pp
import torch
from bae.optim import LM
from bae.utils.pysolvers import PCG
from pypose.autograd.function import psjac
from torch import nn

from torchls import SolveSummary


def raw_projection(
    rotation,
    center,
    point,
    intrinsic,
    center_base,
    center_mask,
    intrinsic_base,
    intrinsic_map,
    observed,
):
    center = center_base + center_mask * (center - center_base)
    k = intrinsic_base + (intrinsic_map @ intrinsic[..., None]).squeeze(-1)
    local = rotation.Act(point - center)
    good = local[..., 2:3] >= torch.finfo(local.dtype).eps
    depth = torch.where(good, local[..., 2:3], 1.0)
    error = k[..., :2].exp() * local[..., :2] / depth + k[..., 2:] - observed
    return torch.where(good, error, 0.0)


@psjac
def huber_factor(
    rotation,
    center,
    point,
    intrinsic,
    center_base,
    center_mask,
    intrinsic_base,
    intrinsic_map,
    observed,
    delta,
):
    error = raw_projection(
        rotation,
        center,
        point,
        intrinsic,
        center_base,
        center_mask,
        intrinsic_base,
        intrinsic_map,
        observed,
    )
    squared = error.square().sum(-1, keepdim=True)
    norm = squared.clamp_min(torch.finfo(error.dtype).tiny).sqrt()
    weight = torch.where(squared > delta.square(), delta / norm, 1.0).sqrt().detach()
    return error * weight


class BAEResidual(nn.Module):
    def __init__(
        self,
        rotations,
        centers,
        rotation_mask,
        center_mask,
        points,
        intrinsic_values,
        camera_ids,
        point_ids,
        intrinsic_ids,
        pixels,
        delta,
    ):
        super().__init__()
        # Camera 0 is always fixed. The baseline coordinate is imposed in the
        # factor/output via center_mask, so it stays exact even after rejection.
        self.base_rotations = rotations.clone()
        self.base_centers = centers.clone()
        self.center_mask = center_mask
        active_rotation = rotation_mask.reshape(len(rotations), 3).bool()
        if not bool(((active_rotation == 0).all(-1) | active_rotation.all(-1)).all()):
            raise ValueError("BAE adapter requires fully fixed or fully free rotations")
        self.active_rotation_ids = torch.where(active_rotation.all(-1))[0]
        self.constant_rotation_ids = torch.where(~active_rotation.any(-1))[0]
        order = torch.cat((self.constant_rotation_ids, self.active_rotation_ids))
        self.rotation_inverse = torch.argsort(order)
        poses = pp.mat2SO3(rotations)
        self.fixed_rotations = poses[self.constant_rotation_ids].clone()
        if len(self.active_rotation_ids):
            self.rotation = pp.Parameter(poses[self.active_rotation_ids].clone(), sjac=True)
        self.center = pp.Parameter(centers[1:].clone(), sjac=True)
        self.point = pp.Parameter(points.clone(), sjac=True)
        mask = intrinsic_values[:, 1].bool()
        if not torch.equal(mask, mask[:1].expand_as(mask)):
            raise ValueError("BAE adapter requires the same free intrinsic coordinates per camera")
        active = torch.where(mask[0])[0]
        self.intrinsic_base = intrinsic_values[:, 0].clone()
        self.intrinsic_map = torch.eye(4, device=centers.device, dtype=centers.dtype)[:, active]
        if len(active):
            self.intrinsic = pp.Parameter(self.intrinsic_base[:, active].clone(), sjac=True)
            self.intrinsic_base[:, active] = 0.0
        else:
            # One dummy constant input keeps the factor signature unchanged.
            self.intrinsic = centers.new_zeros((len(intrinsic_values), 1))
            self.intrinsic_map = centers.new_zeros((4, 1))
        self.camera_ids, self.point_ids, self.intrinsic_ids = camera_ids, point_ids, intrinsic_ids
        self.pixels = pixels
        self.delta = centers.new_tensor([delta])

    def tables(self, tracked):
        centers = self.center if tracked else torch.Tensor(self.center)
        centers = torch.cat((self.base_centers[:1], centers), dim=0)
        rotations = self.fixed_rotations
        if hasattr(self, "rotation"):
            active = self.rotation if tracked else pp.SO3(torch.Tensor(self.rotation))
            rotations = torch.cat((rotations, active), dim=0)
        rotations = rotations[self.rotation_inverse]
        point = self.point if tracked else torch.Tensor(self.point)
        intrinsic = self.intrinsic if tracked else torch.Tensor(self.intrinsic)
        return rotations, centers, point, intrinsic

    def factor_args(self, tracked):
        r, c, p, k = self.tables(tracked)
        ci, pi, ii = self.camera_ids, self.point_ids, self.intrinsic_ids
        return (
            r[ci],
            c[ci],
            p[pi],
            k[ii],
            self.base_centers[ci],
            self.center_mask[ci],
            self.intrinsic_base[ii],
            self.intrinsic_map.expand(len(ci), -1, -1),
            self.pixels,
        )

    def forward(self):
        return huber_factor(*self.factor_args(True), self.delta.expand(len(self.pixels), 1))

    def raw(self):
        return raw_projection(*self.factor_args(False))

    def objective(self):
        squared = self.raw().square().sum(-1)
        norm = squared.clamp_min(torch.finfo(squared.dtype).tiny).sqrt()
        return torch.where(
            squared > self.delta.square(), self.delta * (2 * norm - self.delta), squared
        ).sum()

    def values(self):
        r, c, p, k = self.tables(False)
        matrices = r.matrix().detach().clone()
        # Preserve fixed matrices exactly rather than round-tripping quaternions.
        matrices[self.constant_rotation_ids] = self.base_rotations[self.constant_rotation_ids]
        centers = self.base_centers + self.center_mask * (c - self.base_centers)
        intrinsics = self.intrinsic_base + (self.intrinsic_map @ k[..., None]).squeeze(-1)
        return matrices, centers.detach(), p.detach(), intrinsics.detach()


class _HuberObjective(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input, target=None):
        return [self.model()]

    def loss(self, input, target=None):
        return self.model.objective()


class _GradientStop(Exception):
    pass


class _ObservedPCG(PCG):
    def __init__(self, cfg):
        super().__init__(
            maxiter=cfg.conjugate_gradient.max_iterations, tol=cfg.conjugate_gradient.rtol
        )
        self.threshold = cfg.gradient_tolerance
        self.gradient_norm = math.inf
        self.step_norm = math.inf
        self.stopped = False

    def forward(self, a, b, *args, **kwargs):
        self.gradient_norm = float(b.abs().max())
        if self.gradient_norm <= self.threshold:
            self.stopped = True
            raise _GradientStop()
        result = super().forward(a, b, *args, **kwargs)
        self.step_norm = float(result.norm())
        return result


@torch.no_grad()
def solve_bae(
    rotations,
    centers,
    rotation_mask,
    center_mask,
    points,
    intrinsic_values,
    camera_ids,
    point_ids,
    intrinsic_ids,
    pixels,
    cfg,
    max_iterations,
):
    if cfg.principal_point_prior_sigma is not None and cfg.refine_principal_point:
        raise ValueError("BAE adapter does not yet support a principal-point prior")
    if cfg.linear_solver not in ("auto", "conjugate_gradient"):
        raise ValueError("BAE adapter currently supports LM + full-system PCG only")
    model = BAEResidual(
        rotations,
        centers,
        rotation_mask,
        center_mask,
        points,
        intrinsic_values,
        camera_ids,
        point_ids,
        intrinsic_ids,
        pixels,
        cfg.loss_scale,
    )
    solver = _ObservedPCG(cfg)
    optimizer = LM(
        model,
        solver=solver,
        strategy=pp.optim.strategy.TrustRegion(
            radius=1 / cfg.lambda_initial, low=max(cfg.step_quality_min, 1e-12), up=2.0, down=0.5**4
        ),
        reject=30,
    )
    optimizer.model = _HuberObjective(model)
    initial = 0.5 * float(model.objective())
    summary = SolveSummary(
        initial, initial, cost_history=[initial], linear_solver="bae_pcg", linear_iterations=-1
    )  # native BAE/PyPose does not expose the count
    failures = []

    def quiet(*items, **kwargs):
        if items and isinstance(items[0], Exception) and not isinstance(items[0], _GradientStop):
            failures.append(str(items[0]))

    with patch.object(optimizer_module, "print", quiet, create=True):
        for iteration in range(max_iterations):
            old_cost = summary.final_cost
            saved = [torch.Tensor(p).clone() for p in model.parameters()]
            summary.lambda_history.append(float(optimizer.param_groups[0]["damping"]))
            cost = 0.5 * float(optimizer.step(()))
            summary.iterations = iteration + 1
            summary.gradient_norm = solver.gradient_norm
            summary.rejected_steps += optimizer.reject_count
            if failures:
                raise RuntimeError("BAE linear solver failed: " + failures[-1])
            if solver.stopped:
                summary.termination_reason = "gradient_tolerance"
                break
            if not math.isfinite(cost) or cost > old_cost:
                # Native BAE can return the last increased loss at its reject
                # limit. Do not propagate that failed update into triangulation.
                for param, value in zip(model.parameters(), saved, strict=True):
                    torch.Tensor(param).copy_(value)
                summary.rejected_steps += 1
                summary.termination_reason = "rejected_step_limit"
                break
            summary.accepted_steps += 1
            summary.final_cost = cost
            summary.cost_history.append(cost)
            if old_cost - cost <= cfg.cost_tolerance * max(
                old_cost, torch.finfo(centers.dtype).tiny
            ):
                summary.termination_reason = "cost_tolerance"
                break
            if solver.step_norm <= cfg.parameter_tolerance:
                summary.termination_reason = "parameter_tolerance"
                break
    return (*model.values(), summary)
