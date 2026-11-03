"""GlobalMapper's staged BA and incremental-refinement policy in Python."""

import math
from dataclasses import asdict

import torch
import torch.nn.functional as F

from .bundle_adjustment import (
    BundleAdjustmentResult,
    _compact,
    _error_stats,
    _measurements,
    _observation_matrices,
    _pinhole_parameters,
    _solve_ba,
    reprojection_errors,
)
from .triangulation import IncrementalTriangulator


def normalize_reconstruction(centers, points, registered):
    selected = centers[registered]
    if len(selected) < 2:
        return centers, points
    ordered = selected.sort(dim=0).values
    low, high = (
        math.floor(0.1 * (len(selected) - 1)),
        math.ceil(0.9 * (len(selected) - 1)),
    )
    origin = ordered[low : high + 1].mean(0)
    extent = (ordered[high] - ordered[low]).norm()
    scale = 10 / extent if float(extent) >= torch.finfo(centers.dtype).eps else 1.0
    return (centers - origin) * scale, (points - origin) * scale


def triangulation_mask(centers, points, observations, minimum):
    if not len(points) or minimum == 0:
        return torch.ones(len(points), dtype=torch.bool, device=points.device)
    pid, fid, _ = observations.T
    directions = F.normalize(points[pid] - centers[fid], dim=-1)
    counts = torch.bincount(pid, minlength=len(points))
    order = pid.argsort(stable=True)
    offsets = counts.cumsum(0) - counts
    slots = torch.empty_like(pid)
    slots[order] = torch.arange(len(order), device=points.device) - offsets[pid[order]]
    padded = points.new_zeros((len(points), int(counts.max()), 3))
    valid = torch.zeros(padded.shape[:2], dtype=torch.bool, device=points.device)
    padded[pid, slots], valid[pid, slots] = directions, True
    minimum_cosine = points.new_ones(len(points))
    for slot in range(padded.shape[1]):
        cosine = (padded * padded[:, slot : slot + 1]).sum(-1).abs()
        pairs = valid & valid[:, slot : slot + 1]
        minimum_cosine = torch.minimum(
            minimum_cosine, cosine.masked_fill(~pairs, 1).amin(-1)
        )
    return minimum_cosine <= math.cos(math.radians(minimum))


def filter_reconstruction(
    frames,
    rotations,
    centers,
    points,
    observations,
    intrinsics,
    threshold,
    *,
    error_type="normalized",
    min_tri_angle=0.0,
):
    before = len(observations)
    if not before:
        return points[:0], observations.reshape(0, 3), 0
    pid, fid, _ = observations.T
    local = (rotations[fid] @ (points[pid] - centers[fid])[..., None]).squeeze(-1)
    positive = (local[:, 2] >= 1e-12) & torch.isfinite(local).all(-1)
    if error_type == "normalized":
        k = _observation_matrices(intrinsics, fid)
        xy = (_measurements(frames, observations, points) - k[:, :2, 2]) / k[
            :, [0, 1], [0, 1]
        ]
        error = (local[:, :2] / local[:, 2:].clamp_min(1e-12) - xy).norm(dim=-1)
    elif error_type == "pixel":
        k = _observation_matrices(intrinsics, fid)
        xy = _measurements(frames, observations, points)
        projected = local[:, :2] / local[:, 2:].clamp_min(1e-12)
        projected = projected * k[:, [0, 1], [0, 1]] + k[:, :2, 2]
        error = (projected - xy).norm(dim=-1)
    else:
        raise ValueError("error_type must be pixel or normalized")
    keep = positive & torch.isfinite(error) & (error <= threshold)
    points, observations = _compact(points, observations, keep, 2)
    if len(observations) and min_tri_angle:
        good = triangulation_mask(centers, points, observations, min_tri_angle)
        points, observations = _compact(
            points, observations, good[observations[:, 0]], 2
        )
    return points, observations, before - len(observations)


def global_bundle_adjustment(
    frames,
    rotations,
    centers,
    points3d,
    observations,
    intrinsics,
    registered,
    cfg,
    geometries=None,
):
    rotations = rotations.detach().clone()
    centers = centers.detach().to(rotations).clone()
    points = points3d.detach().to(rotations).clone()
    observations = (
        observations.detach().to(device=points.device, dtype=torch.long).clone()
    )
    intrinsics = intrinsics.detach().to(rotations).clone()
    registered = registered.detach().clone()
    if rotations.shape != (len(frames), 3, 3) or centers.shape != (len(frames), 3):
        raise ValueError("BA expects one rotation and center per frame")
    _pinhole_parameters(intrinsics)
    original_errors, _ = reprojection_errors(
        frames, rotations, centers, points, observations, intrinsics
    )
    stats = {
        "enabled": cfg.enabled,
        "strategy": "glomap",
        "initial": _error_stats(original_errors),
        "stages": [],
        "filters": [],
        "refinements": [],
    }
    summaries = []
    if not cfg.enabled or not len(observations):
        stats["skipped"] = "disabled" if not cfg.enabled else "no_tracks"
        stats["final"] = stats["initial"]
        return BundleAdjustmentResult(
            rotations,
            centers,
            points,
            observations,
            registered,
            intrinsics,
            original_errors,
            summaries,
            stats,
        )
    points, observations = _compact(
        points, observations, registered[observations[:, 1]], 2
    )

    def solve_stage(name, iterations, fixed=False):
        nonlocal rotations, centers, points, intrinsics
        stage_cfg = cfg
        before, _ = reprojection_errors(
            frames, rotations, centers, points, observations, intrinsics
        )
        counts = torch.bincount(observations[:, 0], minlength=len(points))
        admitted = int((counts * (counts >= cfg.min_track_length)).sum())
        initial_k = _pinhole_parameters(intrinsics).tolist()
        rotations, centers, points, intrinsics, summary = _solve_ba(
            frames,
            rotations,
            centers,
            points,
            observations,
            intrinsics,
            stage_cfg,
            fixed,
            iterations,
        )
        after, _ = reprojection_errors(
            frames, rotations, centers, points, observations, intrinsics
        )
        summaries.append(summary)
        stats["stages"].append(
            {
                "name": name,
                "points": len(points),
                "before": _error_stats(before),
                "after": _error_stats(after),
                "solver": asdict(summary),
                "optimized_observations": admitted,
                "refine_principal_point": cfg.refine_principal_point,
                "intrinsics_before": initial_k,
                "intrinsics_after": _pinhole_parameters(intrinsics).tolist(),
            }
        )

    def normalize():
        nonlocal centers, points
        if cfg.normalize:
            centers, points = normalize_reconstruction(centers, points, registered)

    def filter_stage(name, threshold, error_type="normalized", angle=0.0):
        nonlocal points, observations
        old_points, old_obs = len(points), len(observations)
        points, observations, removed = filter_reconstruction(
            frames,
            rotations,
            centers,
            points,
            observations,
            intrinsics,
            threshold,
            error_type=error_type,
            min_tri_angle=angle,
        )
        stats["filters"].append(
            {
                "name": name,
                "error_type": error_type,
                "threshold": threshold,
                "min_tri_angle": angle,
                "points_before": old_points,
                "points_after": len(points),
                "observations_before": old_obs,
                "observations_after": len(observations),
                "removed": removed,
            }
        )
        if not len(observations):
            raise RuntimeError("No tracks remain after GLOMAP filtering")
        return removed

    iteration = 0
    while iteration < cfg.num_rounds:
        if cfg.fixed_rotation_iterations:
            solve_stage(
                f"round_{iteration + 1}_fixed_rotation",
                cfg.fixed_rotation_iterations,
                True,
            )
        solve_stage(f"round_{iteration + 1}_joint", cfg.max_iterations)
        normalize()
        removed = 0
        while iteration < cfg.num_rounds:
            threshold = max(3 - iteration, 1) * cfg.max_normalized_reprojection_error
            removed += filter_stage(f"round_{iteration + 1}", threshold)
            if removed > cfg.filter_min_change * len(points):
                break
            iteration += 1
        iteration += 1
    filter_stage(
        "global_final", cfg.max_normalized_reprojection_error, angle=cfg.min_tri_angle
    )

    if cfg.retriangulate and geometries:
        old_points = len(points)
        triangulator_class = IncrementalTriangulator
        if cfg.triangulation.implementation == "speculative":
            from .triangulation_speculative import SpeculativeTriangulator

            triangulator_class = SpeculativeTriangulator
        elif cfg.triangulation.implementation == "tensor":
            from .triangulation_tensor import TensorTriangulator

            triangulator_class = TensorTriangulator
        triangulator = triangulator_class(
            frames, geometries, registered, cfg.triangulation
        )
        triangulator.update_geometry(rotations, centers, intrinsics)
        added = triangulator.triangulate()
        changed = triangulator.complete_and_merge()
        retriangulated = triangulator.retriangulate()
        points, observations = triangulator.export()
        stats["retriangulation"] = {
            "points_before": old_points,
            "points_after": len(points),
            "observations_after": len(observations),
            "added": added,
            **changed,
            "retriangulated": retriangulated,
            "candidate_batching": (
                dict(triangulator.candidate_cache.stats)
                if triangulator.candidate_cache
                else None
            ),
        }
        if not len(observations):
            raise RuntimeError("No tracks remain after GLOMAP retriangulation")
        for i in range(cfg.retriangulation_max_refinements):
            # IncrementalMapper removes negative depth before each global BA.
            filter_stage(f"refine_{i + 1}_depth", math.inf, "pixel")
            old_obs = len(observations)
            solve_stage(f"refine_{i + 1}_joint", cfg.retriangulation_max_iterations)
            normalize()
            triangulator.load(points, observations)
            triangulator.update_geometry(rotations, centers, intrinsics)
            changed = triangulator.complete_and_merge()
            points, observations = triangulator.export()
            removed = filter_stage(
                f"refine_{i + 1}",
                cfg.refinement_max_reprojection_error,
                "pixel",
                cfg.refinement_min_tri_angle,
            )
            ratio = (changed["completed"] + changed["merged"] + removed) / max(
                old_obs, 1
            )
            stats["refinements"].append(
                {
                    "iteration": i + 1,
                    **changed,
                    "filtered": removed,
                    "change_ratio": ratio,
                }
            )
            if ratio < cfg.retriangulation_min_change:
                break
        filter_stage(
            "retriangulation_final_before_ba", cfg.max_normalized_reprojection_error
        )
        solve_stage("retriangulation_final_joint", cfg.max_iterations)
        normalize()
        filter_stage(
            "retriangulation_final",
            cfg.max_normalized_reprojection_error,
            angle=cfg.min_tri_angle,
        )

    errors, positive = reprojection_errors(
        frames, rotations, centers, points, observations, intrinsics
    )
    stats["final"] = _error_stats(errors)
    stats["registered"] = int(registered.sum())
    stats["positive_depth"] = int(positive.sum())
    return BundleAdjustmentResult(
        rotations,
        centers,
        points,
        observations,
        registered,
        intrinsics,
        errors,
        summaries,
        stats,
    )
