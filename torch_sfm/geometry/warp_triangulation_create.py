"""Fused float64 two-view SVD and scoring, with Torch multi-view eigh.

Sampling, local-refinement acceptance and ownership arbitration follow the Torch
reference. No normal-equation replacement for the two-view SVD is introduced.
Candidate scoring streams observations instead of allocating [jobs,pairs,views].
"""

import math

import torch
import warp as wp

from .warp_svd import smallest_right_vector4

# ruff: noqa: SIM102 -- explicit guards protect indexed accesses in Warp kernels

wp.init()
wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.func
def _finite(value: wp.float64):
    result = value
    if wp.isnan(value):
        result = wp.float64(0.0)
    elif wp.isinf(value):
        result = wp.float64(1.7976931348623157e308)
        if value < wp.float64(0.0):
            result = -result
    return result


@wp.func
def _camera(
    x: wp.vec3d,
    fid: wp.int64,
    rotations: wp.array3d(dtype=wp.float64),
    centers: wp.array2d(dtype=wp.float64),
):
    d = x - wp.vec3d(centers[fid, 0], centers[fid, 1], centers[fid, 2])
    q = wp.vec3d()
    for r in range(3):
        q[r] = (
            rotations[fid, r, 0] * d[0]
            + rotations[fid, r, 1] * d[1]
            + rotations[fid, r, 2] * d[2]
        )
    return q


@wp.func
def _angular(
    x: wp.vec3d,
    node: wp.int64,
    frame: wp.array(dtype=wp.int64),
    valid: wp.array(dtype=wp.bool),
    rays: wp.array2d(dtype=wp.float64),
    rotations: wp.array3d(dtype=wp.float64),
    centers: wp.array2d(dtype=wp.float64),
):
    error = wp.float64(wp.inf)
    if node >= wp.int64(0):
        if valid[node]:
            q = _camera(x, frame[node], rotations, centers)
            if (
                wp.isfinite(q[0])
                and wp.isfinite(q[1])
                and wp.isfinite(q[2])
                and q[2] >= wp.float64(1e-12)
            ):
                ray = wp.vec3d(rays[node, 0], rays[node, 1], rays[node, 2])
                cosine = wp.dot(q, ray) / wp.length(q)
                if wp.isfinite(cosine):
                    error = wp.acos(wp.clamp(cosine, wp.float64(-1.0), wp.float64(1.0)))
    return error


@wp.func
def _projectable(
    x: wp.vec3d,
    node: wp.int64,
    frame: wp.array(dtype=wp.int64),
    valid: wp.array(dtype=wp.bool),
    xy: wp.array2d(dtype=wp.float64),
    rotations: wp.array3d(dtype=wp.float64),
    centers: wp.array2d(dtype=wp.float64),
    calibration: wp.array2d(dtype=wp.float64),
):
    result = False
    if node >= wp.int64(0):
        if valid[node]:
            fid = frame[node]
            q = _camera(x, fid, rotations, centers)
            if (
                wp.isfinite(q[0])
                and wp.isfinite(q[1])
                and wp.isfinite(q[2])
                and q[2] >= wp.float64(1e-12)
            ):
                ex = (
                    q[0] / q[2] * calibration[fid, 0]
                    + calibration[fid, 2]
                    - xy[node, 0]
                )
                ey = (
                    q[1] / q[2] * calibration[fid, 1]
                    + calibration[fid, 3]
                    - xy[node, 1]
                )
                result = wp.isfinite(wp.sqrt(ex * ex + ey * ey))
    return result


@wp.func
def _wide(
    x: wp.vec3d,
    i: wp.int64,
    j: wp.int64,
    frame: wp.array(dtype=wp.int64),
    centers: wp.array2d(dtype=wp.float64),
    cosine: wp.float64,
):
    a, b = frame[i], frame[j]
    u = x - wp.vec3d(centers[a, 0], centers[a, 1], centers[a, 2])
    v = x - wp.vec3d(centers[b, 0], centers[b, 1], centers[b, 2])
    u = u / wp.max(wp.length(u), wp.float64(1e-12))
    v = v / wp.max(wp.length(v), wp.float64(1e-12))
    return wp.abs(wp.dot(u, v)) <= cosine


@wp.kernel
def _prepare(
    nodes: wp.array2d(dtype=wp.int64),
    pairs: wp.array2d(dtype=wp.int64),
    frame: wp.array(dtype=wp.int64),
    rays: wp.array2d(dtype=wp.float64),
    poses: wp.array3d(dtype=wp.float64),
    matrices: wp.array4d(dtype=wp.float64),
):
    b, h = wp.tid()
    for side in range(2):
        node = wp.max(nodes[b, pairs[h, side]], wp.int64(0))
        fid = frame[node]
        for r in range(2):
            uv = rays[node, r] / rays[node, 2]
            for c in range(4):
                matrices[b, h, 2 * side + r, c] = _finite(
                    uv * poses[fid, 2, c] - poses[fid, r, c]
                )


@wp.kernel
def _score(
    nodes: wp.array2d(dtype=wp.int64),
    pairs: wp.array2d(dtype=wp.int64),
    vh: wp.array4d(dtype=wp.float64),
    frame: wp.array(dtype=wp.int64),
    valid: wp.array(dtype=wp.bool),
    rays: wp.array2d(dtype=wp.float64),
    xy: wp.array2d(dtype=wp.float64),
    rotations: wp.array3d(dtype=wp.float64),
    centers: wp.array2d(dtype=wp.float64),
    calibration: wp.array2d(dtype=wp.float64),
    angle: wp.float64,
    cosine: wp.float64,
    xyz: wp.array3d(dtype=wp.float64),
    counts: wp.array2d(dtype=wp.int64),
    errors: wp.array2d(dtype=wp.float64),
):
    b, h = wp.tid()
    denom = vh[b, h, 3, 3]
    homogeneous = wp.abs(denom) > wp.float64(1e-300)
    if not homogeneous:
        denom = wp.float64(1.0)
    x = wp.vec3d(vh[b, h, 3, 0], vh[b, h, 3, 1], vh[b, h, 3, 2]) / denom
    for c in range(3):
        xyz[b, h, c] = x[c]
    a, d = nodes[b, pairs[h, 0]], nodes[b, pairs[h, 1]]
    stable = False
    if homogeneous and a >= wp.int64(0) and d >= wp.int64(0):
        stable = (
            _projectable(x, a, frame, valid, xy, rotations, centers, calibration)
            and _projectable(x, d, frame, valid, xy, rotations, centers, calibration)
            and _wide(x, a, d, frame, centers, cosine)
        )
    count = wp.int64(0)
    error_sum = wp.float64(0.0)
    if stable:
        for k in range(nodes.shape[1]):
            e = _angular(x, nodes[b, k], frame, valid, rays, rotations, centers)
            if e <= angle:
                count += wp.int64(1)
                error_sum += e * e
    counts[b, h] = count
    errors[b, h] = error_sum


@wp.kernel
def _solve_score(
    nodes: wp.array2d(dtype=wp.int64),
    pairs: wp.array2d(dtype=wp.int64),
    poses: wp.array3d(dtype=wp.float64),
    frame: wp.array(dtype=wp.int64),
    valid: wp.array(dtype=wp.bool),
    rays: wp.array2d(dtype=wp.float64),
    xy: wp.array2d(dtype=wp.float64),
    rotations: wp.array3d(dtype=wp.float64),
    centers: wp.array2d(dtype=wp.float64),
    calibration: wp.array2d(dtype=wp.float64),
    angle: wp.float64,
    cosine: wp.float64,
    xyz: wp.array3d(dtype=wp.float64),
    counts: wp.array2d(dtype=wp.int64),
    errors: wp.array2d(dtype=wp.float64),
):
    b, h = wp.tid()
    matrix = wp.mat44d()
    for side in range(2):
        node = wp.max(nodes[b, pairs[h, side]], wp.int64(0))
        fid = frame[node]
        for r in range(2):
            uv = rays[node, r] / rays[node, 2]
            for c in range(4):
                matrix[2 * side + r, c] = _finite(
                    uv * poses[fid, 2, c] - poses[fid, r, c]
                )
    vector = smallest_right_vector4(matrix)
    denom = vector[3]
    homogeneous = wp.abs(denom) > wp.float64(1e-300)
    if not homogeneous:
        denom = wp.float64(1.0)
    x = wp.vec3d(vector[0], vector[1], vector[2]) / denom
    for c in range(3):
        xyz[b, h, c] = x[c]
    a, d = nodes[b, pairs[h, 0]], nodes[b, pairs[h, 1]]
    stable = False
    if homogeneous and a >= wp.int64(0) and d >= wp.int64(0):
        stable = (
            _projectable(x, a, frame, valid, xy, rotations, centers, calibration)
            and _projectable(x, d, frame, valid, xy, rotations, centers, calibration)
            and _wide(x, a, d, frame, centers, cosine)
        )
    count = wp.int64(0)
    error_sum = wp.float64(0.0)
    if stable:
        for k in range(nodes.shape[1]):
            e = _angular(x, nodes[b, k], frame, valid, rays, rotations, centers)
            if e <= angle:
                count += wp.int64(1)
                error_sum += e * e
    counts[b, h] = count
    errors[b, h] = error_sum


@wp.kernel
def _choose(
    nodes: wp.array2d(dtype=wp.int64),
    proposed: wp.array3d(dtype=wp.float64),
    counts: wp.array2d(dtype=wp.int64),
    errors: wp.array2d(dtype=wp.float64),
    frame: wp.array(dtype=wp.int64),
    valid: wp.array(dtype=wp.bool),
    rays: wp.array2d(dtype=wp.float64),
    rotations: wp.array3d(dtype=wp.float64),
    centers: wp.array2d(dtype=wp.float64),
    angle: wp.float64,
    adaptive: int,
    start: int,
    trials: int,
    log_failure: wp.float64,
    xyz: wp.array2d(dtype=wp.float64),
    inside: wp.array2d(dtype=wp.bool),
    best_count: wp.array(dtype=wp.int64),
    best_error: wp.array(dtype=wp.float64),
    needed: wp.array(dtype=wp.int64),
    active: wp.array(dtype=wp.bool),
):
    b = wp.tid()
    if needed[b] <= wp.int64(start):
        return
    best = wp.int32(-1)
    nc = wp.int64(best_count[b])
    ne = wp.float64(best_error[b])
    for h in range(counts.shape[1]):
        count = wp.int64(counts[b, h])
        error = wp.float64(errors[b, h])
        if count > nc or (count == nc and error < ne):
            best = h
            nc = count
            ne = error
    if best >= 0:
        x = wp.vec3d(proposed[b, best, 0], proposed[b, best, 1], proposed[b, best, 2])
        for c in range(3):
            xyz[b, c] = x[c]
        for k in range(nodes.shape[1]):
            # Unstable hypotheses have zero support even if their position
            # happens to project close to some observations.
            inside[b, k] = (
                nc > wp.int64(0)
                and _angular(x, nodes[b, k], frame, valid, rays, rotations, centers)
                <= angle
            )
        best_count[b] = nc
        best_error[b] = ne
    active[b] = nc > wp.int64(2)
    if adaptive != 0:
        total = wp.int64(0)
        for k in range(nodes.shape[1]):
            node = nodes[b, k]
            if node >= wp.int64(0):
                if valid[node]:
                    total += wp.int64(1)
        prob = wp.float64(nc * (nc - wp.int64(1))) / wp.float64(
            wp.max(total * (total - wp.int64(1)), wp.int64(1))
        )
        prob = wp.clamp(prob, wp.float64(1e-12), wp.float64(1.0 - 1e-12))
        budget = wp.ceil(log_failure / wp.log(wp.float64(1.0) - prob))
        needed[b] = wp.int64(wp.clamp(budget, wp.float64(1.0), wp.float64(trials)))


@wp.kernel
def _normal(
    nodes: wp.array2d(dtype=wp.int64),
    inside: wp.array2d(dtype=wp.bool),
    active: wp.array(dtype=wp.bool),
    frame: wp.array(dtype=wp.int64),
    rays: wp.array2d(dtype=wp.float64),
    poses: wp.array3d(dtype=wp.float64),
    matrix: wp.array3d(dtype=wp.float64),
):
    b = wp.tid()
    normal = wp.mat44d()
    if active[b]:
        for k in range(nodes.shape[1]):
            if inside[b, k]:
                node = nodes[b, k]
                fid = frame[node]
                a = wp.mat44d()
                for c in range(4):
                    dot = (
                        rays[node, 0] * poses[fid, 0, c]
                        + rays[node, 1] * poses[fid, 1, c]
                        + rays[node, 2] * poses[fid, 2, c]
                    )
                    for r in range(3):
                        a[r, c] = _finite(poses[fid, r, c] - rays[node, r] * dot)
                for r in range(4):
                    for c in range(4):
                        normal[r, c] += (
                            a[0, r] * a[0, c] + a[1, r] * a[1, c] + a[2, r] * a[2, c]
                        )
    else:
        for c in range(4):
            normal[c, c] = wp.float64(1.0)
    for r in range(4):
        for c in range(4):
            matrix[b, r, c] = normal[r, c]


@wp.kernel
def _refine(
    nodes: wp.array2d(dtype=wp.int64),
    eigenvectors: wp.array3d(dtype=wp.float64),
    frame: wp.array(dtype=wp.int64),
    valid: wp.array(dtype=wp.bool),
    rays: wp.array2d(dtype=wp.float64),
    xy: wp.array2d(dtype=wp.float64),
    rotations: wp.array3d(dtype=wp.float64),
    centers: wp.array2d(dtype=wp.float64),
    calibration: wp.array2d(dtype=wp.float64),
    angle: wp.float64,
    cosine: wp.float64,
    xyz: wp.array2d(dtype=wp.float64),
    inside: wp.array2d(dtype=wp.bool),
    best_count: wp.array(dtype=wp.int64),
    best_error: wp.array(dtype=wp.float64),
    active: wp.array(dtype=wp.bool),
):
    b = wp.tid()
    if not active[b]:
        return
    active[b] = False
    denom = eigenvectors[b, 3, 0]
    if wp.abs(denom) <= wp.float64(1e-300):
        return
    x = (
        wp.vec3d(eigenvectors[b, 0, 0], eigenvectors[b, 1, 0], eigenvectors[b, 2, 0])
        / denom
    )
    stable = wp.isfinite(x[0]) and wp.isfinite(x[1]) and wp.isfinite(x[2])
    wide = wp.bool(False)
    nc = wp.int64(0)
    ne = wp.float64(0.0)
    for k in range(nodes.shape[1]):
        node = nodes[b, k]
        e = _angular(x, node, frame, valid, rays, rotations, centers)
        if e <= angle:
            nc += wp.int64(1)
            ne += e * e
        if inside[b, k]:
            if not _projectable(
                x, node, frame, valid, xy, rotations, centers, calibration
            ):
                stable = False
            if not wide:
                for j in range(k + 1, nodes.shape[1]):
                    if inside[b, j]:
                        if _wide(x, node, nodes[b, j], frame, centers, cosine):
                            wide = True
    if (
        stable
        and wide
        and (nc > best_count[b] or (nc == best_count[b] and ne < best_error[b]))
    ):
        active[b] = nc != best_count[b] and nc > wp.int64(2)
        best_count[b] = nc
        best_error[b] = ne
        for c in range(3):
            xyz[b, c] = x[c]
        for k in range(nodes.shape[1]):
            inside[b, k] = (
                _angular(x, nodes[b, k], frame, valid, rays, rotations, centers)
                <= angle
            )


@wp.kernel
def _finish(inside: wp.array2d(dtype=wp.bool), counts: wp.array(dtype=wp.int64)):
    b, k = wp.tid()
    if counts[b] < wp.int64(2):
        inside[b, k] = False


def estimate(tri, nodes, *, fused=True):
    """Same hypothesis schedule; fused SVD/scoring or the Torch SVD reference."""
    s = tri.storage
    b, width = nodes.shape
    xyz = s.xy.new_zeros(b, 3)
    inside = torch.zeros_like(nodes, dtype=torch.bool)
    if b == 0 or width < 2:
        return xyz, inside
    pairs = (
        torch.triu_indices(width, width, offset=1, device=nodes.device)
        .T[: tri.config.max_num_trials]
        .contiguous()
    )
    adaptive = width > tri.config.exhaustive_max_track_length
    step = tri.config.tensor_hypothesis_batch if adaptive else len(pairs)
    best_count = nodes.new_zeros(b)
    best_error = s.xy.new_full((b,), math.inf)
    needed = nodes.new_full((b,), len(pairs))
    active = torch.zeros(b, dtype=torch.bool, device=nodes.device)
    stream = wp.stream_from_torch(torch.cuda.current_stream(nodes.device))
    arrays = {
        name: wp.from_torch(value)
        for name, value in {
            "nodes": nodes,
            "frame": s.frame_id,
            "valid": s.obs_valid,
            "rays": s.ray,
            "xy": s.xy,
            "rotations": tri.rotations,
            "centers": tri.centers,
            "calibration": tri.calibration,
            "poses": tri.poses,
            "xyz": xyz,
            "inside": inside,
            "best_count": best_count,
            "best_error": best_error,
            "needed": needed,
            "active": active,
        }.items()
    }

    def arr(*names):
        return [arrays[name] for name in names]

    angle = math.radians(tri.config.create_max_angle_error)
    cosine = math.cos(math.radians(tri.config.min_angle))
    for start in range(0, len(pairs), step):
        if start and not bool((needed > start).any()):
            break
        p = pairs[start : start + step]
        if not fused:
            matrices = s.xy.new_empty(b, len(p), 4, 4)
            wp.launch(
                _prepare,
                dim=(b, len(p)),
                inputs=[
                    arrays["nodes"],
                    wp.from_torch(p),
                    *arr("frame", "rays", "poses"),
                    wp.from_torch(matrices),
                ],
                stream=stream,
            )
            _, _, vh = torch.linalg.svd(matrices)
        candidates = s.xy.new_empty(b, len(p), 3)
        counts = nodes.new_empty(b, len(p))
        errors = s.xy.new_empty(b, len(p))
        wp.launch(
            _solve_score if fused else _score,
            dim=(b, len(p)),
            inputs=[
                arrays["nodes"],
                wp.from_torch(p),
                arrays["poses"] if fused else wp.from_torch(vh),
                *arr(
                    "frame",
                    "valid",
                    "rays",
                    "xy",
                    "rotations",
                    "centers",
                    "calibration",
                ),
                angle,
                cosine,
                *[wp.from_torch(t) for t in (candidates, counts, errors)],
            ],
            stream=stream,
        )
        wp.launch(
            _choose,
            dim=b,
            inputs=[
                arrays["nodes"],
                *[wp.from_torch(t) for t in (candidates, counts, errors)],
                *arr("frame", "valid", "rays", "rotations", "centers"),
                angle,
                int(adaptive),
                start,
                len(pairs),
                3 * math.log1p(-tri.config.confidence),
                *arr("xyz", "inside", "best_count", "best_error", "needed", "active"),
            ],
            stream=stream,
        )
    for _ in range(tri.config.local_iterations):
        if not bool(active.any()):
            break
        matrix = s.xy.new_empty(b, 4, 4)
        wp.launch(
            _normal,
            dim=b,
            inputs=[
                *arr("nodes", "inside", "active", "frame", "rays", "poses"),
                wp.from_torch(matrix),
            ],
            stream=stream,
        )
        _, vectors = torch.linalg.eigh(matrix)
        wp.launch(
            _refine,
            dim=b,
            inputs=[
                arrays["nodes"],
                wp.from_torch(vectors),
                *arr(
                    "frame",
                    "valid",
                    "rays",
                    "xy",
                    "rotations",
                    "centers",
                    "calibration",
                ),
                angle,
                cosine,
                *arr("xyz", "inside", "best_count", "best_error", "active"),
            ],
            stream=stream,
        )
    wp.launch(
        _finish, dim=nodes.shape, inputs=arr("inside", "best_count"), stream=stream
    )
    return xyz, inside
