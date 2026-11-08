"""Ragged track checks and deterministic matching on Torch's CUDA stream.

Persistent storage stays N slots long. Only candidate-sized temporary arrays are
allocated; geometry follows track_start/count into track_obs without padding.
The cache belongs to one merge call (fixed cameras, validity and thresholds).
It retains the previous candidate scan: an absent pair cannot reappear unchanged
because merges only union tracks, and every changed/reused slot has a new version.
"""

import torch
import warp as wp

wp.init()
# Match the float64 Torch reference without implicit fused multiply-add changes.
wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def _check_pairs(
    pairs: wp.array2d(dtype=wp.int64),
    versions: wp.array(dtype=wp.int64),
    starts: wp.array(dtype=wp.int64),
    counts: wp.array(dtype=wp.int64),
    observations: wp.array(dtype=wp.int64),
    points: wp.array2d(dtype=wp.float64),
    frame: wp.array(dtype=wp.int64),
    xy: wp.array2d(dtype=wp.float64),
    valid: wp.array(dtype=wp.bool),
    rotations: wp.array3d(dtype=wp.float64),
    centers: wp.array2d(dtype=wp.float64),
    calibration: wp.array2d(dtype=wp.float64),
    threshold: wp.float64,
    capacity: wp.int64,
    old_keys: wp.array(dtype=wp.int64),
    old_versions: wp.array2d(dtype=wp.int64),
    old_good: wp.array(dtype=wp.bool),
    keys: wp.array(dtype=wp.int64),
    pair_versions: wp.array2d(dtype=wp.int64),
    good: wp.array(dtype=wp.bool),
    xyz: wp.array2d(dtype=wp.float64),
    reused: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    a, b = pairs[i, 0], pairs[i, 1]
    key = a * capacity + b
    keys[i] = key
    va, vb = versions[a], versions[b]
    pair_versions[i, 0] = va
    pair_versions[i, 1] = vb
    na, nb = counts[a], counts[b]
    total = wp.float64(na + nb)
    x = (points[a, 0] * wp.float64(na) + points[b, 0] * wp.float64(nb)) / total
    y = (points[a, 1] * wp.float64(na) + points[b, 1] * wp.float64(nb)) / total
    z = (points[a, 2] * wp.float64(na) + points[b, 2] * wp.float64(nb)) / total
    xyz[i, 0] = x
    xyz[i, 1] = y
    xyz[i, 2] = z
    lo, hi = wp.int32(0), wp.int32(old_keys.shape[0])
    while lo < hi:
        mid = (lo + hi) // 2
        if old_keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    reused[i] = False
    if lo < old_keys.shape[0]:  # noqa: SIM102 - keep the bounds guard explicit
        if (
            old_keys[lo] == key
            and old_versions[lo, 0] == va
            and old_versions[lo, 1] == vb
        ):
            good[i] = old_good[lo]
            reused[i] = True
            return
    accepted = wp.isfinite(x) and wp.isfinite(y) and wp.isfinite(z)
    if na <= 0 or nb <= 0:
        accepted = False
    # One CUDA thread per pair; ragged serial loops permit early rejection and
    # avoid both padded scratch tensors and cross-thread reduction barriers.
    for side in range(2):
        point = a
        if side == 1:
            point = b
        j = wp.int64(0)
        while accepted and j < counts[point]:
            obs = observations[starts[point] + j]
            fid = frame[obs]
            dx, dy, dz = x - centers[fid, 0], y - centers[fid, 1], z - centers[fid, 2]
            qx = (
                rotations[fid, 0, 0] * dx
                + rotations[fid, 0, 1] * dy
                + rotations[fid, 0, 2] * dz
            )
            qy = (
                rotations[fid, 1, 0] * dx
                + rotations[fid, 1, 1] * dy
                + rotations[fid, 1, 2] * dz
            )
            qz = (
                rotations[fid, 2, 0] * dx
                + rotations[fid, 2, 1] * dy
                + rotations[fid, 2, 2] * dz
            )
            if (
                not valid[obs]
                or not wp.isfinite(qx)
                or not wp.isfinite(qy)
                or not wp.isfinite(qz)
                or qz < wp.float64(1e-12)
            ):
                accepted = False
            else:
                ex = qx / qz * calibration[fid, 0] + calibration[fid, 2] - xy[obs, 0]
                ey = qy / qz * calibration[fid, 1] + calibration[fid, 3] - xy[obs, 1]
                error = wp.sqrt(ex * ex + ey * ey)
                if not wp.isfinite(error) or error > threshold:
                    accepted = False
            j += wp.int64(1)
    good[i] = accepted


@wp.kernel
def _propose(
    pairs: wp.array2d(dtype=wp.int64),
    good: wp.array(dtype=wp.bool),
    occupied: wp.array(dtype=wp.int32),
    winner: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    a, b = pairs[i, 0], pairs[i, 1]
    if good[i] and occupied[a] == 0 and occupied[b] == 0:
        wp.atomic_min(winner, a, i)
        wp.atomic_min(winner, b, i)


@wp.kernel
def _select(
    pairs: wp.array2d(dtype=wp.int64),
    winner: wp.array(dtype=wp.int32),
    occupied: wp.array(dtype=wp.int32),
    selected: wp.array(dtype=wp.bool),
    changed: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    a, b = pairs[i, 0], pairs[i, 1]
    if winner[a] == i and winner[b] == i:
        occupied[a] = 1
        occupied[b] = 1
        selected[i] = True
        wp.atomic_add(changed, 0, 1)


def check_pairs(tri, pairs, cache=None):
    """Pairs must be unique, canonical and lexicographically sorted."""
    s = tri.storage
    if s.capacity > 3037000499:
        raise ValueError("Point capacity exceeds int64 pair-key range")
    keys = pairs.new_empty(len(pairs))
    versions = torch.empty_like(pairs)
    good = torch.empty(len(pairs), dtype=torch.bool, device=pairs.device)
    reused = torch.empty_like(good)
    xyz = s.xy.new_empty(len(pairs), 3)
    if cache is None:
        cache = (keys[:0], versions[:0], good[:0])
    if len(pairs):
        arrays = [
            pairs,
            s.point_version,
            s.track_start,
            s.point_num_obs,
            s.track_obs,
            s.point_xyz,
            s.frame_id,
            s.xy,
            s.obs_valid,
            tri.rotations,
            tri.centers,
            tri.calibration,
        ]
        wp.launch(
            _check_pairs,
            dim=len(pairs),
            inputs=[
                *[wp.from_torch(t) for t in arrays],
                tri.config.merge_max_reproj_error,
                s.capacity,
                *[
                    wp.from_torch(t)
                    for t in (*cache, keys, versions, good, xyz, reused)
                ],
            ],
            stream=wp.stream_from_torch(torch.cuda.current_stream(pairs.device)),
        )
    return good, xyz, (keys, versions, good), reused


def select_pairs(pairs, good, capacity, maximal=True):
    """Deterministic maximal matching, or one mutual-priority round for parity.

    All subrounds use the same geometry verdicts. The lowest-priority remaining edge
    always wins, so progress is guaranteed. This is maximal, not maximum matching.
    """
    selected = torch.zeros_like(good)
    occupied = torch.zeros(capacity, dtype=torch.int32, device=pairs.device)
    winner = torch.empty_like(occupied)
    changed = occupied.new_empty(1)
    rounds = 0
    if not len(pairs):
        return selected, rounds
    if len(pairs) >= 2147483647:
        raise ValueError("Too many matching candidates for int32 priority")
    stream = wp.stream_from_torch(torch.cuda.current_stream(pairs.device))
    p, g, o, w, s, c = [
        wp.from_torch(t) for t in (pairs, good, occupied, winner, selected, changed)
    ]
    while True:
        winner.fill_(2147483647)
        changed.zero_()
        wp.launch(_propose, dim=len(pairs), inputs=[p, g, o, w], stream=stream)
        wp.launch(_select, dim=len(pairs), inputs=[p, w, o, s, c], stream=stream)
        rounds += 1
        if not maximal or changed.item() == 0:
            break
    return selected, rounds


def merge_tracks(tri):
    s = tri.storage
    merged = 0
    cache = None
    rounds, match_rounds = 0, 0
    evaluated, reused_count, candidates_count = 0, 0, 0
    # Accumulate counters on device; read them only once at the end.
    counters = torch.zeros(2, dtype=torch.int64, device=s.xy.device)
    while True:
        candidates = []
        for src, dst in tri._edge_batches():
            a, b = s.point_id[src], s.point_id[dst]
            valid = (a >= 0) & (b >= 0) & (a != b)
            candidates.append(
                torch.sort(torch.stack((a[valid], b[valid]), -1), dim=-1).values
            )
        if not candidates:
            break
        pairs = torch.unique(torch.cat(candidates), dim=0, sorted=True)
        if not len(pairs):
            break
        good, xyz, next_cache, reused = check_pairs(tri, pairs, cache)
        if tri.config.tensor_merge_cache:
            cache = next_cache
        counters[0] += reused.sum()
        counters[1] += good.sum()
        candidates_count += len(pairs)
        selected, steps = select_pairs(
            pairs,
            good,
            s.capacity,
            maximal=tri.config.tensor_merge_matching == "maximal",
        )
        match_rounds += steps
        chosen = pairs[selected]
        if not len(chosen):
            break
        rounds += 1
        merged += int(s.point_num_obs[chosen].sum())
        s.merge(chosen, xyz[selected])
    reused_count, accepted_checks = counters.tolist()
    evaluated = candidates_count - reused_count
    tri.merge_stats = {
        "backend": "warp",
        "matching": tri.config.tensor_merge_matching,
        "cache": tri.config.tensor_merge_cache,
        "merge_rounds": rounds,
        "matching_rounds": match_rounds,
        "candidate_checks": candidates_count,
        "geometry_checks": evaluated,
        "cache_hits": reused_count,
        "accepted_checks": accepted_checks,
        "padding_slots": 0,
    }
    return merged
