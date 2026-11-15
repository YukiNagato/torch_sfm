"""Exact triangulation-angle predicate on ragged tracks, without padding."""

from typing import Any

import torch
import warp as wp

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})


@wp.kernel
def _angles(
    directions: wp.array2d(dtype=Any),
    order: wp.array(dtype=wp.int64),
    starts: wp.array(dtype=wp.int64),
    counts: wp.array(dtype=wp.int64),
    threshold: Any,
    good: wp.array(dtype=wp.bool),
):
    track = wp.tid()
    start, end = starts[track], starts[track] + counts[track]
    good[track] = threshold >= 1.0
    # A NaN poisons the minimum in the Torch reference, even when another pair
    # passes. Check the whole track before taking an early successful exit.
    i = start
    while i < end:
        a = order[i]
        if (not wp.isfinite(directions[a, 0]) or not wp.isfinite(directions[a, 1])
                or not wp.isfinite(directions[a, 2])):
            good[track] = False
            return
        i += wp.int64(1)
    i = start
    while i < end:
        a = order[i]
        j = i
        while j < end:
            b = order[j]
            cosine = wp.abs(directions[a, 0] * directions[b, 0]
                          + directions[a, 1] * directions[b, 1]
                          + directions[a, 2] * directions[b, 2])
            if cosine <= threshold:
                good[track] = True
                return
            j += wp.int64(1)
        i += wp.int64(1)


def track_angle_mask(directions, order, starts, counts, threshold):
    wp.init()
    result = torch.empty(len(counts), dtype=torch.bool, device=directions.device)
    scalar = wp.float64 if directions.dtype == torch.float64 else wp.float32
    wp.launch(_angles, dim=len(counts),
        inputs=[wp.from_torch(directions), wp.from_torch(order), wp.from_torch(starts),
                wp.from_torch(counts), scalar(threshold), wp.from_torch(result)],
        stream=wp.stream_from_torch(torch.cuda.current_stream(directions.device)))
    return result
