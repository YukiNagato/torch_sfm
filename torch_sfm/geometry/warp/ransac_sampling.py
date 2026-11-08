"""Exact hash-top-k RANSAC samples and point gathering in one Warp launch.

Each hypothesis owns a warp. Lanes keep their best seven keys, then cooperate
to emit the global top k. No dense (pairs, hypotheses, capacity) score tensor.
The 32-bit hash is a permutation, so eligible point keys have no ties.
"""
import warp as wp

wp.init()
wp.set_module_options({'enable_backward': False, 'max_unroll': 1})
Keys = wp.types.vector(length=7, dtype=wp.int64)
Indices = wp.types.vector(length=7, dtype=wp.int32)


@wp.kernel
def _sample(points0: wp.array3d(dtype=wp.float64), points1: wp.array3d(dtype=wp.float64),
            valid: wp.array2d(dtype=wp.bool), counts: wp.array(dtype=wp.int64),
            active: wp.array(dtype=wp.int64), trial: wp.int64, bias: wp.uint32,
            size: int, batch: int,
            out0: wp.array3d(dtype=wp.float64), out1: wp.array3d(dtype=wp.float64)):
    pair, hypothesis, lane = wp.tid()
    target = active[pair]
    keys, indices = Keys(), Indices()
    for k in range(7):
        keys[k] = wp.int64(-1)
    for point in range(lane, valid.shape[1], 32):
        if valid[target, point]:
            counter = (trial + wp.int64(hypothesis))*counts[target] + wp.int64(point)
            bits = wp.uint32(counter) + bias
            bits = (bits ^ (bits >> wp.uint32(16)))*wp.uint32(2246822519)
            bits = (bits ^ (bits >> wp.uint32(13)))*wp.uint32(3266489917)
            bits = bits ^ (bits >> wp.uint32(16))
            key, index = wp.int64(bits), point
            for k in range(size):
                if key > keys[k]:
                    previous_key, previous_index = keys[k], indices[k]
                    keys[k] = key
                    indices[k] = index
                    key, index = previous_key, previous_index
    for k in range(size):
        maximum = wp.tile_extract(wp.tile_max(wp.tile(keys[0])), 0)
        local_lane = int(32)
        if keys[0] == maximum:
            local_lane = lane
        owner = wp.tile_extract(wp.tile_min(wp.tile(local_lane)), 0)
        point = wp.tile_extract(wp.tile(indices[0]), owner)
        if lane == 0:
            row = pair*batch + hypothesis
            for j in range(2):
                out0[row, k, j] = points0[target, point, j]
                out1[row, k, j] = points1[target, point, j]
        if lane == owner:
            for shift in range(size-1):
                keys[shift] = keys[shift+1]
                indices[shift] = indices[shift+1]
            keys[size-1] = wp.int64(-1)
