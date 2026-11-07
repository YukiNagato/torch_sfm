"""Exact hash-top-k RANSAC samples and point gathering in one Warp launch.

Each hypothesis owns a warp. Lanes keep their best seven keys, then cooperate
to emit the global top k. No dense (pairs, hypotheses, capacity) score tensor.
The 32-bit hash is a permutation, so eligible point keys have no ties.
"""
import torch
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


class Sampler:
    """Per-RANSAC immutable views; active pairs must have at least k valid points.

No global mutable cache or cross-stream workspaces. Seeds and trial counters
match the original hash/topk sampler exactly, including 32-bit wraparound.
"""
    def __init__(self, points0, points1, valid, counts, size, seed):
        if size not in (4,5,6,7):
            raise ValueError('Expected RANSAC sample size 4..7')
        if (points0.ndim != 3 or points0.shape[-1] != 2 or points1.shape != points0.shape
                or valid.shape != points0.shape[:2] or counts.shape != (len(points0),)):
            raise ValueError('Invalid RANSAC observation shapes')
        if (not points0.is_cuda or points0.dtype != torch.float64 or points1.dtype != torch.float64
                or valid.dtype != torch.bool or counts.dtype != torch.int64
                or any(t.device != points0.device for t in (points1,valid,counts))):
            raise ValueError('Warp sampling requires CUDA float64 observations and int64 counts')
        if torch.is_grad_enabled() and (points0.requires_grad or points1.requires_grad):
            raise ValueError('Warp sampling is forward only')
        self.references = (points0,points1,valid,counts)
        self.views = [wp.from_torch(t) for t in self.references]
        self.size = size
        self.bias = wp.uint32(((seed % 2**32)*104729+374761393) & 0xffffffff)
        self.stream = wp.stream_from_torch(torch.cuda.current_stream(points0.device))

    def sample(self, active, trial, batch):
        if active.ndim != 1 or active.dtype != torch.int64 or active.device != self.references[0].device:
            raise ValueError('Expected CUDA int64 active pair indices')
        if trial < 0 or batch < 1:
            raise ValueError('Expected nonnegative trial and positive hypothesis batch')
        a = self.references[0].new_empty((len(active)*batch,self.size,2))
        b = torch.empty_like(a)
        if len(active):
            wp.launch(_sample,dim=(len(active),batch,32),block_dim=32,
                inputs=[*self.views,wp.from_torch(active),wp.int64(trial),self.bias,self.size,batch],
                outputs=[wp.from_torch(a),wp.from_torch(b)],stream=self.stream)
        return a,b
