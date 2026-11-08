"""One CUDA launch for six-point sampling, solving and Sampson scoring.

Experimental, forward-only, CUDA float64. One 32-thread block owns one sample.
Native snippets own block-local shared storage and score matching-point tiles.
There is no CPU recovery, output compaction, local refinement or RANSAC loop.
"""
from dataclasses import dataclass
import math

import torch
import warp as wp

from .six_point_math import _nullspace, _eliminate_cooperative, _action_roots, _recover
from .six_point_polynomial import coefficients
from ..geometry.warp_shared_focal_poselib import _indices

wp.set_module_options({'enable_backward': False, 'fast_math': False,
                       'fuse_fp': False, 'max_unroll': 1})
Keys = wp.types.vector(length=6, dtype=wp.int64)
Indices = wp.types.vector(length=6, dtype=wp.int32)


@wp.func_native('''
__shared__ double storage[1176];
return wp::array_t<double>(storage, 1176);
''')
def _scratch() -> wp.array(dtype=wp.float64): ...


@wp.func_native('''
__shared__ int storage[17];
return wp::array_t<int>(storage, 17);
''')
def _integers() -> wp.array(dtype=wp.int32): ...


@wp.func_native('''
__shared__ bool storage[60];
return wp::array_t<bool>(storage, 1, 60);
''')
def _flags() -> wp.array2d(dtype=wp.bool): ...


@wp.func_native('return wp::array_t<double>(a.data + offset, n);')
def _view1(a: wp.array(dtype=wp.float64), offset: int, n: int) -> wp.array(dtype=wp.float64): ...


@wp.func_native('return wp::array_t<double>(a.data + offset, n, m);')
def _view2(a: wp.array(dtype=wp.float64), offset: int, n: int, m: int) -> wp.array2d(dtype=wp.float64): ...


@wp.func_native('return wp::array_t<double>(a.data + offset, n, m, k);')
def _view3(a: wp.array(dtype=wp.float64), offset: int, n: int, m: int, k: int) -> wp.array3d(dtype=wp.float64): ...


@wp.func_native('return wp::array_t<double>(a.data + offset, n, m, k, l);')
def _view4(a: wp.array(dtype=wp.float64), offset: int, n: int, m: int, k: int, l: int) -> wp.array4d(dtype=wp.float64): ...


@wp.func_native('return wp::array_t<int>(a.data + offset, n);')
def _iview1(a: wp.array(dtype=wp.int32), offset: int, n: int) -> wp.array(dtype=wp.int32): ...


@wp.func_native('return wp::array_t<int>(a.data + offset, n, m);')
def _iview2(a: wp.array(dtype=wp.int32), offset: int, n: int, m: int) -> wp.array2d(dtype=wp.int32): ...


@wp.func_native('__syncthreads();')
def _sync(): ...


@wp.func_native('''
if (times.shape[0] && threadIdx.x == 0) wp::index(times, sample, stage) = clock64();
''')
def _mark(times: wp.array2d(dtype=wp.int64), sample: int, stage: int): ...


@wp.func_native('''
// Called collectively by one complete 32-thread block. All barrier paths are
// uniform; validity is a shared, immutable flag throughout scoring.
__shared__ double points[4][128];
__shared__ bool usable[128];
__shared__ double sums[60];
__shared__ int hits[60];
int lane = threadIdx.x;
bool any_valid = wp::index(valid, 0, lane);
if (lane + 32 < 60) any_valid = any_valid || wp::index(valid, 0, lane + 32);
if (__ballot_sync(0xffffffff, any_valid) == 0) return;
for (int s = lane; s < 60; s += 32) { sums[s] = 0.0; hits[s] = 0; }
__syncthreads();
int capacity = mask.shape[1];
for (int begin = 0; begin < capacity; begin += 128) {
    for (int t = lane; t < 128; t += 32) {
        int index = begin + t;
        bool good = index < capacity && wp::index(mask, pair, index);
        usable[t] = good;
        if (good) {
            points[0][t] = wp::index(p0, pair, index, 0);
            points[1][t] = wp::index(p0, pair, index, 1);
            points[2][t] = wp::index(p1, pair, index, 0);
            points[3][t] = wp::index(p1, pair, index, 1);
        }
    }
    __syncthreads();
    for (int s = 0; s < 60; ++s) {
        if (!wp::index(valid, 0, s)) continue;
        double total = 0.0;
        int count = 0;
        for (int t = lane; t < 128; t += 32) {
            if (!usable[t]) continue;
            double x = points[0][t], y = points[1][t];
            double u = points[2][t], v = points[3][t];
            double a = wp::index(models,0,s,0,0)*x + wp::index(models,0,s,0,1)*y + wp::index(models,0,s,0,2);
            double b = wp::index(models,0,s,1,0)*x + wp::index(models,0,s,1,1)*y + wp::index(models,0,s,1,2);
            double c = wp::index(models,0,s,2,0)*x + wp::index(models,0,s,2,1)*y + wp::index(models,0,s,2,2);
            double d = wp::index(models,0,s,0,0)*u + wp::index(models,0,s,1,0)*v + wp::index(models,0,s,2,0);
            double e = wp::index(models,0,s,0,1)*u + wp::index(models,0,s,1,1)*v + wp::index(models,0,s,2,1);
            double numerator = u*a + v*b + c;
            double error = numerator*numerator / fmax(a*a+b*b+d*d+e*e, 1.0e-30);
            if (!isfinite(error)) error = INFINITY;
            total += fmin(error, threshold);
            count += error <= threshold;
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            total += __shfl_down_sync(0xffffffff, total, offset);
            count += __shfl_down_sync(0xffffffff, count, offset);
        }
        if (lane == 0) { sums[s] += total; hits[s] += count; }
    }
    __syncthreads();
}
for (int s = lane; s < 60; s += 32) {
    bool good = wp::index(valid,0,s);
    wp::index(out_cost, sample, s) = good ? sums[s] : INFINITY;
    wp::index(out_count, sample, s) = good ? hits[s] : 0;
}
''')
def _score(p0: wp.array3d(dtype=wp.float64), p1: wp.array3d(dtype=wp.float64),
           mask: wp.array2d(dtype=wp.bool), pair: int,
           models: wp.array4d(dtype=wp.float64), valid: wp.array2d(dtype=wp.bool),
           threshold: wp.float64, sample: int,
           out_cost: wp.array2d(dtype=wp.float64), out_count: wp.array2d(dtype=wp.int32)): ...


def _make_generate(score_inside):
    @wp.kernel(module="unique")
    def generate(p0: wp.array3d(dtype=wp.float64), p1: wp.array3d(dtype=wp.float64),
                  mask: wp.array2d(dtype=wp.bool), sizes: wp.array(dtype=wp.int64),
                  principal0: wp.array2d(dtype=wp.float64), principal1: wp.array2d(dtype=wp.float64),
                  active: wp.array(dtype=wp.int64), trials: int, trial: wp.int64, bias: wp.uint32,
                  mapping: wp.array2d(dtype=wp.int32), action_rows: wp.array(dtype=wp.int32),
                  threshold: wp.float64,
                  out_models: wp.array4d(dtype=wp.float64), out_focals: wp.array2d(dtype=wp.float64),
                  out_valid: wp.array2d(dtype=wp.bool), out_cost: wp.array2d(dtype=wp.float64),
                  out_count: wp.array2d(dtype=wp.int32), out_samples: wp.array2d(dtype=wp.int32),
                  out_status: wp.array(dtype=wp.int32), times: wp.array2d(dtype=wp.int64)):
        sample, lane = wp.tid()
        _mark(times, sample, 0)
        pair = int(active[sample // trials])
        for slot in range(lane, 60, 32):
            out_valid[sample, slot] = False
            out_focals[sample, slot] = wp.float64(0.0)
            out_cost[sample, slot] = wp.float64(wp.inf)
            out_count[sample, slot] = 0
            for i in range(3):
                for j in range(3):
                    out_models[sample, slot, i, j] = wp.float64(0.0)
        if lane < 6:
            out_samples[sample, lane] = -1
        if lane == 0:
            out_status[sample] = 1
        if sizes[pair] < 6:
            return
        memory = _scratch()
        integers = _integers()
        a = _view3(memory, 0, 1, 6, 2)
        b = _view3(memory, 12, 1, 6, 2)
        pp0 = _view2(memory, 24, 1, 2)
        pp1 = _view2(memory, 26, 1, 2)
        basis = _view2(memory, 28, 1, 27)
        scale = _view1(memory, 55, 1)
        coeff = _view2(memory, 56, 1, 280)
        action = _view3(memory, 336, 1, 15, 15)
        roots = _view2(memory, 561, 1, 15)
        models = _view4(memory, 576, 1, 60, 3, 3)
        focals = _view2(memory, 1116, 1, 60)
        counts = _iview1(integers, 0, 1)
        status = _iview1(integers, 1, 1)
        root_status = _iview2(integers, 2, 1, 15)
        valid = _flags()
        if lane < 2:
            pp0[0, lane] = principal0[pair, lane]
            pp1[0, lane] = principal1[pair, lane]
        keys, indices = Keys(), Indices()
        for k in range(6):
            keys[k] = wp.int64(-1)
        for point in range(lane, mask.shape[1], 32):
            if mask[pair, point]:
                counter = (trial + wp.int64(sample % trials)) * sizes[pair] + wp.int64(point)
                bits = wp.uint32(counter) + bias
                bits = (bits ^ (bits >> wp.uint32(16))) * wp.uint32(2246822519)
                bits = (bits ^ (bits >> wp.uint32(13))) * wp.uint32(3266489917)
                bits = bits ^ (bits >> wp.uint32(16))
                key, index = wp.int64(bits), point
                for k in range(6):
                    if key > keys[k]:
                        previous_key, previous_index = keys[k], indices[k]
                        keys[k] = key
                        indices[k] = index
                        key, index = previous_key, previous_index
        for k in range(6):
            maximum = wp.tile_extract(wp.tile_max(wp.tile(keys[0])), 0)
            local_lane = int(32)
            if keys[0] == maximum:
                local_lane = lane
            owner = wp.tile_extract(wp.tile_min(wp.tile(local_lane)), 0)
            point = wp.tile_extract(wp.tile(indices[0]), owner)
            if lane == 0:
                out_samples[sample, k] = point
                for j in range(2):
                    a[0,k,j] = p0[pair,point,j]
                    b[0,k,j] = p1[pair,point,j]
            if lane == owner:
                for shift in range(5):
                    keys[shift] = keys[shift+1]
                    indices[shift] = indices[shift+1]
                keys[5] = wp.int64(-1)
        _sync()
        _mark(times, sample, 1)
        if lane == 0:
            _nullspace(0, a, b, pp0, pp1, basis, scale, status)
        _sync()
        _mark(times, sample, 2)
        coefficients(lane, basis, coeff)
        _sync()
        _mark(times, sample, 3)
        _eliminate_cooperative(0, lane, coeff, mapping, action_rows, action, counts, status)
        _sync()
        _mark(times, sample, 4)
        if lane == 0:
            _action_roots(0, action, roots, counts, status)
            out_status[sample] = status[0]
        _sync()
        _mark(times, sample, 5)
        if lane < 15:
            _recover(0, lane, a, b, pp0, pp1, basis, scale, action, roots,
                     counts, models, focals, valid, root_status)
        _sync()
        _mark(times, sample, 6)
        for slot in range(lane, 60, 32):
            if valid[0, slot]:
                out_valid[sample, slot] = True
                out_focals[sample, slot] = focals[0, slot]
                for i in range(3):
                    for j in range(3):
                        out_models[sample, slot, i, j] = models[0, slot, i, j]
        if wp.static(score_inside):
            _score(p0, p1, mask, pair, models, valid, threshold, sample, out_cost, out_count)
        _mark(times, sample, 7)
    wp.set_module_options({"enable_backward": False, "fast_math": False,
                           "fuse_fp": False, "max_unroll": 1}, module=generate.module)
    return generate


_generate = _make_generate(True)
_generate_unscored = _make_generate(False)

@dataclass
class Hypotheses:
    """Fixed slots [active pair * trials + trial, root * 4 + pose].

Invalid slots contain zero model/focal/count and infinite cost. No full inlier
mask is materialized. Buffers are owned and overwritten by SixPointBatch.run.
"""
    models: torch.Tensor
    focals: torch.Tensor
    valid: torch.Tensor
    costs: torch.Tensor
    inlier_counts: torch.Tensor
    sample_indices: torch.Tensor
    status: torch.Tensor


class SixPointBatch:
    """Prepared input views and reusable output storage; run submits one kernel.

    Observations are padded [pairs, capacity, 2]. mask marks usable matches and
    sizes must equal mask.sum(1). Coordinates marked usable must be finite.
    seed is the effective sampler seed (production SF adds 3 * 1009).
    Active pair indices must be in bounds; their order does not seed sampling.
    This object is not safe for concurrent runs on multiple streams.
    """

    def __init__(self, points0, points1, mask, sizes, principal0, principal1,
                 active, *, trials=240, seed=0, max_error=4.0, profile=False):
        if type(trials) is not int or trials < 1:
            raise ValueError('trials must be a positive integer')
        if not math.isfinite(max_error) or max_error <= 0:
            raise ValueError('max_error must be finite and positive')
        if points0.ndim != 3 or points0.shape[-1] != 2:
            raise ValueError('Expected points [pairs, capacity, 2]')
        pairs, capacity, _ = points0.shape
        refs = (points0, points1, mask, sizes, principal0, principal1, active)
        shapes = (points0.shape, points0.shape, (pairs, capacity), (pairs,),
                  (pairs, 2), (pairs, 2), (active.numel(),))
        dtypes = (torch.float64, torch.float64, torch.bool, torch.int64,
                  torch.float64, torch.float64, torch.int64)
        for tensor, shape, dtype in zip(refs, shapes, dtypes):
            if tensor.shape != shape or tensor.dtype != dtype or not tensor.is_cuda or tensor.device != points0.device:
                raise ValueError('Invalid shape, dtype or device for fused six-point input')
            if tensor.requires_grad:
                raise ValueError('Fused geometry is forward-only')
        self.references = refs
        self.views = [wp.from_torch(x) for x in refs]
        self.mapping = _indices(points0.device)
        self.map_views = [wp.from_torch(x) for x in self.mapping]
        self.trials, self.max_error = trials, max_error
        self.bias = wp.uint32(((seed % 2**32)*104729+374761393) & 0xffffffff)
        n = len(active)*trials
        def empty(shape, dtype=torch.float64):
            return torch.empty(shape, device=points0.device, dtype=dtype)
        self.result = Hypotheses(empty((n,60,3,3)), empty((n,60)), empty((n,60),torch.bool),
                                empty((n,60)), empty((n,60),torch.int32),
                                empty((n,6),torch.int32), empty((n,),torch.int32))
        self.output_views = [wp.from_torch(x) for x in vars(self.result).values()]
        self.timings = empty((n if profile else 0,8),torch.int64)
        self.output_views.append(wp.from_torch(self.timings))

    @torch.no_grad()
    def run(self, trial=0):
        if type(trial) is not int or trial < 0:
            raise ValueError('trial must be a nonnegative integer')
        if len(self.result.status):
            wp.launch(_generate, dim=(len(self.result.status),32), block_dim=32,
                      inputs=[*self.views,self.trials,wp.int64(trial),self.bias,*self.map_views,
                              wp.float64(self.max_error**2)], outputs=self.output_views,
                      stream=wp.stream_from_torch(torch.cuda.current_stream(self.references[0].device)))
        return self.result
