"""Two launches: sample/solve, then independently scheduled Sampson scoring.

The solver is identical to the fused prototype with scoring compiled out.
Each scoring block handles one sample, packs its valid model slots in shared
memory, and distributes models across warps. Matching tiles are shared by all
warps. Packing stays within the block: no global prefix sum or CPU readback.
"""
import torch
import warp as wp

from .six_point import SixPointBatch, _generate_unscored

wp.set_module_options({'enable_backward': False, 'fast_math': False,
                       'fuse_fp': True, 'max_unroll': 1})
# This module compiles only the independent scorer. Solver kernels retain
# their own FMA settings; scoring follows the original ModelScorer default.


@wp.func_native('''
__shared__ double points[4][128];
__shared__ bool usable[128];
__shared__ int slots[60];
__shared__ int count;
__shared__ double matrices[60][9];
int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
int warps = blockDim.x / 32;
for (int slot = tid; slot < 60; slot += blockDim.x) {
    wp::index(costs, sample, slot) = INFINITY;
    wp::index(inliers, sample, slot) = 0;
}
if (tid == 0) {
    int n = 0;
    for (int slot = 0; slot < 60; ++slot) {
        if (wp::index(valid, sample, slot)) slots[n++] = slot;
    }
    count = n;
}
__syncthreads();
if (count == 0) return;
for (int i = tid; i < count * 9; i += blockDim.x) {
    int model = i / 9, elem = i % 9;
    matrices[model][elem] = wp::index(models, sample, slots[model], elem/3, elem%3);
}
__syncthreads();
int pair = static_cast<int>(wp::index(active, sample / trials));
int capacity = mask.shape[1];
for (int begin = 0; begin < capacity; begin += 128) {
    for (int t = tid; t < 128; t += blockDim.x) {
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
    for (int model = warp; model < count; model += warps) {
        double total = 0.0;
        int hits = 0;
        for (int t = lane; t < 128; t += 32) {
            if (!usable[t]) continue;
            double x = points[0][t], y = points[1][t];
            double u = points[2][t], v = points[3][t];
            double a = matrices[model][0]*x + matrices[model][1]*y + matrices[model][2];
            double b = matrices[model][3]*x + matrices[model][4]*y + matrices[model][5];
            double c = matrices[model][6]*x + matrices[model][7]*y + matrices[model][8];
            double d = matrices[model][0]*u + matrices[model][3]*v + matrices[model][6];
            double e = matrices[model][1]*u + matrices[model][4]*v + matrices[model][7];
            double numerator = u*a + v*b + c;
            double error = numerator*numerator / fmax(a*a+b*b+d*d+e*e, 1.0e-30);
            if (!isfinite(error)) error = INFINITY;
            total += fmin(error, threshold);
            hits += error <= threshold;
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            total += __shfl_down_sync(0xffffffff, total, offset);
            hits += __shfl_down_sync(0xffffffff, hits, offset);
        }
        if (lane == 0) {
            int slot = slots[model];
            double previous = begin == 0 ? 0.0 : wp::index(costs, sample, slot);
            wp::index(costs, sample, slot) = previous + total;
            wp::index(inliers, sample, slot) += hits;
        }
    }
    __syncthreads();
}
''')
def _score_block(p0: wp.array3d(dtype=wp.float64), p1: wp.array3d(dtype=wp.float64),
                 mask: wp.array2d(dtype=wp.bool), active: wp.array(dtype=wp.int64),
                 trials: int, sample: int, models: wp.array4d(dtype=wp.float64),
                 valid: wp.array2d(dtype=wp.bool), threshold: wp.float64,
                 costs: wp.array2d(dtype=wp.float64), inliers: wp.array2d(dtype=wp.int32)): ...


@wp.kernel
def _score(p0: wp.array3d(dtype=wp.float64), p1: wp.array3d(dtype=wp.float64),
           mask: wp.array2d(dtype=wp.bool), active: wp.array(dtype=wp.int64),
           trials: int, models: wp.array4d(dtype=wp.float64), valid: wp.array2d(dtype=wp.bool),
           threshold: wp.float64, costs: wp.array2d(dtype=wp.float64),
           inliers: wp.array2d(dtype=wp.int32)):
    sample, lane = wp.tid()
    _score_block(p0,p1,mask,active,trials,sample,models,valid,threshold,costs,inliers)


class SplitSixPointBatch(SixPointBatch):
    """Same buffers and numerical solver as SixPointBatch; run launches twice.

    generate() and score() are exposed for separate measurement. score() must
    follow generate() on the same CUDA stream (or an explicit caller dependency).
    score_block_size can be 32/64/128/256 independently of solver scheduling.
    All output buffers are overwritten; concurrent use is unsupported.
    """

    def __init__(self, *args, score_block_size=128, **kwargs):
        if score_block_size not in (32,64,128,256):
            raise ValueError('score_block_size must be 32, 64, 128 or 256')
        super().__init__(*args, **kwargs)
        self.score_block_size = score_block_size
        self.score_inputs = [*self.views[:3],self.views[6],self.trials,
                             self.output_views[0],self.output_views[2],
                             wp.float64(self.max_error**2)]

    @torch.no_grad()
    def generate(self, trial=0):
        if type(trial) is not int or trial < 0:
            raise ValueError('trial must be a nonnegative integer')
        if len(self.result.status):
            wp.launch(_generate_unscored, dim=(len(self.result.status),32),block_dim=32,
                      inputs=[*self.views,self.trials,wp.int64(trial),self.bias,*self.map_views,
                              wp.float64(self.max_error**2)],outputs=self.output_views,
                      stream=wp.stream_from_torch(torch.cuda.current_stream(self.references[0].device)))
        return self.result

    @torch.no_grad()
    def score(self):
        if len(self.result.status):
            wp.launch(_score,dim=(len(self.result.status),self.score_block_size),
                      block_dim=self.score_block_size,inputs=self.score_inputs,
                      outputs=[self.output_views[3],self.output_views[4]],
                      stream=wp.stream_from_torch(torch.cuda.current_stream(self.references[0].device)))
        return self.result

    @torch.no_grad()
    def run(self, trial=0):
        self.generate(trial)
        return self.score()
