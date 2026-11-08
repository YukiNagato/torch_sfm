"""Four-launch shared-focal six-point sampling, solving, and scoring.

Scalar preparation and roots/recovery surround a warp-cooperative LU kernel.
All buffers are allocated once; failed samples stay invalid without CPU repair.
"""
import torch
import warp as wp

from ..geometry.warp_shared_focal_poselib import _eliminate_cooperative
from .six_point_layouts import _initialize, _finish, _sample_workspace, OPTIONS
from .six_point_scalar_polynomial import coefficients
from .six_point_split import SplitSixPointBatch
from .six_point import Hypotheses

wp.set_module_options(OPTIONS)


@wp.kernel
def _prepare(p0: wp.array3d(dtype=wp.float64), p1: wp.array3d(dtype=wp.float64),
             mask: wp.array2d(dtype=wp.bool), sizes: wp.array(dtype=wp.int64),
             principal0: wp.array2d(dtype=wp.float64), principal1: wp.array2d(dtype=wp.float64),
             active: wp.array(dtype=wp.int64), trials: int, trial: wp.int64, bias: wp.uint32,
             a: wp.array3d(dtype=wp.float64), b: wp.array3d(dtype=wp.float64),
             pp0: wp.array2d(dtype=wp.float64), pp1: wp.array2d(dtype=wp.float64),
             basis: wp.array2d(dtype=wp.float64), scales: wp.array(dtype=wp.float64),
             counts: wp.array(dtype=wp.int32), root_status: wp.array2d(dtype=wp.int32),
             coeff: wp.array2d(dtype=wp.float64),
             models: wp.array4d(dtype=wp.float64), focals: wp.array2d(dtype=wp.float64),
             valid: wp.array2d(dtype=wp.bool), costs: wp.array2d(dtype=wp.float64),
             inliers: wp.array2d(dtype=wp.int32), samples: wp.array2d(dtype=wp.int32),
             status: wp.array(dtype=wp.int32)):
    sample = wp.tid()
    _initialize(sample,p0,p1,mask,sizes,principal0,principal1,active,trials,trial,bias,
                a,b,pp0,pp1,basis,scales,counts,root_status,models,focals,valid,costs,inliers,samples,status)
    if status[sample] == 0:
        coefficients(sample,basis,coeff)


@wp.kernel
def _solve(a: wp.array3d(dtype=wp.float64), b: wp.array3d(dtype=wp.float64),
           pp0: wp.array2d(dtype=wp.float64), pp1: wp.array2d(dtype=wp.float64),
           basis: wp.array2d(dtype=wp.float64), scales: wp.array(dtype=wp.float64),
           action: wp.array3d(dtype=wp.float64), roots: wp.array2d(dtype=wp.float64),
           counts: wp.array(dtype=wp.int32), root_status: wp.array2d(dtype=wp.int32),
           models: wp.array4d(dtype=wp.float64), focals: wp.array2d(dtype=wp.float64),
           valid: wp.array2d(dtype=wp.bool), status: wp.array(dtype=wp.int32)):
    sample = wp.tid()
    _finish(sample,a,b,pp0,pp1,basis,scales,action,roots,counts,root_status,models,focals,valid,status)


class CooperativeSixPointBatch(SplitSixPointBatch):
    """Reusable batch: prepare -> eliminate -> solve -> score.

    run(trial=0) launches exactly four kernels for a nonempty batch. generate()
    launches the three solver kernels, and score() scores their fixed slots.
    The individual stages must run in this order on the same CUDA stream.
    Scalar stages use solver_block_size threads/block (default 128); LU always
    uses one 32-thread block/sample. Scoring enables FMA independently.
    Buffers are overwritten on every run. Concurrent use is unsupported.
    """

    def __init__(self, *args, solver_block_size=128, **kwargs):
        if type(solver_block_size) is not int or solver_block_size not in (32,64,128,256):
            raise ValueError('solver_block_size must be 32, 64, 128 or 256')
        super().__init__(*args, **kwargs)
        self.solver_block_size = solver_block_size
        self.workspace = _sample_workspace(len(self.result.status), self.references[0].device)
        self.workspace_views = [wp.from_torch(x) for x in self.workspace]
        self._result_storage = tuple(vars(self.result).values())
        self._workspace_storage = self.workspace
        self._max_pairs = len(self.references[6])
        self._max_trials = self.trials
        self._bind_views()

    def _bind_views(self):
        w = self.workspace_views
        self.prepare_workspace = [w[i] for i in (0,1,2,3,4,5,8,9,10)]
        self.eliminate_inputs = [w[10],*self.map_views,w[6],w[8],self.output_views[6]]
        self.solve_inputs = w[:10]
        self.solve_outputs = [self.output_views[i] for i in (0,1,2,6)]

    def set_batch(self, active, trials=None):
        """Rebind smaller active/trial prefixes without allocating GPU buffers."""
        trials = self._max_trials if trials is None else trials
        if type(trials) is not int or not 1 <= trials <= self._max_trials:
            raise ValueError('trials exceeds the allocated batch capacity')
        if (active.ndim != 1 or active.dtype != torch.int64 or active.device != self.references[0].device
                or len(active) > self._max_pairs):
            raise ValueError('Invalid active pair prefix')
        self.trials = trials
        self.references = (*self.references[:6], active)
        self.views[6] = wp.from_torch(active)
        n = len(active)*trials
        self.result = Hypotheses(*(x[:n] for x in self._result_storage))
        self.output_views = [wp.from_torch(x) for x in vars(self.result).values()] + [wp.from_torch(self.timings)]
        self.workspace = tuple(x[:n] for x in self._workspace_storage)
        self.workspace_views = [wp.from_torch(x) for x in self.workspace]
        self.score_inputs = [*self.views[:3], self.views[6], self.trials,
                             self.output_views[0], self.output_views[2], wp.float64(self.max_error**2)]
        self._bind_views()

    @torch.no_grad()
    def prepare(self, trial=0):
        if type(trial) is not int or trial < 0:
            raise ValueError('trial must be a nonnegative integer')
        if len(self.result.status):
            wp.launch(_prepare, dim=len(self.result.status), block_dim=self.solver_block_size,
                      inputs=[*self.views,self.trials,wp.int64(trial),self.bias,*self.prepare_workspace],
                      outputs=self.output_views[:7],
                      stream=wp.stream_from_torch(torch.cuda.current_stream(self.references[0].device)))
        return self.result

    @torch.no_grad()
    def eliminate(self):
        if len(self.result.status):
            wp.launch(_eliminate_cooperative, dim=(len(self.result.status),32), block_dim=32,
                      inputs=self.eliminate_inputs,
                      stream=wp.stream_from_torch(torch.cuda.current_stream(self.references[0].device)))
        return self.result

    @torch.no_grad()
    def solve(self):
        if len(self.result.status):
            wp.launch(_solve, dim=len(self.result.status), block_dim=self.solver_block_size,
                      inputs=self.solve_inputs, outputs=self.solve_outputs,
                      stream=wp.stream_from_torch(torch.cuda.current_stream(self.references[0].device)))
        return self.result

    @torch.no_grad()
    def generate(self, trial=0):
        self.prepare(trial)
        self.eliminate()
        return self.solve()
