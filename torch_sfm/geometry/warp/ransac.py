"""Adaptive shared-focal RANSAC with persistent buffers and GPU incumbent state."""
from dataclasses import dataclass
import math
import torch
import warp as wp
from .six_point_cooperative import CooperativeSixPointBatch
from .local_optimization import LocalOptimizer, _winner, _matrix, _error, BLOCK

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':True,'max_unroll':1})


from .schedule import _initialize, _schedule, _log1p


@wp.kernel
def _commit(candidates: wp.array4d(dtype=wp.float64),candidate_focals: wp.array2d(dtype=wp.float64),
            candidate_costs: wp.array2d(dtype=wp.float64),candidate_valid: wp.array2d(dtype=wp.bool),
            batch_trials: int,points0: wp.array3d(dtype=wp.float64),points1: wp.array3d(dtype=wp.float64),
            valid: wp.array2d(dtype=wp.bool),active: wp.array(dtype=wp.int64),threshold: wp.float64,
            minimum: int,maximum: int,failure_log: wp.float64,
            limits: wp.array(dtype=wp.int64),
            sizes: wp.array(dtype=wp.int64),required: wp.array(dtype=wp.int64),
            models: wp.array3d(dtype=wp.float64),masks: wp.array2d(dtype=wp.bool),
            found: wp.array(dtype=wp.bool),focals: wp.array(dtype=wp.float64),
            costs: wp.array(dtype=wp.float64),trials: wp.array(dtype=wp.int64)):
    pair,lane=wp.tid();target=int(active[pair])
    best,index=_winner(candidate_costs,candidate_valid,batch_trials,pair,lane)
    flag=int(0)
    if lane==0:flag=int(index<batch_trials*60 and wp.isfinite(best) and best<costs[target])
    accept=wp.tile_extract(wp.tile_sum(wp.tile(flag)),0)!=0
    model=wp.mat33d()
    if accept:
        sample=pair*batch_trials+index//60;slot=index%60
        model=_matrix(candidates,sample,slot)
        if lane==0:
            for i in range(3):
                for j in range(3):models[target,i,j]=model[i,j]
            focals[target]=candidate_focals[sample,slot];costs[target]=best;found[target]=True
    partial=int(0)
    for j in range(lane,valid.shape[1],BLOCK):
        inside=masks[target,j]
        if accept:
            inside=False
            if valid[target,j]:
                inside=_error(model,points0[target,j,0],points0[target,j,1],points1[target,j,0],points1[target,j,1])<=threshold
            masks[target,j]=inside
        partial+=int(inside)
    inliers=wp.tile_extract(wp.tile_sum(wp.tile(partial)),0)
    if lane==0:
        total=wp.float64(sizes[target]);probability=wp.float64(1.0)
        for j in range(6):
            probability*=wp.clamp((wp.float64(inliers)-wp.float64(j))/(total-wp.float64(j)),wp.float64(0.0),wp.float64(1.0))
        estimate=wp.float64(maximum)
        if probability>=wp.float64(1.0):estimate=wp.float64(minimum)
        elif probability>wp.float64(0.0):
            estimate=wp.ceil(failure_log/_log1p(-wp.min(probability,wp.float64(1.0-1e-15))))
        required[target]=wp.min(limits[target],wp.int64(wp.clamp(estimate,wp.float64(minimum),wp.float64(maximum))))
        trials[target]+=wp.int64(batch_trials)


@dataclass
class RansacResult:
    models: torch.Tensor
    masks: torch.Tensor
    found: torch.Tensor
    focals: torch.Tensor
    costs: torch.Tensor
    trials: torch.Tensor
    required: torch.Tensor


class SharedFocalRansac:
    """Complete adaptive SF solve; bounded batches, no CPU numerical fallback.

    The only iterative host read is the next active-pair count (one int32).
    Each round submits four minimal-solver kernels, two LO kernels, a combined
    best/commit/budget kernel, and stable scheduling. Output uses original pair
    order. Old outputs and all workspaces are reused by subsequent run() calls.
    """
    def __init__(self,points0,points1,valid,principal0,principal1,cfg,min_inlier_ratio=None):
        self.cfg=cfg;self.n=len(points0);device=points0.device
        def empty(shape,dtype=torch.float64):return torch.empty(shape,device=device,dtype=dtype)
        self.sizes=empty((self.n,),torch.int64)
        self.limits=empty((self.n,),torch.int64)
        self.initial_limits=empty((0,),torch.int64)
        if min_inlier_ratio is not None:
            if min_inlier_ratio.shape!=(self.n,) or min_inlier_ratio.device!=device:
                raise ValueError('min_inlier_ratio must contain one value per pair on the input device')
            probability=min_inlier_ratio.pow(6)
            budget=torch.ceil(math.log1p(-cfg.confidence)/torch.log1p(-probability.clamp_max(1-1e-15)))
            self.initial_limits=budget.clamp(cfg.min_trials,cfg.max_trials).long()
        self.limit_views=[wp.from_torch(self.initial_limits),wp.from_torch(self.limits)]
        self.active=empty((self.n,),torch.int64);self.history=empty((self.n,));self.count=empty((1,),torch.int32)
        self.result=RansacResult(empty((self.n,3,3)),empty(valid.shape,torch.bool),empty((self.n,),torch.bool),
            empty((self.n,)),empty((self.n,)),empty((self.n,),torch.int64),empty((self.n,),torch.int64))
        self.frontend=CooperativeSixPointBatch(points0,points1,valid,self.sizes,principal0,principal1,self.active,
            trials=min(cfg.batch_size,cfg.max_trials),seed=cfg.seed+3*1009,max_error=cfg.max_error)
        self.lo=LocalOptimizer(self.frontend)
        self.state_views=[wp.from_torch(x) for x in (self.sizes,self.result.required,self.result.models,self.result.masks,
            self.result.found,self.result.focals,self.result.costs,self.result.trials)]
        self.schedule_views=[wp.from_torch(x) for x in (self.result.required,self.result.costs,self.active,self.history,self.count)]
        self.rounds=[]

    @torch.no_grad()
    def run(self):
        self.rounds=[]
        if not self.n:return self.result
        cfg=self.cfg;front=self.frontend
        stream=wp.stream_from_torch(torch.cuda.current_stream(self.sizes.device))
        minimum=max(6,cfg.min_num_inliers)
        if front.references[0].shape[1]<8:minimum=front.references[0].shape[1]+1
        wp.launch(_initialize,dim=(self.n,BLOCK),block_dim=BLOCK,
            inputs=[front.views[2],minimum,cfg.max_trials,*self.limit_views,*self.state_views],stream=stream)
        trial=0
        while trial<cfg.max_trials:
            wp.launch(_schedule,dim=1,block_dim=BLOCK,
                inputs=[*self.schedule_views[:2],trial,*self.schedule_views[2:]],stream=stream)
            active_count=int(self.count.item())
            if active_count==0:break
            batch=min(cfg.batch_size,cfg.max_trials-trial)
            front.set_batch(self.active[:active_count],batch);self.lo.bind()
            front.run(trial);self.lo.refine(self.history[:active_count],choose_best=False)
            o=front.output_views;v=front.views
            wp.launch(_commit,dim=(active_count,BLOCK),block_dim=BLOCK,
                inputs=[o[0],o[1],o[3],o[2],batch,v[0],v[1],v[2],v[6],wp.float64(cfg.max_error**2),
                        cfg.min_trials,cfg.max_trials,wp.float64(3.*math.log1p(-cfg.confidence)),
                        self.limit_views[1],*self.state_views],stream=stream)
            self.rounds.append(dict(trial=trial,active_pairs=active_count,samples=active_count*batch))
            trial+=batch
        return self.result


def ransac_batch(points0,points1,valid,inverse0,inverse1,cfg,min_inlier_ratio=None):
    principal0=-inverse0[:,:2,2]/inverse0.diagonal(dim1=-2,dim2=-1)[:,:2]
    principal1=-inverse1[:,:2,2]/inverse1.diagonal(dim1=-2,dim2=-1)[:,:2]
    result=SharedFocalRansac(points0,points1,valid,principal0,principal1,cfg,min_inlier_ratio).run()
    return result.models,result.masks,result.found
