"""E/F/H adaptive RANSAC with fixed slots and persistent five-round LO."""
import math
import torch
import warp as wp
from .minimal import MinimalBatch
from .schedule import _initialize, _schedule, _log1p as _log1p64
from .scoring import score, vote, error, BLOCK
from .refit import refit
from .six_point import _sync
from .ransac import RansacResult
from ..geometry.warp_relative_pose import _matrix
from ..geometry.warp_ransac import _log1p
from ..geometry.essential_scoring import prepare_essential_observations
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False,'max_unroll':1})

@wp.kernel
def commit(candidates:wp.array3d(dtype=wp.float64),costs:wp.array(dtype=wp.float64),poses:wp.array3d(dtype=wp.float64),
           active:wp.array(dtype=wp.int64),stride:int,kind:int,batch:int,
           p0:wp.array3d(dtype=wp.float64),p1:wp.array3d(dtype=wp.float64),
           pixel0:wp.array3d(dtype=wp.float64),pixel1:wp.array3d(dtype=wp.float64),valid:wp.array2d(dtype=wp.bool),
           ray0:wp.array3d(dtype=wp.float64),ray1:wp.array3d(dtype=wp.float64),jac0:wp.array4d(dtype=wp.float64),jac1:wp.array4d(dtype=wp.float64),
           threshold:wp.float64,minimum:int,maximum:int,log_failure:wp.float64,double_budget:bool,
           models:wp.array3d(dtype=wp.float64),masks:wp.array2d(dtype=wp.bool),best:wp.array(dtype=wp.float64),found:wp.array(dtype=wp.bool),
           sizes:wp.array(dtype=wp.int64),limits:wp.array(dtype=wp.int64),required:wp.array(dtype=wp.int64),trials:wp.array(dtype=wp.int64),
           design:wp.array3d(dtype=wp.float64),fitted:wp.array3d(dtype=wp.float64),good:wp.array(dtype=wp.bool),proposal:wp.array2d(dtype=wp.bool)):
    pair,lane=wp.tid();owner=int(active[pair]);value=wp.float64(wp.inf);index=int(stride)
    for i in range(lane,stride,BLOCK):
        v=costs[pair*stride+i]
        if v<value or (v==value and i<index):value=v;index=i
    smallest=wp.tile_extract(wp.tile_min(wp.tile(value)),0)
    if value!=smallest:index=stride
    winner=wp.tile_extract(wp.tile_min(wp.tile(index)),0)
    improve=smallest<best[owner] and winner<stride
    if improve:
        selected=pair*stride+winner;model=_matrix(candidates,selected);r=_matrix(poses,selected)
        t=wp.vec3d(poses[selected,3,0],poses[selected,3,1],poses[selected,3,2])
        for j in range(lane,valid.shape[1],BLOCK):
            inside=False
            if valid[owner,j]:inside=error(model,kind,r,t,pixel0,pixel1,ray0,ray1,jac0,jac1,owner,j)<=threshold
            masks[owner,j]=inside
        _sync()
        for turn in range(5):
            refit(p0,p1,masks,owner,lane,kind,design,fitted,good)
            if not good[owner]:break
            candidate=_matrix(fitted,owner);r,t=wp.mat33d(),wp.vec3d();safe=True
            if kind==0:r,t,safe=vote(candidate,ray0,ray1,valid,owner,lane)
            cost=wp.float64(0.0)
            for j in range(lane,valid.shape[1],BLOCK):
                inside=False
                if valid[owner,j]:
                    e=wp.float64(wp.inf)
                    if safe:e=error(candidate,kind,r,t,pixel0,pixel1,ray0,ray1,jac0,jac1,owner,j)
                    cost+=wp.min(e,threshold);inside=e<=threshold
                proposal[owner,j]=inside
            total=wp.tile_extract(wp.tile_sum(wp.tile(cost)),0)
            if not wp.isfinite(total) or total>=smallest:break
            smallest=total;model=candidate
            for j in range(lane,valid.shape[1],BLOCK):masks[owner,j]=proposal[owner,j]
            _sync()
        if lane==0:
            best[owner]=smallest;found[owner]=True
            for i in range(3):
                for j in range(3):models[owner,i,j]=model[i,j]
    hits=wp.int64(0)
    for j in range(lane,valid.shape[1],BLOCK):hits+=wp.int64(masks[owner,j])
    hits=wp.tile_extract(wp.tile_sum(wp.tile(hits)),0)
    if lane==0:
        sample_size=int(5)
        if kind==1:sample_size=7
        elif kind==2:sample_size=4
        ratio=wp.float32(hits)/wp.float32(sizes[owner]);probability=wp.pow(ratio,wp.float32(sample_size))
        estimate=wp.int64(maximum)
        if probability>=wp.float32(1.0):estimate=wp.int64(minimum)
        elif probability>wp.float32(0.0):
            x=wp.ceil(wp.float32(log_failure)/_log1p(-probability))
            if wp.isfinite(x):estimate=wp.int64(wp.clamp(x,wp.float32(minimum),wp.float32(maximum)))
        if double_budget:
            p=wp.pow(wp.float64(hits)/wp.float64(sizes[owner]),wp.float64(sample_size))
            estimate=wp.int64(maximum)
            if p>=wp.float64(1.0):estimate=wp.int64(minimum)
            elif p>wp.float64(0.0):
                x64=wp.ceil(log_failure/_log1p64(-wp.min(p,wp.float64(1.0-1e-15))))
                if wp.isfinite(x64):estimate=wp.int64(wp.clamp(x64,wp.float64(minimum),wp.float64(maximum)))
        required[owner]=wp.min(estimate,limits[owner]);trials[owner]+=wp.int64(batch)

class CalibratedRansac:
    """One bounded pair batch; E, F and H instances have independent state."""
    def __init__(self,p0,p1,valid,pixel0,pixel1,inverse0,inverse1,kind,cfg,min_inlier_ratio=None):
        if kind not in ('E','F','H'):
            raise ValueError('Expected E, F or H')
        if (p0.ndim!=3 or p0.shape[-1]!=2 or any(x.shape!=p0.shape for x in (p1,pixel0,pixel1))
                or valid.shape!=p0.shape[:2] or valid.dtype!=torch.bool
                or inverse0.shape!=(len(p0),3,3) or inverse1.shape!=inverse0.shape):
            raise ValueError('Invalid calibrated RANSAC observation shapes')
        values=(p0,p1,pixel0,pixel1,inverse0,inverse1)
        if (not p0.is_cuda or any(x.device!=p0.device or x.dtype!=torch.float64 for x in values)
                or valid.device!=p0.device):
            raise ValueError('Warp calibrated RANSAC requires CUDA float64 inputs on one device')
        if torch.is_grad_enabled() and any(x.requires_grad for x in values):
            raise ValueError('Warp calibrated RANSAC is forward only')
        self.cfg=cfg;self.kind=kind;self.n=len(p0);self.points=(p0,p1,pixel0,pixel1,valid)
        def empty(shape,dtype=torch.float64):return torch.empty(shape,device=p0.device,dtype=dtype)
        self.sizes=empty((self.n,),torch.int64);self.limits=empty((self.n,),torch.int64)
        self.active=empty((self.n,),torch.int64);self.history=empty((self.n,));self.count=empty((1,),torch.int32)
        self.result=RansacResult(empty((self.n,3,3)),empty(valid.shape,torch.bool),empty((self.n,),torch.bool),empty((self.n,)),empty((self.n,)),empty((self.n,),torch.int64),empty((self.n,),torch.int64))
        self.initial_limits=empty((0,),torch.int64)
        size={'E':5,'F':7,'H':4}[kind]
        if min_inlier_ratio is not None:
            probability=min_inlier_ratio.pow(size)
            budget=torch.ceil(math.log1p(-cfg.confidence)/torch.log1p(-probability.clamp_max(1-1e-15)))
            self.initial_limits=budget.clamp(cfg.min_trials,cfg.max_trials).long()
        self.frontend=MinimalBatch(p0,p1,valid,self.sizes,self.active,cfg,kind)
        if kind=='E':self.observations=prepare_essential_observations(pixel0,pixel1,inverse0,inverse1)
        else:self.observations=(empty((0,0,3)),empty((0,0,3)),empty((0,0,3,2)),empty((0,0,3,2)))
        self.design=empty((self.n,max(9,valid.shape[1]*(2 if kind=='H' else 1)),9))
        self.fitted=empty((self.n,3,3));self.good=empty((self.n,),torch.bool);self.proposal=empty(valid.shape,torch.bool)
        self.rounds=[]

    @torch.no_grad()
    def run(self):
        self.rounds=[]
        if not self.n:return self.result
        c=self.cfg;r=self.result;p0,p1,x0,x1,valid=self.points
        stream=wp.stream_from_torch(torch.cuda.current_stream(p0.device))
        def launch(kernel,dim,args,block=BLOCK):
            wp.launch(kernel,dim=dim,block_dim=block,inputs=[wp.from_torch(v) if isinstance(v,torch.Tensor) else v for v in args],stream=stream)
        minimum=max(self.frontend.size,c.min_num_inliers)
        if valid.shape[1]<8:minimum=valid.shape[1]+1
        launch(_initialize,(self.n,BLOCK),[valid,minimum,c.max_trials,self.initial_limits,self.limits,self.sizes,r.required,r.models,r.masks,r.found,r.focals,r.costs,r.trials])
        trial=0;kind={'E':0,'F':1,'H':2}[self.kind]
        while trial<c.max_trials:
            launch(_schedule,1,[r.required,r.costs,trial,self.active,self.history,self.count])
            n=int(self.count.item())
            if not n:break
            active=self.active[:n];batch=min(c.batch_size,c.max_trials-trial)
            models,good,costs,poses=self.frontend.run(active,trial,batch)
            stride=batch*self.frontend.slots
            launch(score,(len(models),BLOCK),[models,good,active,stride,kind,x0,x1,valid,*self.observations,wp.float64(c.max_error**2),costs,poses])
            launch(commit,(n,BLOCK),[models,costs,poses,active,stride,kind,batch,p0,p1,x0,x1,valid,*self.observations,
                wp.float64(c.max_error**2),c.min_trials,c.max_trials,wp.float64(math.log1p(-c.confidence)),torch.get_default_dtype()==torch.float64,
                r.models,r.masks,r.costs,r.found,self.sizes,self.limits,r.required,r.trials,self.design,self.fitted,self.good,self.proposal])
            self.rounds.append(dict(trial=trial,active_pairs=n,samples=n*batch));trial+=batch
        return r


def enabled(cfg, points):
    compatible=(points.is_cuda and points.dtype==torch.float64 and cfg.backend!='torch'
                and cfg.essential_cheirality)
    if cfg.calibrated_frontend=='warp' and not compatible:
        raise ValueError('Warp calibrated frontend requires CUDA float64 and essential cheirality')
    return compatible and (cfg.calibrated_frontend=='warp' or
        (cfg.calibrated_frontend=='auto' and cfg.five_point_backend=='warp_poselib'))
