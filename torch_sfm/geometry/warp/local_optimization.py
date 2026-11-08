"""GPU record selection and persistent block-local LO/LM, without host polling."""
from dataclasses import dataclass
import math
import torch
import warp as wp

from .essential_pose import _decompose_matrix
from .shared_focal_local import _error
from .local_lm import refine
from .six_point import _sync

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':True,'max_unroll':8})
BLOCK=64


@wp.kernel
def _select(valid: wp.array2d(dtype=wp.bool),costs: wp.array2d(dtype=wp.float64),
            history: wp.array(dtype=wp.float64),trials: int,capacity: int,threshold: wp.float64,
            queue: wp.array2d(dtype=wp.int32),counts: wp.array(dtype=wp.int32)):
    pair=wp.tid()
    minimum=wp.float64(wp.inf);count=int(0)
    # Keep the old offset-cummin comparison arithmetic. No cap on record count.
    offset=wp.float64(pair)*(wp.float64(capacity)*threshold+wp.float64(1.0))
    for local in range(trials*60):
        sample=pair*trials+local//60;slot=local%60
        if valid[sample,slot]:
            cost=costs[sample,slot];adjusted=cost-offset
            if adjusted<minimum and cost<history[pair]:
                queue[pair,count]=local;count+=1
            minimum=wp.min(minimum,adjusted)
    counts[pair]=count


@wp.func
def _matrix(models: wp.array4d(dtype=wp.float64),sample: int,slot: int):
    result=wp.mat33d()
    for i in range(3):
        for j in range(3):result[i,j]=models[sample,slot,i,j]
    return result


@wp.func
def _pose(model: wp.mat33d,f: wp.float64,p: wp.vec2d,q: wp.vec2d,
          a: wp.array3d(dtype=wp.float64),b: wp.array3d(dtype=wp.float64),
          mask: wp.array2d(dtype=wp.bool),owner: int,worker: int,lane: int):
    r0,r1,t=wp.mat33d(),wp.mat33d(),wp.vec3d();good=False
    if lane==0:
        z,o=wp.float64(0.0),wp.float64(1.0)
        k0=wp.mat33d(f,z,p[0],z,f,p[1],z,z,o)
        k1=wp.mat33d(f,z,q[0],z,f,q[1],z,z,o)
        r0,r1,t,good=_decompose_matrix(wp.transpose(k1)*model*k0)
    for i in range(3):
        t[i]=wp.tile_extract(wp.tile(t[i]),0)
        for j in range(3):
            r0[i,j]=wp.tile_extract(wp.tile(r0[i,j]),0)
            r1[i,j]=wp.tile_extract(wp.tile(r1[i,j]),0)
    good=wp.tile_extract(wp.tile(int(good)),0)!=0
    # Do not alias the basis R/t with the loop-carried winner in Warp's SSA.
    chosen_r,chosen_t=wp.mat33d(r0),wp.vec3d(t)
    best_count=int(-1);best_cost=wp.float64(0.0)
    for candidate in range(4):
        r=r0
        if candidate>=2:r=r1
        direction=t
        if candidate%2==1:direction=-direction
        center=-(wp.transpose(r)*direction)
        count=int(0);cost=wp.float64(0.0)
        for point in range(lane,a.shape[1],BLOCK):
            if mask[worker,point]:
                x=wp.vec3d((a[owner,point,0]-p[0])/f,(a[owner,point,1]-p[1])/f,wp.float64(1.0))
                y=wp.vec3d((b[owner,point,0]-q[0])/f,(b[owner,point,1]-q[1])/f,wp.float64(1.0))
                x=x/wp.max(wp.length(x),wp.float64(1e-12))
                y=wp.transpose(r)*(y/wp.max(wp.length(y),wp.float64(1e-12)))
                cosine=wp.dot(x,y);den=wp.max(wp.float64(1.0)-cosine*cosine,wp.float64(1e-15))
                ac,bc=wp.dot(x,center),wp.dot(y,center)
                d0,d1=(ac-cosine*bc)/den,(cosine*ac-bc)/den
                xyz=(d0*x+center+d1*y)*wp.float64(.5)
                positive=d0>wp.float64(1e-8) and d1>wp.float64(1e-8)
                positive=positive and wp.isfinite(xyz[0]) and wp.isfinite(xyz[1]) and wp.isfinite(xyz[2])
                error=xyz/wp.max(wp.length(xyz),wp.float64(1e-12))-x
                count+=int(positive);cost+=wp.dot(error,error)*wp.float64(int(positive))
        total=wp.tile_extract(wp.tile_sum(wp.tile(count)),0)
        score=wp.tile_extract(wp.tile_sum(wp.tile(cost)),0)
        if total>best_count or (total==best_count and (score<best_cost or (wp.isnan(score) and not wp.isnan(best_cost)))):
            best_count,best_cost,chosen_r,chosen_t=total,score,r,direction
    return chosen_r,chosen_t,good


@wp.kernel
def _local(points0: wp.array3d(dtype=wp.float64),points1: wp.array3d(dtype=wp.float64),
           valid: wp.array2d(dtype=wp.bool),p0: wp.array2d(dtype=wp.float64),p1: wp.array2d(dtype=wp.float64),
           active: wp.array(dtype=wp.int64),trials: int,queue: wp.array2d(dtype=wp.int32),
           counts: wp.array(dtype=wp.int32),workers: int,lo_iterations: int,lm_iterations: int,
           threshold: wp.float64,accepted_scale: wp.float64,lower: wp.float64,upper: wp.float64,
           models: wp.array4d(dtype=wp.float64),focals: wp.array2d(dtype=wp.float64),
           costs: wp.array2d(dtype=wp.float64),inliers: wp.array2d(dtype=wp.int32),
           mask: wp.array2d(dtype=wp.bool),proposal: wp.array2d(dtype=wp.bool),stats: wp.array2d(dtype=wp.int32)):
    worker,lane=wp.tid();pair=worker//workers;owner=int(active[pair])
    p=wp.vec2d(p0[owner,0],p0[owner,1]);q=wp.vec2d(p1[owner,0],p1[owner,1])
    entered,visits,steps,accepted,failed=int(0),int(0),int(0),int(0),int(0)
    for job in range(worker%workers,counts[pair],workers):
        local=queue[pair,job];sample=pair*trials+local//60;slot=local%60
        model=_matrix(models,sample,slot);f=focals[sample,slot];cost=costs[sample,slot]
        support=int(0)
        for point in range(lane,points0.shape[1],BLOCK):
            inside=False
            if valid[owner,point]:
                error=_error(model,points0[owner,point,0],points0[owner,point,1],points1[owner,point,0],points1[owner,point,1])
                inside=error<=threshold
            mask[worker,point]=inside;support+=int(inside)
        count=wp.tile_extract(wp.tile_sum(wp.tile(support)),0)
        if count>=8:entered+=1
        for turn in range(lo_iterations):
            if count<8:break
            r,t,good=_pose(model,f,p,q,points0,points1,mask,owner,worker,lane)
            if not good:failed+=1;break
            fitted,new_focal,done=refine(r,t,f,f,p,q,points0,points1,mask,owner,worker,lane,lm_iterations,lower,upper,accepted_scale)
            visits+=1;steps+=done
            total=wp.float64(0.0);support=int(0)
            for point in range(lane,points0.shape[1],BLOCK):
                inside=False
                if valid[owner,point]:
                    error=_error(fitted,points0[owner,point,0],points0[owner,point,1],points1[owner,point,0],points1[owner,point,1])
                    total+=wp.min(error,threshold);inside=error<=threshold
                proposal[worker,point]=inside;support+=int(inside)
            new_cost=wp.tile_extract(wp.tile_sum(wp.tile(total)),0)
            new_count=wp.tile_extract(wp.tile_sum(wp.tile(support)),0)
            if not wp.isfinite(new_cost) or new_cost>=cost:break
            model,f,cost,count=fitted,new_focal,new_cost,new_count;accepted+=1
            for point in range(lane,points0.shape[1],BLOCK):mask[worker,point]=proposal[worker,point]
            _sync()
        if lane==0:
            for i in range(3):
                for j in range(3):models[sample,slot,i,j]=model[i,j]
            focals[sample,slot]=f;costs[sample,slot]=cost;inliers[sample,slot]=count
        _sync()
    if lane==0:
        stats[worker,0]=entered;stats[worker,1]=visits;stats[worker,2]=steps
        stats[worker,3]=accepted;stats[worker,4]=failed


@wp.func
def _winner(costs: wp.array2d(dtype=wp.float64),valid: wp.array2d(dtype=wp.bool),trials: int,pair: int,lane: int):
    best=wp.float64(wp.inf);chosen=int(trials*60)
    for local in range(lane,trials*60,BLOCK):
        sample=pair*trials+local//60;slot=local%60
        if valid[sample,slot]:
            value=costs[sample,slot]
            if value<best or (value==best and local<chosen):best,chosen=value,local
    minimum=wp.tile_extract(wp.tile_min(wp.tile(best)),0)
    if best!=minimum:chosen=trials*60
    winner=wp.tile_extract(wp.tile_min(wp.tile(chosen)),0)
    return minimum,winner


@wp.kernel
def _best(models: wp.array4d(dtype=wp.float64),focals: wp.array2d(dtype=wp.float64),
          costs: wp.array2d(dtype=wp.float64),valid: wp.array2d(dtype=wp.bool),trials: int,
          points0: wp.array3d(dtype=wp.float64),points1: wp.array3d(dtype=wp.float64),
          mask: wp.array2d(dtype=wp.bool),active: wp.array(dtype=wp.int64),threshold: wp.float64,
          output: wp.array3d(dtype=wp.float64),focal: wp.array(dtype=wp.float64),score: wp.array(dtype=wp.float64),
          inliers: wp.array2d(dtype=wp.bool),index: wp.array(dtype=wp.int32)):
    pair,lane=wp.tid()
    minimum,winner=_winner(costs,valid,trials,pair,lane)
    good=winner<trials*60 and wp.isfinite(minimum)
    model=wp.mat33d();f=wp.float64(0.0)
    if good:model=_matrix(models,pair*trials+winner//60,winner%60);f=focals[pair*trials+winner//60,winner%60]
    if lane==0:
        score[pair]=minimum;focal[pair]=f;index[pair]=-1
        if good:index[pair]=winner
        for i in range(3):
            for j in range(3):output[pair,i,j]=model[i,j]
    owner=int(active[pair])
    for point in range(lane,points0.shape[1],BLOCK):
        inside=False
        if good and mask[owner,point]:
            inside=_error(model,points0[owner,point,0],points0[owner,point,1],points1[owner,point,0],points1[owner,point,1])<=threshold
        inliers[pair,point]=inside


@dataclass
class LocalResult:
    models: torch.Tensor
    focals: torch.Tensor
    costs: torch.Tensor
    inliers: torch.Tensor
    indices: torch.Tensor


class LocalOptimizer:
    """Attach to a prepared SixPointBatch. refine() adds exactly three launches.

    Selection is frozen before LO. Eight persistent blocks per pair process
    every selected candidate in order-strided queues, with no candidate cap.
    history is a CUDA float64 cost per active pair; omitted means infinity.
    Result is this batch's best, not a stored incumbent from previous rounds.
    Candidate buffers are refined in place; call frontend.run() before reuse.
    """
    def __init__(self,frontend,*,workers_per_pair=8,lo_iterations=10,lm_iterations=25,focal_ratio_bounds=(1e-6,1e6)):
        for name,value in [('workers_per_pair',workers_per_pair),('lo_iterations',lo_iterations),('lm_iterations',lm_iterations)]:
            if type(value) is not int or value<1:raise ValueError(f'{name} must be a positive integer')
        lo,hi=focal_ratio_bounds
        if not (math.isfinite(lo) and math.isfinite(hi) and 0<lo<hi):raise ValueError('Invalid focal_ratio_bounds')
        self.frontend=frontend;self.workers=workers_per_pair;self.lo_iterations=lo_iterations;self.lm_iterations=lm_iterations
        self.bounds=(lo,hi);self.n=len(frontend.references[6]);device=frontend.references[0].device
        capacity=frontend.references[0].shape[1];workers=self.n*self.workers
        def empty(shape,dtype=torch.float64):return torch.empty(shape,dtype=dtype,device=device)
        self.queue=empty((self.n,frontend.trials*60),torch.int32);self.counts=empty((self.n,),torch.int32)
        self.mask=empty((workers,capacity),torch.bool);self.proposal=empty((workers,capacity),torch.bool)
        self.stats=empty((workers,5),torch.int32)
        self.history=torch.full((self.n,),torch.inf,dtype=torch.float64,device=device)
        self.result=LocalResult(empty((self.n,3,3)),empty((self.n,)),empty((self.n,)),
                                empty((self.n,capacity),torch.bool),empty((self.n,),torch.int32))
        self.views=[wp.from_torch(x) for x in (self.queue,self.counts,self.mask,self.proposal,self.stats)]
        self.result_views=[wp.from_torch(x) for x in vars(self.result).values()]
        self.accepted_scale=float(torch.tensor(1/3,dtype=torch.get_default_dtype()))
        self._storage=(self.queue,self.counts,self.mask,self.proposal,self.stats,self.history)
        self._result_storage=tuple(vars(self.result).values())

    def bind(self):
        """Follow frontend.set_batch(), keeping the initially allocated buffers."""
        n=len(self.frontend.references[6])
        if n>len(self._storage[0]) or self.frontend.trials*60>self._storage[0].shape[1]:
            raise ValueError('Batch exceeds allocated LO workspace')
        self.n=n
        queue,counts,mask,proposal,stats,history=self._storage
        self.queue,self.counts=queue[:n],counts[:n]
        self.mask,self.proposal,self.stats=mask[:n*self.workers],proposal[:n*self.workers],stats[:n*self.workers]
        self.history=history[:n]
        self.result=LocalResult(*(x[:n] for x in self._result_storage))
        self.views=[wp.from_torch(x) for x in (self.queue,self.counts,self.mask,self.proposal,self.stats)]
        self.result_views=[wp.from_torch(x) for x in vars(self.result).values()]

    @torch.no_grad()
    def refine(self,history=None,*,choose_best=True):
        if history is None:history=self.history
        if history.shape!=(self.n,) or history.dtype!=torch.float64 or history.device!=self.history.device:
            raise ValueError('history must be CUDA float64 with one cost per active pair')
        if self.n:
            plan=self.frontend;v=plan.views;o=plan.output_views;queue,counts,mask,proposal,stats=self.views
            stream=wp.stream_from_torch(torch.cuda.current_stream(self.history.device));threshold=wp.float64(plan.max_error**2)
            wp.launch(_select,dim=self.n,block_dim=BLOCK,inputs=[o[2],o[3],wp.from_torch(history),plan.trials,self.mask.shape[1],threshold,queue,counts],stream=stream)
            wp.launch(_local,dim=(self.n*self.workers,BLOCK),block_dim=BLOCK,
                inputs=[v[0],v[1],v[2],v[4],v[5],v[6],plan.trials,queue,counts,self.workers,self.lo_iterations,self.lm_iterations,
                        threshold,wp.float64(self.accepted_scale),wp.float64(self.bounds[0]),wp.float64(self.bounds[1]),
                        o[0],o[1],o[3],o[4],mask,proposal,stats],stream=stream)
            if choose_best:
                wp.launch(_best,dim=(self.n,BLOCK),block_dim=BLOCK,
                    inputs=[o[0],o[1],o[3],o[2],plan.trials,v[0],v[1],v[2],v[6],threshold,*self.result_views],stream=stream)
        return self.result

    @torch.no_grad()
    def run(self,trial=0,history=None):
        self.frontend.run(trial)
        return self.refine(history)
