"""Fixed-capacity E/F/H minimal solvers, without packing or numerical retries."""
import torch
import warp as wp
from ..geometry import warp_five_point_poselib as five
from ..geometry import warp_seven_point as seven
from ..geometry import warp_homography as homo
from ..geometry.warp_nullspace import _nullspace
from ..geometry.warp_ransac_sampling import _sample

class MinimalBatch:
    def __init__(self, points0, points1, valid, sizes, active, cfg, kind):
        self.kind=kind;self.size,self.slots,self.salt={'E':(5,10,0),'F':(7,3,1),'H':(4,1,2)}[kind]
        self.max_trials=min(cfg.batch_size,cfg.max_trials);n=len(points0)*self.max_trials
        def empty(shape,dtype=torch.float64):return torch.empty(shape,device=points0.device,dtype=dtype)
        self.a=empty((n,self.size,2));self.b=empty(self.a.shape)
        self.models=empty((n,self.slots,3,3));self.valid=empty((n,self.slots),torch.bool)
        self.costs=empty((n*self.slots,));self.poses=empty((n*self.slots,4,3))
        self.inputs=[wp.from_torch(x) for x in (points0,points1,valid,sizes)]
        self.active=active;self.cfg=cfg
        if kind=='E':
            self.work=[empty((n,4,9)),empty((n,),torch.int32),empty((n,3,13)),empty((n,10)),empty((n,),torch.int32),empty((n,10),torch.int32)]
        else:
            rows=7 if kind=='F' else 8
            self.work=[empty((n,rows,9)),empty((n,2,3)),empty((n,9,9-rows)),empty((n,),torch.bool),empty((n,4)),empty((n,),torch.bool)]

    def run(self, active, trial, batch):
        n=len(active)*batch;self.n=n;self.batch=batch
        stream=wp.stream_from_torch(torch.cuda.current_stream(self.a.device))
        def launch(kernel,dim,args,block=32):
            wp.launch(kernel,dim=dim,block_dim=block,inputs=[wp.from_torch(x) if isinstance(x,torch.Tensor) else x for x in args],stream=stream)
        a,b=self.a[:n],self.b[:n];models,valid=self.models[:n],self.valid[:n]
        bias=wp.uint32((((self.cfg.seed+self.salt*1009)%2**32)*104729+374761393)&0xffffffff)
        launch(_sample,(len(active),batch,32),[*self.inputs,active,wp.int64(trial),bias,self.size,batch,a,b])
        w=[x[:n] for x in self.work]
        if self.kind=='E':
            basis,status,matrices,roots,counts,root_status=w
            launch(five._nullspace,n,[a,b,basis,status])
            launch(five._eliminate,n,[basis,status,matrices,roots,counts])
            launch(five._recover,(n,10),[a,b,basis,matrices,roots,counts,True,models,valid,root_status])
        else:
            design,transforms,vectors,rank,coeff,weak=w
            launch(seven._prepare if self.kind=='F' else homo._prepare,n,[a,b,design,transforms])
            launch(_nullspace,n,[design,vectors,rank,design.shape[1]])
            if self.kind=='F':
                launch(seven._coefficients,n,[vectors,rank,coeff,weak])
                launch(seven._models,n,[vectors,rank,coeff,transforms,models,valid])
            else:
                launch(homo._finish,n,[vectors.reshape(n,9),rank,transforms,models.reshape(n,3,3),valid.reshape(n)])
        return models.reshape(-1,3,3),valid.reshape(-1),self.costs[:n*self.slots],self.poses[:n*self.slots]
