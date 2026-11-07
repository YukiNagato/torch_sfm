"""Bounded CUDA graph reuse for repeatedly fitted shared-focal hypotheses."""
from collections import OrderedDict
import torch


class _Graph:
    def __init__(self,args,capacity,points,iterations,bounds,backend):
        from .fundamental_refinement import _refine_shared_focal_lm
        if backend=='warp':
            from .warp_shared_focal_refinement import refine as implementation
        else:implementation=_refine_shared_focal_lm
        reference=args[2];device=reference.device;dtype=reference.dtype
        eye=torch.eye(3,device=device,dtype=dtype).repeat(capacity,1,1)
        direction=reference.new_zeros((capacity,3));direction[:,0]=1
        focal=reference.new_ones(capacity)
        self.inputs=[eye,direction,focal,focal.clone(),reference.new_zeros((capacity,points,2)),
            reference.new_zeros((capacity,points,2)),torch.zeros((capacity,points),device=device,dtype=torch.bool),
            reference.new_zeros((capacity,2)),reference.new_zeros((capacity,2))]
        self.stream=torch.cuda.current_stream(device)
        side=torch.cuda.Stream(device=device);side.wait_stream(self.stream)
        with torch.cuda.stream(side):
            for _ in range(2):implementation(*self.inputs,iterations,bounds)
        self.stream.wait_stream(side)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=side):
            self.outputs=implementation(*self.inputs,iterations,bounds)

    def run(self,args):
        count,points=args[4].shape[:2]
        self.inputs[6].zero_()
        for i,(target,source) in enumerate(zip(self.inputs,args)):
            if i in (4,5,6):target[:count,:points].copy_(source)
            else:target[:count].copy_(source)
        self.graph.replay()
        return tuple(value[:count].clone() for value in self.outputs)


_graphs=OrderedDict()
_calls=OrderedDict()


def run(args,iterations,bounds,backend='torch'):
    from .fundamental_refinement import _refine_shared_focal_lm
    if backend=='warp':
        from .warp_shared_focal_refinement import refine as implementation
    else:implementation=_refine_shared_focal_lm
    count,points=args[4].shape[:2];reference=args[2]
    capacity=max(64,1<<(count-1).bit_length());points=1<<(points-1).bit_length()
    key=(reference.device,reference.dtype,torch.cuda.current_stream(reference.device).cuda_stream,
         capacity,points,iterations,tuple(bounds),backend,torch.get_default_dtype())
    if key not in _graphs:
        _calls[key]=_calls.get(key,0)+1;_calls.move_to_end(key)
        if len(_calls)>32:_calls.popitem(last=False)
        if _calls[key]<3:return implementation(*args,iterations,bounds)
        if len(_graphs)>=4:
            _,old=_graphs.popitem(last=False);old.stream.synchronize()
        _graphs[key]=_Graph(args,capacity,points,iterations,bounds,backend)
    _graphs.move_to_end(key)
    return _graphs[key].run(args)
