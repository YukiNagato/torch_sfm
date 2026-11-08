# BSD 3-Clause License
# Copyright (c) 2020, Viktor Larsson. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Two experimental solver layouts, each followed by the same scoring kernel.

ThreadSixPointBatch: one thread owns sampling and the complete minimal solver.
HybridSixPointBatch: sample-parallel scalar phases, warp-cooperative elimination
inside the same block, and shared coefficient storage. No CPU fallback.
"""
from functools import lru_cache

import torch
import warp as wp

from .six_point import Keys, Indices, _sync
from .six_point_math import _nullspace, _action_roots, _recover
from .six_point_scalar_polynomial import coefficients
from .six_point_split import SplitSixPointBatch

OPTIONS = {'enable_backward':False,'fast_math':False,'fuse_fp':False,'max_unroll':1}
wp.set_module_options(OPTIONS)


@wp.func
def _initialize(sample: int, p0: wp.array3d(dtype=wp.float64), p1: wp.array3d(dtype=wp.float64),
                mask: wp.array2d(dtype=wp.bool), sizes: wp.array(dtype=wp.int64),
                principal0: wp.array2d(dtype=wp.float64), principal1: wp.array2d(dtype=wp.float64),
                active: wp.array(dtype=wp.int64), trials: int, trial: wp.int64, bias: wp.uint32,
                a: wp.array3d(dtype=wp.float64), b: wp.array3d(dtype=wp.float64),
                pp0: wp.array2d(dtype=wp.float64), pp1: wp.array2d(dtype=wp.float64),
                basis: wp.array2d(dtype=wp.float64), scales: wp.array(dtype=wp.float64),
                counts: wp.array(dtype=wp.int32), root_status: wp.array2d(dtype=wp.int32),
                models: wp.array4d(dtype=wp.float64), focals: wp.array2d(dtype=wp.float64),
                valid: wp.array2d(dtype=wp.bool), costs: wp.array2d(dtype=wp.float64),
                inliers: wp.array2d(dtype=wp.int32), samples: wp.array2d(dtype=wp.int32),
                status: wp.array(dtype=wp.int32)):
    status[sample] = 1
    counts[sample] = 0
    for k in range(15):
        root_status[sample,k] = 0
    for slot in range(60):
        focals[sample,slot] = wp.float64(0.0)
        valid[sample,slot] = False
        costs[sample,slot] = wp.float64(wp.inf)
        inliers[sample,slot] = 0
        for i in range(3):
            for j in range(3):
                models[sample,slot,i,j] = wp.float64(0.0)
    for k in range(6):
        samples[sample,k] = -1
    pair = int(active[sample//trials])
    if sizes[pair]<6:
        return
    keys,indices = Keys(),Indices()
    for k in range(6):
        keys[k] = wp.int64(-1)
    for point in range(mask.shape[1]):
        if mask[pair,point]:
            counter = (trial+wp.int64(sample%trials))*sizes[pair]+wp.int64(point)
            bits = wp.uint32(counter)+bias
            bits = (bits^(bits>>wp.uint32(16)))*wp.uint32(2246822519)
            bits = (bits^(bits>>wp.uint32(13)))*wp.uint32(3266489917)
            bits = bits^(bits>>wp.uint32(16))
            key,index = wp.int64(bits),point
            for k in range(6):
                if key>keys[k]:
                    previous_key,previous_index = keys[k],indices[k]
                    keys[k] = key
                    indices[k] = index
                    key,index = previous_key,previous_index
    for k in range(6):
        point = indices[k]
        samples[sample,k] = point
        for j in range(2):
            a[sample,k,j] = p0[pair,point,j]
            b[sample,k,j] = p1[pair,point,j]
    for j in range(2):
        pp0[sample,j] = principal0[pair,j]
        pp1[sample,j] = principal1[pair,j]
    _nullspace(sample,a,b,pp0,pp1,basis,scales,status)


@wp.func_native('''
wp::index(counts,sample) = 0;
if (wp::index(status,sample)) return;
double c[31][31], rhs[31][15];
for (int col=0; col<31; ++col) for (int i=0; i<31; ++i) {
    int ix=wp::index(mapping,col,i);
    c[i][col]=ix>=0 ? wp::index(coeff,sample,ix) : 0.0;
}
for (int col=0; col<15; ++col) for (int i=0; i<31; ++i) {
    int ix=wp::index(mapping,col+31,i);
    rhs[i][col]=ix>=0 ? wp::index(coeff,sample,ix) : 0.0;
}
for (int k=0; k<31; ++k) {
    int row=k;
    double pivot=fabs(c[k][k]);
    for (int i=k+1; i<31; ++i) if (fabs(c[i][k])>pivot) {
        pivot=fabs(c[i][k]); row=i;
    }
    if (pivot==0.0 || !isfinite(pivot)) { wp::index(status,sample)=2; return; }
    for (int col=k; col<31; ++col) {
        double t=c[k][col]; c[k][col]=c[row][col]; c[row][col]=t;
    }
    for (int col=0; col<15; ++col) {
        double t=rhs[k][col]; rhs[k][col]=rhs[row][col]; rhs[row][col]=t;
    }
    double inv=1.0/c[k][k];
    for (int i=k+1; i<31; ++i) {
        double factor=c[i][k]*inv;
        for (int col=k+1; col<31; ++col) c[i][col]-=factor*c[k][col];
        for (int col=0; col<15; ++col) rhs[i][col]-=factor*rhs[k][col];
    }
}
for (int i=30; i>=0; --i) for (int col=0; col<15; ++col) {
    double value=rhs[i][col];
    for (int k=i+1; k<31; ++k) value-=c[i][k]*rhs[k][col];
    rhs[i][col]=value/c[i][i];
}
for (int i=0; i<15; ++i) for (int col=0; col<15; ++col) {
    int row=wp::index(action_rows,i);
    wp::index(action,sample,i,col)=row<8 ? -rhs[23+row][col] : (row-8==col ? 1.0 : 0.0);
}
''')
def _eliminate_serial(coeff: wp.array2d(dtype=wp.float64), mapping: wp.array2d(dtype=wp.int32),
                       action_rows: wp.array(dtype=wp.int32), action: wp.array3d(dtype=wp.float64),
                       counts: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32), sample: int): ...


@wp.func_native('''
// All lanes of this warp participate; different warps may own different samples.
int col=threadIdx.x%32;
if (col==0) wp::index(counts,sample)=0;
if (wp::index(status,sample)) return;
double c[31]={0.0}, rhs[31]={0.0};
for (int i=0; i<31; ++i) {
    if (col<31) {
        int ix=wp::index(mapping,col,i);
        if (ix>=0) c[i]=wp::index(coeff,local_sample,ix);
    }
    if (col<15) {
        int ix=wp::index(mapping,col+31,i);
        if (ix>=0) rhs[i]=wp::index(coeff,local_sample,ix);
    }
}
for (int k=0; k<31; ++k) {
    int row=k; double pivot=0.0;
    if (col==k) {
        pivot=fabs(c[k]);
        for (int i=k+1; i<31; ++i) if (fabs(c[i])>pivot) {pivot=fabs(c[i]);row=i;}
    }
    row=__shfl_sync(0xffffffff,row,k);
    pivot=__shfl_sync(0xffffffff,pivot,k);
    if (pivot==0.0 || !isfinite(pivot)) { if (col==0) wp::index(status,sample)=2; return; }
    if (col>=k) {double t=c[k];c[k]=c[row];c[row]=t;}
    double t=rhs[k];rhs[k]=rhs[row];rhs[row]=t;
    double diagonal=__shfl_sync(0xffffffff,c[k],k), inv=1.0/diagonal;
    for (int i=k+1; i<31; ++i) {
        double factor=__shfl_sync(0xffffffff,c[i],k)*inv;
        if (col>k) c[i]-=factor*c[k];
        if (col<15) rhs[i]-=factor*rhs[k];
    }
}
for (int i=30; i>=0; --i) {
    double value=rhs[i];
    for (int k=i+1; k<31; ++k) value-=__shfl_sync(0xffffffff,c[i],k)*rhs[k];
    rhs[i]=value/__shfl_sync(0xffffffff,c[i],i);
}
if (col<15) for (int i=0; i<15; ++i) {
    int row=wp::index(action_rows,i);
    wp::index(action,sample,i,col)=row<8 ? -rhs[23+row] : (row-8==col ? 1.0 : 0.0);
}
''')
def _eliminate_warp(coeff: wp.array2d(dtype=wp.float64), mapping: wp.array2d(dtype=wp.int32),
                     action_rows: wp.array(dtype=wp.int32), action: wp.array3d(dtype=wp.float64),
                     counts: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32),
                     sample: int, local_sample: int): ...


@wp.func
def _finish(sample: int, a: wp.array3d(dtype=wp.float64), b: wp.array3d(dtype=wp.float64),
             p0: wp.array2d(dtype=wp.float64), p1: wp.array2d(dtype=wp.float64),
             basis: wp.array2d(dtype=wp.float64), scales: wp.array(dtype=wp.float64),
             action: wp.array3d(dtype=wp.float64), roots: wp.array2d(dtype=wp.float64),
             counts: wp.array(dtype=wp.int32), root_status: wp.array2d(dtype=wp.int32),
             models: wp.array4d(dtype=wp.float64), focals: wp.array2d(dtype=wp.float64),
             valid: wp.array2d(dtype=wp.bool), status: wp.array(dtype=wp.int32)):
    _action_roots(sample,action,roots,counts,status)
    for root in range(counts[sample]):
        _recover(sample,root,a,b,p0,p1,basis,scales,action,roots,counts,models,focals,valid,root_status)


@lru_cache(maxsize=None)
def _make_solver(hybrid,group_size,block_size):
    tile_rows = 307 if hybrid else 1
    # A spare column avoids power-of-two bank strides when a warp reads
    # different coefficients of the same sample during elimination.
    tile_cols = group_size+1 if hybrid else 1
    tile_type = wp.tile[wp.float64,tile_rows,tile_cols]

    @wp.func_native(f'''
    wp::array_t<double> out(tile.data.ptr + offset*{tile_cols}, {tile_cols}, cols);
    out.strides[0]=sizeof(double); out.strides[1]={tile_cols}*sizeof(double);
    return out;
    ''')
    def shared_view(tile: tile_type, offset: int, cols: int) -> wp.array2d(dtype=wp.float64): ...

    @wp.kernel(module='unique')
    def solve(p0: wp.array3d(dtype=wp.float64), p1: wp.array3d(dtype=wp.float64),
              mask: wp.array2d(dtype=wp.bool), sizes: wp.array(dtype=wp.int64),
              principal0: wp.array2d(dtype=wp.float64), principal1: wp.array2d(dtype=wp.float64),
              active: wp.array(dtype=wp.int64), trials: int, trial: wp.int64, bias: wp.uint32,
              mapping: wp.array2d(dtype=wp.int32), action_rows: wp.array(dtype=wp.int32),
              a: wp.array3d(dtype=wp.float64), b: wp.array3d(dtype=wp.float64),
              pp0: wp.array2d(dtype=wp.float64), pp1: wp.array2d(dtype=wp.float64),
              basis: wp.array2d(dtype=wp.float64), scales: wp.array(dtype=wp.float64),
              action: wp.array3d(dtype=wp.float64), roots: wp.array2d(dtype=wp.float64),
              counts: wp.array(dtype=wp.int32), root_status: wp.array2d(dtype=wp.int32),
              coeff: wp.array2d(dtype=wp.float64),
              models: wp.array4d(dtype=wp.float64), focals: wp.array2d(dtype=wp.float64),
              valid: wp.array2d(dtype=wp.bool), costs: wp.array2d(dtype=wp.float64),
              inliers: wp.array2d(dtype=wp.int32), samples: wp.array2d(dtype=wp.int32),
              status: wp.array(dtype=wp.int32)):
        block,lane = wp.tid()
        sample = block*block_size+lane
        if wp.static(hybrid):
            sample = block*group_size+lane
        live = sample<status.shape[0]
        if wp.static(hybrid):
            live = live and lane<group_size
        if live:
            _initialize(sample,p0,p1,mask,sizes,principal0,principal1,active,trials,trial,bias,
                        a,b,pp0,pp1,basis,scales,counts,root_status,models,focals,valid,costs,inliers,samples,status)
        if wp.static(hybrid):
            tile = wp.tile_zeros(shape=(tile_rows,tile_cols),dtype=wp.float64,storage='shared')
            local_basis = shared_view(tile,0,27)
            local_coeff = shared_view(tile,27,280)
            if live:
                if status[sample]==0:
                    for j in range(27):
                        local_basis[lane,j] = basis[sample,j]
                    coefficients(lane,local_basis,local_coeff)
            _sync()
            for turn in range(wp.static((group_size+block_size//32-1)//(block_size//32))):
                local_sample = turn*(block_size//32)+lane//32
                target = block*group_size+local_sample
                if local_sample<group_size and target<status.shape[0]:
                    _eliminate_warp(local_coeff,mapping,action_rows,action,counts,status,target,local_sample)
            _sync()
        else:
            if live:
                if status[sample]==0:
                    coefficients(sample,basis,coeff)
                    _eliminate_serial(coeff,mapping,action_rows,action,counts,status,sample)
        if live:
            _finish(sample,a,b,pp0,pp1,basis,scales,action,roots,counts,root_status,models,focals,valid,status)

    wp.set_module_options(OPTIONS,module=solve.module)
    return solve


def _sample_workspace(n, device, *, coefficients=True):
    """Sample-contiguous intermediate arrays shared by scalar solver layouts."""
    def soa(tail, dtype=torch.float64):
        return torch.empty((*tail,n),dtype=dtype,device=device).movedim(-1,0)
    return (soa((6,2)),soa((6,2)),soa((2,)),soa((2,)),soa((27,)),soa(()),
            soa((15,15)),soa((15,)),soa((),torch.int32),soa((15,),torch.int32),
            soa((280,)) if coefficients else soa((0,)))


class ThreadSixPointBatch(SplitSixPointBatch):
    """One solver launch, one scoring launch; serial arithmetic per sample.

    Global intermediates use sample-contiguous (SoA) storage for coalesced
    access between independent threads. Serial LU scratch is CUDA local memory.
    """
    def __init__(self,*args,solver_block_size=128,**kwargs):
        if solver_block_size not in (32,64,128,256):
            raise ValueError('solver_block_size must be 32, 64, 128 or 256')
        super().__init__(*args,**kwargs)
        self.solver_block_size=solver_block_size
        self.group_size=solver_block_size
        self.hybrid=False
        self.solver=_make_solver(False,solver_block_size,solver_block_size)
        self._allocate_workspace()

    def _allocate_workspace(self):
        n=len(self.result.status);device=self.references[0].device
        self.workspace=_sample_workspace(n,device,coefficients=not self.hybrid)
        self.workspace_views=[wp.from_torch(x) for x in self.workspace]

    @torch.no_grad()
    def generate(self,trial=0):
        if type(trial) is not int or trial<0:
            raise ValueError('trial must be a nonnegative integer')
        n=len(self.result.status)
        if n:
            wp.launch(self.solver,dim=((n+self.group_size-1)//self.group_size,self.solver_block_size),
                      block_dim=self.solver_block_size,
                      inputs=[*self.views,self.trials,wp.int64(trial),self.bias,*self.map_views,*self.workspace_views],
                      outputs=self.output_views[:7],
                      stream=wp.stream_from_torch(torch.cuda.current_stream(self.references[0].device)))
        return self.result


class HybridSixPointBatch(ThreadSixPointBatch):
    """Scalar phases and cooperative LU switch roles within one block.

    group_size selects simultaneous samples (8/16/32). The block has 32/64/128
    threads: scalar phases use one thread per sample; LU uses one warp per
    sample in successive rounds. Coefficients remain in dynamic shared memory.
    """
    def __init__(self,*args,group_size=16,solver_block_size=64,**kwargs):
        if type(group_size) is not int or group_size not in (8,16,32):
            raise ValueError('group_size must be 8, 16 or 32')
        if type(solver_block_size) is not int or solver_block_size not in (32,64,128):
            raise ValueError('solver_block_size must be 32, 64 or 128')
        SplitSixPointBatch.__init__(self,*args,**kwargs)
        self.solver_block_size=solver_block_size
        self.group_size=group_size
        self.hybrid=True
        self.solver=_make_solver(True,group_size,solver_block_size)
        self._allocate_workspace()
