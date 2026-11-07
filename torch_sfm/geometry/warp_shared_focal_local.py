"""Score an LO fit and commit its state in one cooperative kernel."""
import torch
import warp as wp
from .warp_relative_pose import _matrix

wp.set_module_options({'enable_backward':False, 'fast_math':False, 'max_unroll':1})
BLOCK=128


@wp.func
def _error(f: wp.mat33d, x: wp.float64, y: wp.float64, u: wp.float64, v: wp.float64):
    # Same evaluation order and denominator guard as warp_two_view._score.
    a=f[0,0]*x+f[0,1]*y+f[0,2]
    b=f[1,0]*x+f[1,1]*y+f[1,2]
    c=f[2,0]*x+f[2,1]*y+f[2,2]
    d=f[0,0]*u+f[1,0]*v+f[2,0]
    e=f[0,1]*u+f[1,1]*v+f[2,1]
    numerator=u*a+v*b+c
    error=numerator*numerator/wp.max(a*a+b*b+d*d+e*e,wp.float64(1e-30))
    if not wp.isfinite(error):error=wp.float64(wp.inf)
    return error


@wp.kernel
def _update(fitted: wp.array3d(dtype=wp.float64), new_focal: wp.array(dtype=wp.float64),
            sub: wp.array(dtype=wp.int64), pair: wp.array(dtype=wp.int64), ids: wp.array(dtype=wp.int64),
            points0: wp.array3d(dtype=wp.float64), points1: wp.array3d(dtype=wp.float64),
            valid: wp.array2d(dtype=wp.bool), threshold: wp.float64,
            model: wp.array3d(dtype=wp.float64), focal: wp.array(dtype=wp.float64),
            score: wp.array(dtype=wp.float64), mask: wp.array2d(dtype=wp.bool),
            active: wp.array(dtype=wp.bool), errors: wp.array2d(dtype=wp.float64)):
    i,lane=wp.tid();slot=sub[i];owner=pair[slot];f=_matrix(fitted,i)
    cost=wp.float64(0.0)
    for j in range(lane,points0.shape[1],BLOCK):
        if valid[owner,j]:
            error=_error(f,points0[owner,j,0],points0[owner,j,1],points1[owner,j,0],points1[owner,j,1])
            cost+=wp.min(error,threshold)
    total=wp.tile_extract(wp.tile_sum(wp.tile(cost)),0)
    accept=wp.isfinite(total) and total<score[slot]
    # All lanes must read the old score before lane zero commits it.
    accepted=wp.tile_extract(wp.tile_sum(wp.tile(int(accept))),0)>0
    if lane==0:
        active[slot]=accepted
        if accepted:
            focal[slot]=new_focal[i];score[slot]=total
            for row in range(3):
                for col in range(3):model[slot,row,col]=f[row,col]
    if accepted:
        target=ids[slot]
        for j in range(lane,points0.shape[1],BLOCK):
            error=wp.float64(0.0)
            if valid[owner,j]:
                error=_error(f,points0[owner,j,0],points0[owner,j,1],points1[owner,j,0],points1[owner,j,1])
            errors[target,j]=error
            mask[slot,j]=valid[owner,j] and error<=threshold


def update(fitted,new_focal,sub,pair,ids,points0,points1,valid,threshold,model,focal,score,mask,active,errors):
    wp.launch(_update,dim=(len(sub),BLOCK),block_dim=BLOCK,
              inputs=[wp.from_torch(v) for v in (fitted,new_focal,sub,pair,ids,points0,points1,valid)]
              +[wp.float64(threshold)]
              +[wp.from_torch(v) for v in (model,focal,score,mask,active,errors)],
              stream=wp.stream_from_torch(torch.cuda.current_stream(fitted.device)))
