"""Masked normalization and E/F/H design matrices in one float64 kernel."""
import torch
import warp as wp

wp.init()
wp.set_module_options({'enable_backward': False, 'fast_math': False,
                       'fuse_fp': False, 'max_unroll': 1})
BLOCK = 128


@wp.kernel
def _prepare(points0: wp.array3d(dtype=wp.float64),
             points1: wp.array3d(dtype=wp.float64),
             mask: wp.array2d(dtype=wp.bool), homography: bool,
             design: wp.array3d(dtype=wp.float64),
             transform0: wp.array3d(dtype=wp.float64),
             transform1: wp.array3d(dtype=wp.float64)):
    pair, lane = wp.tid()
    count = wp.float64(0.0)
    x0, y0 = wp.float64(0.0), wp.float64(0.0)
    x1, y1 = wp.float64(0.0), wp.float64(0.0)
    for j in range(lane,points0.shape[1],BLOCK):
        weight = wp.float64(mask[pair,j])
        count += weight
        x0 += points0[pair,j,0]*weight
        y0 += points0[pair,j,1]*weight
        x1 += points1[pair,j,0]*weight
        y1 += points1[pair,j,1]*weight
    count = wp.max(wp.tile_extract(wp.tile_sum(wp.tile(count)),0),wp.float64(1.0))
    x0 = wp.tile_extract(wp.tile_sum(wp.tile(x0)),0)/count
    y0 = wp.tile_extract(wp.tile_sum(wp.tile(y0)),0)/count
    x1 = wp.tile_extract(wp.tile_sum(wp.tile(x1)),0)/count
    y1 = wp.tile_extract(wp.tile_sum(wp.tile(y1)),0)/count
    distance0, distance1 = wp.float64(0.0), wp.float64(0.0)
    for j in range(lane,points0.shape[1],BLOCK):
        weight = wp.float64(mask[pair,j])
        distance0 += wp.length(wp.vec2d(points0[pair,j,0]-x0,points0[pair,j,1]-y0))*weight
        distance1 += wp.length(wp.vec2d(points1[pair,j,0]-x1,points1[pair,j,1]-y1))*weight
    distance0 = wp.tile_extract(wp.tile_sum(wp.tile(distance0)),0)
    distance1 = wp.tile_extract(wp.tile_sum(wp.tile(distance1)),0)
    scale0 = wp.sqrt(wp.float64(2.0))*count/wp.max(distance0,wp.float64(1e-12))
    scale1 = wp.sqrt(wp.float64(2.0))*count/wp.max(distance1,wp.float64(1e-12))
    if lane == 0:
        for row in range(3):
            for col in range(3):
                transform0[pair,row,col] = wp.float64(0.0)
                transform1[pair,row,col] = wp.float64(0.0)
        transform0[pair,0,0] = scale0
        transform0[pair,1,1] = scale0
        transform0[pair,0,2] = -x0*scale0
        transform0[pair,1,2] = -y0*scale0
        transform0[pair,2,2] = wp.float64(1.0)
        transform1[pair,0,0] = scale1
        transform1[pair,1,1] = scale1
        transform1[pair,0,2] = -x1*scale1
        transform1[pair,1,2] = -y1*scale1
        transform1[pair,2,2] = wp.float64(1.0)
    for j in range(lane,points0.shape[1],BLOCK):
        weight = wp.float64(mask[pair,j])
        a = wp.vec3d((points0[pair,j,0]-x0)*scale0,
                    (points0[pair,j,1]-y0)*scale0,wp.float64(1.0))
        b = wp.vec3d((points1[pair,j,0]-x1)*scale1,
                    (points1[pair,j,1]-y1)*scale1,wp.float64(1.0))
        if homography:
            for k in range(3):
                design[pair,2*j,k] = -a[k]*weight
                design[pair,2*j,k+3] = wp.float64(0.0)
                design[pair,2*j,k+6] = b[0]*a[k]*weight
                design[pair,2*j+1,k] = wp.float64(0.0)
                design[pair,2*j+1,k+3] = -a[k]*weight
                design[pair,2*j+1,k+6] = b[1]*a[k]*weight
        else:
            for row in range(3):
                for col in range(3):
                    design[pair,j,3*row+col] = b[row]*a[col]*weight


def prepare(points0, points1, mask, kind):
    """Return (design, T0, T1), with the same masked Hartley normalization."""
    if (points0.ndim != 3 or points0.shape[-1] != 2 or points1.shape != points0.shape
            or mask.shape != points0.shape[:2] or mask.dtype != torch.bool):
        raise ValueError('Expected matching (batch, points, 2) arrays and a boolean mask')
    if (not points0.is_cuda or points0.dtype != torch.float64
            or points1.device != points0.device or points1.dtype != points0.dtype
            or mask.device != points0.device):
        raise ValueError('Warp local refit preparation requires CUDA float64')
    n, m = mask.shape
    design = points0.new_empty((n,m*(2 if kind == 'H' else 1),9))
    transform0 = points0.new_empty((n,3,3))
    transform1 = torch.empty_like(transform0)
    if n:
        wp.launch(_prepare,dim=(n,BLOCK),block_dim=BLOCK,
                  inputs=[wp.from_torch(points0),wp.from_torch(points1),wp.from_torch(mask),kind=='H',
                          wp.from_torch(design),wp.from_torch(transform0),wp.from_torch(transform1)],
                  stream=wp.stream_from_torch(torch.cuda.current_stream(points0.device)))
    return design,transform0,transform1
