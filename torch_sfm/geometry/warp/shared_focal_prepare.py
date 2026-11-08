"""Fuse shared-focal camera/point construction while preserving solver arithmetic."""
import torch
import warp as wp

wp.init()
wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})


@wp.kernel
def _prepare(points0: wp.array3d(dtype=wp.float64), points1: wp.array3d(dtype=wp.float64),
             focal: wp.array(dtype=wp.float64), principal0: wp.array2d(dtype=wp.float64),
             principal1: wp.array2d(dtype=wp.float64), inverse0: wp.array3d(dtype=wp.float64),
             inverse1: wp.array3d(dtype=wp.float64), x: wp.array3d(dtype=wp.float64),
             y: wp.array3d(dtype=wp.float64)):
    i, j = wp.tid()
    f = focal[i]
    if j == 0:
        for row in range(3):
            for col in range(3):
                a, b = wp.float64(0.0), wp.float64(0.0)
                if row == col:
                    a = wp.float64(1.0)
                    if row < 2:
                        a = wp.float64(1.0)/f
                    b = a
                elif row < 2 and col == 2:
                    a = -principal0[i,row]/f
                    b = -principal1[i,row]/f
                inverse0[i,row,col] = a
                inverse1[i,row,col] = b
    if j < points0.shape[1]:
        for axis in range(2):
            x[i,j,axis] = points0[i,j,axis]
            y[i,j,axis] = points1[i,j,axis]
        x[i,j,2] = wp.float64(1.0)
        y[i,j,2] = wp.float64(1.0)


def prepare(matrix, points0, points1, focal, principal0, principal1):
    """Forward-only CUDA float64 preparation; focal must be finite and positive.

    The same shared focal and per-view principal points used by the LM define
    both camera matrices. Outputs own their storage and use the current stream.
    """
    n = len(focal)
    if (matrix.shape != (n,3,3) or points0.ndim != 3 or points0.shape[0] != n
            or points0.shape[-1] != 2 or points1.shape != points0.shape
            or principal0.shape != (n,2) or principal1.shape != (n,2) or focal.ndim != 1):
        raise ValueError('Invalid shared-focal initialization shapes')
    values = (matrix, points0, points1, focal, principal0, principal1)
    if not focal.is_cuda or any(t.device != focal.device or t.dtype != torch.float64 for t in values):
        raise ValueError('Shared-focal preparation requires CUDA float64 inputs')
    if torch.is_grad_enabled() and any(t.requires_grad for t in values):
        raise ValueError('Shared-focal preparation is forward only')
    inverse = focal.new_empty((2,n,3,3))
    inverse0,inverse1 = inverse.unbind(0)
    coordinates = points0.new_empty((2,*points0.shape[:2],3))
    a,b = coordinates.unbind(0)
    if n:
        wp.launch(_prepare, dim=(n,max(points0.shape[1],1)),
                  inputs=[wp.from_torch(t) for t in (points0,points1,focal,principal0,principal1,inverse0,inverse1,a,b)],
                  stream=wp.stream_from_torch(torch.cuda.current_stream(focal.device)))
    # Retain the original inverse and GEMM order. Replacing inv(inv(K)) by K
    # changes rounding and can alter LM acceptance on ill-conditioned pairs.
    # Positive finite focal makes these triangular matrices nonsingular. Use
    # the same inverse kernels without the synchronous host error check.
    cameras = torch.linalg.inv_ex(inverse.reshape(2*n,3,3),check_errors=False).inverse
    k0,k1 = cameras.reshape(2,n,3,3).unbind(0)
    essential = k1.transpose(-1,-2)@matrix@k0
    rays = coordinates.reshape(2*n,points0.shape[1],3)@inverse.reshape(2*n,3,3).transpose(-1,-2)
    rays0,rays1 = rays.reshape(2,n,points0.shape[1],3).unbind(0)
    return essential,rays0,rays1
