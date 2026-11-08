"""Float64 four-point DLT preparation and denormalization around existing QR."""
import torch
import warp as wp

wp.init()
wp.set_module_options({'enable_backward': False, 'fast_math': False,
                       'fuse_fp': False, 'max_unroll': 1})


@wp.kernel
def _prepare(points0: wp.array3d(dtype=wp.float64),
             points1: wp.array3d(dtype=wp.float64),
             design: wp.array3d(dtype=wp.float64),
             transforms: wp.array3d(dtype=wp.float64)):
    sample = wp.tid()
    center0, center1 = wp.vec2d(), wp.vec2d()
    for i in range(4):
        center0 += wp.vec2d(points0[sample,i,0],points0[sample,i,1])
        center1 += wp.vec2d(points1[sample,i,0],points1[sample,i,1])
    center0 /= wp.float64(4.0)
    center1 /= wp.float64(4.0)
    distance0, distance1 = wp.float64(0.0), wp.float64(0.0)
    for i in range(4):
        distance0 += wp.length(wp.vec2d(points0[sample,i,0],points0[sample,i,1])-center0)
        distance1 += wp.length(wp.vec2d(points1[sample,i,0],points1[sample,i,1])-center1)
    scale0 = wp.sqrt(wp.float64(2.0))/wp.max(distance0/wp.float64(4.0),wp.float64(1e-12))
    scale1 = wp.sqrt(wp.float64(2.0))/wp.max(distance1/wp.float64(4.0),wp.float64(1e-12))
    transforms[sample,0,0] = scale0
    transforms[sample,0,1] = -center0[0]*scale0
    transforms[sample,0,2] = -center0[1]*scale0
    transforms[sample,1,0] = scale1
    transforms[sample,1,1] = -center1[0]*scale1
    transforms[sample,1,2] = -center1[1]*scale1
    for i in range(4):
        a = (wp.vec2d(points0[sample,i,0],points0[sample,i,1])-center0)*scale0
        b = (wp.vec2d(points1[sample,i,0],points1[sample,i,1])-center1)*scale1
        x = wp.vec3d(a[0],a[1],wp.float64(1.0))
        for k in range(3):
            design[sample,2*i,k] = -x[k]
            design[sample,2*i,k+3] = wp.float64(0.0)
            design[sample,2*i,k+6] = b[0]*x[k]
            design[sample,2*i+1,k] = wp.float64(0.0)
            design[sample,2*i+1,k+3] = -x[k]
            design[sample,2*i+1,k+6] = b[1]*x[k]


@wp.func
def _determinant(matrix: wp.mat33d):
    # Partial-pivot LU, as in Torch's determinant. Avoid a cofactor expansion
    # with unnecessary cancellation in large translated image coordinates.
    a = matrix
    determinant = wp.float64(1.0)
    for k in range(3):
        pivot = k
        for row in range(k+1,3):
            if wp.abs(a[row,k]) > wp.abs(a[pivot,k]):
                pivot = row
        if pivot != k:
            for col in range(3):
                value = a[k,col]
                a[k,col] = a[pivot,col]
                a[pivot,col] = value
            determinant = -determinant
        diagonal = a[k,k]
        if diagonal == wp.float64(0.0):
            return wp.float64(0.0)
        determinant *= diagonal
        for row in range(k+1,3):
            factor = a[row,k]/diagonal
            for col in range(k+1,3):
                a[row,col] -= factor*a[k,col]
    return determinant


@wp.kernel
def _finish(vectors: wp.array2d(dtype=wp.float64), rank: wp.array(dtype=wp.bool),
            transforms: wp.array3d(dtype=wp.float64), models: wp.array3d(dtype=wp.float64),
            valid: wp.array(dtype=wp.bool)):
    sample = wp.tid()
    valid[sample] = False
    if not rank[sample]:
        return
    # Q = H_normalized T0, followed by the upper-triangular solve T1 H = Q.
    q = wp.mat33d()
    for row in range(3):
        q[row,0] = vectors[sample,3*row]*transforms[sample,0,0]
        q[row,1] = vectors[sample,3*row+1]*transforms[sample,0,0]
        q[row,2] = (vectors[sample,3*row]*transforms[sample,0,1]
                    +vectors[sample,3*row+1]*transforms[sample,0,2])+vectors[sample,3*row+2]
    h = wp.mat33d()
    for col in range(3):
        h[2,col] = q[2,col]
        h[0,col] = (q[0,col]-transforms[sample,1,1]*q[2,col])/transforms[sample,1,0]
        h[1,col] = (q[1,col]-transforms[sample,1,2]*q[2,col])/transforms[sample,1,0]
    valid[sample] = wp.abs(_determinant(h)) > wp.float64(1e-12)
    for row in range(3):
        for col in range(3):
            models[sample,row,col] = h[row,col]


def homography_dlt(points0, points1, return_owners=False):
    from .nullspace import right_null_vector8
    n = len(points0)
    if points0.shape != (n,4,2) or points1.shape != points0.shape:
        raise ValueError('Expected matching batches of four 2D points')
    if (not points0.is_cuda or points0.dtype != torch.float64
            or points1.device != points0.device or points1.dtype != points0.dtype):
        raise ValueError('Warp four-point DLT requires CUDA float64')
    if not n:
        models = points0.new_empty((0,3,3))
        owners = torch.empty(0,device=points0.device,dtype=torch.long)
        return (models,owners) if return_owners else models
    stream = wp.stream_from_torch(torch.cuda.current_stream(points0.device))
    design = points0.new_empty((n,8,9))
    transforms = points0.new_empty((n,2,3))
    wp.launch(_prepare,dim=n,inputs=[wp.from_torch(v) for v in
        (points0,points1,design,transforms)],stream=stream)
    vectors,rank = right_null_vector8(design)
    models = points0.new_empty((n,3,3))
    valid = torch.empty(n,device=points0.device,dtype=torch.bool)
    wp.launch(_finish,dim=n,inputs=[wp.from_torch(v) for v in
        (vectors,rank,transforms,models,valid)],stream=stream)
    owners = torch.where(valid)[0]
    result = models[owners]
    return (result,owners) if return_owners else result
