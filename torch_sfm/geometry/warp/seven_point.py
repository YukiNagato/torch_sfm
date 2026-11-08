"""Fused float64 seven-point preparation and cubic/model evaluation.

The existing Householder nullspace and its rank/chart policies are retained.
Only the surrounding arithmetic is fused; model slots remain sample-major.
"""
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
    center0 = wp.vec2d()
    center1 = wp.vec2d()
    for i in range(7):
        center0 += wp.vec2d(points0[sample,i,0], points0[sample,i,1])
        center1 += wp.vec2d(points1[sample,i,0], points1[sample,i,1])
    center0 /= wp.float64(7.0)
    center1 /= wp.float64(7.0)
    distance0 = wp.float64(0.0)
    distance1 = wp.float64(0.0)
    for i in range(7):
        distance0 += wp.length(wp.vec2d(points0[sample,i,0], points0[sample,i,1])-center0)
        distance1 += wp.length(wp.vec2d(points1[sample,i,0], points1[sample,i,1])-center1)
    scale0 = wp.sqrt(wp.float64(2.0))/wp.max(distance0/wp.float64(7.0), wp.float64(1e-12))
    scale1 = wp.sqrt(wp.float64(2.0))/wp.max(distance1/wp.float64(7.0), wp.float64(1e-12))
    transforms[sample,0,0] = scale0
    transforms[sample,0,1] = -center0[0]*scale0
    transforms[sample,0,2] = -center0[1]*scale0
    transforms[sample,1,0] = scale1
    transforms[sample,1,1] = -center1[0]*scale1
    transforms[sample,1,2] = -center1[1]*scale1
    for i in range(7):
        a = (wp.vec2d(points0[sample,i,0], points0[sample,i,1])-center0)*scale0
        b = (wp.vec2d(points1[sample,i,0], points1[sample,i,1])-center1)*scale1
        x = wp.vec3d(a[0], a[1], wp.float64(1.0))
        y = wp.vec3d(b[0], b[1], wp.float64(1.0))
        for j in range(3):
            for k in range(3):
                design[sample,i,3*j+k] = y[j]*x[k]


@wp.kernel
def _coefficients(vectors: wp.array3d(dtype=wp.float64),
                  rank: wp.array(dtype=wp.bool),
                  coefficients: wp.array2d(dtype=wp.float64),
                  weak: wp.array(dtype=wp.bool)):
    i = wp.tid()
    a0 = wp.vec3d(vectors[i,0,0], vectors[i,1,0], vectors[i,2,0])
    a1 = wp.vec3d(vectors[i,3,0], vectors[i,4,0], vectors[i,5,0])
    a2 = wp.vec3d(vectors[i,6,0], vectors[i,7,0], vectors[i,8,0])
    b0 = wp.vec3d(vectors[i,0,1], vectors[i,1,1], vectors[i,2,1])
    b1 = wp.vec3d(vectors[i,3,1], vectors[i,4,1], vectors[i,5,1])
    b2 = wp.vec3d(vectors[i,6,1], vectors[i,7,1], vectors[i,8,1])
    aa = wp.cross(a1,a2)
    bb = wp.cross(b1,b2)
    ab = wp.cross(a1,b2)+wp.cross(b1,a2)
    c = wp.vec4d(wp.dot(a0,aa), wp.dot(b0,aa)+wp.dot(a0,ab),
                 wp.dot(a0,bb)+wp.dot(b0,ab), wp.dot(b0,bb))
    maximum = wp.float64(0.0)
    for j in range(4):
        coefficients[i,j] = c[j]
        maximum = wp.max(maximum,wp.abs(c[j]))
    weak[i] = rank[i] and (wp.abs(c[3]) <= wp.float64(1e-8)*maximum or wp.abs(c[3]) <= wp.float64(1e-10))


@wp.kernel
def _models(vectors: wp.array3d(dtype=wp.float64),
            rank: wp.array(dtype=wp.bool),
            coefficients: wp.array2d(dtype=wp.float64),
            transforms: wp.array3d(dtype=wp.float64),
            models: wp.array4d(dtype=wp.float64),
            valid: wp.array2d(dtype=wp.bool)):
    i = wp.tid()
    for j in range(3):
        valid[i,j] = False
    leading = coefficients[i,3]
    if not rank[i] or not (wp.abs(leading) > wp.float64(1e-12)):
        return
    c = coefficients[i,0]/leading
    b = coefficients[i,1]/leading
    a = coefficients[i,2]/leading
    p = b-a*a/wp.float64(3.0)
    q = wp.float64(2.0)*a*a*a/wp.float64(27.0)-a*b/wp.float64(3.0)+c
    half_q = q/wp.float64(2.0)
    third_p = p/wp.float64(3.0)
    cube_p = third_p*third_p*third_p
    discriminant = half_q*half_q+cube_p
    multiple = wp.abs(discriminant) <= wp.float64(1e-14)*wp.max(half_q*half_q+wp.abs(cube_p),wp.float64(1e-30))
    value = -half_q-wp.copysign(wp.sqrt(wp.max(discriminant,wp.float64(0.0))),q)
    u = wp.sign(value)*wp.pow(wp.abs(value),wp.float64(1.0)/wp.float64(3.0))
    v = wp.float64(0.0)
    if wp.abs(u) > wp.float64(1e-15):
        v = -p/(wp.float64(3.0)*u)
    one_root = u+v-a/wp.float64(3.0)
    radius = wp.sqrt(wp.max(-third_p,wp.float64(1e-30)))
    angle = wp.acos(wp.clamp(-q/(wp.float64(2.0)*radius*radius*radius),wp.float64(-1.0),wp.float64(1.0)))/wp.float64(3.0)
    t0 = wp.mat33d(transforms[i,0,0],wp.float64(0.0),transforms[i,0,1],
                  wp.float64(0.0),transforms[i,0,0],transforms[i,0,2],
                  wp.float64(0.0),wp.float64(0.0),wp.float64(1.0))
    t1 = wp.mat33d(transforms[i,1,0],wp.float64(0.0),transforms[i,1,1],
                  wp.float64(0.0),transforms[i,1,0],transforms[i,1,2],
                  wp.float64(0.0),wp.float64(0.0),wp.float64(1.0))
    for slot in range(3):
        root = wp.float64(2.0)*radius*wp.cos(angle-wp.float64(slot)*wp.float64(2.0943951023931953))-a/wp.float64(3.0)
        if discriminant >= wp.float64(0.0):
            root = one_root
        if multiple and slot > 0:
            root = -u-a/wp.float64(3.0)
        for step in range(2):
            residual = ((root+a)*root+b)*root+c
            derivative = (wp.float64(3.0)*root+wp.float64(2.0)*a)*root+b
            if wp.abs(derivative) > wp.float64(1e-15):
                root -= residual/derivative
        if not ((discriminant < wp.float64(0.0) or slot == 0 or multiple) and wp.isfinite(root)):
            continue
        matrix = wp.mat33d()
        for j in range(3):
            for k in range(3):
                matrix[j,k] = vectors[i,3*j+k,0]+root*vectors[i,3*j+k,1]
        matrix = wp.transpose(t1)*matrix*t0
        norm2 = wp.float64(0.0)
        for j in range(3):
            for k in range(3):
                norm2 += matrix[j,k]*matrix[j,k]
        norm = wp.max(wp.sqrt(norm2),wp.float64(1e-12))
        for j in range(3):
            for k in range(3):
                models[i,slot,j,k] = matrix[j,k]/norm
        valid[i,slot] = True


def fundamental_seven_point(points0, points1, return_owners=False):
    from .nullspace import right_nullspace
    from ..two_view import _seven_point_coefficients

    n = len(points0)
    if points0.shape != (n,7,2) or points1.shape != points0.shape:
        raise ValueError('Expected matching batches of seven 2D points')
    if (not points0.is_cuda or points0.dtype != torch.float64
            or points1.device != points0.device or points1.dtype != points0.dtype):
        raise ValueError('Warp seven-point solver requires CUDA float64 inputs')
    if not n:
        models = points0.new_empty((0,3,3))
        owners = torch.empty(0,device=points0.device,dtype=torch.long)
        return (models,owners) if return_owners else models
    stream = wp.stream_from_torch(torch.cuda.current_stream(points0.device))
    design = points0.new_empty((n,7,9))
    transforms = points0.new_empty((n,2,3))
    wp.launch(_prepare,dim=n,inputs=[wp.from_torch(v) for v in
              (points0,points1,design,transforms)],stream=stream)
    vectors,rank = right_nullspace(design)
    coefficients = points0.new_empty((n,4))
    weak = torch.empty(n,device=points0.device,dtype=torch.bool)
    wp.launch(_coefficients,dim=n,inputs=[wp.from_torch(v) for v in
              (vectors,rank,coefficients,weak)],stream=stream)
    # Preserve the pre-existing weak-leading-coefficient chart policy.
    fallback = torch.where(weak)[0]
    if len(fallback):
        _,singular,vh = torch.linalg.svd(design[fallback],full_matrices=True)
        vectors[fallback] = vh[:,-2:].transpose(-1,-2)
        rank[fallback] = singular[:,-1] > 1e-10*singular[:,0]
        coefficients[fallback] = _seven_point_coefficients(vh[:,-2:].reshape(-1,2,3,3))
    models = points0.new_empty((n,3,3,3))
    valid = torch.empty((n,3),device=points0.device,dtype=torch.bool)
    wp.launch(_models,dim=n,inputs=[wp.from_torch(v) for v in
              (vectors,rank,coefficients,transforms,models,valid)],stream=stream)
    slots = torch.where(valid.flatten())[0]
    result = models.reshape(-1,3,3)[slots]
    return (result,slots//3) if return_owners else result
