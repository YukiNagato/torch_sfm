"""Float64 one-sided Jacobi decomposition for essential cheirality voting.

Rotate the original columns (never E.T E), then complete proper U/V frames.
Only the set of four poses is significant. The caller uses the reference SVD
for ambiguous cheirality votes, because singular-vector signs can reorder it.
"""
import torch
import warp as wp

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False,
                       'max_unroll': 1})


@wp.func
def _decompose_matrix(model: wp.mat33d):
    scale = wp.float64(0.0)
    finite = wp.bool(True)
    a, v = wp.mat33d(), wp.mat33d()
    for i in range(3):
        v[i, i] = wp.float64(1.0)
        for j in range(3):
            value = model[i,j]
            finite = finite and wp.isfinite(value)
            scale = wp.max(scale, wp.abs(value))
    if not finite or scale == wp.float64(0.0):
        return wp.mat33d(),wp.mat33d(),wp.vec3d(),False
    for i in range(3):
        for j in range(3):
            a[i, j] = model[i,j] / scale
    for sweep in range(24):
        changed = wp.bool(False)
        for p in range(2):
            for q in range(p + 1, 3):
                alpha, beta, gamma = wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
                for i in range(3):
                    alpha += a[i, p] * a[i, p]
                    beta += a[i, q] * a[i, q]
                    gamma += a[i, p] * a[i, q]
                if wp.abs(gamma) > wp.float64(2e-15) * wp.sqrt(alpha) * wp.sqrt(beta):
                    delta, twice = beta - alpha, wp.float64(2.0) * gamma
                    pair_scale = wp.max(wp.abs(delta), wp.abs(twice))
                    x, y = delta / pair_scale, twice / pair_scale
                    sign = wp.float64(1.0)
                    if x < wp.float64(0.0):
                        sign = wp.float64(-1.0)
                    tangent = y / (x + sign * wp.sqrt(x*x + y*y))
                    cosine = wp.float64(1.0) / wp.sqrt(wp.float64(1.0) + tangent*tangent)
                    sine = cosine * tangent
                    for i in range(3):
                        ap, aq, vp, vq = a[i, p], a[i, q], v[i, p], v[i, q]
                        a[i, p] = cosine*ap - sine*aq
                        a[i, q] = sine*ap + cosine*aq
                        v[i, p] = cosine*vp - sine*vq
                        v[i, q] = sine*vp + cosine*vq
                    changed = True
        if not changed:
            break
    norms = wp.vec3d()
    for j in range(3):
        for i in range(3):
            norms[j] += a[i, j]*a[i, j]
    # Sorting columns only; an arbitrary rotation inside the repeated leading
    # singular subspace does not change the set of essential poses.
    for p in range(2):
        for q in range(p + 1, 3):
            if norms[q] > norms[p]:
                tmp = norms[p]; norms[p] = norms[q]; norms[q] = tmp
                for i in range(3):
                    tmp = a[i, p]; a[i, p] = a[i, q]; a[i, q] = tmp
                    tmp = v[i, p]; v[i, p] = v[i, q]; v[i, q] = tmp
    good = norms[1] > wp.float64(1e-24)*norms[0]
    good = good and norms[1] - norms[2] > wp.float64(1e-8)*norms[0]
    if not good:
        return wp.mat33d(),wp.mat33d(),wp.vec3d(),False
    u0 = wp.vec3d(a[0,0],a[1,0],a[2,0]) / wp.sqrt(norms[0])
    u1 = wp.vec3d(a[0,1],a[1,1],a[2,1]) / wp.sqrt(norms[1])
    v0, v1 = wp.vec3d(v[0,0],v[1,0],v[2,0]), wp.vec3d(v[0,1],v[1,1],v[2,1])
    u2, v2 = wp.cross(u0,u1), wp.cross(v0,v1)
    # Residual guard catches a failure to converge as well as ill-conditioned
    # null spaces; reference SVD is retained for these exceptional matrices.
    av2 = wp.vec3d()
    for i in range(3):
        for j in range(3):
            av2[i] += model[i,j]/scale*v2[j]
    residual = av2 - wp.dot(av2,u2)*u2
    good = wp.abs(wp.dot(u0,u1)) < wp.float64(2e-12)
    good = good and wp.length(residual) < wp.float64(2e-12)*wp.sqrt(norms[0])
    r0,r1=wp.mat33d(),wp.mat33d()
    for i in range(3):
        for j in range(3):
            cross=u0[i]*v1[j]-u1[i]*v0[j]
            outer=u2[i]*v2[j]
            r0[i,j]=outer+cross
            r1[i,j]=outer-cross
    return r0,r1,u2,good


@wp.kernel
def _decompose(models: wp.array3d(dtype=wp.float64),
               rotations: wp.array4d(dtype=wp.float64),
               translation: wp.array2d(dtype=wp.float64), safe: wp.array(dtype=wp.bool)):
    m=wp.tid()
    e=wp.mat33d()
    for i in range(3):
        for j in range(3):e[i,j]=models[m,i,j]
    r0,r1,t,good=_decompose_matrix(e)
    safe[m]=good
    for i in range(3):
        translation[m,i]=t[i]
        for j in range(3):
            rotations[m,0,i,j]=r0[i,j]
            rotations[m,1,i,j]=r1[i,j]


def decompose(models):
    rotations = models.new_zeros((len(models),2,3,3))
    translation = models.new_zeros((len(models),3))
    safe = torch.empty(len(models), device=models.device, dtype=torch.bool)
    wp.launch(_decompose, dim=len(models), inputs=[wp.from_torch(models)],
              outputs=[wp.from_torch(x) for x in (rotations,translation,safe)],
              stream=wp.stream_from_torch(torch.cuda.current_stream(models.device)))
    return rotations, translation, safe
