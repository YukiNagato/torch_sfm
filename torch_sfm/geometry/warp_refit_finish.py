"""Fused float64 E/F/H refit projection, denormalization and validity.

The nine-column QR/SVD rank decision is retained. Small 3x3 projections use
one-sided Jacobi on the original matrix, without normal equations or retries.
"""
import torch
import warp as wp

wp.init()
wp.set_module_options({'enable_backward': False, 'fast_math': False,
                       'fuse_fp': False, 'max_unroll': 1})


@wp.func
def _project(model: wp.mat33d, essential: bool):
    scale = wp.float64(0.0)
    for i in range(3):
        for j in range(3):
            scale = wp.max(scale, wp.abs(model[i, j]))
    if scale == wp.float64(0.0):
        return wp.mat33d()
    a, v = wp.mat33d(), wp.mat33d()
    for i in range(3):
        v[i, i] = wp.float64(1.0)
        for j in range(3):
            a[i, j] = model[i, j] / scale
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
    for p in range(2):
        for q in range(p + 1, 3):
            if norms[q] > norms[p]:
                tmp = norms[p]; norms[p] = norms[q]; norms[q] = tmp
                for i in range(3):
                    tmp = a[i, p]; a[i, p] = a[i, q]; a[i, q] = tmp
                    tmp = v[i, p]; v[i, p] = v[i, q]; v[i, q] = tmp
    result = wp.mat33d()
    if not essential:
        # Retaining the two leading components avoids cancellation when the
        # input is close to rank one.
        for i in range(3):
            for j in range(3):
                result[i, j] = scale*(a[i, 0]*v[j, 0] + a[i, 1]*v[j, 1])
    else:
        s0, s1 = wp.sqrt(norms[0]), wp.sqrt(norms[1])
        u0 = wp.vec3d(a[0, 0], a[1, 0], a[2, 0])/s0
        u1 = wp.vec3d()
        if s1 > wp.float64(0.0):
            u1 = wp.vec3d(a[0, 1], a[1, 1], a[2, 1])/s1
        else:
            # A rank-one matrix has a non-unique nearest essential matrix.
            # Complete an orthonormal basis deterministically.
            axis = wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0))
            if wp.abs(u0[0]) > wp.float64(0.8):
                axis = wp.vec3d(wp.float64(0.0), wp.float64(1.0), wp.float64(0.0))
            u1 = wp.normalize(wp.cross(u0, axis))
        mean = scale*(s0+s1)/wp.float64(2.0)
        for i in range(3):
            for j in range(3):
                result[i, j] = mean*(u0[i]*v[j, 0] + u1[i]*v[j, 1])
    return result


@wp.kernel
def _finish(vh: wp.array3d(dtype=wp.float64), singular: wp.array2d(dtype=wp.float64),
            transform0: wp.array3d(dtype=wp.float64), transform1: wp.array3d(dtype=wp.float64),
            mask: wp.array2d(dtype=wp.bool), kind: int,
            output: wp.array3d(dtype=wp.float64), good: wp.array(dtype=wp.bool)):
    pair, lane = wp.tid()
    count = int(0)
    for j in range(lane, mask.shape[1], 32):
        count += int(mask[pair, j])
    count = wp.tile_extract(wp.tile_sum(wp.tile(count)), 0)
    if lane == 0:
        model, t0, t1 = wp.mat33d(), wp.mat33d(), wp.mat33d()
        finite_inputs = wp.bool(True)
        for i in range(3):
            for j in range(3):
                model[i, j] = vh[pair, 8, 3*i+j]
                t0[i, j] = transform0[pair, i, j]
                t1[i, j] = transform1[pair, i, j]
                finite_inputs = finite_inputs and wp.isfinite(model[i, j]) and wp.isfinite(t0[i, j]) and wp.isfinite(t1[i, j])
        valid = finite_inputs and count >= 8 and singular[pair, 7] > wp.float64(1e-10)*singular[pair, 0]
        if kind == 2:
            inverse = wp.mat33d()
            inverse[0, 0] = wp.float64(1.0)/t1[0, 0]
            inverse[1, 1] = wp.float64(1.0)/t1[1, 1]
            inverse[0, 2] = -t1[0, 2]/t1[0, 0]
            inverse[1, 2] = -t1[1, 2]/t1[1, 1]
            inverse[2, 2] = wp.float64(1.0)
            model = inverse*(model*t0)
            valid = valid and wp.abs(wp.determinant(model)) > wp.float64(1e-12)
        else:
            model = wp.transpose(t1)*_project(model, False)*t0
            if kind == 0:
                model = _project(model, True)
        for i in range(3):
            for j in range(3):
                valid = valid and wp.isfinite(model[i, j])
                output[pair, i, j] = model[i, j]
        good[pair] = valid


def finish(vh, singular, transform0, transform1, mask, kind):
    """Finish a 9x9 refit SVD in one launch; CUDA float64, forward only."""
    if kind not in ('E', 'F', 'H'):
        raise ValueError('Expected E, F or H')
    n = len(vh)
    if (vh.shape != (n, 9, 9) or singular.shape != (n, 9)
            or transform0.shape != (n, 3, 3) or transform1.shape != (n, 3, 3)
            or mask.ndim != 2 or mask.shape[0] != n or mask.dtype != torch.bool):
        raise ValueError('Invalid refit finish shapes')
    values = (vh, singular, transform0, transform1)
    if not vh.is_cuda or any(x.device != vh.device or x.dtype != torch.float64 for x in values) or mask.device != vh.device:
        raise ValueError('Warp refit finish requires CUDA float64 inputs')
    if torch.is_grad_enabled() and any(x.requires_grad for x in values):
        raise ValueError('Warp refit finish is forward only')
    output = vh.new_empty((n, 3, 3))
    good = torch.empty(n, device=vh.device, dtype=torch.bool)
    if n:
        wp.launch(_finish, dim=(n, 32), block_dim=32,
            inputs=[wp.from_torch(x) for x in (*values, mask)]+[{'E':0, 'F':1, 'H':2}[kind]],
            outputs=[wp.from_torch(output), wp.from_torch(good)],
            stream=wp.stream_from_torch(torch.cuda.current_stream(vh.device)))
    return output, good
