"""Small float64 decompositions callable from Warp kernels."""

import warp as wp

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.func
def smallest_right_vector4(matrix: wp.mat44d):
    """Unit right singular vector for the smallest singular value of a 4x4.

    Cyclic one-sided Jacobi rotates the original columns, without forming A.T A.
    Scaling avoids overflow for finite inputs. Repeated/zero singular values
    have no unique vector; any vector in their minimizing subspace is valid.
    """
    scale = wp.float64(0.0)
    for i in range(4):
        for j in range(4):
            scale = wp.max(scale, wp.abs(matrix[i, j]))
    a = wp.mat44d()
    if scale > wp.float64(0.0):
        for i in range(4):
            for j in range(4):
                a[i, j] = matrix[i, j] / scale
    v = wp.mat44d()
    for i in range(4):
        v[i, i] = wp.float64(1.0)
    for sweep in range(32):
        changed = wp.bool(False)
        for p in range(3):
            for q in range(p + 1, 4):
                alpha = wp.float64(0.0)
                beta = wp.float64(0.0)
                gamma = wp.float64(0.0)
                for i in range(4):
                    alpha += a[i, p] * a[i, p]
                    beta += a[i, q] * a[i, q]
                    gamma += a[i, p] * a[i, q]
                threshold = wp.float64(2.0e-15) * wp.sqrt(alpha) * wp.sqrt(beta)
                if wp.abs(gamma) > threshold:
                    # Stable Jacobi tangent; scale the pair before squaring.
                    delta = beta - alpha
                    twice_gamma = wp.float64(2.0) * gamma
                    pair_scale = wp.max(wp.abs(delta), wp.abs(twice_gamma))
                    x = delta / pair_scale
                    y = twice_gamma / pair_scale
                    sign = wp.float64(1.0)
                    if x < wp.float64(0.0):
                        sign = wp.float64(-1.0)
                    tangent = y / (x + sign * wp.sqrt(x * x + y * y))
                    cosine = wp.float64(1.0) / wp.sqrt(
                        wp.float64(1.0) + tangent * tangent
                    )
                    sine = cosine * tangent
                    for i in range(4):
                        ap, aq = a[i, p], a[i, q]
                        a[i, p] = cosine * ap - sine * aq
                        a[i, q] = sine * ap + cosine * aq
                        vp, vq = v[i, p], v[i, q]
                        v[i, p] = cosine * vp - sine * vq
                        v[i, q] = sine * vp + cosine * vq
                    changed = True
        if not changed:
            break
    best = wp.int32(0)
    minimum = wp.float64(wp.inf)
    for j in range(4):
        norm = wp.float64(0.0)
        for i in range(4):
            norm += a[i, j] * a[i, j]
        if norm < minimum:
            minimum = norm
            best = j
    result = wp.vec4d(v[0, best], v[1, best], v[2, best], v[3, best])
    return result / wp.length(result)
