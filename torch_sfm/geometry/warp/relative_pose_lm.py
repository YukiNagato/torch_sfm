"""Block-local relative-pose LM; one launch for all iterations.

Uses the existing pixel-Sampson objective, right SO(3) / S2 retraction,
float64 arithmetic and LM acceptance rules. No runtime reference solve.
"""

import warp as wp
import torch

from .relative_pose import _matrix, _skew, _tangent, Vec5

wp.set_module_options({'enable_backward': False, 'max_unroll': 8, 'fast_math': False})
Mat5 = wp.types.matrix(shape=(5, 5), dtype=wp.float64)
Coefficients = wp.types.matrix(shape=(6, 9), dtype=wp.float64)
Vec21 = wp.types.vector(length=21, dtype=wp.float64)
BLOCK = 64


@wp.func
def _coefficients(r: wp.mat33d, t: wp.vec3d, inv0: wp.mat33d, inv1t: wp.mat33d):
    result = Coefficients()
    essential = _skew(t) * r
    first, second = _tangent(t)
    for k in range(6):
        derivative = essential
        if k > 0 and k < 4:
            unit = wp.vec3d(wp.float64(0.0))
            unit[k - 1] = wp.float64(1.0)
            derivative = essential * _skew(unit)
        elif k == 4:
            derivative = _skew(first) * r
        elif k == 5:
            derivative = _skew(second) * r
        f = inv1t * derivative * inv0
        for row in range(3):
            for col in range(3):
                result[k, 3 * row + col] = f[row, col]
    return result


@wp.func
def _coefficient(c: Coefficients, k: int):
    return wp.mat33d(c[k, 0], c[k, 1], c[k, 2], c[k, 3], c[k, 4],
                     c[k, 5], c[k, 6], c[k, 7], c[k, 8])


@wp.func
def _terms(c: Coefficients, a: wp.vec3d, b: wp.vec3d):
    f = _coefficient(c, 0)
    l1, l0 = f * a, wp.transpose(f) * b
    numerator = wp.dot(b, l1)
    norm2 = l0[0] * l0[0] + l0[1] * l0[1] + l1[0] * l1[0] + l1[1] * l1[1]
    inv = wp.float64(1.0) / wp.sqrt(wp.max(norm2, wp.float64(1e-20)))
    j = Vec5()
    for k in range(5):
        df = _coefficient(c, k + 1)
        dl1, dl0 = df * a, wp.transpose(df) * b
        half = wp.float64(0.0)
        if norm2 > wp.float64(1e-20):
            half = l0[0] * dl0[0] + l0[1] * dl0[1] + l1[0] * dl1[0] + l1[1] * dl1[1]
        j[k] = wp.dot(b, dl1) * inv - numerator * inv * inv * inv * half
    return numerator * inv, j


@wp.func
def _residual(f: wp.mat33d, a: wp.vec3d, b: wp.vec3d):
    l1, l0 = f * a, wp.transpose(f) * b
    norm2 = l0[0] * l0[0] + l0[1] * l0[1] + l1[0] * l1[0] + l1[1] * l1[1]
    return wp.dot(b, l1) / wp.sqrt(wp.max(norm2, wp.float64(1e-20)))


@wp.func
def _solve(normal: Mat5, rhs: Vec5):
    # Partial-pivot elimination, matching the LU class of torch.solve_ex.
    a, b = normal, rhs
    good = wp.bool(True)
    for col in range(5):
        pivot = col
        largest = wp.abs(a[col, col])
        for row in range(col + 1, 5):
            if wp.abs(a[row, col]) > largest:
                largest = wp.abs(a[row, col])
                pivot = row
        if largest == wp.float64(0.0) or not wp.isfinite(largest):
            good = False
            break
        for k in range(col, 5):
            temp = a[col, k]
            a[col, k] = a[pivot, k]
            a[pivot, k] = temp
        temp = b[col]
        b[col] = b[pivot]
        b[pivot] = temp
        for row in range(col + 1, 5):
            factor = a[row, col] / a[col, col]
            for k in range(col + 1, 5):
                a[row, k] -= factor * a[col, k]
            b[row] -= factor * b[col]
    delta = Vec5()
    if good:
        for step in range(5):
            row = 4 - step
            value = b[row]
            for col in range(row + 1, 5):
                value -= a[row, col] * delta[col]
            delta[row] = value / a[row, row]
            good = good and wp.isfinite(delta[row])
    if not good:
        delta = Vec5()
    return delta, good


@wp.func
def _retract(r: wp.mat33d, t: wp.vec3d, d: Vec5):
    theta2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    a, b = wp.float64(1.0), wp.float64(0.5)
    if theta2 < wp.float64(1e-4):
        a = wp.float64(1.0) - theta2 / wp.float64(6.0) + theta2 * theta2 / wp.float64(120.0)
        b = wp.float64(0.5) - theta2 / wp.float64(24.0) + theta2 * theta2 / wp.float64(720.0)
    else:
        theta = wp.sqrt(theta2)
        a = wp.sin(theta) / theta
        half_sinc = wp.sin(theta * wp.float64(0.5)) / (theta * wp.float64(0.5))
        b = wp.float64(0.5) * half_sinc * half_sinc
    omega = _skew(wp.vec3d(d[0], d[1], d[2]))
    eye = wp.mat33d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0),
                   wp.float64(0.0), wp.float64(1.0), wp.float64(0.0),
                   wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
    proposed_r = r * (eye + a * omega + b * (omega * omega))
    first, second = _tangent(t)
    proposed_t = t + first * d[3] + second * d[4]
    proposed_t = proposed_t / wp.max(wp.length(proposed_t), wp.float64(1e-12))
    return proposed_r, proposed_t


@wp.func
def refine_pair(i: int, lane: int, rotations: wp.array3d(dtype=wp.float64), translations: wp.array2d(dtype=wp.float64),
            x0: wp.array3d(dtype=wp.float64), x1: wp.array3d(dtype=wp.float64),
            inverse0: wp.array3d(dtype=wp.float64), inverse1: wp.array3d(dtype=wp.float64),
            valid: wp.array2d(dtype=wp.bool), iterations: int):
    r = _matrix(rotations, i)
    t = wp.vec3d(translations[i, 0], translations[i, 1], translations[i, 2])
    inv0, inv1t = _matrix(inverse0, i), wp.transpose(_matrix(inverse1, i))
    support = int(0)
    for p in range(lane, x0.shape[1], BLOCK):
        support += int(valid[i, p])
    count = wp.tile_extract(wp.tile_sum(wp.tile(support)), 0)
    damping = wp.float64(1e-3)
    for iteration in range(iterations):
        if count < 5:
            break
        coefficients = _coefficients(r, t, inv0, inv1t)
        partial = Vec21()
        for p in range(lane, x0.shape[1], BLOCK):
            if valid[i, p]:
                a = wp.vec3d(x0[i, p, 0], x0[i, p, 1], x0[i, p, 2])
                b = wp.vec3d(x1[i, p, 0], x1[i, p, 1], x1[i, p, 2])
                residual, j = _terms(coefficients, a, b)
                index = int(0)
                for row in range(5):
                    for col in range(row, 5):
                        partial[index] += j[row] * j[col]
                        index += 1
                for row in range(5):
                    partial[15 + row] += j[row] * residual
                partial[20] += residual * residual
        totals = Vec21()
        for k in range(21):
            totals[k] = wp.tile_extract(wp.tile_sum(wp.tile(partial[k])), 0)
        normal, gradient = Mat5(), Vec5()
        index = int(0)
        for row in range(5):
            gradient[row] = totals[15 + row]
            for col in range(row, 5):
                value = totals[index]
                index += 1
                if row == col:
                    value += damping * wp.max(value, wp.float64(1e-8)) + wp.float64(1e-12)
                normal[row, col] = value
                normal[col, row] = value
        delta, usable = _solve(normal, -gradient)
        proposed_r, proposed_t = _retract(r, t, delta)
        f = inv1t * (_skew(proposed_t) * proposed_r) * inv0
        trial_sum = wp.float64(0.0)
        if usable:
            for p in range(lane, x0.shape[1], BLOCK):
                if valid[i, p]:
                    a = wp.vec3d(x0[i, p, 0], x0[i, p, 1], x0[i, p, 2])
                    b = wp.vec3d(x1[i, p, 0], x1[i, p, 1], x1[i, p, 2])
                    residual = _residual(f, a, b)
                    trial_sum += residual * residual
        proposed_cost = wp.tile_extract(wp.tile_sum(wp.tile(trial_sum)), 0)
        accept = usable and wp.isfinite(proposed_cost) and proposed_cost < totals[20]
        multiplier = wp.float64(10.0)
        if accept:
            r, t = proposed_r, proposed_t
            multiplier = wp.float64(1.0 / 3.0)
        damping = wp.clamp(damping * multiplier, wp.float64(1e-12), wp.float64(1e12))
        gradmax, norm2 = wp.float64(0.0), wp.float64(0.0)
        for k in range(5):
            gradmax = wp.max(gradmax, wp.abs(gradient[k]))
            norm2 += delta[k] * delta[k]
        small = wp.sqrt(norm2) < wp.float64(1e-9) or gradmax < wp.float64(1e-8)
        converged = small or (accept and totals[20] - proposed_cost < wp.float64(1e-9) * wp.max(totals[20], wp.float64(1.0)))
        if usable and converged:
            break
    if lane == 0:
        for row in range(3):
            translations[i, row] = t[row]
            for col in range(3):
                rotations[i, row, col] = r[row, col]


@wp.kernel
def _refine(rotations: wp.array3d(dtype=wp.float64), translations: wp.array2d(dtype=wp.float64),
            x0: wp.array3d(dtype=wp.float64), x1: wp.array3d(dtype=wp.float64),
            inverse0: wp.array3d(dtype=wp.float64), inverse1: wp.array3d(dtype=wp.float64),
            valid: wp.array2d(dtype=wp.bool), iterations: int):
    i, lane = wp.tid()
    refine_pair(i,lane,rotations,translations,x0,x1,inverse0,inverse1,valid,iterations)


def refine(rotations, translations, x0, x1, inverse0, inverse1, valid, iterations):
    """Refine independent pairs in place on the current Torch stream."""
    if len(rotations) and iterations > 0:
        wp.launch(_refine, dim=(len(rotations), BLOCK), block_dim=BLOCK,
            inputs=[wp.from_torch(v) for v in
                    (rotations, translations, x0, x1, inverse0, inverse1, valid)] + [iterations],
            stream=wp.stream_from_torch(torch.cuda.current_stream(rotations.device)))
    return rotations, translations
