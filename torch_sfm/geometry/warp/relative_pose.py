"""Fused pixel-Sampson normal equations; Torch owns all buffers and the LM solve."""

import torch
import warp as wp

wp.init()
wp.set_module_options({'enable_backward': False})

Vec5 = wp.types.vector(length=5, dtype=wp.float64)
BLOCK = 128


@wp.func
def _matrix(a: wp.array3d(dtype=wp.float64), i: int):
    return wp.mat33d(a[i, 0, 0], a[i, 0, 1], a[i, 0, 2],
                     a[i, 1, 0], a[i, 1, 1], a[i, 1, 2],
                     a[i, 2, 0], a[i, 2, 1], a[i, 2, 2])


@wp.func
def _coefficient(a: wp.array4d(dtype=wp.float64), i: int, k: int):
    return wp.mat33d(a[i, k, 0, 0], a[i, k, 0, 1], a[i, k, 0, 2],
                     a[i, k, 1, 0], a[i, k, 1, 1], a[i, k, 1, 2],
                     a[i, k, 2, 0], a[i, k, 2, 1], a[i, k, 2, 2])


@wp.func
def _skew(v: wp.vec3d):
    return wp.mat33d(wp.float64(0.0), -v[2], v[1], v[2], wp.float64(0.0), -v[0], -v[1], v[0], wp.float64(0.0))


@wp.func
def _tangent(t: wp.vec3d):
    axis = int(0)
    if wp.abs(t[1]) < wp.abs(t[axis]):
        axis = 1
    if wp.abs(t[2]) < wp.abs(t[axis]):
        axis = 2
    unit = wp.vec3d(wp.float64(0.0))
    unit[axis] = wp.float64(1.0)
    first = wp.cross(t, unit)
    first = first / wp.max(wp.length(first), wp.float64(1e-12))
    return first, wp.cross(first, t)


@wp.kernel
def _prepare(rotations: wp.array3d(dtype=wp.float64), translations: wp.array2d(dtype=wp.float64),
             inverse0: wp.array3d(dtype=wp.float64), inverse1: wp.array3d(dtype=wp.float64),
             jacobian: int, coefficients: wp.array4d(dtype=wp.float64)):
    i = wp.tid()
    r = _matrix(rotations, i)
    t = wp.vec3d(translations[i, 0], translations[i, 1], translations[i, 2])
    inv0, inv1t = _matrix(inverse0, i), wp.transpose(_matrix(inverse1, i))
    essential = _skew(t) * r
    first, second = _tangent(t)
    for k in range(6):
        if k == 0 or jacobian != 0:
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
                    coefficients[i, k, row, col] = f[row, col]


@wp.func
def _terms(coefficients: wp.array4d(dtype=wp.float64), x0: wp.array3d(dtype=wp.float64),
           x1: wp.array3d(dtype=wp.float64), i: int, p: int, jacobian: int):
    a = wp.vec3d(x0[i, p, 0], x0[i, p, 1], x0[i, p, 2])
    b = wp.vec3d(x1[i, p, 0], x1[i, p, 1], x1[i, p, 2])
    f = _coefficient(coefficients, i, 0)
    line1, line0 = f * a, wp.transpose(f) * b
    numerator = wp.dot(b, line1)
    norm2 = line0[0] * line0[0] + line0[1] * line0[1] + line1[0] * line1[0] + line1[1] * line1[1]
    inverse_norm = wp.float64(1.0) / wp.sqrt(wp.max(norm2, wp.float64(1e-20)))
    residual = numerator * inverse_norm
    j = Vec5(wp.float64(0.0))
    if jacobian != 0:
        for k in range(5):
            df = _coefficient(coefficients, i, k + 1)
            dl1, dl0 = df * a, wp.transpose(df) * b
            half_norm = wp.float64(0.0)
            if norm2 > wp.float64(1e-20):
                half_norm = line0[0] * dl0[0] + line0[1] * dl0[1] + line1[0] * dl1[0] + line1[1] * dl1[1]
            j[k] = wp.dot(b, dl1) * inverse_norm - numerator * inverse_norm * inverse_norm * inverse_norm * half_norm
    return residual, j


@wp.kernel
def _partials(coefficients: wp.array4d(dtype=wp.float64), x0: wp.array3d(dtype=wp.float64),
              x1: wp.array3d(dtype=wp.float64), valid: wp.array2d(dtype=wp.bool),
              jacobian: int, partials: wp.array3d(dtype=wp.float64)):
    i, chunk, lane = wp.tid()
    p = chunk * BLOCK + lane
    residual = wp.float64(0.0)
    j = Vec5(wp.float64(0.0))
    if p < x0.shape[1]:
        if valid[i, p]:
            residual, j = _terms(coefficients, x0, x1, i, p, jacobian)
    if jacobian != 0:
        index = int(0)
        for row in range(5):
            for col in range(row, 5):
                value = wp.tile(j[row] * j[col])
                wp.tile_store(partials[i, chunk], wp.tile_sum(value), offset=index)
                index += 1
        for row in range(5):
            value = wp.tile(j[row] * residual)
            wp.tile_store(partials[i, chunk], wp.tile_sum(value), offset=15 + row)
    value = wp.tile(residual * residual)
    wp.tile_store(partials[i, chunk], wp.tile_sum(value), offset=20)


@wp.kernel
def _finish(partials: wp.array3d(dtype=wp.float64), jacobian: int,
            normal: wp.array3d(dtype=wp.float64), gradient: wp.array2d(dtype=wp.float64),
            cost: wp.array(dtype=wp.float64)):
    i, index = wp.tid()
    if jacobian != 0 or index == 20:
        value = wp.float64(0.0)
        for chunk in range(partials.shape[1]):
            value += partials[i, chunk, index]
        if index < 15:
            position = int(0)
            for row in range(5):
                for col in range(row, 5):
                    if position == index:
                        normal[i, row, col] = value
                        normal[i, col, row] = value
                    position += 1
        elif index < 20:
            gradient[i, index - 15] = value
        else:
            cost[i] = value


@wp.kernel
def _evaluate(coefficients: wp.array4d(dtype=wp.float64), x0: wp.array3d(dtype=wp.float64),
              x1: wp.array3d(dtype=wp.float64), valid: wp.array2d(dtype=wp.bool),
              residual: wp.array2d(dtype=wp.float64), jacobian: wp.array3d(dtype=wp.float64)):
    i, p = wp.tid()
    r = wp.float64(0.0)
    j = Vec5(wp.float64(0.0))
    if valid[i, p]:
        r, j = _terms(coefficients, x0, x1, i, p, 1)
    residual[i, p] = r
    for k in range(5):
        jacobian[i, p, k] = j[k]


@wp.kernel
def _system(normal: wp.array3d(dtype=wp.float64), gradient: wp.array2d(dtype=wp.float64),
            damping: wp.array(dtype=wp.float64), system: wp.array3d(dtype=wp.float64),
            rhs: wp.array2d(dtype=wp.float64)):
    i = wp.tid()
    for row in range(5):
        rhs[i, row] = -gradient[i, row]
        for col in range(5):
            value = normal[i, row, col]
            if row == col:
                value += damping[i] * wp.max(value, wp.float64(1e-8)) + wp.float64(1e-12)
            system[i, row, col] = value


@wp.kernel
def _propose(rotations: wp.array3d(dtype=wp.float64), translations: wp.array2d(dtype=wp.float64),
             delta: wp.array2d(dtype=wp.float64), info: wp.array(dtype=wp.int32),
             gradient: wp.array2d(dtype=wp.float64), active: wp.array(dtype=wp.bool),
             proposed_r: wp.array3d(dtype=wp.float64), proposed_t: wp.array2d(dtype=wp.float64),
             usable: wp.array(dtype=wp.bool), small: wp.array(dtype=wp.bool)):
    i = wp.tid()
    good = active[i] and info[i] == 0
    for k in range(5):
        good = good and wp.isfinite(delta[i, k])
    d = Vec5(wp.float64(0.0))
    gradmax, norm2 = wp.float64(0.0), wp.float64(0.0)
    for k in range(5):
        if good:
            d[k] = delta[i, k]
        gradmax = wp.max(gradmax, wp.abs(gradient[i, k]))
        norm2 += d[k] * d[k]
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
    r = _matrix(rotations, i) * (eye + a * omega + b * (omega * omega))
    t = wp.vec3d(translations[i, 0], translations[i, 1], translations[i, 2])
    first, second = _tangent(t)
    t = t + first * d[3] + second * d[4]
    t = t / wp.max(wp.length(t), wp.float64(1e-12))
    for row in range(3):
        proposed_t[i, row] = t[row]
        for col in range(3):
            proposed_r[i, row, col] = r[row, col]
    usable[i] = good
    small[i] = wp.sqrt(norm2) < wp.float64(1e-9) or gradmax < wp.float64(1e-8)


@wp.kernel
def _accept(rotations: wp.array3d(dtype=wp.float64), translations: wp.array2d(dtype=wp.float64),
            proposed_r: wp.array3d(dtype=wp.float64), proposed_t: wp.array2d(dtype=wp.float64),
            cost: wp.array(dtype=wp.float64), proposed_cost: wp.array(dtype=wp.float64),
            usable: wp.array(dtype=wp.bool), small: wp.array(dtype=wp.bool),
            damping: wp.array(dtype=wp.float64), active: wp.array(dtype=wp.bool)):
    i = wp.tid()
    accept = usable[i] and wp.isfinite(proposed_cost[i]) and proposed_cost[i] < cost[i]
    scale = wp.float64(10.0)
    if accept:
        scale = wp.float64(1.0 / 3.0)
        for row in range(3):
            translations[i, row] = proposed_t[i, row]
            for col in range(3):
                rotations[i, row, col] = proposed_r[i, row, col]
    damping[i] = wp.clamp(damping[i] * scale, wp.float64(1e-12), wp.float64(1e12))
    converged = small[i] or (accept and cost[i] - proposed_cost[i] < wp.float64(1e-9) * wp.max(cost[i], wp.float64(1.0)))
    active[i] = active[i] and not (usable[i] and converged)


def _prepare_coefficients(rotations, translations, x0, inverse0, inverse1, jacobian):
    stream = wp.stream_from_torch(torch.cuda.current_stream(rotations.device))
    coefficients = x0.new_empty((len(rotations), 6, 3, 3))
    wp.launch(_prepare, dim=len(rotations), inputs=[wp.from_torch(v) for v in
        (rotations, translations, inverse0, inverse1)] + [int(jacobian)],
        outputs=[wp.from_torch(coefficients)], stream=stream)
    return coefficients, stream


def residual_jacobian(rotations, translations, x0, x1, inverse0, inverse1, valid):
    coefficients, stream = _prepare_coefficients(rotations, translations, x0, inverse0, inverse1, True)
    residual, jacobian = x0.new_empty(x0.shape[:2]), x0.new_empty((*x0.shape[:2], 5))
    wp.launch(_evaluate, dim=x0.shape[:2], inputs=[wp.from_torch(v) for v in (coefficients, x0, x1, valid)],
        outputs=[wp.from_torch(residual), wp.from_torch(jacobian)], stream=stream)
    return residual, jacobian


def normal_equations(rotations, translations, x0, x1, inverse0, inverse1, valid, jacobian=True):
    coefficients, stream = _prepare_coefficients(rotations, translations, x0, inverse0, inverse1, jacobian)
    pairs, points = x0.shape[:2]
    chunks = (points + BLOCK - 1) // BLOCK
    partials = x0.new_empty((pairs, chunks, 21))
    normal, gradient, cost = x0.new_empty((pairs, 5, 5)), x0.new_empty((pairs, 5)), x0.new_empty((pairs,))
    wp.launch(_partials, dim=(pairs, chunks, BLOCK), block_dim=BLOCK,
        inputs=[wp.from_torch(v) for v in (coefficients, x0, x1, valid)] + [int(jacobian)],
        outputs=[wp.from_torch(partials)], stream=stream)
    wp.launch(_finish, dim=(pairs, 21), inputs=[wp.from_torch(partials), int(jacobian)],
        outputs=[wp.from_torch(v) for v in (normal, gradient, cost)], stream=stream)
    return (normal, gradient, cost) if jacobian else cost


def refine(rotations, translations, x0, x1, inverse0, inverse1, valid, iterations):
    """Torch solve_ex between fused steps; preserve the reference LM decisions."""
    pairs = len(rotations)
    damping = x0.new_full((pairs,), 1e-3)
    active = valid.sum(-1) >= 5
    system, rhs = x0.new_empty((pairs, 5, 5)), x0.new_empty((pairs, 5))
    proposed_r, proposed_t = torch.empty_like(rotations), torch.empty_like(translations)
    usable, small = torch.empty_like(active), torch.empty_like(active)
    delta = x0.new_empty((pairs, 5))
    info = torch.empty((pairs,), device=x0.device, dtype=torch.int32)
    chunks = (x0.shape[1] + BLOCK - 1) // BLOCK
    coefficients = x0.new_empty((pairs, 6, 3, 3))
    partials = x0.new_empty((pairs, chunks, 21))
    normal, gradient = x0.new_empty((pairs, 5, 5)), x0.new_empty((pairs, 5))
    cost, proposed_cost = x0.new_empty((pairs,)), x0.new_empty((pairs,))
    # The numerical loop reuses both Torch workspaces and their zero-copy Warp
    # views. Reconstructing these descriptors per step would add host overhead.
    values = {name: wp.from_torch(value) for name, value in dict(
        rotations=rotations, translations=translations, x0=x0, x1=x1,
        inverse0=inverse0, inverse1=inverse1, valid=valid, damping=damping, active=active,
        system=system, rhs=rhs, proposed_r=proposed_r, proposed_t=proposed_t,
        usable=usable, small=small, delta=delta, info=info, coefficients=coefficients,
        partials=partials, normal=normal, gradient=gradient, cost=cost, proposed_cost=proposed_cost).items()}

    def arrays(*names):
        return [values[name] for name in names]

    prepare = arrays('rotations', 'translations', 'inverse0', 'inverse1') + [1]
    prepare_proposed = arrays('proposed_r', 'proposed_t', 'inverse0', 'inverse1') + [0]
    partial_inputs = arrays('coefficients', 'x0', 'x1', 'valid')
    coefficient_out, partial_out = arrays('coefficients'), arrays('partials')
    normal_out, cost_out = arrays('normal', 'gradient', 'cost'), arrays('normal', 'gradient', 'proposed_cost')
    system_inputs, system_out = arrays('normal', 'gradient', 'damping'), arrays('system', 'rhs')
    propose_inputs = arrays('rotations', 'translations', 'delta', 'info', 'gradient', 'active')
    propose_out = arrays('proposed_r', 'proposed_t', 'usable', 'small')
    accept_inputs = arrays('rotations', 'translations', 'proposed_r', 'proposed_t', 'cost',
                          'proposed_cost', 'usable', 'small', 'damping', 'active')
    stream = wp.stream_from_torch(torch.cuda.current_stream(rotations.device))
    for _ in range(iterations):
        wp.launch(_prepare, dim=pairs, inputs=prepare, outputs=coefficient_out, stream=stream)
        wp.launch(_partials, dim=(pairs, chunks, BLOCK), block_dim=BLOCK,
            inputs=partial_inputs + [1], outputs=partial_out, stream=stream)
        wp.launch(_finish, dim=(pairs, 21), inputs=partial_out + [1], outputs=normal_out, stream=stream)
        wp.launch(_system, dim=pairs, inputs=system_inputs, outputs=system_out, stream=stream)
        torch.linalg.solve_ex(system, rhs, out=(delta, info))
        wp.launch(_propose, dim=pairs, inputs=propose_inputs, outputs=propose_out, stream=stream)
        wp.launch(_prepare, dim=pairs, inputs=prepare_proposed, outputs=coefficient_out, stream=stream)
        wp.launch(_partials, dim=(pairs, chunks, BLOCK), block_dim=BLOCK,
            inputs=partial_inputs + [0], outputs=partial_out, stream=stream)
        wp.launch(_finish, dim=(pairs, 21), inputs=partial_out + [0], outputs=cost_out, stream=stream)
        wp.launch(_accept, dim=pairs, inputs=accept_inputs, stream=stream)
    return rotations, translations
