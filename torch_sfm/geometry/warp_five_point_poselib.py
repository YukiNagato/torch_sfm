# BSD 3-Clause License
# Copyright (c) 2020, Viktor Larsson. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""PoseLib five-point minimal solver, entirely on CUDA with bounded work.

Full-pivot Householder QR, trace/determinant constraints, LU, degree-ten Sturm
roots, and PoseLib's 2x2 / column-pivoted 3x2 QR recovery. Optional COLMAP-style
all-five cheirality filtering. Failed samples/roots are dropped, never replayed
through Torch linalg or a CPU solver. Not bitwise equivalent to Eigen.
Generated fixed polynomial expressions follow PoseLib/solvers/relpose_5pt.cc.
"""
import torch
import warp as wp
from .warp_essential_pose import _decompose_matrix

wp.init()
wp.set_module_options({'enable_backward': False, 'fast_math': False,
                       'fuse_fp': False, 'max_unroll': 1})
M95 = wp.types.matrix(shape=(9,5), dtype=wp.float64)
M49 = wp.types.matrix(shape=(4,9), dtype=wp.float64)
M910 = wp.types.matrix(shape=(9,10), dtype=wp.float64)
M1020 = wp.types.matrix(shape=(10,20), dtype=wp.float64)
M313 = wp.types.matrix(shape=(3,13), dtype=wp.float64)
M311 = wp.types.matrix(shape=(3,11), dtype=wp.float64)
V4 = wp.types.vector(length=4, dtype=wp.float64)
V5 = wp.types.vector(length=5, dtype=wp.float64)
I5 = wp.types.vector(length=5, dtype=wp.int32)
V9 = wp.types.vector(length=9, dtype=wp.float64)
V10 = wp.types.vector(length=10, dtype=wp.float64)
V11 = wp.types.vector(length=11, dtype=wp.float64)
V20 = wp.types.vector(length=20, dtype=wp.float64)
V30 = wp.types.vector(length=30, dtype=wp.float64)
I11 = wp.types.vector(length=11, dtype=wp.int32)

@wp.func
def _o1(a: V4, b: V4):
    c = V10()
    c[0] = a[0] * b[0]
    c[1] = a[0] * b[1] + a[1] * b[0]
    c[2] = a[0] * b[2] + a[2] * b[0]
    c[3] = a[0] * b[3] + a[3] * b[0]
    c[4] = a[1] * b[1]
    c[5] = a[1] * b[2] + a[2] * b[1]
    c[6] = a[1] * b[3] + a[3] * b[1]
    c[7] = a[2] * b[2]
    c[8] = a[2] * b[3] + a[3] * b[2]
    c[9] = a[3] * b[3]
    return c


@wp.func
def _o2(a: V10, b: V4):
    c = V20()
    c[0] = a[0] * b[0]
    c[1] = a[4] * b[1]
    c[2] = a[0] * b[1] + a[1] * b[0]
    c[3] = a[1] * b[1] + a[4] * b[0]
    c[4] = a[0] * b[2] + a[2] * b[0]
    c[5] = a[0] * b[3] + a[3] * b[0]
    c[6] = a[4] * b[2] + a[5] * b[1]
    c[7] = a[4] * b[3] + a[6] * b[1]
    c[8] = a[1] * b[2] + a[2] * b[1] + a[5] * b[0]
    c[9] = a[1] * b[3] + a[3] * b[1] + a[6] * b[0]
    c[10] = a[2] * b[2] + a[7] * b[0]
    c[11] = a[2] * b[3] + a[3] * b[2] + a[8] * b[0]
    c[12] = a[3] * b[3] + a[9] * b[0]
    c[13] = a[5] * b[2] + a[7] * b[1]
    c[14] = a[5] * b[3] + a[6] * b[2] + a[8] * b[1]
    c[15] = a[6] * b[3] + a[9] * b[1]
    c[16] = a[7] * b[2]
    c[17] = a[7] * b[3] + a[8] * b[2]
    c[18] = a[8] * b[3] + a[9] * b[2]
    c[19] = a[9] * b[3]
    return c


@wp.func
def _entry(n: M49, i: int, j: int):
    return V4(n[0,3*j+i],n[1,3*j+i],n[2,3*j+i],n[3,3*j+i])


@wp.func
def _constraints(n: M49):
    c = M1020()
    d = _o1(_entry(n,0,1),_entry(n,1,2))-_o1(_entry(n,0,2),_entry(n,1,1))
    row = _o2(d,_entry(n,2,0))
    d = _o1(_entry(n,0,2),_entry(n,1,0))-_o1(_entry(n,0,0),_entry(n,1,2))
    row += _o2(d,_entry(n,2,1))
    d = _o1(_entry(n,0,0),_entry(n,1,1))-_o1(_entry(n,0,1),_entry(n,1,0))
    row += _o2(d,_entry(n,2,2))
    for k in range(20):
        c[9,k] = row[k]
    eet = M910()
    for i in range(3):
        for j in range(i,3):
            d = _o1(_entry(n,i,0),_entry(n,j,0))
            d += _o1(_entry(n,i,1),_entry(n,j,1))
            d += _o1(_entry(n,i,2),_entry(n,j,2))
            for k in range(10):
                eet[3*i+j,k] = d[k]
                eet[3*j+i,k] = d[k]
    for k in range(10):
        trace = wp.float64(.5)*(eet[0,k]+eet[4,k]+eet[8,k])
        eet[0,k] -= trace
        eet[4,k] -= trace
        eet[8,k] -= trace
    for i in range(3):
        for j in range(3):
            row = V20()
            for p in range(3):
                d = V10()
                for k in range(10):
                    d[k] = eet[3*i+p,k]
                row += _o2(d,_entry(n,p,j))
            for k in range(20):
                c[3*i+j,k] = row[k]
    return c


@wp.kernel
def _nullspace(a: wp.array3d(dtype=wp.float64), b: wp.array3d(dtype=wp.float64),
               basis: wp.array3d(dtype=wp.float64), status: wp.array(dtype=wp.int32)):
    sample = wp.tid()
    status[sample] = 1
    q = M95()
    finite = wp.bool(True)
    for k in range(5):
        x = wp.normalize(wp.vec3d(a[sample,k,0],a[sample,k,1],wp.float64(1.0)))
        y = wp.normalize(wp.vec3d(b[sample,k,0],b[sample,k,1],wp.float64(1.0)))
        for i in range(3):
            for j in range(3):
                q[3*i+j,k] = x[i]*y[j]
                finite = finite and wp.isfinite(q[3*i+j,k])
    if not finite:
        return
    tau = V5()
    swaps = I5()
    biggest = wp.float64(0.0)
    for k in range(5):
        pivot = wp.float64(0.0)
        row, col = int(k), int(k)
        # Column-major tie order, matching Eigen's maximum-coefficient scan.
        for j in range(k, 5):
            for i in range(k, 9):
                if wp.abs(q[i, j]) > pivot:
                    pivot = wp.abs(q[i, j]); row = i; col = j
        if k == 0:
            biggest = pivot
        if not wp.isfinite(pivot) or pivot <= wp.float64(1.1102230246251565e-15)*biggest:
            return
        swaps[k] = row
        for j in range(k, 5):
            tmp = q[k, j]; q[k, j] = q[row, j]; q[row, j] = tmp
        for i in range(9):
            tmp = q[i, k]; q[i, k] = q[i, col]; q[i, col] = tmp
        tail = wp.float64(0.0)
        for i in range(k+1, 9):
            tail += q[i, k]*q[i, k]
        alpha = q[k, k]
        if tail > wp.float64(0.0):
            beta = wp.sqrt(alpha*alpha+tail)
            if alpha >= wp.float64(0.0):
                beta = -beta
            tau[k] = (beta-alpha)/beta
            for i in range(k+1, 9):
                q[i, k] /= alpha-beta
            q[k, k] = beta
            for j in range(k+1, 5):
                dot = q[k, j]
                for i in range(k+1, 9):
                    dot += q[i, k]*q[i, j]
                dot *= tau[k]
                q[k, j] -= dot
                for i in range(k+1, 9):
                    q[i, j] -= q[i, k]*dot
    for slot in range(4):
        v = V9()
        v[5+slot] = wp.float64(1.0)
        for step in range(5):
            k = 4-step
            dot = v[k]
            for i in range(k+1, 9):
                dot += q[i, k]*v[i]
            dot *= tau[k]
            v[k] -= dot
            for i in range(k+1, 9):
                v[i] -= q[i, k]*dot
            row = swaps[k]
            tmp = v[k]; v[k] = v[row]; v[row] = tmp
        for j in range(9):
            basis[sample, slot, j] = v[j]
    status[sample] = 0


@wp.func
def _polyval(p: V11, x: wp.float64):
    value = wp.float64(1.0)
    for step in range(10):
        value = x*value+p[9-step]
    return value


@wp.func
def _derivative(p: V11, x: wp.float64):
    # PoseLib evaluates the normalized monic derivative, then multiplies by 10.
    value = wp.float64(1.0)
    for step in range(9):
        i = 8-step
        value = x*value+p[i+1]*(wp.float64(i+1)/wp.float64(10.0))
    return wp.float64(10.0)*value


@wp.func
def _signchanges(s: V30, x: wp.float64):
    f2 = s[29]
    f1 = s[27]+x*s[28]
    count = int(0)
    if (f1 < wp.float64(0.0)) != (f2 < wp.float64(0.0)):
        count += 1
    good = wp.bool(not wp.isnan(f1) and not wp.isnan(f2))
    for step in range(9):
        i = 8-step
        f0 = (s[3*i]+x*s[3*i+1])*f1+s[3*i+2]*f2
        good = good and not wp.isnan(f0)
        if (f0 < wp.float64(0.0)) != (f1 < wp.float64(0.0)):
            count += 1
        f2 = f1; f1 = f0
    if not good:
        count = -1
    return count


@wp.func
def _refine_root(p: V11, left: wp.float64, right: wp.float64):
    a, b = left, right
    fa, fb = _polyval(p, a), _polyval(p, b)
    good = wp.bool((fa < wp.float64(0.0)) != (fb < wp.float64(0.0)))
    if good:
        for iteration in range(30):
            if wp.abs(a-b) < wp.float64(1e-3):
                break
            c = (a+b)*wp.float64(.5)
            fc = _polyval(p, c)
            radicand = fc*fc-fa*fb
            if radicand <= wp.float64(0.0) or not wp.isfinite(radicand):
                # No alternate solver; failed bracketing invalidates this root.
                good = False
                break
            s = wp.sqrt(radicand)
            d = c+(c-a)*fc/s
            if fa < fb:
                d = c+(a-c)*fc/s
            fd = _polyval(p, d)
            if (fd >= wp.float64(0.0) and fc < wp.float64(0.0)) or (fd < wp.float64(0.0) and fc > wp.float64(0.0)):
                a = c; fa = fc; b = d; fb = fd
            elif (fd >= wp.float64(0.0) and fa < wp.float64(0.0)) or (fd < wp.float64(0.0) and fa > wp.float64(0.0)):
                b = d; fb = fd
            else:
                a = d; fa = fd
    x = (a+b)*wp.float64(.5)
    if good:
        for iteration in range(10):
            fx = _polyval(p, x)
            if wp.abs(fx) < wp.float64(1e-12):
                break
            derivative = _derivative(p, x)
            if derivative == wp.float64(0.0) or not wp.isfinite(derivative):
                good = False
                break
            dx = fx/derivative
            x -= dx
            if wp.abs(dx) < wp.float64(1e-12):
                break
    return x, good and wp.isfinite(x)


@wp.func
def _sturm(p: V11):
    f, s = M311(), V30()
    for j in range(11):
        f[0, j] = p[j]
    for j in range(10):
        f[1, j] = p[j+1]*(wp.float64(j+1)/wp.float64(10.0))
    good = wp.bool(True)
    i1, i2, i3 = int(0), int(1), int(2)
    for i in range(9):
        q1 = f[i1, 10-i]*f[i2, 9-i]
        q0 = f[i1, 9-i]*f[i2, 9-i]-f[i1, 10-i]*f[i2, 8-i]
        f[i3, 0] = f[i1, 0]-q0*f[i2, 0]
        for j in range(1, 9-i):
            f[i3, j] = f[i1, j]-q1*f[i2, j-1]-q0*f[i2, j]
        c = -wp.abs(f[i3, 8-i])
        if c == wp.float64(0.0) or not wp.isfinite(c):
            good = False
            break
        inv = wp.float64(1.0)/c
        for j in range(9-i):
            f[i3, j] *= inv
        tmp = i1; i1 = i2; i2 = i3; i3 = tmp
        s[3*i] = q0; s[3*i+1] = q1; s[3*i+2] = c
    s[27] = f[i1, 0]; s[28] = f[i1, 1]; s[29] = f[i2, 0]
    bound = wp.float64(0.0)
    for j in range(10):
        bound = wp.max(bound, wp.abs(p[j]))
    bound += wp.float64(1.0)
    sa, sb = _signchanges(s, -bound), _signchanges(s, bound)
    good = good and sa >= 0 and sb >= 0 and sa >= sb
    roots, left, right = V10(), V11(), V11()
    signs_a, signs_b, depths = I11(), I11(), I11()
    count, size, processed = int(0), int(0), int(0)
    if good and sa > sb:
        left[0] = -bound; right[0] = bound
        signs_a[0] = sa; signs_b[0] = sb; size = 1
    # At most 10 nonempty disjoint intervals are live. Depth limit matches
    # PoseLib's 300; the work cap also bounds pathological sign inconsistencies.
    while size > 0 and processed < 9000:
        size -= 1; processed += 1
        a, b = left[size], right[size]
        sa, sb, depth = signs_a[size], signs_b[size], depths[size]
        if depth > 300:
            good = False
            continue
        root = wp.float64(0.0)
        found = wp.bool(False)
        if b-a < wp.float64(1e-12):
            root = b; found = True
        elif sa-sb == 1:
            root, found = _refine_root(p, a, b)
            if not found:
                good = False
        elif sa-sb > 1:
            c = (a+b)*wp.float64(.5)
            sc = _signchanges(s, c)
            if sc < sb or sc > sa or size > 8:
                good = False
                continue
            # Push right before left, preserving ascending root traversal.
            if sc > sb:
                left[size] = c; right[size] = b; depths[size] = depth+1
                signs_a[size] = sc; signs_b[size] = sb; size += 1
            if sa > sc:
                left[size] = a; right[size] = c; depths[size] = depth+1
                signs_a[size] = sa; signs_b[size] = sc; size += 1
        if found and count < 10:
            roots[count] = root; count += 1
    if size > 0:
        good = False
    return roots, count, good


@wp.func
def _determinant(a: M313):
    p = V11()
    p[0] = (a[0, 12] * a[1, 3] * a[2, 7] - a[0, 12] * a[1, 7] * a[2, 3] - a[0, 3] * a[2, 7] * a[1, 12] +
           a[0, 7] * a[2, 3] * a[1, 12] + a[0, 3] * a[1, 7] * a[2, 12] - a[0, 7] * a[1, 3] * a[2, 12])
    p[1] = (a[0, 11] * a[1, 3] * a[2, 7] - a[0, 11] * a[1, 7] * a[2, 3] + a[0, 12] * a[1, 2] * a[2, 7] +
           a[0, 12] * a[1, 3] * a[2, 6] - a[0, 12] * a[1, 6] * a[2, 3] - a[0, 12] * a[1, 7] * a[2, 2] -
           a[0, 2] * a[2, 7] * a[1, 12] - a[0, 3] * a[2, 6] * a[1, 12] - a[0, 3] * a[2, 7] * a[1, 11] +
           a[0, 6] * a[2, 3] * a[1, 12] + a[0, 7] * a[2, 2] * a[1, 12] + a[0, 7] * a[2, 3] * a[1, 11] +
           a[0, 2] * a[1, 7] * a[2, 12] + a[0, 3] * a[1, 6] * a[2, 12] + a[0, 3] * a[1, 7] * a[2, 11] -
           a[0, 6] * a[1, 3] * a[2, 12] - a[0, 7] * a[1, 2] * a[2, 12] - a[0, 7] * a[1, 3] * a[2, 11])
    p[2] = (a[0, 10] * a[1, 3] * a[2, 7] - a[0, 10] * a[1, 7] * a[2, 3] + a[0, 11] * a[1, 2] * a[2, 7] +
           a[0, 11] * a[1, 3] * a[2, 6] - a[0, 11] * a[1, 6] * a[2, 3] - a[0, 11] * a[1, 7] * a[2, 2] +
           a[1, 1] * a[0, 12] * a[2, 7] + a[0, 12] * a[1, 2] * a[2, 6] + a[0, 12] * a[1, 3] * a[2, 5] -
           a[0, 12] * a[1, 5] * a[2, 3] - a[0, 12] * a[1, 6] * a[2, 2] - a[0, 12] * a[1, 7] * a[2, 1] -
           a[0, 1] * a[2, 7] * a[1, 12] - a[0, 2] * a[2, 6] * a[1, 12] - a[0, 2] * a[2, 7] * a[1, 11] -
           a[0, 3] * a[2, 5] * a[1, 12] - a[0, 3] * a[2, 6] * a[1, 11] - a[0, 3] * a[2, 7] * a[1, 10] +
           a[0, 5] * a[2, 3] * a[1, 12] + a[0, 6] * a[2, 2] * a[1, 12] + a[0, 6] * a[2, 3] * a[1, 11] +
           a[0, 7] * a[2, 1] * a[1, 12] + a[0, 7] * a[2, 2] * a[1, 11] + a[0, 7] * a[2, 3] * a[1, 10] +
           a[0, 1] * a[1, 7] * a[2, 12] + a[0, 2] * a[1, 6] * a[2, 12] + a[0, 2] * a[1, 7] * a[2, 11] +
           a[0, 3] * a[1, 5] * a[2, 12] + a[0, 3] * a[1, 6] * a[2, 11] + a[0, 3] * a[1, 7] * a[2, 10] -
           a[0, 5] * a[1, 3] * a[2, 12] - a[0, 6] * a[1, 2] * a[2, 12] - a[0, 6] * a[1, 3] * a[2, 11] -
           a[0, 7] * a[1, 1] * a[2, 12] - a[0, 7] * a[1, 2] * a[2, 11] - a[0, 7] * a[1, 3] * a[2, 10])
    p[3] = (a[0, 3] * a[1, 7] * a[2, 9] - a[0, 3] * a[1, 9] * a[2, 7] - a[0, 7] * a[1, 3] * a[2, 9] +
           a[0, 7] * a[1, 9] * a[2, 3] + a[0, 9] * a[1, 3] * a[2, 7] - a[0, 9] * a[1, 7] * a[2, 3] +
           a[0, 10] * a[1, 2] * a[2, 7] + a[0, 10] * a[1, 3] * a[2, 6] - a[0, 10] * a[1, 6] * a[2, 3] -
           a[0, 10] * a[1, 7] * a[2, 2] + a[1, 0] * a[0, 12] * a[2, 7] + a[0, 11] * a[1, 1] * a[2, 7] +
           a[0, 11] * a[1, 2] * a[2, 6] + a[0, 11] * a[1, 3] * a[2, 5] - a[0, 11] * a[1, 5] * a[2, 3] -
           a[0, 11] * a[1, 6] * a[2, 2] - a[0, 11] * a[1, 7] * a[2, 1] + a[1, 1] * a[0, 12] * a[2, 6] +
           a[0, 12] * a[1, 2] * a[2, 5] + a[0, 12] * a[1, 3] * a[2, 4] - a[0, 12] * a[1, 4] * a[2, 3] -
           a[0, 12] * a[1, 5] * a[2, 2] - a[0, 12] * a[1, 6] * a[2, 1] - a[0, 12] * a[1, 7] * a[2, 0] -
           a[0, 0] * a[2, 7] * a[1, 12] - a[0, 1] * a[2, 6] * a[1, 12] - a[0, 1] * a[2, 7] * a[1, 11] -
           a[0, 2] * a[2, 5] * a[1, 12] - a[0, 2] * a[2, 6] * a[1, 11] - a[0, 2] * a[2, 7] * a[1, 10] -
           a[0, 3] * a[2, 4] * a[1, 12] - a[0, 3] * a[2, 5] * a[1, 11] - a[0, 3] * a[2, 6] * a[1, 10] +
           a[0, 4] * a[2, 3] * a[1, 12] + a[0, 5] * a[2, 2] * a[1, 12] + a[0, 5] * a[2, 3] * a[1, 11] +
           a[0, 6] * a[2, 1] * a[1, 12] + a[0, 6] * a[2, 2] * a[1, 11] + a[0, 6] * a[2, 3] * a[1, 10] +
           a[0, 7] * a[2, 0] * a[1, 12] + a[0, 7] * a[2, 1] * a[1, 11] + a[0, 7] * a[2, 2] * a[1, 10] +
           a[0, 0] * a[1, 7] * a[2, 12] + a[0, 1] * a[1, 6] * a[2, 12] + a[0, 1] * a[1, 7] * a[2, 11] +
           a[0, 2] * a[1, 5] * a[2, 12] + a[0, 2] * a[1, 6] * a[2, 11] + a[0, 2] * a[1, 7] * a[2, 10] +
           a[0, 3] * a[1, 4] * a[2, 12] + a[0, 3] * a[1, 5] * a[2, 11] + a[0, 3] * a[1, 6] * a[2, 10] -
           a[0, 4] * a[1, 3] * a[2, 12] - a[0, 5] * a[1, 2] * a[2, 12] - a[0, 5] * a[1, 3] * a[2, 11] -
           a[0, 6] * a[1, 1] * a[2, 12] - a[0, 6] * a[1, 2] * a[2, 11] - a[0, 6] * a[1, 3] * a[2, 10] -
           a[0, 7] * a[1, 0] * a[2, 12] - a[0, 7] * a[1, 1] * a[2, 11] - a[0, 7] * a[1, 2] * a[2, 10])
    p[4] = (a[0, 2] * a[1, 7] * a[2, 9] - a[0, 2] * a[1, 9] * a[2, 7] + a[0, 3] * a[1, 6] * a[2, 9] +
           a[0, 3] * a[1, 7] * a[2, 8] - a[0, 3] * a[1, 8] * a[2, 7] - a[0, 3] * a[1, 9] * a[2, 6] -
           a[0, 6] * a[1, 3] * a[2, 9] + a[0, 6] * a[1, 9] * a[2, 3] - a[0, 7] * a[1, 2] * a[2, 9] -
           a[0, 7] * a[1, 3] * a[2, 8] + a[0, 7] * a[1, 8] * a[2, 3] + a[0, 7] * a[1, 9] * a[2, 2] +
           a[0, 8] * a[1, 3] * a[2, 7] - a[0, 8] * a[1, 7] * a[2, 3] + a[0, 9] * a[1, 2] * a[2, 7] +
           a[0, 9] * a[1, 3] * a[2, 6] - a[0, 9] * a[1, 6] * a[2, 3] - a[0, 9] * a[1, 7] * a[2, 2] +
           a[0, 10] * a[1, 1] * a[2, 7] + a[0, 10] * a[1, 2] * a[2, 6] + a[0, 10] * a[1, 3] * a[2, 5] -
           a[0, 10] * a[1, 5] * a[2, 3] - a[0, 10] * a[1, 6] * a[2, 2] - a[0, 10] * a[1, 7] * a[2, 1] +
           a[1, 0] * a[0, 11] * a[2, 7] + a[1, 0] * a[0, 12] * a[2, 6] + a[0, 11] * a[1, 1] * a[2, 6] +
           a[0, 11] * a[1, 2] * a[2, 5] + a[0, 11] * a[1, 3] * a[2, 4] - a[0, 11] * a[1, 4] * a[2, 3] -
           a[0, 11] * a[1, 5] * a[2, 2] - a[0, 11] * a[1, 6] * a[2, 1] - a[0, 11] * a[1, 7] * a[2, 0] +
           a[1, 1] * a[0, 12] * a[2, 5] + a[0, 12] * a[1, 2] * a[2, 4] - a[0, 12] * a[1, 4] * a[2, 2] -
           a[0, 12] * a[1, 5] * a[2, 1] - a[0, 12] * a[1, 6] * a[2, 0] - a[0, 0] * a[2, 6] * a[1, 12] -
           a[0, 0] * a[2, 7] * a[1, 11] - a[0, 1] * a[2, 5] * a[1, 12] - a[0, 1] * a[2, 6] * a[1, 11] -
           a[0, 1] * a[2, 7] * a[1, 10] - a[0, 2] * a[2, 4] * a[1, 12] - a[0, 2] * a[2, 5] * a[1, 11] -
           a[0, 2] * a[2, 6] * a[1, 10] - a[0, 3] * a[2, 4] * a[1, 11] - a[0, 3] * a[2, 5] * a[1, 10] +
           a[0, 4] * a[2, 2] * a[1, 12] + a[0, 4] * a[2, 3] * a[1, 11] + a[0, 5] * a[2, 1] * a[1, 12] +
           a[0, 5] * a[2, 2] * a[1, 11] + a[0, 5] * a[2, 3] * a[1, 10] + a[0, 6] * a[2, 0] * a[1, 12] +
           a[0, 6] * a[2, 1] * a[1, 11] + a[0, 6] * a[2, 2] * a[1, 10] + a[0, 7] * a[2, 0] * a[1, 11] +
           a[0, 7] * a[2, 1] * a[1, 10] + a[0, 0] * a[1, 6] * a[2, 12] + a[0, 0] * a[1, 7] * a[2, 11] +
           a[0, 1] * a[1, 5] * a[2, 12] + a[0, 1] * a[1, 6] * a[2, 11] + a[0, 1] * a[1, 7] * a[2, 10] +
           a[0, 2] * a[1, 4] * a[2, 12] + a[0, 2] * a[1, 5] * a[2, 11] + a[0, 2] * a[1, 6] * a[2, 10] +
           a[0, 3] * a[1, 4] * a[2, 11] + a[0, 3] * a[1, 5] * a[2, 10] - a[0, 4] * a[1, 2] * a[2, 12] -
           a[0, 4] * a[1, 3] * a[2, 11] - a[0, 5] * a[1, 1] * a[2, 12] - a[0, 5] * a[1, 2] * a[2, 11] -
           a[0, 5] * a[1, 3] * a[2, 10] - a[0, 6] * a[1, 0] * a[2, 12] - a[0, 6] * a[1, 1] * a[2, 11] -
           a[0, 6] * a[1, 2] * a[2, 10] - a[0, 7] * a[1, 0] * a[2, 11] - a[0, 7] * a[1, 1] * a[2, 10])
    p[5] = (a[0, 1] * a[1, 7] * a[2, 9] - a[0, 1] * a[1, 9] * a[2, 7] + a[0, 2] * a[1, 6] * a[2, 9] +
           a[0, 2] * a[1, 7] * a[2, 8] - a[0, 2] * a[1, 8] * a[2, 7] - a[0, 2] * a[1, 9] * a[2, 6] +
           a[0, 3] * a[1, 5] * a[2, 9] + a[0, 3] * a[1, 6] * a[2, 8] - a[0, 3] * a[1, 8] * a[2, 6] -
           a[0, 3] * a[1, 9] * a[2, 5] - a[0, 5] * a[1, 3] * a[2, 9] + a[0, 5] * a[1, 9] * a[2, 3] -
           a[0, 6] * a[1, 2] * a[2, 9] - a[0, 6] * a[1, 3] * a[2, 8] + a[0, 6] * a[1, 8] * a[2, 3] +
           a[0, 6] * a[1, 9] * a[2, 2] - a[0, 7] * a[1, 1] * a[2, 9] - a[0, 7] * a[1, 2] * a[2, 8] +
           a[0, 7] * a[1, 8] * a[2, 2] + a[0, 7] * a[1, 9] * a[2, 1] + a[0, 8] * a[1, 2] * a[2, 7] +
           a[0, 8] * a[1, 3] * a[2, 6] - a[0, 8] * a[1, 6] * a[2, 3] - a[0, 8] * a[1, 7] * a[2, 2] +
           a[0, 9] * a[1, 1] * a[2, 7] + a[0, 9] * a[1, 2] * a[2, 6] + a[0, 9] * a[1, 3] * a[2, 5] -
           a[0, 9] * a[1, 5] * a[2, 3] - a[0, 9] * a[1, 6] * a[2, 2] - a[0, 9] * a[1, 7] * a[2, 1] +
           a[0, 10] * a[1, 0] * a[2, 7] + a[0, 10] * a[1, 1] * a[2, 6] + a[0, 10] * a[1, 2] * a[2, 5] +
           a[0, 10] * a[1, 3] * a[2, 4] - a[0, 10] * a[1, 4] * a[2, 3] - a[0, 10] * a[1, 5] * a[2, 2] -
           a[0, 10] * a[1, 6] * a[2, 1] - a[0, 10] * a[1, 7] * a[2, 0] + a[1, 0] * a[0, 11] * a[2, 6] +
           a[1, 0] * a[0, 12] * a[2, 5] + a[0, 11] * a[1, 1] * a[2, 5] + a[0, 11] * a[1, 2] * a[2, 4] -
           a[0, 11] * a[1, 4] * a[2, 2] - a[0, 11] * a[1, 5] * a[2, 1] - a[0, 11] * a[1, 6] * a[2, 0] +
           a[1, 1] * a[0, 12] * a[2, 4] - a[0, 12] * a[1, 4] * a[2, 1] - a[0, 12] * a[1, 5] * a[2, 0] -
           a[0, 0] * a[2, 5] * a[1, 12] - a[0, 0] * a[2, 6] * a[1, 11] - a[0, 0] * a[2, 7] * a[1, 10] -
           a[0, 1] * a[2, 4] * a[1, 12] - a[0, 1] * a[2, 5] * a[1, 11] - a[0, 1] * a[2, 6] * a[1, 10] -
           a[0, 2] * a[2, 4] * a[1, 11] - a[0, 2] * a[2, 5] * a[1, 10] - a[0, 3] * a[2, 4] * a[1, 10] +
           a[0, 4] * a[2, 1] * a[1, 12] + a[0, 4] * a[2, 2] * a[1, 11] + a[0, 4] * a[2, 3] * a[1, 10] +
           a[0, 5] * a[2, 0] * a[1, 12] + a[0, 5] * a[2, 1] * a[1, 11] + a[0, 5] * a[2, 2] * a[1, 10] +
           a[0, 6] * a[2, 0] * a[1, 11] + a[0, 6] * a[2, 1] * a[1, 10] + a[0, 7] * a[2, 0] * a[1, 10] +
           a[0, 0] * a[1, 5] * a[2, 12] + a[0, 0] * a[1, 6] * a[2, 11] + a[0, 0] * a[1, 7] * a[2, 10] +
           a[0, 1] * a[1, 4] * a[2, 12] + a[0, 1] * a[1, 5] * a[2, 11] + a[0, 1] * a[1, 6] * a[2, 10] +
           a[0, 2] * a[1, 4] * a[2, 11] + a[0, 2] * a[1, 5] * a[2, 10] + a[0, 3] * a[1, 4] * a[2, 10] -
           a[0, 4] * a[1, 1] * a[2, 12] - a[0, 4] * a[1, 2] * a[2, 11] - a[0, 4] * a[1, 3] * a[2, 10] -
           a[0, 5] * a[1, 0] * a[2, 12] - a[0, 5] * a[1, 1] * a[2, 11] - a[0, 5] * a[1, 2] * a[2, 10] -
           a[0, 6] * a[1, 0] * a[2, 11] - a[0, 6] * a[1, 1] * a[2, 10] - a[0, 7] * a[1, 0] * a[2, 10])
    p[6] = (a[0, 0] * a[1, 7] * a[2, 9] - a[0, 0] * a[1, 9] * a[2, 7] + a[0, 1] * a[1, 6] * a[2, 9] +
           a[0, 1] * a[1, 7] * a[2, 8] - a[0, 1] * a[1, 8] * a[2, 7] - a[0, 1] * a[1, 9] * a[2, 6] +
           a[0, 2] * a[1, 5] * a[2, 9] + a[0, 2] * a[1, 6] * a[2, 8] - a[0, 2] * a[1, 8] * a[2, 6] -
           a[0, 2] * a[1, 9] * a[2, 5] + a[0, 3] * a[1, 4] * a[2, 9] + a[0, 3] * a[1, 5] * a[2, 8] -
           a[0, 3] * a[1, 8] * a[2, 5] - a[0, 3] * a[1, 9] * a[2, 4] - a[0, 4] * a[1, 3] * a[2, 9] +
           a[0, 4] * a[1, 9] * a[2, 3] - a[0, 5] * a[1, 2] * a[2, 9] - a[0, 5] * a[1, 3] * a[2, 8] +
           a[0, 5] * a[1, 8] * a[2, 3] + a[0, 5] * a[1, 9] * a[2, 2] - a[0, 6] * a[1, 1] * a[2, 9] -
           a[0, 6] * a[1, 2] * a[2, 8] + a[0, 6] * a[1, 8] * a[2, 2] + a[0, 6] * a[1, 9] * a[2, 1] -
           a[0, 7] * a[1, 0] * a[2, 9] - a[0, 7] * a[1, 1] * a[2, 8] + a[0, 7] * a[1, 8] * a[2, 1] +
           a[0, 7] * a[1, 9] * a[2, 0] + a[0, 8] * a[1, 1] * a[2, 7] + a[0, 8] * a[1, 2] * a[2, 6] +
           a[0, 8] * a[1, 3] * a[2, 5] - a[0, 8] * a[1, 5] * a[2, 3] - a[0, 8] * a[1, 6] * a[2, 2] -
           a[0, 8] * a[1, 7] * a[2, 1] + a[0, 9] * a[1, 0] * a[2, 7] + a[0, 9] * a[1, 1] * a[2, 6] +
           a[0, 9] * a[1, 2] * a[2, 5] + a[0, 9] * a[1, 3] * a[2, 4] - a[0, 9] * a[1, 4] * a[2, 3] -
           a[0, 9] * a[1, 5] * a[2, 2] - a[0, 9] * a[1, 6] * a[2, 1] - a[0, 9] * a[1, 7] * a[2, 0] +
           a[0, 10] * a[1, 0] * a[2, 6] + a[0, 10] * a[1, 1] * a[2, 5] + a[0, 10] * a[1, 2] * a[2, 4] -
           a[0, 10] * a[1, 4] * a[2, 2] - a[0, 10] * a[1, 5] * a[2, 1] - a[0, 10] * a[1, 6] * a[2, 0] +
           a[1, 0] * a[0, 11] * a[2, 5] + a[1, 0] * a[0, 12] * a[2, 4] + a[0, 11] * a[1, 1] * a[2, 4] -
           a[0, 11] * a[1, 4] * a[2, 1] - a[0, 11] * a[1, 5] * a[2, 0] - a[0, 12] * a[1, 4] * a[2, 0] -
           a[0, 0] * a[2, 4] * a[1, 12] - a[0, 0] * a[2, 5] * a[1, 11] - a[0, 0] * a[2, 6] * a[1, 10] -
           a[0, 1] * a[2, 4] * a[1, 11] - a[0, 1] * a[2, 5] * a[1, 10] - a[0, 2] * a[2, 4] * a[1, 10] +
           a[0, 4] * a[2, 0] * a[1, 12] + a[0, 4] * a[2, 1] * a[1, 11] + a[0, 4] * a[2, 2] * a[1, 10] +
           a[0, 5] * a[2, 0] * a[1, 11] + a[0, 5] * a[2, 1] * a[1, 10] + a[0, 6] * a[2, 0] * a[1, 10] +
           a[0, 0] * a[1, 4] * a[2, 12] + a[0, 0] * a[1, 5] * a[2, 11] + a[0, 0] * a[1, 6] * a[2, 10] +
           a[0, 1] * a[1, 4] * a[2, 11] + a[0, 1] * a[1, 5] * a[2, 10] + a[0, 2] * a[1, 4] * a[2, 10] -
           a[0, 4] * a[1, 0] * a[2, 12] - a[0, 4] * a[1, 1] * a[2, 11] - a[0, 4] * a[1, 2] * a[2, 10] -
           a[0, 5] * a[1, 0] * a[2, 11] - a[0, 5] * a[1, 1] * a[2, 10] - a[0, 6] * a[1, 0] * a[2, 10])
    p[7] = (a[0, 0] * a[1, 6] * a[2, 9] + a[0, 0] * a[1, 7] * a[2, 8] - a[0, 0] * a[1, 8] * a[2, 7] -
           a[0, 0] * a[1, 9] * a[2, 6] + a[0, 1] * a[1, 5] * a[2, 9] + a[0, 1] * a[1, 6] * a[2, 8] -
           a[0, 1] * a[1, 8] * a[2, 6] - a[0, 1] * a[1, 9] * a[2, 5] + a[0, 2] * a[1, 4] * a[2, 9] +
           a[0, 2] * a[1, 5] * a[2, 8] - a[0, 2] * a[1, 8] * a[2, 5] - a[0, 2] * a[1, 9] * a[2, 4] +
           a[0, 3] * a[1, 4] * a[2, 8] - a[0, 3] * a[1, 8] * a[2, 4] - a[0, 4] * a[1, 2] * a[2, 9] -
           a[0, 4] * a[1, 3] * a[2, 8] + a[0, 4] * a[1, 8] * a[2, 3] + a[0, 4] * a[1, 9] * a[2, 2] -
           a[0, 5] * a[1, 1] * a[2, 9] - a[0, 5] * a[1, 2] * a[2, 8] + a[0, 5] * a[1, 8] * a[2, 2] +
           a[0, 5] * a[1, 9] * a[2, 1] - a[0, 6] * a[1, 0] * a[2, 9] - a[0, 6] * a[1, 1] * a[2, 8] +
           a[0, 6] * a[1, 8] * a[2, 1] + a[0, 6] * a[1, 9] * a[2, 0] - a[0, 7] * a[1, 0] * a[2, 8] +
           a[0, 7] * a[1, 8] * a[2, 0] + a[0, 8] * a[1, 0] * a[2, 7] + a[0, 8] * a[1, 1] * a[2, 6] +
           a[0, 8] * a[1, 2] * a[2, 5] + a[0, 8] * a[1, 3] * a[2, 4] - a[0, 8] * a[1, 4] * a[2, 3] -
           a[0, 8] * a[1, 5] * a[2, 2] - a[0, 8] * a[1, 6] * a[2, 1] - a[0, 8] * a[1, 7] * a[2, 0] +
           a[0, 9] * a[1, 0] * a[2, 6] + a[0, 9] * a[1, 1] * a[2, 5] + a[0, 9] * a[1, 2] * a[2, 4] -
           a[0, 9] * a[1, 4] * a[2, 2] - a[0, 9] * a[1, 5] * a[2, 1] - a[0, 9] * a[1, 6] * a[2, 0] +
           a[0, 10] * a[1, 0] * a[2, 5] + a[0, 10] * a[1, 1] * a[2, 4] - a[0, 10] * a[1, 4] * a[2, 1] -
           a[0, 10] * a[1, 5] * a[2, 0] + a[1, 0] * a[0, 11] * a[2, 4] - a[0, 11] * a[1, 4] * a[2, 0] -
           a[0, 0] * a[2, 4] * a[1, 11] - a[0, 0] * a[2, 5] * a[1, 10] - a[0, 1] * a[2, 4] * a[1, 10] +
           a[0, 4] * a[2, 0] * a[1, 11] + a[0, 4] * a[2, 1] * a[1, 10] + a[0, 5] * a[2, 0] * a[1, 10] +
           a[0, 0] * a[1, 4] * a[2, 11] + a[0, 0] * a[1, 5] * a[2, 10] + a[0, 1] * a[1, 4] * a[2, 10] -
           a[0, 4] * a[1, 0] * a[2, 11] - a[0, 4] * a[1, 1] * a[2, 10] - a[0, 5] * a[1, 0] * a[2, 10])
    p[8] = (a[0, 0] * a[1, 5] * a[2, 9] + a[0, 0] * a[1, 6] * a[2, 8] - a[0, 0] * a[1, 8] * a[2, 6] -
           a[0, 0] * a[1, 9] * a[2, 5] + a[0, 1] * a[1, 4] * a[2, 9] + a[0, 1] * a[1, 5] * a[2, 8] -
           a[0, 1] * a[1, 8] * a[2, 5] - a[0, 1] * a[1, 9] * a[2, 4] + a[0, 2] * a[1, 4] * a[2, 8] -
           a[0, 2] * a[1, 8] * a[2, 4] - a[0, 4] * a[1, 1] * a[2, 9] - a[0, 4] * a[1, 2] * a[2, 8] +
           a[0, 4] * a[1, 8] * a[2, 2] + a[0, 4] * a[1, 9] * a[2, 1] - a[0, 5] * a[1, 0] * a[2, 9] -
           a[0, 5] * a[1, 1] * a[2, 8] + a[0, 5] * a[1, 8] * a[2, 1] + a[0, 5] * a[1, 9] * a[2, 0] -
           a[0, 6] * a[1, 0] * a[2, 8] + a[0, 6] * a[1, 8] * a[2, 0] + a[0, 8] * a[1, 0] * a[2, 6] +
           a[0, 8] * a[1, 1] * a[2, 5] + a[0, 8] * a[1, 2] * a[2, 4] - a[0, 8] * a[1, 4] * a[2, 2] -
           a[0, 8] * a[1, 5] * a[2, 1] - a[0, 8] * a[1, 6] * a[2, 0] + a[0, 9] * a[1, 0] * a[2, 5] +
           a[0, 9] * a[1, 1] * a[2, 4] - a[0, 9] * a[1, 4] * a[2, 1] - a[0, 9] * a[1, 5] * a[2, 0] +
           a[0, 10] * a[1, 0] * a[2, 4] - a[0, 10] * a[1, 4] * a[2, 0] - a[0, 0] * a[2, 4] * a[1, 10] +
           a[0, 4] * a[2, 0] * a[1, 10] + a[0, 0] * a[1, 4] * a[2, 10] - a[0, 4] * a[1, 0] * a[2, 10])
    p[9] = (a[0, 0] * a[1, 4] * a[2, 9] + a[0, 0] * a[1, 5] * a[2, 8] - a[0, 0] * a[1, 8] * a[2, 5] -
           a[0, 0] * a[1, 9] * a[2, 4] + a[0, 1] * a[1, 4] * a[2, 8] - a[0, 1] * a[1, 8] * a[2, 4] -
           a[0, 4] * a[1, 0] * a[2, 9] - a[0, 4] * a[1, 1] * a[2, 8] + a[0, 4] * a[1, 8] * a[2, 1] +
           a[0, 4] * a[1, 9] * a[2, 0] - a[0, 5] * a[1, 0] * a[2, 8] + a[0, 5] * a[1, 8] * a[2, 0] +
           a[0, 8] * a[1, 0] * a[2, 5] + a[0, 8] * a[1, 1] * a[2, 4] - a[0, 8] * a[1, 4] * a[2, 1] -
           a[0, 8] * a[1, 5] * a[2, 0] + a[0, 9] * a[1, 0] * a[2, 4] - a[0, 9] * a[1, 4] * a[2, 0])
    p[10] = (a[0, 0] * a[1, 4] * a[2, 8] - a[0, 0] * a[1, 8] * a[2, 4] - a[0, 4] * a[1, 0] * a[2, 8] +
            a[0, 4] * a[1, 8] * a[2, 0] + a[0, 8] * a[1, 0] * a[2, 4] - a[0, 8] * a[1, 4] * a[2, 0])

    return p


@wp.kernel
def _eliminate(basis: wp.array3d(dtype=wp.float64), status: wp.array(dtype=wp.int32),
               matrices: wp.array3d(dtype=wp.float64), roots: wp.array2d(dtype=wp.float64),
               counts: wp.array(dtype=wp.int32)):
    sample = wp.tid()
    counts[sample] = 0
    if status[sample] != 0:
        return
    n = M49()
    for i in range(4):
        for j in range(9):
            n[i,j] = basis[sample,i,j]
    c = _constraints(n)
    for k in range(10):
        pivot = wp.abs(c[k,k]); row = int(k)
        for i in range(k+1,10):
            if wp.abs(c[i,k]) > pivot:
                pivot = wp.abs(c[i,k]); row = i
        if pivot == wp.float64(0.0) or not wp.isfinite(pivot):
            status[sample] = 2
            return
        for j in range(k,20):
            tmp = c[k,j]; c[k,j] = c[row,j]; c[row,j] = tmp
        inv = wp.float64(1.0)/c[k,k]
        for i in range(k+1,10):
            factor = c[i,k]*inv
            for j in range(k+1,20):
                c[i,j] -= factor*c[k,j]
    for step in range(10):
        i = 9-step
        for j in range(10,20):
            value = c[i,j]
            for k in range(i+1,10):
                value -= c[i,k]*c[k,j]
            c[i,j] = value/c[i,i]
    a = M313()
    for i in range(3):
        for j in range(3):
            a[i,j+1] = c[4+2*i,10+j]
            a[i,5+j] = c[4+2*i,13+j]
        for j in range(4):
            a[i,9+j] = c[4+2*i,16+j]
        for j in range(3):
            a[i,j] -= c[5+2*i,10+j]
            a[i,4+j] -= c[5+2*i,13+j]
        for j in range(4):
            a[i,8+j] -= c[5+2*i,16+j]
    p = _determinant(a)
    if p[10] == wp.float64(0.0) or not wp.isfinite(p[10]):
        status[sample] = 3
        return
    lead = p[10]
    for j in range(11):
        p[j] /= lead
        if not wp.isfinite(p[j]):
            status[sample] = 3
            return
    values, count, good = _sturm(p)
    if not good:
        status[sample] = 4
    counts[sample] = count
    for j in range(count):
        roots[sample,j] = values[j]
    for i in range(3):
        for j in range(13):
            matrices[sample,i,j] = a[i,j]


@wp.func
def _coordinates(a: M313, z: wp.float64):
    b = wp.mat33d()  # augmented 3x2 system + RHS
    z2, z3, z4 = z*z, z*z*z, (z*z)*(z*z)
    for i in range(3):
        b[i,0] = a[i,0]*z3+a[i,1]*z2+a[i,2]*z+a[i,3]
        b[i,1] = a[i,4]*z3+a[i,5]*z2+a[i,6]*z+a[i,7]
        b[i,2] = a[i,8]*z4+a[i,9]*z3+a[i,10]*z2+a[i,11]*z+a[i,12]
    determinant = b[0,0]*b[1,1]-b[0,1]*b[1,0]
    x,y = wp.float64(0.0),wp.float64(0.0)
    qr = wp.bool(determinant == wp.float64(0.0))
    if not qr:
        inv = wp.float64(1.0)/determinant
        x = (b[1,1]*inv)*b[0,2]-(b[0,1]*inv)*b[1,2]
        y = -(b[1,0]*inv)*b[0,2]+(b[0,0]*inv)*b[1,2]
        qr = not wp.isfinite(x) or not wp.isfinite(y) or wp.abs(b[2,0]*x+b[2,1]*y-b[2,2]) > wp.float64(1e-6)
    good = wp.bool(True)
    if qr:
        norm0, norm1 = wp.float64(0.0),wp.float64(0.0)
        for i in range(3):
            norm0 += b[i,0]*b[i,0]; norm1 += b[i,1]*b[i,1]
        swap = norm1 > norm0
        if swap:
            for i in range(3):
                tmp=b[i,0];b[i,0]=b[i,1];b[i,1]=tmp
        for k in range(2):
            tail=wp.float64(0.0)
            for i in range(k+1,3):
                tail += b[i,k]*b[i,k]
            alpha=b[k,k]
            if tail > wp.float64(0.0):
                beta=wp.sqrt(alpha*alpha+tail)
                if alpha >= wp.float64(0.0):beta=-beta
                tau=(beta-alpha)/beta
                for i in range(k+1,3):b[i,k]/=alpha-beta
                b[k,k]=beta
                for j in range(k+1,3):
                    dot=b[k,j]
                    for i in range(k+1,3):dot+=b[i,k]*b[i,j]
                    dot*=tau;b[k,j]-=dot
                    for i in range(k+1,3):b[i,j]-=b[i,k]*dot
        good = b[0,0] != wp.float64(0.0) and b[1,1] != wp.float64(0.0)
        if good:
            y=b[1,2]/b[1,1];x=(b[0,2]-b[0,1]*y)/b[0,0]
            if swap:
                tmp=x;x=y;y=tmp
    return -x,-y,good and wp.isfinite(x) and wp.isfinite(y),qr


@wp.kernel
def _recover(a: wp.array3d(dtype=wp.float64), b: wp.array3d(dtype=wp.float64),
             basis: wp.array3d(dtype=wp.float64), matrices: wp.array3d(dtype=wp.float64),
             roots: wp.array2d(dtype=wp.float64), counts: wp.array(dtype=wp.int32),
             cheirality: wp.bool, output: wp.array4d(dtype=wp.float64),
             valid: wp.array2d(dtype=wp.bool), root_status: wp.array2d(dtype=wp.int32)):
    sample,root=wp.tid()
    valid[sample,root]=False;root_status[sample,root]=0
    if root >= counts[sample]:return
    matrix=M313()
    for i in range(3):
        for j in range(13):matrix[i,j]=matrices[sample,i,j]
    z=roots[sample,root]
    x,y,good,qr=_coordinates(matrix,z)
    if not good:
        root_status[sample,root]=5
        return
    inv=wp.float64(1.0)/wp.sqrt(x*x+y*y+z*z+wp.float64(1.0))
    e=wp.mat33d()
    for i in range(3):
        for j in range(3):
            k=3*j+i
            e[i,j]=(basis[sample,0,k]*x+basis[sample,1,k]*y+basis[sample,2,k]*z+basis[sample,3,k])*inv
            good=good and wp.isfinite(e[i,j])
    if not good:
        root_status[sample,root]=5
        return
    if cheirality:
        r0,r1,t,good=_decompose_matrix(e)
        accepted=wp.bool(False)
        if good:
            for pose in range(4):
                r=r0
                if pose >= 2:r=r1
                translation=t
                if pose % 2 == 1:translation=-t
                positive=wp.bool(True)
                for k in range(5):
                    u=wp.normalize(wp.vec3d(a[sample,k,0],a[sample,k,1],wp.float64(1.0)))
                    v=wp.normalize(wp.vec3d(b[sample,k,0],b[sample,k,1],wp.float64(1.0)))
                    ru=r*u;cosine=-wp.dot(ru,v)
                    rhs1,rhs2=-wp.dot(ru,translation),wp.dot(v,translation)
                    positive=positive and rhs1-cosine*rhs2 > wp.float64(0.0) and rhs2-cosine*rhs1 > wp.float64(0.0)
                accepted=accepted or positive
        if not accepted:
            root_status[sample,root]=6
            return
    for i in range(3):
        for j in range(3):output[sample,root,i,j]=e[i,j]
    valid[sample,root]=True
    if qr:root_status[sample,root]=7


def essential_five_point(points0,points1,return_owners=False,*,cheirality=True,return_diagnostics=False):
    """Solve (B,5,2) normalized-image correspondences; return packed E/owners.

    Diagnostics stay on GPU: sample 0 success, 1 input/rank, 2 LU,
    3 polynomial, 4 incomplete Sturm; root 5 recovery failure, 6 cheirality
    rejection, 7 accepted using the internal 3x2 QR. No CPU recovery.
    """
    if points0.shape != points1.shape or points0.ndim != 3 or points0.shape[1:] != (5,2):
        raise ValueError('Expected matching (B,5,2) normalized-image coordinates')
    if any(not t.is_cuda or t.dtype != torch.float64 or t.device != points0.device for t in (points0,points1)):
        raise ValueError('warp_poselib requires CUDA float64 inputs on the same device')
    a,b=points0.contiguous(),points1.contiguous();n=len(a)
    basis=a.new_empty((n,4,9));matrices=a.new_empty((n,3,13));roots=a.new_empty((n,10))
    counts=torch.empty(n,device=a.device,dtype=torch.int32);status=torch.empty_like(counts)
    result=a.new_empty((n,10,3,3));valid=torch.empty((n,10),device=a.device,dtype=torch.bool)
    root_status=torch.empty((n,10),device=a.device,dtype=torch.int32)
    if n:
        stream=wp.stream_from_torch(torch.cuda.current_stream(a.device))
        def launch(kernel,dim,args):
            wp.launch(kernel,dim=dim,inputs=[wp.from_torch(t) if isinstance(t,torch.Tensor) else t for t in args],block_dim=32,stream=stream)
        launch(_nullspace,n,[a,b,basis,status])
        launch(_eliminate,n,[basis,status,matrices,roots,counts])
        launch(_recover,(n,10),[a,b,basis,matrices,roots,counts,cheirality,result,valid,root_status])
    owners,slots=torch.where(valid)
    result=result[owners,slots]
    if return_diagnostics:return result,owners,dict(sample_status=status,root_status=root_status,root_count=counts)
    return (result,owners) if return_owners else result
