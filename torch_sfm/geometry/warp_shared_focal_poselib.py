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

"""PoseLib-style float64 six-point solver entirely on Warp CUDA.

Ports the elimination, Danilevsky/Sturm, structured QR recovery and essential
decomposition in PoseLib's relpose_6pt_focal.cc, misc/sturm.h and essential.cc.
The full-pivot Householder nullspace follows the algorithm used by Eigen.
No LAPACK, Torch linalg, or CPU recovery: a failed sample/root emits nothing.
Floating-point ordering differs from Eigen; this is not bitwise equivalence.

Five launches: normalization/QR, polynomial coefficients, warp-cooperative LU,
Danilevsky/Sturm, and parallel root recovery/cheirality. The 31x31 left matrix
and 31x15 right-hand sides are distributed by column over one warp per sample.
Outputs stay padded until final packing.
"""
from functools import lru_cache

import torch
import warp as wp

wp.init()
wp.set_module_options({'enable_backward': False, 'fast_math': False,
                       'fuse_fp': False, 'max_unroll': 1})
M96 = wp.types.matrix(shape=(9, 6), dtype=wp.float64)
M15 = wp.types.matrix(shape=(15, 15), dtype=wp.float64)
M316 = wp.types.matrix(shape=(3, 16), dtype=wp.float64)
M78 = wp.types.matrix(shape=(7, 8), dtype=wp.float64)
V6 = wp.types.vector(length=6, dtype=wp.float64)
I6 = wp.types.vector(length=6, dtype=wp.int32)
I8 = wp.types.vector(length=8, dtype=wp.int32)
V7 = wp.types.vector(length=7, dtype=wp.float64)
V9 = wp.types.vector(length=9, dtype=wp.float64)
V15 = wp.types.vector(length=15, dtype=wp.float64)
V16 = wp.types.vector(length=16, dtype=wp.float64)
I16 = wp.types.vector(length=16, dtype=wp.int32)
V45 = wp.types.vector(length=45, dtype=wp.float64)
V31 = wp.types.vector(length=31, dtype=wp.float64)


@wp.kernel
def _nullspace(a: wp.array3d(dtype=wp.float64), b: wp.array3d(dtype=wp.float64),
               p0: wp.array2d(dtype=wp.float64), p1: wp.array2d(dtype=wp.float64),
               basis: wp.array2d(dtype=wp.float64), scales: wp.array(dtype=wp.float64),
               status: wp.array(dtype=wp.int32)):
    sample = wp.tid()
    status[sample] = 1
    for i in range(27):
        basis[sample, i] = wp.float64(0.0)
    scale = wp.float64(0.0)
    for k in range(6):
        offset0 = wp.vec2d(a[sample, k, 0]-p0[sample, 0], a[sample, k, 1]-p0[sample, 1])
        offset1 = wp.vec2d(b[sample, k, 0]-p1[sample, 0], b[sample, k, 1]-p1[sample, 1])
        scale += (wp.length(offset0)+wp.length(offset1))/wp.float64(12.0)
    scales[sample] = scale
    if not wp.isfinite(scale) or scale <= wp.float64(1e-12):
        return
    q = M96()
    for k in range(6):
        x = wp.normalize(wp.vec3d((a[sample, k, 0]-p0[sample, 0])/scale,
                                  (a[sample, k, 1]-p0[sample, 1])/scale, wp.float64(1.0)))
        y = wp.normalize(wp.vec3d((b[sample, k, 0]-p1[sample, 0])/scale,
                                  (b[sample, k, 1]-p1[sample, 1])/scale, wp.float64(1.0)))
        for i in range(3):
            for j in range(3):
                q[3*i+j, k] = x[i]*y[j]
    tau = V6()
    swaps = I6()
    biggest = wp.float64(0.0)
    for k in range(6):
        pivot = wp.float64(0.0)
        row, col = int(k), int(k)
        # Column-major tie order, matching Eigen's maximum-coefficient scan.
        for j in range(k, 6):
            for i in range(k, 9):
                if wp.abs(q[i, j]) > pivot:
                    pivot = wp.abs(q[i, j]); row = i; col = j
        if k == 0:
            biggest = pivot
        if not wp.isfinite(pivot) or pivot <= wp.float64(1.3322676295501878e-15)*biggest:
            return
        swaps[k] = row
        for j in range(k, 6):
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
            for j in range(k+1, 6):
                dot = q[k, j]
                for i in range(k+1, 9):
                    dot += q[i, k]*q[i, j]
                dot *= tau[k]
                q[k, j] -= dot
                for i in range(k+1, 9):
                    q[i, j] -= q[i, k]*dot
    for slot in range(3):
        v = V9()
        v[6+slot] = wp.float64(1.0)
        for step in range(6):
            k = 5-step
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
            basis[sample, slot*9+j] = v[j]
    status[sample] = 0


@wp.func
def _charpoly(source: M15):
    a = source
    good = wp.bool(True)
    for step in range(14):
        i = 14-step
        pivot_index = i-1
        pivot = wp.abs(a[i, i-1])
        for j in range(i-1):
            if wp.abs(a[i, j]) > pivot:
                pivot = wp.abs(a[i, j]); pivot_index = j
        if pivot == wp.float64(0.0) or not wp.isfinite(pivot):
            good = False
            break
        for j in range(15):
            tmp = a[i-1, j]; a[i-1, j] = a[pivot_index, j]; a[pivot_index, j] = tmp
        for j in range(15):
            tmp = a[j, i-1]; a[j, i-1] = a[j, pivot_index]; a[j, pivot_index] = tmp
        v, vinv, new_row, col = V15(), V15(), V15(), V15()
        for j in range(15):
            v[j] = a[i, j]
        for j in range(15):
            for k in range(15):
                new_row[j] += v[k]*a[k, j]
        for j in range(15):
            a[i-1, j] = new_row[j]
            vinv[j] = -v[j]/v[i-1]
            col[j] = a[j, i-1]
        vinv[i-1] = wp.float64(1.0)/v[i-1]-wp.float64(1.0)
        for j in range(i+1):
            for k in range(15):
                a[j, k] += col[j]*vinv[k]
        for j in range(15):
            a[i, j] = wp.float64(0.0)
        a[i, i-1] = wp.float64(1.0)
    p = V16()
    p[15] = wp.float64(1.0)
    for j in range(15):
        p[j] = -a[0, 14-j]
        good = good and wp.isfinite(p[j])
    return p, good


@wp.func
def _polyval(p: V16, x: wp.float64):
    value = wp.float64(1.0)
    for step in range(15):
        value = x*value+p[14-step]
    return value


@wp.func
def _derivative(p: V16, x: wp.float64):
    # PoseLib evaluates the normalized monic derivative, then multiplies by 15.
    value = wp.float64(1.0)
    for step in range(14):
        i = 13-step
        value = x*value+p[i+1]*(wp.float64(i+1)/wp.float64(15.0))
    return wp.float64(15.0)*value


@wp.func
def _signchanges(s: V45, x: wp.float64):
    f2 = s[44]
    f1 = s[42]+x*s[43]
    count = int(0)
    if (f1 < wp.float64(0.0)) != (f2 < wp.float64(0.0)):
        count += 1
    good = wp.bool(not wp.isnan(f1) and not wp.isnan(f2))
    for step in range(14):
        i = 13-step
        f0 = (s[3*i]+x*s[3*i+1])*f1+s[3*i+2]*f2
        good = good and not wp.isnan(f0)
        if (f0 < wp.float64(0.0)) != (f1 < wp.float64(0.0)):
            count += 1
        f2 = f1; f1 = f0
    if not good:
        count = -1
    return count


@wp.func
def _refine_root(p: V16, left: wp.float64, right: wp.float64):
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
def _sturm(p: V16):
    f, s = M316(), V45()
    for j in range(16):
        f[0, j] = p[j]
    for j in range(15):
        f[1, j] = p[j+1]*(wp.float64(j+1)/wp.float64(15.0))
    good = wp.bool(True)
    i1, i2, i3 = int(0), int(1), int(2)
    for i in range(14):
        q1 = f[i1, 15-i]*f[i2, 14-i]
        q0 = f[i1, 14-i]*f[i2, 14-i]-f[i1, 15-i]*f[i2, 13-i]
        f[i3, 0] = f[i1, 0]-q0*f[i2, 0]
        for j in range(1, 14-i):
            f[i3, j] = f[i1, j]-q1*f[i2, j-1]-q0*f[i2, j]
        c = -wp.abs(f[i3, 13-i])
        if c == wp.float64(0.0) or not wp.isfinite(c):
            good = False
            break
        inv = wp.float64(1.0)/c
        for j in range(14-i):
            f[i3, j] *= inv
        tmp = i1; i1 = i2; i2 = i3; i3 = tmp
        s[3*i] = q0; s[3*i+1] = q1; s[3*i+2] = c
    s[42] = f[i1, 0]; s[43] = f[i1, 1]; s[44] = f[i2, 0]
    bound = wp.float64(0.0)
    for j in range(15):
        bound = wp.max(bound, wp.abs(p[j]))
    bound += wp.float64(1.0)
    sa, sb = _signchanges(s, -bound), _signchanges(s, bound)
    good = good and sa >= 0 and sb >= 0 and sa >= sb
    roots, left, right = V15(), V16(), V16()
    signs_a, signs_b, depths = I16(), I16(), I16()
    count, size, processed = int(0), int(0), int(0)
    if good and sa > sb:
        left[0] = -bound; right[0] = bound
        signs_a[0] = sa; signs_b[0] = sb; size = 1
    # At most 15 nonempty disjoint intervals are live. Depth limit matches
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
            if sc < sb or sc > sa or size > 13:
                good = False
                continue
            # Push right before left, preserving ascending root traversal.
            if sc > sb:
                left[size] = c; right[size] = b; depths[size] = depth+1
                signs_a[size] = sc; signs_b[size] = sb; size += 1
            if sa > sc:
                left[size] = a; right[size] = c; depths[size] = depth+1
                signs_a[size] = sa; signs_b[size] = sc; size += 1
        if found and count < 15:
            roots[count] = root; count += 1
    if size > 0:
        good = False
    return roots, count, good


@wp.kernel
def _eliminate_cooperative(coefficients: wp.array2d(dtype=wp.float64),
               mapping: wp.array2d(dtype=wp.int32), action_rows: wp.array(dtype=wp.int32),
               action: wp.array3d(dtype=wp.float64),
               counts: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32)):
    sample, col = wp.tid()
    if col == 0:
        counts[sample] = 0
    if status[sample] != 0:
        return
    # Lanes 0..30 own left-hand columns; 0..14 also own RHS columns.
    # All 32 lanes participate in broadcasts, including the otherwise idle lane.
    c = V31()
    rhs = V31()
    for i in range(31):
        if col < 31:
            index = mapping[col,i]
            if index >= 0:
                c[i] = coefficients[sample,index]
        if col < 15:
            index = mapping[col+31,i]
            if index >= 0:
                rhs[i] = coefficients[sample,index]
    for k in range(31):
        row = int(k)
        pivot = wp.float64(0.0)
        if col == k:
            pivot = wp.abs(c[k])
            for i in range(k+1,31):
                if wp.abs(c[i]) > pivot:
                    pivot = wp.abs(c[i])
                    row = i
        # The pivot column scans rows in the same order as the serial solver.
        row = wp.tile_extract(wp.tile(row),k)
        pivot = wp.tile_extract(wp.tile(pivot),k)
        if pivot == wp.float64(0.0) or not wp.isfinite(pivot):
            if col == 0:
                status[sample] = 2
            return
        if col >= k:
            tmp = c[k]
            c[k] = c[row]
            c[row] = tmp
        tmp = rhs[k]
        rhs[k] = rhs[row]
        rhs[row] = tmp
        diagonal = wp.tile_extract(wp.tile(c[k]),k)
        inv = wp.float64(1.0)/diagonal
        # Broadcast only the needed scalar, avoiding a full 31-vector replica.
        for i in range(k+1,31):
            entry = wp.tile_extract(wp.tile(c[i]),k)
            factor = entry*inv
            if col > k:
                c[i] -= factor*c[k]
            if col < 15:
                rhs[i] -= factor*rhs[k]
    # Ascending k preserves the serial back-substitution accumulation order.
    for step in range(31):
        i = 30-step
        value = rhs[i]
        for k in range(i+1,31):
            aik = wp.tile_extract(wp.tile(c[i]),k)
            value -= aik*rhs[k]
        diagonal = wp.tile_extract(wp.tile(c[i]),i)
        rhs[i] = value/diagonal
    if col < 15:
        for i in range(15):
            row = action_rows[i]
            value = wp.float64(0.0)
            if row < 8:
                value = -rhs[23+row]
            elif row-8 == col:
                value = wp.float64(1.0)
            action[sample,i,col] = value


@wp.kernel
def _action_roots(action: wp.array3d(dtype=wp.float64), roots: wp.array2d(dtype=wp.float64),
           counts: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32)):
    sample = wp.tid()
    if status[sample] != 0:
        return
    am = M15()
    for i in range(15):
        for j in range(15):
            am[i,j] = action[sample,i,j]
    polynomial, ok = _charpoly(am)
    if not ok:
        status[sample] = 3
        return
    real, count, ok = _sturm(polynomial)
    # Successfully isolated roots remain usable even when a separate interval
    # failed a bounded-work guard. No roots are repaired or recomputed on CPU.
    if not ok:
        status[sample] = 4
    counts[sample] = count
    for i in range(count):
        roots[sample, i] = real[i]


@wp.func
def _coordinates(am: M15, z: wp.float64):
    rows = I8(2, 3, 4, 6, 8, 9, 11, 14)
    a = M78()
    z2, z3 = z*z, z*z*z
    for i in range(7):
        r = rows[i]
        a[i, 0] = am[r, 2]; a[i, 1] = am[r, 6]
        a[i, 2] = z*am[r, 4]+am[r, 5]
        a[i, 3] = am[r, 1]+z*am[r, 3]
        a[i, 4] = am[r, 14]
        a[i, 5] = z*am[r, 11]+am[r, 13]
        a[i, 6] = z2*am[r, 9]+z*am[r, 10]+am[r, 12]
        a[i, 7] = -(am[r, 0]+z*am[r, 7]+z2*am[r, 8])
    a[0, 0] -= z; a[3, 1] -= z; a[2, 2] -= z2; a[1, 3] -= z2
    a[6, 5] -= z2; a[5, 6] -= z3; a[4, 7] += z3
    good = wp.bool(True)
    for k in range(7):
        tail = wp.float64(0.0)
        for i in range(k+1, 7):
            tail += a[i, k]*a[i, k]
        alpha = a[k, k]
        if tail > wp.float64(0.0):
            beta = wp.sqrt(alpha*alpha+tail)
            if alpha >= wp.float64(0.0):
                beta = -beta
            tau = (beta-alpha)/beta
            for i in range(k+1, 7):
                a[i, k] /= alpha-beta
            a[k, k] = beta
            for j in range(k+1, 8):
                dot = a[k, j]
                for i in range(k+1, 7):
                    dot += a[i, k]*a[i, j]
                dot *= tau
                a[k, j] -= dot
                for i in range(k+1, 7):
                    a[i, j] -= a[i, k]*dot
        if a[k, k] == wp.float64(0.0) or not wp.isfinite(a[k, k]):
            good = False
            break
    x = V7()
    if good:
        for step in range(7):
            i = 6-step
            value = a[i, 7]
            for j in range(i+1, 7):
                value -= a[i, j]*x[j]
            x[i] = value/a[i, i]
            good = good and wp.isfinite(x[i])
    return x[3], x[6], good


@wp.func
def _poses(e: wp.mat33d):
    c0, c1, c2 = wp.vec3d(e[0,0], e[1,0], e[2,0]), wp.vec3d(e[0,1], e[1,1], e[2,1]), wp.vec3d(e[0,2], e[1,2], e[2,2])
    u12, u13, u23 = wp.cross(c0,c1), wp.cross(c0,c2), wp.cross(c1,c2)
    n12, n13, n23 = wp.dot(u12,u12), wp.dot(u13,u13), wp.dot(u23,u23)
    u1, u2 = c1, u23
    if n12 > n13:
        if n12 > n23:
            u1 = c0; u2 = u12
    elif n13 > n23:
        u1 = c0; u2 = u13
    good = wp.bool(wp.length(u1) > wp.float64(0.0) and wp.length(u2) > wp.float64(0.0))
    u1, u2 = wp.normalize(u1), wp.normalize(u2)
    u0 = -wp.cross(u2, u1)
    v0, v1 = wp.transpose(e)*u1, -(wp.transpose(e)*u0)
    v0 = wp.normalize(v0)
    v1 -= wp.dot(v0, v1)*v0
    good = good and wp.length(v1) > wp.float64(0.0)
    v1 = wp.normalize(v1)
    v2 = wp.cross(v0, v1)
    r0, r1 = wp.mat33d(), wp.mat33d()
    for i in range(3):
        for j in range(3):
            r0[i,j] = u0[i]*v0[j]+u1[i]*v1[j]+u2[i]*v2[j]
            r1[i,j] = -u0[i]*v0[j]-u1[i]*v1[j]+u2[i]*v2[j]
            good = good and wp.isfinite(r0[i,j]) and wp.isfinite(r1[i,j])
    return r0, r1, u2, good


@wp.kernel
def _recover(a: wp.array3d(dtype=wp.float64), b: wp.array3d(dtype=wp.float64),
             p0: wp.array2d(dtype=wp.float64), p1: wp.array2d(dtype=wp.float64),
             basis: wp.array2d(dtype=wp.float64), scales: wp.array(dtype=wp.float64),
             action: wp.array3d(dtype=wp.float64), roots: wp.array2d(dtype=wp.float64),
             counts: wp.array(dtype=wp.int32), output: wp.array4d(dtype=wp.float64),
             focals: wp.array2d(dtype=wp.float64), valid: wp.array2d(dtype=wp.bool),
             root_status: wp.array2d(dtype=wp.int32)):
    sample, root = wp.tid()
    for pose in range(4):
        valid[sample, root*4+pose] = False
    root_status[sample, root] = 0
    if root >= counts[sample]:
        return
    am = M15()
    for i in range(15):
        for j in range(15):
            am[i,j] = action[sample,i,j]
    z = roots[sample, root]
    x, w, good = _coordinates(am, z)
    if not good:
        root_status[sample, root] = 5
        return
    if w < wp.float64(1e-8) or not wp.isfinite(w):
        root_status[sample, root] = 6
        return
    f = wp.sqrt(wp.float64(1.0)/w)
    e = wp.mat33d()
    norm2 = wp.float64(0.0)
    for j in range(3):
        for i in range(3):
            k = 3*j+i
            value = basis[sample,k]+x*basis[sample,9+k]+z*basis[sample,18+k]
            e[i,j] = value
            norm2 += value*value
    if norm2 <= wp.float64(0.0) or not wp.isfinite(norm2):
        root_status[sample, root] = 7
        return
    norm = wp.sqrt(norm2)
    for i in range(3):
        for j in range(3):
            e[i,j] /= norm
            if i < 2:
                e[i,j] *= f
            if j < 2:
                e[i,j] *= f
    r0, r1, t, good = _poses(e)
    if not good:
        root_status[sample, root] = 7
        return
    scale = scales[sample]
    pixel_focal = f*scale
    accepted = int(0)
    for pose in range(4):
        r = r0
        if pose >= 2:
            r = r1
        sign = wp.float64(1.0)
        if pose == 1 or pose == 2:
            sign = wp.float64(-1.0)
        translation = sign*t
        positive = wp.bool(True)
        for k in range(6):
            u = wp.normalize(wp.vec3d((a[sample,k,0]-p0[sample,0])/pixel_focal,
                                       (a[sample,k,1]-p0[sample,1])/pixel_focal, wp.float64(1.0)))
            v = wp.normalize(wp.vec3d((b[sample,k,0]-p1[sample,0])/pixel_focal,
                                       (b[sample,k,1]-p1[sample,1])/pixel_focal, wp.float64(1.0)))
            ru = r*u
            cosine = -wp.dot(ru,v)
            rhs1, rhs2 = -wp.dot(ru,translation), wp.dot(v,translation)
            positive = positive and rhs1-cosine*rhs2 > wp.float64(0.0) and rhs2-cosine*rhs1 > wp.float64(0.0)
        if positive:
            tx = wp.mat33d(wp.float64(0.0),-translation[2],translation[1],
                           translation[2],wp.float64(0.0),-translation[0],
                           -translation[1],translation[0],wp.float64(0.0))
            ki0 = wp.mat33d(wp.float64(1.0)/pixel_focal,wp.float64(0.0),-p0[sample,0]/pixel_focal,
                            wp.float64(0.0),wp.float64(1.0)/pixel_focal,-p0[sample,1]/pixel_focal,
                            wp.float64(0.0),wp.float64(0.0),wp.float64(1.0))
            ki1 = wp.mat33d(wp.float64(1.0)/pixel_focal,wp.float64(0.0),-p1[sample,0]/pixel_focal,
                            wp.float64(0.0),wp.float64(1.0)/pixel_focal,-p1[sample,1]/pixel_focal,
                            wp.float64(0.0),wp.float64(0.0),wp.float64(1.0))
            fundamental = wp.transpose(ki1)*(tx*r)*ki0
            norm2 = wp.float64(0.0)
            for i in range(3):
                for j in range(3):
                    norm2 += fundamental[i,j]*fundamental[i,j]
            if wp.isfinite(norm2) and norm2 > wp.float64(0.0) and wp.isfinite(pixel_focal):
                slot = root*4+pose
                for i in range(3):
                    for j in range(3):
                        output[sample,slot,i,j] = fundamental[i,j]/wp.sqrt(norm2)
                focals[sample,slot] = pixel_focal
                valid[sample,slot] = True
                accepted += 1
    if accepted == 0:
        root_status[sample, root] = 8


@lru_cache(maxsize=8)
def _indices(device):
    from . import shared_focal_polynomial as p
    entries = [(c, i % 31, i // 31) for c, i in zip(p.COEFFS0_IND, p.C0_IND)]
    entries += [(c, i % 31, 31+i // 31) for c, i in zip(p.COEFFS1_IND, p.C1_IND)]
    mapping = [[-1]*31 for _ in range(46)]
    for coefficient,row,column in entries:
        mapping[column][row] = coefficient
    return (torch.tensor(mapping, device=device, dtype=torch.int32),
            torch.tensor(p.AM_IND, device=device, dtype=torch.int32))


@torch.no_grad()
def shared_focal_six_point(points0, points1, principal0, principal1, *, return_diagnostics=False):
    """Return pixel F, positive pixel focal and sample owner; never CPU-repair.

    Inputs must be CUDA float64 (B,6,2), (B,6,2), (B,2), (B,2). Sample status:
    0 processed; 1 invalid/rank-deficient; 2 LU; 3 characteristic polynomial;
    4 incomplete Sturm isolation. Root status: 5 QR; 6 nonpositive/out-of-range
    inverse focal squared; 7 invalid essential decomposition; 8 cheirality.
    Status tensors are optional diagnostics, never read back by the solver.
    """
    n = len(points0)
    values = (points0, points1, principal0, principal1)
    if any(v.device != points0.device or not v.is_cuda or v.dtype != torch.float64 for v in values):
        raise ValueError('warp_poselib requires CUDA float64 inputs on one device')
    if points0.shape != (n,6,2) or points1.shape != (n,6,2) or principal0.shape != (n,2) or principal1.shape != (n,2):
        raise ValueError('Expected point batches (B,6,2) and principal points (B,2)')
    sample_status = torch.empty(n, device=points0.device, dtype=torch.int32)
    root_status = torch.empty((n,15), device=points0.device, dtype=torch.int32)
    counts = torch.empty_like(sample_status)
    if not n:
        result = (points0.new_empty((0,3,3)), points0.new_empty(0),
                  torch.empty(0, device=points0.device, dtype=torch.long))
        return (*result, dict(sample_status=sample_status, root_status=root_status, root_counts=counts)) if return_diagnostics else result
    values = tuple(v.contiguous() for v in values)
    basis = points0.new_empty((n,27)); scales = points0.new_empty(n)
    action = points0.new_empty((n,15,15)); roots = points0.new_empty((n,15))
    output = points0.new_empty((n,60,3,3)); focals = points0.new_empty((n,60))
    valid = torch.empty((n,60), device=points0.device, dtype=torch.bool)
    indices, action_rows = _indices(points0.device)
    stream = wp.stream_from_torch(torch.cuda.current_stream(points0.device))
    wp.launch(_nullspace, dim=n, block_dim=32,
              inputs=[wp.from_torch(v) for v in (*values,basis,scales,sample_status)], stream=stream)
    from .warp_shared_focal_polynomial import coefficients
    coeff = coefficients(basis)
    wp.launch(_eliminate_cooperative, dim=(n,32), block_dim=32,
              inputs=[wp.from_torch(v) for v in (coeff,indices,action_rows,action,counts,sample_status)], stream=stream)
    wp.launch(_action_roots, dim=n, block_dim=32,
              inputs=[wp.from_torch(v) for v in (action,roots,counts,sample_status)], stream=stream)
    wp.launch(_recover, dim=(n,15), block_dim=32,
              inputs=[wp.from_torch(v) for v in (*values,basis,scales,action,roots,counts,output,focals,valid,root_status)], stream=stream)
    owner, slot = torch.where(valid)
    result = output[owner,slot], focals[owner,slot], owner
    if return_diagnostics:
        return *result, dict(sample_status=sample_status, root_status=root_status, root_counts=counts)
    return result
