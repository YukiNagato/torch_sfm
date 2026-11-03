# Sturm construction adapted from PoseLib/misc/sturm.h (BSD-3-Clause).
# Copyright (c) 2020, Viktor Larsson
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of the copyright holder nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Degree-ten real roots on CUDA, one bounded Sturm solve per polynomial."""

import torch
import warp as wp

wp.init()
wp.set_module_options({'enable_backward': False, 'max_unroll': 1})

Poly = wp.types.vector(length=11, dtype=wp.float64)
Sequence = wp.types.vector(length=30, dtype=wp.float64)


@wp.func
def _sequence(poly: Poly):
    first = poly
    second = Poly(wp.float64(0.0))
    for i in range(10):
        second[i] = poly[i + 1] * wp.float64(i + 1) / wp.float64(10.0)
    sequence = Sequence(wp.float64(0.0))
    ok = bool(True)
    for i in range(9):
        q1 = first[10 - i] * second[9 - i]
        q0 = first[9 - i] * second[9 - i] - first[10 - i] * second[8 - i]
        third = Poly(wp.float64(0.0))
        third[0] = first[0] - q0 * second[0]
        for j in range(1, 9 - i):
            third[j] = first[j] - q1 * second[j - 1] - q0 * second[j]
        scale = -wp.abs(third[8 - i])
        if not wp.isfinite(scale) or wp.abs(scale) < wp.float64(1e-18):
            ok = False
            break
        for j in range(9 - i):
            third[j] = third[j] / scale
        first = second
        second = third
        sequence[3 * i] = q0
        sequence[3 * i + 1] = q1
        sequence[3 * i + 2] = scale
    sequence[27] = first[0]
    sequence[28] = first[1]
    sequence[29] = second[0]
    return sequence, ok


@wp.func
def _variations(sequence: Sequence, x: wp.float64):
    next2 = sequence[29]
    next1 = sequence[27] + x * sequence[28]
    changes = int(next1 < wp.float64(0.0)) ^ int(next2 < wp.float64(0.0))
    for j in range(9):
        i = 8 - j
        value = (sequence[3 * i] + x * sequence[3 * i + 1]) * next1 + sequence[3 * i + 2] * next2
        changes += int(value < wp.float64(0.0)) ^ int(next1 < wp.float64(0.0))
        next2 = next1
        next1 = value
    return changes


@wp.kernel
def _determinant_polynomial(a: wp.array3d(dtype=wp.float64), coefficients: wp.array2d(dtype=wp.float64)):
    sample = wp.tid()
    poly = Poly(wp.float64(0.0))
    for i in range(4):
        for j in range(4):
            for k in range(5):
                v = (a[sample, 0, i] * a[sample, 1, 4 + j] * a[sample, 2, 8 + k]
                    + a[sample, 0, 4 + j] * a[sample, 1, 8 + k] * a[sample, 2, i]
                    + a[sample, 0, 8 + k] * a[sample, 1, i] * a[sample, 2, 4 + j]
                    - a[sample, 0, 8 + k] * a[sample, 1, 4 + j] * a[sample, 2, i]
                    - a[sample, 0, 4 + j] * a[sample, 1, i] * a[sample, 2, 8 + k]
                    - a[sample, 0, i] * a[sample, 1, 8 + k] * a[sample, 2, 4 + j])
                poly[10 - i - j - k] += v
    for i in range(11):
        coefficients[sample, i] = poly[i]


@wp.kernel
def _solve(coefficients: wp.array2d(dtype=wp.float64),
           roots: wp.array2d(dtype=wp.float64), counts: wp.array(dtype=wp.int32)):
    sample = wp.tid()
    poly = Poly(wp.float64(0.0))
    largest = wp.float64(0.0)
    for i in range(11):
        largest = wp.max(largest, wp.abs(coefficients[sample, i]))
    leading = coefficients[sample, 10]
    counts[sample] = -1
    if not wp.isfinite(largest) or wp.abs(leading) < wp.float64(1e-14) * largest or leading == wp.float64(0.0):
        return
    bound = wp.float64(1.0)
    for i in range(11):
        poly[i] = coefficients[sample, i] / leading
        if i < 10:
            bound = wp.max(bound, wp.float64(1.0) + wp.abs(poly[i]))
    sequence, ok = _sequence(poly)
    if not ok:
        return
    left_signs = _variations(sequence, -bound)
    right_signs = _variations(sequence, bound)
    count = left_signs - right_signs
    if count < 0 or count > 10 or count % 2 != 0:
        return
    previous = -bound
    for root in range(count):
        lo = -bound
        hi = bound + wp.float64(0.0)
        for iteration in range(100):
            mid = wp.float64(0.5) * (lo + hi)
            if _variations(sequence, mid) >= left_signs - root:
                lo = mid
            else:
                hi = mid
            if hi - lo <= wp.float64(1e-12) * (wp.float64(1.0) + wp.abs(mid)):
                break
        value = wp.float64(0.5) * (lo + hi)
        roots[sample, root] = value
        polynomial = wp.float64(1.0)
        magnitude = wp.float64(1.0)
        for j in range(10):
            polynomial = polynomial * value + poly[9 - j]
            magnitude = magnitude * wp.abs(value) + wp.abs(poly[9 - j])
        if not wp.isfinite(value) or wp.abs(polynomial) > wp.float64(1e-8) * wp.max(magnitude, wp.float64(1.0)):
            counts[sample] = -3
            return
        if root > 0 and value - previous < wp.float64(1e-7) * (wp.float64(1.0) + wp.abs(value)):
            counts[sample] = -4
            return  # Near-multiple roots use the reference solver.
        previous = value
    counts[sample] = count


def real_roots(coefficients):
    roots = coefficients.new_zeros((len(coefficients), 10))
    counts = torch.empty(len(coefficients), device=coefficients.device, dtype=torch.int32)
    if len(coefficients):
        wp.launch(_solve, dim=len(coefficients),
            inputs=[wp.from_torch(coefficients)], outputs=[wp.from_torch(roots), wp.from_torch(counts)],
            block_dim=32, stream=wp.stream_from_torch(torch.cuda.current_stream(coefficients.device)))
    return roots, counts


def determinant_roots(a):
    coefficients = a.new_empty((len(a), 11))
    if len(a):
        wp.launch(_determinant_polynomial, dim=len(a), inputs=[wp.from_torch(a)],
            outputs=[wp.from_torch(coefficients)], block_dim=32,
            stream=wp.stream_from_torch(torch.cuda.current_stream(a.device)))
    return real_roots(coefficients)
