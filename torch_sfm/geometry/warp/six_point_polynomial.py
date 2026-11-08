# BSD 3-Clause License
#
# Copyright (c) 2020, Viktor Larsson
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Generated from PoseLib/solvers/relpose_6pt_focal.cc; do not edit coefficients.

import warp as wp

wp.init()
wp.set_module_options({"enable_backward": False})

@wp.func
def coefficients(lane: int, data: wp.array2d(dtype=wp.float64), output: wp.array2d(dtype=wp.float64)):
    row = int(0)
    d0 = data[row, 0]
    d1 = data[row, 1]
    d2 = data[row, 2]
    d3 = data[row, 3]
    d4 = data[row, 4]
    d5 = data[row, 5]
    d6 = data[row, 6]
    d7 = data[row, 7]
    d8 = data[row, 8]
    d9 = data[row, 9]
    d10 = data[row, 10]
    d11 = data[row, 11]
    d12 = data[row, 12]
    d13 = data[row, 13]
    d14 = data[row, 14]
    d15 = data[row, 15]
    d16 = data[row, 16]
    d17 = data[row, 17]
    d18 = data[row, 18]
    d19 = data[row, 19]
    d20 = data[row, 20]
    d21 = data[row, 21]
    d22 = data[row, 22]
    d23 = data[row, 23]
    d24 = data[row, 24]
    d25 = data[row, 25]
    d26 = data[row, 26]
    if lane == 0:
        output[row, 0] = wp.float64(2.0) * d11 * d15 * d17 - d9 * (d17*d17)
    if lane == 1:
        output[row, 1] = -(d17*d17) * d18 + wp.float64(2.0) * d15 * d17 * d20 + wp.float64(2.0) * d11 * d17 * d24 + wp.float64(2.0) * d11 * d15 * d26 - wp.float64(2.0) * d9 * d17 * d26
    if lane == 2:
        output[row, 2] = wp.float64(2.0) * d17 * d20 * d24 - wp.float64(2.0) * d17 * d18 * d26 + wp.float64(2.0) * d15 * d20 * d26 + wp.float64(2.0) * d11 * d24 * d26 - d9 * (d26*d26)
    if lane == 3:
        output[row, 3] = wp.float64(2.0) * d20 * d24 * d26 - d18 * (d26*d26)
    if lane == 4:
        output[row, 4] = d9 * (d11*d11) + wp.float64(2.0) * d11 * d12 * d14 - d9 * (d14*d14) + d9 * (d15*d15) + wp.float64(2.0) * d10 * d15 * d16 - d9 * (d16*d16)
    if lane == 5:
        output[row, 5] = (d11*d11) * d18 - (d14*d14) * d18 + (d15*d15) * d18 - (d16*d16) * d18 + wp.float64(2.0) * d15 * d16 * d19 + wp.float64(2.0) * d9 * d11 * d20 + wp.float64(2.0) * d12 * d14 * d20 + wp.float64(2.0) * d11 * d14 * d21 + wp.float64(2.0) * d11 * d12 * d23 - wp.float64(2.0) * d9 * d14 * d23 + wp.float64(2.0) * d9 * d15 * d24 + wp.float64(2.0) * d10 * d16 * d24 + wp.float64(2.0) * d10 * d15 * d25 - wp.float64(2.0) * d9 * d16 * d25
    if lane == 6:
        output[row, 6] = wp.float64(2.0) * d11 * d18 * d20 + d9 * (d20*d20) + wp.float64(2.0) * d14 * d20 * d21 - wp.float64(2.0) * d14 * d18 * d23 + wp.float64(2.0) * d12 * d20 * d23 + wp.float64(2.0) * d11 * d21 * d23 - d9 * (d23*d23) + wp.float64(2.0) * d15 * d18 * d24 + wp.float64(2.0) * d16 * d19 * d24 + d9 * (d24*d24) - wp.float64(2.0) * d16 * d18 * d25 + wp.float64(2.0) * d15 * d19 * d25 + wp.float64(2.0) * d10 * d24 * d25 - d9 * (d25*d25)
    if lane == 7:
        output[row, 7] = d18 * (d20*d20) + wp.float64(2.0) * d20 * d21 * d23 - d18 * (d23*d23) + d18 * (d24*d24) + wp.float64(2.0) * d19 * d24 * d25 - d18 * (d25*d25)
    if lane == 8:
        output[row, 8] = wp.float64(2.0) * d8 * d11 * d15 - wp.float64(2.0) * d8 * d9 * d17 + wp.float64(2.0) * d6 * d11 * d17 + wp.float64(2.0) * d2 * d15 * d17 - d0 * (d17*d17)
    if lane == 9:
        output[row, 9] = -wp.float64(2.0) * d8 * d17 * d18 + wp.float64(2.0) * d8 * d15 * d20 + wp.float64(2.0) * d6 * d17 * d20 + wp.float64(2.0) * d8 * d11 * d24 + wp.float64(2.0) * d2 * d17 * d24 - wp.float64(2.0) * d8 * d9 * d26 + wp.float64(2.0) * d6 * d11 * d26 + wp.float64(2.0) * d2 * d15 * d26 - wp.float64(2.0) * d0 * d17 * d26
    if lane == 10:
        output[row, 10] = wp.float64(2.0) * d8 * d20 * d24 - wp.float64(2.0) * d8 * d18 * d26 + wp.float64(2.0) * d6 * d20 * d26 + wp.float64(2.0) * d2 * d24 * d26 - d0 * (d26*d26)
    if lane == 11:
        output[row, 11] = (d9*d9*d9) + d9 * (d10*d10) + d9 * (d12*d12) + wp.float64(2.0) * d10 * d12 * d13 - d9 * (d13*d13)
    if lane == 12:
        output[row, 12] = wp.float64(3.0) * (d9*d9) * d18 + (d10*d10) * d18 + (d12*d12) * d18 - (d13*d13) * d18 + wp.float64(2.0) * d9 * d10 * d19 + wp.float64(2.0) * d12 * d13 * d19 + wp.float64(2.0) * d9 * d12 * d21 + wp.float64(2.0) * d10 * d13 * d21 + wp.float64(2.0) * d10 * d12 * d22 - wp.float64(2.0) * d9 * d13 * d22
    if lane == 13:
        output[row, 13] = wp.float64(3.0) * d9 * (d18*d18) + wp.float64(2.0) * d10 * d18 * d19 + d9 * (d19*d19) + wp.float64(2.0) * d12 * d18 * d21 + wp.float64(2.0) * d13 * d19 * d21 + d9 * (d21*d21) - wp.float64(2.0) * d13 * d18 * d22 + wp.float64(2.0) * d12 * d19 * d22 + wp.float64(2.0) * d10 * d21 * d22 - d9 * (d22*d22)
    if lane == 14:
        output[row, 14] = (d18*d18*d18) + d18 * (d19*d19) + d18 * (d21*d21) + wp.float64(2.0) * d19 * d21 * d22 - d18 * (d22*d22)
    if lane == 15:
        output[row, 15] = wp.float64(2.0) * d2 * d9 * d11 + d0 * (d11*d11) + wp.float64(2.0) * d5 * d11 * d12 - wp.float64(2.0) * d5 * d9 * d14 + wp.float64(2.0) * d3 * d11 * d14 + wp.float64(2.0) * d2 * d12 * d14 - d0 * (d14*d14) + wp.float64(2.0) * d6 * d9 * d15 + wp.float64(2.0) * d7 * d10 * d15 + d0 * (d15*d15) - wp.float64(2.0) * d7 * d9 * d16 + wp.float64(2.0) * d6 * d10 * d16 + wp.float64(2.0) * d1 * d15 * d16 - d0 * (d16*d16)
    if lane == 16:
        output[row, 16] = wp.float64(2.0) * d2 * d11 * d18 - wp.float64(2.0) * d5 * d14 * d18 + wp.float64(2.0) * d6 * d15 * d18 - wp.float64(2.0) * d7 * d16 * d18 + wp.float64(2.0) * d7 * d15 * d19 + wp.float64(2.0) * d6 * d16 * d19 + wp.float64(2.0) * d2 * d9 * d20 + wp.float64(2.0) * d0 * d11 * d20 + wp.float64(2.0) * d5 * d12 * d20 + wp.float64(2.0) * d3 * d14 * d20 + wp.float64(2.0) * d5 * d11 * d21 + wp.float64(2.0) * d2 * d14 * d21 - wp.float64(2.0) * d5 * d9 * d23 + wp.float64(2.0) * d3 * d11 * d23 + wp.float64(2.0) * d2 * d12 * d23 - wp.float64(2.0) * d0 * d14 * d23 + wp.float64(2.0) * d6 * d9 * d24 + wp.float64(2.0) * d7 * d10 * d24 + wp.float64(2.0) * d0 * d15 * d24 + wp.float64(2.0) * d1 * d16 * d24 - wp.float64(2.0) * d7 * d9 * d25 + wp.float64(2.0) * d6 * d10 * d25 + wp.float64(2.0) * d1 * d15 * d25 - wp.float64(2.0) * d0 * d16 * d25
    if lane == 17:
        output[row, 17] = wp.float64(2.0) * d2 * d18 * d20 + d0 * (d20*d20) + wp.float64(2.0) * d5 * d20 * d21 - wp.float64(2.0) * d5 * d18 * d23 + wp.float64(2.0) * d3 * d20 * d23 + wp.float64(2.0) * d2 * d21 * d23 - d0 * (d23*d23) + wp.float64(2.0) * d6 * d18 * d24 + wp.float64(2.0) * d7 * d19 * d24 + d0 * (d24*d24) - wp.float64(2.0) * d7 * d18 * d25 + wp.float64(2.0) * d6 * d19 * d25 + wp.float64(2.0) * d1 * d24 * d25 - d0 * (d25*d25)
    if lane == 18:
        output[row, 18] = -(d8*d8) * d9 + wp.float64(2.0) * d6 * d8 * d11 + wp.float64(2.0) * d2 * d8 * d15 + wp.float64(2.0) * d2 * d6 * d17 - wp.float64(2.0) * d0 * d8 * d17
    if lane == 19:
        output[row, 19] = -(d8*d8) * d18 + wp.float64(2.0) * d6 * d8 * d20 + wp.float64(2.0) * d2 * d8 * d24 + wp.float64(2.0) * d2 * d6 * d26 - wp.float64(2.0) * d0 * d8 * d26
    if lane == 20:
        output[row, 20] = wp.float64(3.0) * d0 * (d9*d9) + wp.float64(2.0) * d1 * d9 * d10 + d0 * (d10*d10) + wp.float64(2.0) * d3 * d9 * d12 + wp.float64(2.0) * d4 * d10 * d12 + d0 * (d12*d12) - wp.float64(2.0) * d4 * d9 * d13 + wp.float64(2.0) * d3 * d10 * d13 + wp.float64(2.0) * d1 * d12 * d13 - d0 * (d13*d13)
    if lane == 21:
        output[row, 21] = wp.float64(6.0) * d0 * d9 * d18 + wp.float64(2.0) * d1 * d10 * d18 + wp.float64(2.0) * d3 * d12 * d18 - wp.float64(2.0) * d4 * d13 * d18 + wp.float64(2.0) * d1 * d9 * d19 + wp.float64(2.0) * d0 * d10 * d19 + wp.float64(2.0) * d4 * d12 * d19 + wp.float64(2.0) * d3 * d13 * d19 + wp.float64(2.0) * d3 * d9 * d21 + wp.float64(2.0) * d4 * d10 * d21 + wp.float64(2.0) * d0 * d12 * d21 + wp.float64(2.0) * d1 * d13 * d21 - wp.float64(2.0) * d4 * d9 * d22 + wp.float64(2.0) * d3 * d10 * d22 + wp.float64(2.0) * d1 * d12 * d22 - wp.float64(2.0) * d0 * d13 * d22
    if lane == 22:
        output[row, 22] = wp.float64(3.0) * d0 * (d18*d18) + wp.float64(2.0) * d1 * d18 * d19 + d0 * (d19*d19) + wp.float64(2.0) * d3 * d18 * d21 + wp.float64(2.0) * d4 * d19 * d21 + d0 * (d21*d21) - wp.float64(2.0) * d4 * d18 * d22 + wp.float64(2.0) * d3 * d19 * d22 + wp.float64(2.0) * d1 * d21 * d22 - d0 * (d22*d22)
    if lane == 23:
        output[row, 23] = (d2*d2) * d9 - (d5*d5) * d9 + (d6*d6) * d9 - (d7*d7) * d9 + wp.float64(2.0) * d6 * d7 * d10 + wp.float64(2.0) * d0 * d2 * d11 + wp.float64(2.0) * d3 * d5 * d11 + wp.float64(2.0) * d2 * d5 * d12 + wp.float64(2.0) * d2 * d3 * d14 - wp.float64(2.0) * d0 * d5 * d14 + wp.float64(2.0) * d0 * d6 * d15 + wp.float64(2.0) * d1 * d7 * d15 + wp.float64(2.0) * d1 * d6 * d16 - wp.float64(2.0) * d0 * d7 * d16
    if lane == 24:
        output[row, 24] = (d2*d2) * d18 - (d5*d5) * d18 + (d6*d6) * d18 - (d7*d7) * d18 + wp.float64(2.0) * d6 * d7 * d19 + wp.float64(2.0) * d0 * d2 * d20 + wp.float64(2.0) * d3 * d5 * d20 + wp.float64(2.0) * d2 * d5 * d21 + wp.float64(2.0) * d2 * d3 * d23 - wp.float64(2.0) * d0 * d5 * d23 + wp.float64(2.0) * d0 * d6 * d24 + wp.float64(2.0) * d1 * d7 * d24 + wp.float64(2.0) * d1 * d6 * d25 - wp.float64(2.0) * d0 * d7 * d25
    if lane == 25:
        output[row, 25] = wp.float64(2.0) * d2 * d6 * d8 - d0 * (d8*d8)
    if lane == 26:
        output[row, 26] = wp.float64(3.0) * (d0*d0) * d9 + (d1*d1) * d9 + (d3*d3) * d9 - (d4*d4) * d9 + wp.float64(2.0) * d0 * d1 * d10 + wp.float64(2.0) * d3 * d4 * d10 + wp.float64(2.0) * d0 * d3 * d12 + wp.float64(2.0) * d1 * d4 * d12 + wp.float64(2.0) * d1 * d3 * d13 - wp.float64(2.0) * d0 * d4 * d13
    if lane == 27:
        output[row, 27] = wp.float64(3.0) * (d0*d0) * d18 + (d1*d1) * d18 + (d3*d3) * d18 - (d4*d4) * d18 + wp.float64(2.0) * d0 * d1 * d19 + wp.float64(2.0) * d3 * d4 * d19 + wp.float64(2.0) * d0 * d3 * d21 + wp.float64(2.0) * d1 * d4 * d21 + wp.float64(2.0) * d1 * d3 * d22 - wp.float64(2.0) * d0 * d4 * d22
    if lane == 28:
        output[row, 28] = d0 * (d2*d2) + wp.float64(2.0) * d2 * d3 * d5 - d0 * (d5*d5) + d0 * (d6*d6) + wp.float64(2.0) * d1 * d6 * d7 - d0 * (d7*d7)
    if lane == 29:
        output[row, 29] = (d0*d0*d0) + d0 * (d1*d1) + d0 * (d3*d3) + wp.float64(2.0) * d1 * d3 * d4 - d0 * (d4*d4)
    if lane == 30:
        output[row, 30] = wp.float64(2.0) * d11 * d16 * d17 - d10 * (d17*d17)
    if lane == 31:
        output[row, 31] = -(d17*d17) * d19 + wp.float64(2.0) * d16 * d17 * d20 + wp.float64(2.0) * d11 * d17 * d25 + wp.float64(2.0) * d11 * d16 * d26 - wp.float64(2.0) * d10 * d17 * d26
    if lane == 0:
        output[row, 32] = wp.float64(2.0) * d17 * d20 * d25 - wp.float64(2.0) * d17 * d19 * d26 + wp.float64(2.0) * d16 * d20 * d26 + wp.float64(2.0) * d11 * d25 * d26 - d10 * (d26*d26)
    if lane == 1:
        output[row, 33] = wp.float64(2.0) * d20 * d25 * d26 - d19 * (d26*d26)
    if lane == 2:
        output[row, 34] = d10 * (d11*d11) + wp.float64(2.0) * d11 * d13 * d14 - d10 * (d14*d14) - d10 * (d15*d15) + wp.float64(2.0) * d9 * d15 * d16 + d10 * (d16*d16)
    if lane == 3:
        output[row, 35] = wp.float64(2.0) * d15 * d16 * d18 + (d11*d11) * d19 - (d14*d14) * d19 - (d15*d15) * d19 + (d16*d16) * d19 + wp.float64(2.0) * d10 * d11 * d20 + wp.float64(2.0) * d13 * d14 * d20 + wp.float64(2.0) * d11 * d14 * d22 + wp.float64(2.0) * d11 * d13 * d23 - wp.float64(2.0) * d10 * d14 * d23 - wp.float64(2.0) * d10 * d15 * d24 + wp.float64(2.0) * d9 * d16 * d24 + wp.float64(2.0) * d9 * d15 * d25 + wp.float64(2.0) * d10 * d16 * d25
    if lane == 4:
        output[row, 36] = wp.float64(2.0) * d11 * d19 * d20 + d10 * (d20*d20) + wp.float64(2.0) * d14 * d20 * d22 - wp.float64(2.0) * d14 * d19 * d23 + wp.float64(2.0) * d13 * d20 * d23 + wp.float64(2.0) * d11 * d22 * d23 - d10 * (d23*d23) + wp.float64(2.0) * d16 * d18 * d24 - wp.float64(2.0) * d15 * d19 * d24 - d10 * (d24*d24) + wp.float64(2.0) * d15 * d18 * d25 + wp.float64(2.0) * d16 * d19 * d25 + wp.float64(2.0) * d9 * d24 * d25 + d10 * (d25*d25)
    if lane == 5:
        output[row, 37] = d19 * (d20*d20) + wp.float64(2.0) * d20 * d22 * d23 - d19 * (d23*d23) - d19 * (d24*d24) + wp.float64(2.0) * d18 * d24 * d25 + d19 * (d25*d25)
    if lane == 6:
        output[row, 38] = wp.float64(2.0) * d8 * d11 * d16 - wp.float64(2.0) * d8 * d10 * d17 + wp.float64(2.0) * d7 * d11 * d17 + wp.float64(2.0) * d2 * d16 * d17 - d1 * (d17*d17)
    if lane == 7:
        output[row, 39] = -wp.float64(2.0) * d8 * d17 * d19 + wp.float64(2.0) * d8 * d16 * d20 + wp.float64(2.0) * d7 * d17 * d20 + wp.float64(2.0) * d8 * d11 * d25 + wp.float64(2.0) * d2 * d17 * d25 - wp.float64(2.0) * d8 * d10 * d26 + wp.float64(2.0) * d7 * d11 * d26 + wp.float64(2.0) * d2 * d16 * d26 - wp.float64(2.0) * d1 * d17 * d26
    if lane == 8:
        output[row, 40] = wp.float64(2.0) * d8 * d20 * d25 - wp.float64(2.0) * d8 * d19 * d26 + wp.float64(2.0) * d7 * d20 * d26 + wp.float64(2.0) * d2 * d25 * d26 - d1 * (d26*d26)
    if lane == 9:
        output[row, 41] = (d9*d9) * d10 + (d10*d10*d10) - d10 * (d12*d12) + wp.float64(2.0) * d9 * d12 * d13 + d10 * (d13*d13)
    if lane == 10:
        output[row, 42] = wp.float64(2.0) * d9 * d10 * d18 + wp.float64(2.0) * d12 * d13 * d18 + (d9*d9) * d19 + wp.float64(3.0) * (d10*d10) * d19 - (d12*d12) * d19 + (d13*d13) * d19 - wp.float64(2.0) * d10 * d12 * d21 + wp.float64(2.0) * d9 * d13 * d21 + wp.float64(2.0) * d9 * d12 * d22 + wp.float64(2.0) * d10 * d13 * d22
    if lane == 11:
        output[row, 43] = d10 * (d18*d18) + wp.float64(2.0) * d9 * d18 * d19 + wp.float64(3.0) * d10 * (d19*d19) + wp.float64(2.0) * d13 * d18 * d21 - wp.float64(2.0) * d12 * d19 * d21 - d10 * (d21*d21) + wp.float64(2.0) * d12 * d18 * d22 + wp.float64(2.0) * d13 * d19 * d22 + wp.float64(2.0) * d9 * d21 * d22 + d10 * (d22*d22)
    if lane == 12:
        output[row, 44] = (d18*d18) * d19 + (d19*d19*d19) - d19 * (d21*d21) + wp.float64(2.0) * d18 * d21 * d22 + d19 * (d22*d22)
    if lane == 13:
        output[row, 45] = wp.float64(2.0) * d2 * d10 * d11 + d1 * (d11*d11) + wp.float64(2.0) * d5 * d11 * d13 - wp.float64(2.0) * d5 * d10 * d14 + wp.float64(2.0) * d4 * d11 * d14 + wp.float64(2.0) * d2 * d13 * d14 - d1 * (d14*d14) + wp.float64(2.0) * d7 * d9 * d15 - wp.float64(2.0) * d6 * d10 * d15 - d1 * (d15*d15) + wp.float64(2.0) * d6 * d9 * d16 + wp.float64(2.0) * d7 * d10 * d16 + wp.float64(2.0) * d0 * d15 * d16 + d1 * (d16*d16)
    if lane == 14:
        output[row, 46] = wp.float64(2.0) * d7 * d15 * d18 + wp.float64(2.0) * d6 * d16 * d18 + wp.float64(2.0) * d2 * d11 * d19 - wp.float64(2.0) * d5 * d14 * d19 - wp.float64(2.0) * d6 * d15 * d19 + wp.float64(2.0) * d7 * d16 * d19 + wp.float64(2.0) * d2 * d10 * d20 + wp.float64(2.0) * d1 * d11 * d20 + wp.float64(2.0) * d5 * d13 * d20 + wp.float64(2.0) * d4 * d14 * d20 + wp.float64(2.0) * d5 * d11 * d22 + wp.float64(2.0) * d2 * d14 * d22 - wp.float64(2.0) * d5 * d10 * d23 + wp.float64(2.0) * d4 * d11 * d23 + wp.float64(2.0) * d2 * d13 * d23 - wp.float64(2.0) * d1 * d14 * d23 + wp.float64(2.0) * d7 * d9 * d24 - wp.float64(2.0) * d6 * d10 * d24 - wp.float64(2.0) * d1 * d15 * d24 + wp.float64(2.0) * d0 * d16 * d24 + wp.float64(2.0) * d6 * d9 * d25 + wp.float64(2.0) * d7 * d10 * d25 + wp.float64(2.0) * d0 * d15 * d25 + wp.float64(2.0) * d1 * d16 * d25
    if lane == 15:
        output[row, 47] = wp.float64(2.0) * d2 * d19 * d20 + d1 * (d20*d20) + wp.float64(2.0) * d5 * d20 * d22 - wp.float64(2.0) * d5 * d19 * d23 + wp.float64(2.0) * d4 * d20 * d23 + wp.float64(2.0) * d2 * d22 * d23 - d1 * (d23*d23) + wp.float64(2.0) * d7 * d18 * d24 - wp.float64(2.0) * d6 * d19 * d24 - d1 * (d24*d24) + wp.float64(2.0) * d6 * d18 * d25 + wp.float64(2.0) * d7 * d19 * d25 + wp.float64(2.0) * d0 * d24 * d25 + d1 * (d25*d25)
    if lane == 16:
        output[row, 48] = -(d8*d8) * d10 + wp.float64(2.0) * d7 * d8 * d11 + wp.float64(2.0) * d2 * d8 * d16 + wp.float64(2.0) * d2 * d7 * d17 - wp.float64(2.0) * d1 * d8 * d17
    if lane == 17:
        output[row, 49] = -(d8*d8) * d19 + wp.float64(2.0) * d7 * d8 * d20 + wp.float64(2.0) * d2 * d8 * d25 + wp.float64(2.0) * d2 * d7 * d26 - wp.float64(2.0) * d1 * d8 * d26
    if lane == 18:
        output[row, 50] = d1 * (d9*d9) + wp.float64(2.0) * d0 * d9 * d10 + wp.float64(3.0) * d1 * (d10*d10) + wp.float64(2.0) * d4 * d9 * d12 - wp.float64(2.0) * d3 * d10 * d12 - d1 * (d12*d12) + wp.float64(2.0) * d3 * d9 * d13 + wp.float64(2.0) * d4 * d10 * d13 + wp.float64(2.0) * d0 * d12 * d13 + d1 * (d13*d13)
    if lane == 19:
        output[row, 51] = wp.float64(2.0) * d1 * d9 * d18 + wp.float64(2.0) * d0 * d10 * d18 + wp.float64(2.0) * d4 * d12 * d18 + wp.float64(2.0) * d3 * d13 * d18 + wp.float64(2.0) * d0 * d9 * d19 + wp.float64(6.0) * d1 * d10 * d19 - wp.float64(2.0) * d3 * d12 * d19 + wp.float64(2.0) * d4 * d13 * d19 + wp.float64(2.0) * d4 * d9 * d21 - wp.float64(2.0) * d3 * d10 * d21 - wp.float64(2.0) * d1 * d12 * d21 + wp.float64(2.0) * d0 * d13 * d21 + wp.float64(2.0) * d3 * d9 * d22 + wp.float64(2.0) * d4 * d10 * d22 + wp.float64(2.0) * d0 * d12 * d22 + wp.float64(2.0) * d1 * d13 * d22
    if lane == 20:
        output[row, 52] = d1 * (d18*d18) + wp.float64(2.0) * d0 * d18 * d19 + wp.float64(3.0) * d1 * (d19*d19) + wp.float64(2.0) * d4 * d18 * d21 - wp.float64(2.0) * d3 * d19 * d21 - d1 * (d21*d21) + wp.float64(2.0) * d3 * d18 * d22 + wp.float64(2.0) * d4 * d19 * d22 + wp.float64(2.0) * d0 * d21 * d22 + d1 * (d22*d22)
    if lane == 21:
        output[row, 53] = wp.float64(2.0) * d6 * d7 * d9 + (d2*d2) * d10 - (d5*d5) * d10 - (d6*d6) * d10 + (d7*d7) * d10 + wp.float64(2.0) * d1 * d2 * d11 + wp.float64(2.0) * d4 * d5 * d11 + wp.float64(2.0) * d2 * d5 * d13 + wp.float64(2.0) * d2 * d4 * d14 - wp.float64(2.0) * d1 * d5 * d14 - wp.float64(2.0) * d1 * d6 * d15 + wp.float64(2.0) * d0 * d7 * d15 + wp.float64(2.0) * d0 * d6 * d16 + wp.float64(2.0) * d1 * d7 * d16
    if lane == 22:
        output[row, 54] = wp.float64(2.0) * d6 * d7 * d18 + (d2*d2) * d19 - (d5*d5) * d19 - (d6*d6) * d19 + (d7*d7) * d19 + wp.float64(2.0) * d1 * d2 * d20 + wp.float64(2.0) * d4 * d5 * d20 + wp.float64(2.0) * d2 * d5 * d22 + wp.float64(2.0) * d2 * d4 * d23 - wp.float64(2.0) * d1 * d5 * d23 - wp.float64(2.0) * d1 * d6 * d24 + wp.float64(2.0) * d0 * d7 * d24 + wp.float64(2.0) * d0 * d6 * d25 + wp.float64(2.0) * d1 * d7 * d25
    if lane == 23:
        output[row, 55] = wp.float64(2.0) * d2 * d7 * d8 - d1 * (d8*d8)
    if lane == 24:
        output[row, 56] = wp.float64(2.0) * d0 * d1 * d9 + wp.float64(2.0) * d3 * d4 * d9 + (d0*d0) * d10 + wp.float64(3.0) * (d1*d1) * d10 - (d3*d3) * d10 + (d4*d4) * d10 - wp.float64(2.0) * d1 * d3 * d12 + wp.float64(2.0) * d0 * d4 * d12 + wp.float64(2.0) * d0 * d3 * d13 + wp.float64(2.0) * d1 * d4 * d13
    if lane == 25:
        output[row, 57] = wp.float64(2.0) * d0 * d1 * d18 + wp.float64(2.0) * d3 * d4 * d18 + (d0*d0) * d19 + wp.float64(3.0) * (d1*d1) * d19 - (d3*d3) * d19 + (d4*d4) * d19 - wp.float64(2.0) * d1 * d3 * d21 + wp.float64(2.0) * d0 * d4 * d21 + wp.float64(2.0) * d0 * d3 * d22 + wp.float64(2.0) * d1 * d4 * d22
    if lane == 26:
        output[row, 58] = d1 * (d2*d2) + wp.float64(2.0) * d2 * d4 * d5 - d1 * (d5*d5) - d1 * (d6*d6) + wp.float64(2.0) * d0 * d6 * d7 + d1 * (d7*d7)
    if lane == 27:
        output[row, 59] = (d0*d0) * d1 + (d1*d1*d1) - d1 * (d3*d3) + wp.float64(2.0) * d0 * d3 * d4 + d1 * (d4*d4)
    if lane == 28:
        output[row, 60] = d11 * (d17*d17)
    if lane == 29:
        output[row, 61] = (d17*d17) * d20 + wp.float64(2.0) * d11 * d17 * d26
    if lane == 30:
        output[row, 62] = wp.float64(2.0) * d17 * d20 * d26 + d11 * (d26*d26)
    if lane == 31:
        output[row, 63] = d20 * (d26*d26)
    if lane == 0:
        output[row, 64] = (d11*d11*d11) + d11 * (d14*d14) - d11 * (d15*d15) - d11 * (d16*d16) + wp.float64(2.0) * d9 * d15 * d17 + wp.float64(2.0) * d10 * d16 * d17
    if lane == 1:
        output[row, 65] = wp.float64(2.0) * d15 * d17 * d18 + wp.float64(2.0) * d16 * d17 * d19 + wp.float64(3.0) * (d11*d11) * d20 + (d14*d14) * d20 - (d15*d15) * d20 - (d16*d16) * d20 + wp.float64(2.0) * d11 * d14 * d23 - wp.float64(2.0) * d11 * d15 * d24 + wp.float64(2.0) * d9 * d17 * d24 - wp.float64(2.0) * d11 * d16 * d25 + wp.float64(2.0) * d10 * d17 * d25 + wp.float64(2.0) * d9 * d15 * d26 + wp.float64(2.0) * d10 * d16 * d26
    if lane == 2:
        output[row, 66] = wp.float64(3.0) * d11 * (d20*d20) + wp.float64(2.0) * d14 * d20 * d23 + d11 * (d23*d23) + wp.float64(2.0) * d17 * d18 * d24 - wp.float64(2.0) * d15 * d20 * d24 - d11 * (d24*d24) + wp.float64(2.0) * d17 * d19 * d25 - wp.float64(2.0) * d16 * d20 * d25 - d11 * (d25*d25) + wp.float64(2.0) * d15 * d18 * d26 + wp.float64(2.0) * d16 * d19 * d26 + wp.float64(2.0) * d9 * d24 * d26 + wp.float64(2.0) * d10 * d25 * d26
    if lane == 3:
        output[row, 67] = (d20*d20*d20) + d20 * (d23*d23) - d20 * (d24*d24) - d20 * (d25*d25) + wp.float64(2.0) * d18 * d24 * d26 + wp.float64(2.0) * d19 * d25 * d26
    if lane == 4:
        output[row, 68] = wp.float64(2.0) * d8 * d11 * d17 + d2 * (d17*d17)
    if lane == 5:
        output[row, 69] = wp.float64(2.0) * d8 * d17 * d20 + wp.float64(2.0) * d8 * d11 * d26 + wp.float64(2.0) * d2 * d17 * d26
    if lane == 6:
        output[row, 70] = wp.float64(2.0) * d8 * d20 * d26 + d2 * (d26*d26)
    if lane == 7:
        output[row, 71] = (d9*d9) * d11 + (d10*d10) * d11 - d11 * (d12*d12) - d11 * (d13*d13) + wp.float64(2.0) * d9 * d12 * d14 + wp.float64(2.0) * d10 * d13 * d14
    if lane == 8:
        output[row, 72] = wp.float64(2.0) * d9 * d11 * d18 + wp.float64(2.0) * d12 * d14 * d18 + wp.float64(2.0) * d10 * d11 * d19 + wp.float64(2.0) * d13 * d14 * d19 + (d9*d9) * d20 + (d10*d10) * d20 - (d12*d12) * d20 - (d13*d13) * d20 - wp.float64(2.0) * d11 * d12 * d21 + wp.float64(2.0) * d9 * d14 * d21 - wp.float64(2.0) * d11 * d13 * d22 + wp.float64(2.0) * d10 * d14 * d22 + wp.float64(2.0) * d9 * d12 * d23 + wp.float64(2.0) * d10 * d13 * d23
    if lane == 9:
        output[row, 73] = d11 * (d18*d18) + d11 * (d19*d19) + wp.float64(2.0) * d9 * d18 * d20 + wp.float64(2.0) * d10 * d19 * d20 + wp.float64(2.0) * d14 * d18 * d21 - wp.float64(2.0) * d12 * d20 * d21 - d11 * (d21*d21) + wp.float64(2.0) * d14 * d19 * d22 - wp.float64(2.0) * d13 * d20 * d22 - d11 * (d22*d22) + wp.float64(2.0) * d12 * d18 * d23 + wp.float64(2.0) * d13 * d19 * d23 + wp.float64(2.0) * d9 * d21 * d23 + wp.float64(2.0) * d10 * d22 * d23
    if lane == 10:
        output[row, 74] = (d18*d18) * d20 + (d19*d19) * d20 - d20 * (d21*d21) - d20 * (d22*d22) + wp.float64(2.0) * d18 * d21 * d23 + wp.float64(2.0) * d19 * d22 * d23
    if lane == 11:
        output[row, 75] = wp.float64(3.0) * d2 * (d11*d11) + wp.float64(2.0) * d5 * d11 * d14 + d2 * (d14*d14) + wp.float64(2.0) * d8 * d9 * d15 - wp.float64(2.0) * d6 * d11 * d15 - d2 * (d15*d15) + wp.float64(2.0) * d8 * d10 * d16 - wp.float64(2.0) * d7 * d11 * d16 - d2 * (d16*d16) + wp.float64(2.0) * d6 * d9 * d17 + wp.float64(2.0) * d7 * d10 * d17 + wp.float64(2.0) * d0 * d15 * d17 + wp.float64(2.0) * d1 * d16 * d17
    if lane == 12:
        output[row, 76] = wp.float64(2.0) * d8 * d15 * d18 + wp.float64(2.0) * d6 * d17 * d18 + wp.float64(2.0) * d8 * d16 * d19 + wp.float64(2.0) * d7 * d17 * d19 + wp.float64(6.0) * d2 * d11 * d20 + wp.float64(2.0) * d5 * d14 * d20 - wp.float64(2.0) * d6 * d15 * d20 - wp.float64(2.0) * d7 * d16 * d20 + wp.float64(2.0) * d5 * d11 * d23 + wp.float64(2.0) * d2 * d14 * d23 + wp.float64(2.0) * d8 * d9 * d24 - wp.float64(2.0) * d6 * d11 * d24 - wp.float64(2.0) * d2 * d15 * d24 + wp.float64(2.0) * d0 * d17 * d24 + wp.float64(2.0) * d8 * d10 * d25 - wp.float64(2.0) * d7 * d11 * d25 - wp.float64(2.0) * d2 * d16 * d25 + wp.float64(2.0) * d1 * d17 * d25 + wp.float64(2.0) * d6 * d9 * d26 + wp.float64(2.0) * d7 * d10 * d26 + wp.float64(2.0) * d0 * d15 * d26 + wp.float64(2.0) * d1 * d16 * d26
    if lane == 13:
        output[row, 77] = wp.float64(3.0) * d2 * (d20*d20) + wp.float64(2.0) * d5 * d20 * d23 + d2 * (d23*d23) + wp.float64(2.0) * d8 * d18 * d24 - wp.float64(2.0) * d6 * d20 * d24 - d2 * (d24*d24) + wp.float64(2.0) * d8 * d19 * d25 - wp.float64(2.0) * d7 * d20 * d25 - d2 * (d25*d25) + wp.float64(2.0) * d6 * d18 * d26 + wp.float64(2.0) * d7 * d19 * d26 + wp.float64(2.0) * d0 * d24 * d26 + wp.float64(2.0) * d1 * d25 * d26
    if lane == 14:
        output[row, 78] = (d8*d8) * d11 + wp.float64(2.0) * d2 * d8 * d17
    if lane == 15:
        output[row, 79] = (d8*d8) * d20 + wp.float64(2.0) * d2 * d8 * d26
    if lane == 16:
        output[row, 80] = d2 * (d9*d9) + d2 * (d10*d10) + wp.float64(2.0) * d0 * d9 * d11 + wp.float64(2.0) * d1 * d10 * d11 + wp.float64(2.0) * d5 * d9 * d12 - wp.float64(2.0) * d3 * d11 * d12 - d2 * (d12*d12) + wp.float64(2.0) * d5 * d10 * d13 - wp.float64(2.0) * d4 * d11 * d13 - d2 * (d13*d13) + wp.float64(2.0) * d3 * d9 * d14 + wp.float64(2.0) * d4 * d10 * d14 + wp.float64(2.0) * d0 * d12 * d14 + wp.float64(2.0) * d1 * d13 * d14
    if lane == 17:
        output[row, 81] = wp.float64(2.0) * d2 * d9 * d18 + wp.float64(2.0) * d0 * d11 * d18 + wp.float64(2.0) * d5 * d12 * d18 + wp.float64(2.0) * d3 * d14 * d18 + wp.float64(2.0) * d2 * d10 * d19 + wp.float64(2.0) * d1 * d11 * d19 + wp.float64(2.0) * d5 * d13 * d19 + wp.float64(2.0) * d4 * d14 * d19 + wp.float64(2.0) * d0 * d9 * d20 + wp.float64(2.0) * d1 * d10 * d20 - wp.float64(2.0) * d3 * d12 * d20 - wp.float64(2.0) * d4 * d13 * d20 + wp.float64(2.0) * d5 * d9 * d21 - wp.float64(2.0) * d3 * d11 * d21 - wp.float64(2.0) * d2 * d12 * d21 + wp.float64(2.0) * d0 * d14 * d21 + wp.float64(2.0) * d5 * d10 * d22 - wp.float64(2.0) * d4 * d11 * d22 - wp.float64(2.0) * d2 * d13 * d22 + wp.float64(2.0) * d1 * d14 * d22 + wp.float64(2.0) * d3 * d9 * d23 + wp.float64(2.0) * d4 * d10 * d23 + wp.float64(2.0) * d0 * d12 * d23 + wp.float64(2.0) * d1 * d13 * d23
    if lane == 18:
        output[row, 82] = d2 * (d18*d18) + d2 * (d19*d19) + wp.float64(2.0) * d0 * d18 * d20 + wp.float64(2.0) * d1 * d19 * d20 + wp.float64(2.0) * d5 * d18 * d21 - wp.float64(2.0) * d3 * d20 * d21 - d2 * (d21*d21) + wp.float64(2.0) * d5 * d19 * d22 - wp.float64(2.0) * d4 * d20 * d22 - d2 * (d22*d22) + wp.float64(2.0) * d3 * d18 * d23 + wp.float64(2.0) * d4 * d19 * d23 + wp.float64(2.0) * d0 * d21 * d23 + wp.float64(2.0) * d1 * d22 * d23
    if lane == 19:
        output[row, 83] = wp.float64(2.0) * d6 * d8 * d9 + wp.float64(2.0) * d7 * d8 * d10 + wp.float64(3.0) * (d2*d2) * d11 + (d5*d5) * d11 - (d6*d6) * d11 - (d7*d7) * d11 + wp.float64(2.0) * d2 * d5 * d14 - wp.float64(2.0) * d2 * d6 * d15 + wp.float64(2.0) * d0 * d8 * d15 - wp.float64(2.0) * d2 * d7 * d16 + wp.float64(2.0) * d1 * d8 * d16 + wp.float64(2.0) * d0 * d6 * d17 + wp.float64(2.0) * d1 * d7 * d17
    if lane == 20:
        output[row, 84] = wp.float64(2.0) * d6 * d8 * d18 + wp.float64(2.0) * d7 * d8 * d19 + wp.float64(3.0) * (d2*d2) * d20 + (d5*d5) * d20 - (d6*d6) * d20 - (d7*d7) * d20 + wp.float64(2.0) * d2 * d5 * d23 - wp.float64(2.0) * d2 * d6 * d24 + wp.float64(2.0) * d0 * d8 * d24 - wp.float64(2.0) * d2 * d7 * d25 + wp.float64(2.0) * d1 * d8 * d25 + wp.float64(2.0) * d0 * d6 * d26 + wp.float64(2.0) * d1 * d7 * d26
    if lane == 21:
        output[row, 85] = d2 * (d8*d8)
    if lane == 22:
        output[row, 86] = wp.float64(2.0) * d0 * d2 * d9 + wp.float64(2.0) * d3 * d5 * d9 + wp.float64(2.0) * d1 * d2 * d10 + wp.float64(2.0) * d4 * d5 * d10 + (d0*d0) * d11 + (d1*d1) * d11 - (d3*d3) * d11 - (d4*d4) * d11 - wp.float64(2.0) * d2 * d3 * d12 + wp.float64(2.0) * d0 * d5 * d12 - wp.float64(2.0) * d2 * d4 * d13 + wp.float64(2.0) * d1 * d5 * d13 + wp.float64(2.0) * d0 * d3 * d14 + wp.float64(2.0) * d1 * d4 * d14
    if lane == 23:
        output[row, 87] = wp.float64(2.0) * d0 * d2 * d18 + wp.float64(2.0) * d3 * d5 * d18 + wp.float64(2.0) * d1 * d2 * d19 + wp.float64(2.0) * d4 * d5 * d19 + (d0*d0) * d20 + (d1*d1) * d20 - (d3*d3) * d20 - (d4*d4) * d20 - wp.float64(2.0) * d2 * d3 * d21 + wp.float64(2.0) * d0 * d5 * d21 - wp.float64(2.0) * d2 * d4 * d22 + wp.float64(2.0) * d1 * d5 * d22 + wp.float64(2.0) * d0 * d3 * d23 + wp.float64(2.0) * d1 * d4 * d23
    if lane == 24:
        output[row, 88] = (d2*d2*d2) + d2 * (d5*d5) - d2 * (d6*d6) - d2 * (d7*d7) + wp.float64(2.0) * d0 * d6 * d8 + wp.float64(2.0) * d1 * d7 * d8
    if lane == 25:
        output[row, 89] = (d0*d0) * d2 + (d1*d1) * d2 - d2 * (d3*d3) - d2 * (d4*d4) + wp.float64(2.0) * d0 * d3 * d5 + wp.float64(2.0) * d1 * d4 * d5
    if lane == 26:
        output[row, 90] = wp.float64(2.0) * d14 * d15 * d17 - d12 * (d17*d17)
    if lane == 27:
        output[row, 91] = -(d17*d17) * d21 + wp.float64(2.0) * d15 * d17 * d23 + wp.float64(2.0) * d14 * d17 * d24 + wp.float64(2.0) * d14 * d15 * d26 - wp.float64(2.0) * d12 * d17 * d26
    if lane == 28:
        output[row, 92] = wp.float64(2.0) * d17 * d23 * d24 - wp.float64(2.0) * d17 * d21 * d26 + wp.float64(2.0) * d15 * d23 * d26 + wp.float64(2.0) * d14 * d24 * d26 - d12 * (d26*d26)
    if lane == 29:
        output[row, 93] = wp.float64(2.0) * d23 * d24 * d26 - d21 * (d26*d26)
    if lane == 30:
        output[row, 94] = -(d11*d11) * d12 + wp.float64(2.0) * d9 * d11 * d14 + d12 * (d14*d14) + d12 * (d15*d15) + wp.float64(2.0) * d13 * d15 * d16 - d12 * (d16*d16)
    if lane == 31:
        output[row, 95] = wp.float64(2.0) * d11 * d14 * d18 - wp.float64(2.0) * d11 * d12 * d20 + wp.float64(2.0) * d9 * d14 * d20 - (d11*d11) * d21 + (d14*d14) * d21 + (d15*d15) * d21 - (d16*d16) * d21 + wp.float64(2.0) * d15 * d16 * d22 + wp.float64(2.0) * d9 * d11 * d23 + wp.float64(2.0) * d12 * d14 * d23 + wp.float64(2.0) * d12 * d15 * d24 + wp.float64(2.0) * d13 * d16 * d24 + wp.float64(2.0) * d13 * d15 * d25 - wp.float64(2.0) * d12 * d16 * d25
    if lane == 0:
        output[row, 96] = wp.float64(2.0) * d14 * d18 * d20 - d12 * (d20*d20) - wp.float64(2.0) * d11 * d20 * d21 + wp.float64(2.0) * d11 * d18 * d23 + wp.float64(2.0) * d9 * d20 * d23 + wp.float64(2.0) * d14 * d21 * d23 + d12 * (d23*d23) + wp.float64(2.0) * d15 * d21 * d24 + wp.float64(2.0) * d16 * d22 * d24 + d12 * (d24*d24) - wp.float64(2.0) * d16 * d21 * d25 + wp.float64(2.0) * d15 * d22 * d25 + wp.float64(2.0) * d13 * d24 * d25 - d12 * (d25*d25)
    if lane == 1:
        output[row, 97] = -(d20*d20) * d21 + wp.float64(2.0) * d18 * d20 * d23 + d21 * (d23*d23) + d21 * (d24*d24) + wp.float64(2.0) * d22 * d24 * d25 - d21 * (d25*d25)
    if lane == 2:
        output[row, 98] = wp.float64(2.0) * d8 * d14 * d15 - wp.float64(2.0) * d8 * d12 * d17 + wp.float64(2.0) * d6 * d14 * d17 + wp.float64(2.0) * d5 * d15 * d17 - d3 * (d17*d17)
    if lane == 3:
        output[row, 99] = -wp.float64(2.0) * d8 * d17 * d21 + wp.float64(2.0) * d8 * d15 * d23 + wp.float64(2.0) * d6 * d17 * d23 + wp.float64(2.0) * d8 * d14 * d24 + wp.float64(2.0) * d5 * d17 * d24 - wp.float64(2.0) * d8 * d12 * d26 + wp.float64(2.0) * d6 * d14 * d26 + wp.float64(2.0) * d5 * d15 * d26 - wp.float64(2.0) * d3 * d17 * d26
    if lane == 4:
        output[row, 100] = wp.float64(2.0) * d8 * d23 * d24 - wp.float64(2.0) * d8 * d21 * d26 + wp.float64(2.0) * d6 * d23 * d26 + wp.float64(2.0) * d5 * d24 * d26 - d3 * (d26*d26)
    if lane == 5:
        output[row, 101] = (d9*d9) * d12 - (d10*d10) * d12 + (d12*d12*d12) + wp.float64(2.0) * d9 * d10 * d13 + d12 * (d13*d13)
    if lane == 6:
        output[row, 102] = wp.float64(2.0) * d9 * d12 * d18 + wp.float64(2.0) * d10 * d13 * d18 - wp.float64(2.0) * d10 * d12 * d19 + wp.float64(2.0) * d9 * d13 * d19 + (d9*d9) * d21 - (d10*d10) * d21 + wp.float64(3.0) * (d12*d12) * d21 + (d13*d13) * d21 + wp.float64(2.0) * d9 * d10 * d22 + wp.float64(2.0) * d12 * d13 * d22
    if lane == 7:
        output[row, 103] = d12 * (d18*d18) + wp.float64(2.0) * d13 * d18 * d19 - d12 * (d19*d19) + wp.float64(2.0) * d9 * d18 * d21 - wp.float64(2.0) * d10 * d19 * d21 + wp.float64(3.0) * d12 * (d21*d21) + wp.float64(2.0) * d10 * d18 * d22 + wp.float64(2.0) * d9 * d19 * d22 + wp.float64(2.0) * d13 * d21 * d22 + d12 * (d22*d22)
    if lane == 8:
        output[row, 104] = (d18*d18) * d21 - (d19*d19) * d21 + (d21*d21*d21) + wp.float64(2.0) * d18 * d19 * d22 + d21 * (d22*d22)
    if lane == 9:
        output[row, 105] = wp.float64(2.0) * d5 * d9 * d11 - d3 * (d11*d11) - wp.float64(2.0) * d2 * d11 * d12 + wp.float64(2.0) * d2 * d9 * d14 + wp.float64(2.0) * d0 * d11 * d14 + wp.float64(2.0) * d5 * d12 * d14 + d3 * (d14*d14) + wp.float64(2.0) * d6 * d12 * d15 + wp.float64(2.0) * d7 * d13 * d15 + d3 * (d15*d15) - wp.float64(2.0) * d7 * d12 * d16 + wp.float64(2.0) * d6 * d13 * d16 + wp.float64(2.0) * d4 * d15 * d16 - d3 * (d16*d16)
    if lane == 10:
        output[row, 106] = wp.float64(2.0) * d5 * d11 * d18 + wp.float64(2.0) * d2 * d14 * d18 + wp.float64(2.0) * d5 * d9 * d20 - wp.float64(2.0) * d3 * d11 * d20 - wp.float64(2.0) * d2 * d12 * d20 + wp.float64(2.0) * d0 * d14 * d20 - wp.float64(2.0) * d2 * d11 * d21 + wp.float64(2.0) * d5 * d14 * d21 + wp.float64(2.0) * d6 * d15 * d21 - wp.float64(2.0) * d7 * d16 * d21 + wp.float64(2.0) * d7 * d15 * d22 + wp.float64(2.0) * d6 * d16 * d22 + wp.float64(2.0) * d2 * d9 * d23 + wp.float64(2.0) * d0 * d11 * d23 + wp.float64(2.0) * d5 * d12 * d23 + wp.float64(2.0) * d3 * d14 * d23 + wp.float64(2.0) * d6 * d12 * d24 + wp.float64(2.0) * d7 * d13 * d24 + wp.float64(2.0) * d3 * d15 * d24 + wp.float64(2.0) * d4 * d16 * d24 - wp.float64(2.0) * d7 * d12 * d25 + wp.float64(2.0) * d6 * d13 * d25 + wp.float64(2.0) * d4 * d15 * d25 - wp.float64(2.0) * d3 * d16 * d25
    if lane == 11:
        output[row, 107] = wp.float64(2.0) * d5 * d18 * d20 - d3 * (d20*d20) - wp.float64(2.0) * d2 * d20 * d21 + wp.float64(2.0) * d2 * d18 * d23 + wp.float64(2.0) * d0 * d20 * d23 + wp.float64(2.0) * d5 * d21 * d23 + d3 * (d23*d23) + wp.float64(2.0) * d6 * d21 * d24 + wp.float64(2.0) * d7 * d22 * d24 + d3 * (d24*d24) - wp.float64(2.0) * d7 * d21 * d25 + wp.float64(2.0) * d6 * d22 * d25 + wp.float64(2.0) * d4 * d24 * d25 - d3 * (d25*d25)
    if lane == 12:
        output[row, 108] = -(d8*d8) * d12 + wp.float64(2.0) * d6 * d8 * d14 + wp.float64(2.0) * d5 * d8 * d15 + wp.float64(2.0) * d5 * d6 * d17 - wp.float64(2.0) * d3 * d8 * d17
    if lane == 13:
        output[row, 109] = -(d8*d8) * d21 + wp.float64(2.0) * d6 * d8 * d23 + wp.float64(2.0) * d5 * d8 * d24 + wp.float64(2.0) * d5 * d6 * d26 - wp.float64(2.0) * d3 * d8 * d26
    if lane == 14:
        output[row, 110] = d3 * (d9*d9) + wp.float64(2.0) * d4 * d9 * d10 - d3 * (d10*d10) + wp.float64(2.0) * d0 * d9 * d12 - wp.float64(2.0) * d1 * d10 * d12 + wp.float64(3.0) * d3 * (d12*d12) + wp.float64(2.0) * d1 * d9 * d13 + wp.float64(2.0) * d0 * d10 * d13 + wp.float64(2.0) * d4 * d12 * d13 + d3 * (d13*d13)
    if lane == 15:
        output[row, 111] = wp.float64(2.0) * d3 * d9 * d18 + wp.float64(2.0) * d4 * d10 * d18 + wp.float64(2.0) * d0 * d12 * d18 + wp.float64(2.0) * d1 * d13 * d18 + wp.float64(2.0) * d4 * d9 * d19 - wp.float64(2.0) * d3 * d10 * d19 - wp.float64(2.0) * d1 * d12 * d19 + wp.float64(2.0) * d0 * d13 * d19 + wp.float64(2.0) * d0 * d9 * d21 - wp.float64(2.0) * d1 * d10 * d21 + wp.float64(6.0) * d3 * d12 * d21 + wp.float64(2.0) * d4 * d13 * d21 + wp.float64(2.0) * d1 * d9 * d22 + wp.float64(2.0) * d0 * d10 * d22 + wp.float64(2.0) * d4 * d12 * d22 + wp.float64(2.0) * d3 * d13 * d22
    if lane == 16:
        output[row, 112] = d3 * (d18*d18) + wp.float64(2.0) * d4 * d18 * d19 - d3 * (d19*d19) + wp.float64(2.0) * d0 * d18 * d21 - wp.float64(2.0) * d1 * d19 * d21 + wp.float64(3.0) * d3 * (d21*d21) + wp.float64(2.0) * d1 * d18 * d22 + wp.float64(2.0) * d0 * d19 * d22 + wp.float64(2.0) * d4 * d21 * d22 + d3 * (d22*d22)
    if lane == 17:
        output[row, 113] = wp.float64(2.0) * d2 * d5 * d9 - wp.float64(2.0) * d2 * d3 * d11 + wp.float64(2.0) * d0 * d5 * d11 - (d2*d2) * d12 + (d5*d5) * d12 + (d6*d6) * d12 - (d7*d7) * d12 + wp.float64(2.0) * d6 * d7 * d13 + wp.float64(2.0) * d0 * d2 * d14 + wp.float64(2.0) * d3 * d5 * d14 + wp.float64(2.0) * d3 * d6 * d15 + wp.float64(2.0) * d4 * d7 * d15 + wp.float64(2.0) * d4 * d6 * d16 - wp.float64(2.0) * d3 * d7 * d16
    if lane == 18:
        output[row, 114] = wp.float64(2.0) * d2 * d5 * d18 - wp.float64(2.0) * d2 * d3 * d20 + wp.float64(2.0) * d0 * d5 * d20 - (d2*d2) * d21 + (d5*d5) * d21 + (d6*d6) * d21 - (d7*d7) * d21 + wp.float64(2.0) * d6 * d7 * d22 + wp.float64(2.0) * d0 * d2 * d23 + wp.float64(2.0) * d3 * d5 * d23 + wp.float64(2.0) * d3 * d6 * d24 + wp.float64(2.0) * d4 * d7 * d24 + wp.float64(2.0) * d4 * d6 * d25 - wp.float64(2.0) * d3 * d7 * d25
    if lane == 19:
        output[row, 115] = wp.float64(2.0) * d5 * d6 * d8 - d3 * (d8*d8)
    if lane == 20:
        output[row, 116] = wp.float64(2.0) * d0 * d3 * d9 + wp.float64(2.0) * d1 * d4 * d9 - wp.float64(2.0) * d1 * d3 * d10 + wp.float64(2.0) * d0 * d4 * d10 + (d0*d0) * d12 - (d1*d1) * d12 + wp.float64(3.0) * (d3*d3) * d12 + (d4*d4) * d12 + wp.float64(2.0) * d0 * d1 * d13 + wp.float64(2.0) * d3 * d4 * d13
    if lane == 21:
        output[row, 117] = wp.float64(2.0) * d0 * d3 * d18 + wp.float64(2.0) * d1 * d4 * d18 - wp.float64(2.0) * d1 * d3 * d19 + wp.float64(2.0) * d0 * d4 * d19 + (d0*d0) * d21 - (d1*d1) * d21 + wp.float64(3.0) * (d3*d3) * d21 + (d4*d4) * d21 + wp.float64(2.0) * d0 * d1 * d22 + wp.float64(2.0) * d3 * d4 * d22
    if lane == 22:
        output[row, 118] = -(d2*d2) * d3 + wp.float64(2.0) * d0 * d2 * d5 + d3 * (d5*d5) + d3 * (d6*d6) + wp.float64(2.0) * d4 * d6 * d7 - d3 * (d7*d7)
    if lane == 23:
        output[row, 119] = (d0*d0) * d3 - (d1*d1) * d3 + (d3*d3*d3) + wp.float64(2.0) * d0 * d1 * d4 + d3 * (d4*d4)
    if lane == 24:
        output[row, 120] = wp.float64(2.0) * d14 * d16 * d17 - d13 * (d17*d17)
    if lane == 25:
        output[row, 121] = -(d17*d17) * d22 + wp.float64(2.0) * d16 * d17 * d23 + wp.float64(2.0) * d14 * d17 * d25 + wp.float64(2.0) * d14 * d16 * d26 - wp.float64(2.0) * d13 * d17 * d26
    if lane == 26:
        output[row, 122] = wp.float64(2.0) * d17 * d23 * d25 - wp.float64(2.0) * d17 * d22 * d26 + wp.float64(2.0) * d16 * d23 * d26 + wp.float64(2.0) * d14 * d25 * d26 - d13 * (d26*d26)
    if lane == 27:
        output[row, 123] = wp.float64(2.0) * d23 * d25 * d26 - d22 * (d26*d26)
    if lane == 28:
        output[row, 124] = -(d11*d11) * d13 + wp.float64(2.0) * d10 * d11 * d14 + d13 * (d14*d14) - d13 * (d15*d15) + wp.float64(2.0) * d12 * d15 * d16 + d13 * (d16*d16)
    if lane == 29:
        output[row, 125] = wp.float64(2.0) * d11 * d14 * d19 - wp.float64(2.0) * d11 * d13 * d20 + wp.float64(2.0) * d10 * d14 * d20 + wp.float64(2.0) * d15 * d16 * d21 - (d11*d11) * d22 + (d14*d14) * d22 - (d15*d15) * d22 + (d16*d16) * d22 + wp.float64(2.0) * d10 * d11 * d23 + wp.float64(2.0) * d13 * d14 * d23 - wp.float64(2.0) * d13 * d15 * d24 + wp.float64(2.0) * d12 * d16 * d24 + wp.float64(2.0) * d12 * d15 * d25 + wp.float64(2.0) * d13 * d16 * d25
    if lane == 30:
        output[row, 126] = wp.float64(2.0) * d14 * d19 * d20 - d13 * (d20*d20) - wp.float64(2.0) * d11 * d20 * d22 + wp.float64(2.0) * d11 * d19 * d23 + wp.float64(2.0) * d10 * d20 * d23 + wp.float64(2.0) * d14 * d22 * d23 + d13 * (d23*d23) + wp.float64(2.0) * d16 * d21 * d24 - wp.float64(2.0) * d15 * d22 * d24 - d13 * (d24*d24) + wp.float64(2.0) * d15 * d21 * d25 + wp.float64(2.0) * d16 * d22 * d25 + wp.float64(2.0) * d12 * d24 * d25 + d13 * (d25*d25)
    if lane == 31:
        output[row, 127] = -(d20*d20) * d22 + wp.float64(2.0) * d19 * d20 * d23 + d22 * (d23*d23) - d22 * (d24*d24) + wp.float64(2.0) * d21 * d24 * d25 + d22 * (d25*d25)
    if lane == 0:
        output[row, 128] = wp.float64(2.0) * d8 * d14 * d16 - wp.float64(2.0) * d8 * d13 * d17 + wp.float64(2.0) * d7 * d14 * d17 + wp.float64(2.0) * d5 * d16 * d17 - d4 * (d17*d17)
    if lane == 1:
        output[row, 129] = -wp.float64(2.0) * d8 * d17 * d22 + wp.float64(2.0) * d8 * d16 * d23 + wp.float64(2.0) * d7 * d17 * d23 + wp.float64(2.0) * d8 * d14 * d25 + wp.float64(2.0) * d5 * d17 * d25 - wp.float64(2.0) * d8 * d13 * d26 + wp.float64(2.0) * d7 * d14 * d26 + wp.float64(2.0) * d5 * d16 * d26 - wp.float64(2.0) * d4 * d17 * d26
    if lane == 2:
        output[row, 130] = wp.float64(2.0) * d8 * d23 * d25 - wp.float64(2.0) * d8 * d22 * d26 + wp.float64(2.0) * d7 * d23 * d26 + wp.float64(2.0) * d5 * d25 * d26 - d4 * (d26*d26)
    if lane == 3:
        output[row, 131] = wp.float64(2.0) * d9 * d10 * d12 - (d9*d9) * d13 + (d10*d10) * d13 + (d12*d12) * d13 + (d13*d13*d13)
    if lane == 4:
        output[row, 132] = wp.float64(2.0) * d10 * d12 * d18 - wp.float64(2.0) * d9 * d13 * d18 + wp.float64(2.0) * d9 * d12 * d19 + wp.float64(2.0) * d10 * d13 * d19 + wp.float64(2.0) * d9 * d10 * d21 + wp.float64(2.0) * d12 * d13 * d21 - (d9*d9) * d22 + (d10*d10) * d22 + (d12*d12) * d22 + wp.float64(3.0) * (d13*d13) * d22
    if lane == 5:
        output[row, 133] = -d13 * (d18*d18) + wp.float64(2.0) * d12 * d18 * d19 + d13 * (d19*d19) + wp.float64(2.0) * d10 * d18 * d21 + wp.float64(2.0) * d9 * d19 * d21 + d13 * (d21*d21) - wp.float64(2.0) * d9 * d18 * d22 + wp.float64(2.0) * d10 * d19 * d22 + wp.float64(2.0) * d12 * d21 * d22 + wp.float64(3.0) * d13 * (d22*d22)
    if lane == 6:
        output[row, 134] = wp.float64(2.0) * d18 * d19 * d21 - (d18*d18) * d22 + (d19*d19) * d22 + (d21*d21) * d22 + (d22*d22*d22)
    if lane == 7:
        output[row, 135] = wp.float64(2.0) * d5 * d10 * d11 - d4 * (d11*d11) - wp.float64(2.0) * d2 * d11 * d13 + wp.float64(2.0) * d2 * d10 * d14 + wp.float64(2.0) * d1 * d11 * d14 + wp.float64(2.0) * d5 * d13 * d14 + d4 * (d14*d14) + wp.float64(2.0) * d7 * d12 * d15 - wp.float64(2.0) * d6 * d13 * d15 - d4 * (d15*d15) + wp.float64(2.0) * d6 * d12 * d16 + wp.float64(2.0) * d7 * d13 * d16 + wp.float64(2.0) * d3 * d15 * d16 + d4 * (d16*d16)
    if lane == 8:
        output[row, 136] = wp.float64(2.0) * d5 * d11 * d19 + wp.float64(2.0) * d2 * d14 * d19 + wp.float64(2.0) * d5 * d10 * d20 - wp.float64(2.0) * d4 * d11 * d20 - wp.float64(2.0) * d2 * d13 * d20 + wp.float64(2.0) * d1 * d14 * d20 + wp.float64(2.0) * d7 * d15 * d21 + wp.float64(2.0) * d6 * d16 * d21 - wp.float64(2.0) * d2 * d11 * d22 + wp.float64(2.0) * d5 * d14 * d22 - wp.float64(2.0) * d6 * d15 * d22 + wp.float64(2.0) * d7 * d16 * d22 + wp.float64(2.0) * d2 * d10 * d23 + wp.float64(2.0) * d1 * d11 * d23 + wp.float64(2.0) * d5 * d13 * d23 + wp.float64(2.0) * d4 * d14 * d23 + wp.float64(2.0) * d7 * d12 * d24 - wp.float64(2.0) * d6 * d13 * d24 - wp.float64(2.0) * d4 * d15 * d24 + wp.float64(2.0) * d3 * d16 * d24 + wp.float64(2.0) * d6 * d12 * d25 + wp.float64(2.0) * d7 * d13 * d25 + wp.float64(2.0) * d3 * d15 * d25 + wp.float64(2.0) * d4 * d16 * d25
    if lane == 9:
        output[row, 137] = wp.float64(2.0) * d5 * d19 * d20 - d4 * (d20*d20) - wp.float64(2.0) * d2 * d20 * d22 + wp.float64(2.0) * d2 * d19 * d23 + wp.float64(2.0) * d1 * d20 * d23 + wp.float64(2.0) * d5 * d22 * d23 + d4 * (d23*d23) + wp.float64(2.0) * d7 * d21 * d24 - wp.float64(2.0) * d6 * d22 * d24 - d4 * (d24*d24) + wp.float64(2.0) * d6 * d21 * d25 + wp.float64(2.0) * d7 * d22 * d25 + wp.float64(2.0) * d3 * d24 * d25 + d4 * (d25*d25)
    if lane == 10:
        output[row, 138] = -(d8*d8) * d13 + wp.float64(2.0) * d7 * d8 * d14 + wp.float64(2.0) * d5 * d8 * d16 + wp.float64(2.0) * d5 * d7 * d17 - wp.float64(2.0) * d4 * d8 * d17
    if lane == 11:
        output[row, 139] = -(d8*d8) * d22 + wp.float64(2.0) * d7 * d8 * d23 + wp.float64(2.0) * d5 * d8 * d25 + wp.float64(2.0) * d5 * d7 * d26 - wp.float64(2.0) * d4 * d8 * d26
    if lane == 12:
        output[row, 140] = -d4 * (d9*d9) + wp.float64(2.0) * d3 * d9 * d10 + d4 * (d10*d10) + wp.float64(2.0) * d1 * d9 * d12 + wp.float64(2.0) * d0 * d10 * d12 + d4 * (d12*d12) - wp.float64(2.0) * d0 * d9 * d13 + wp.float64(2.0) * d1 * d10 * d13 + wp.float64(2.0) * d3 * d12 * d13 + wp.float64(3.0) * d4 * (d13*d13)
    if lane == 13:
        output[row, 141] = -wp.float64(2.0) * d4 * d9 * d18 + wp.float64(2.0) * d3 * d10 * d18 + wp.float64(2.0) * d1 * d12 * d18 - wp.float64(2.0) * d0 * d13 * d18 + wp.float64(2.0) * d3 * d9 * d19 + wp.float64(2.0) * d4 * d10 * d19 + wp.float64(2.0) * d0 * d12 * d19 + wp.float64(2.0) * d1 * d13 * d19 + wp.float64(2.0) * d1 * d9 * d21 + wp.float64(2.0) * d0 * d10 * d21 + wp.float64(2.0) * d4 * d12 * d21 + wp.float64(2.0) * d3 * d13 * d21 - wp.float64(2.0) * d0 * d9 * d22 + wp.float64(2.0) * d1 * d10 * d22 + wp.float64(2.0) * d3 * d12 * d22 + wp.float64(6.0) * d4 * d13 * d22
    if lane == 14:
        output[row, 142] = -d4 * (d18*d18) + wp.float64(2.0) * d3 * d18 * d19 + d4 * (d19*d19) + wp.float64(2.0) * d1 * d18 * d21 + wp.float64(2.0) * d0 * d19 * d21 + d4 * (d21*d21) - wp.float64(2.0) * d0 * d18 * d22 + wp.float64(2.0) * d1 * d19 * d22 + wp.float64(2.0) * d3 * d21 * d22 + wp.float64(3.0) * d4 * (d22*d22)
    if lane == 15:
        output[row, 143] = wp.float64(2.0) * d2 * d5 * d10 - wp.float64(2.0) * d2 * d4 * d11 + wp.float64(2.0) * d1 * d5 * d11 + wp.float64(2.0) * d6 * d7 * d12 - (d2*d2) * d13 + (d5*d5) * d13 - (d6*d6) * d13 + (d7*d7) * d13 + wp.float64(2.0) * d1 * d2 * d14 + wp.float64(2.0) * d4 * d5 * d14 - wp.float64(2.0) * d4 * d6 * d15 + wp.float64(2.0) * d3 * d7 * d15 + wp.float64(2.0) * d3 * d6 * d16 + wp.float64(2.0) * d4 * d7 * d16
    if lane == 16:
        output[row, 144] = wp.float64(2.0) * d2 * d5 * d19 - wp.float64(2.0) * d2 * d4 * d20 + wp.float64(2.0) * d1 * d5 * d20 + wp.float64(2.0) * d6 * d7 * d21 - (d2*d2) * d22 + (d5*d5) * d22 - (d6*d6) * d22 + (d7*d7) * d22 + wp.float64(2.0) * d1 * d2 * d23 + wp.float64(2.0) * d4 * d5 * d23 - wp.float64(2.0) * d4 * d6 * d24 + wp.float64(2.0) * d3 * d7 * d24 + wp.float64(2.0) * d3 * d6 * d25 + wp.float64(2.0) * d4 * d7 * d25
    if lane == 17:
        output[row, 145] = wp.float64(2.0) * d5 * d7 * d8 - d4 * (d8*d8)
    if lane == 18:
        output[row, 146] = wp.float64(2.0) * d1 * d3 * d9 - wp.float64(2.0) * d0 * d4 * d9 + wp.float64(2.0) * d0 * d3 * d10 + wp.float64(2.0) * d1 * d4 * d10 + wp.float64(2.0) * d0 * d1 * d12 + wp.float64(2.0) * d3 * d4 * d12 - (d0*d0) * d13 + (d1*d1) * d13 + (d3*d3) * d13 + wp.float64(3.0) * (d4*d4) * d13
    if lane == 19:
        output[row, 147] = wp.float64(2.0) * d1 * d3 * d18 - wp.float64(2.0) * d0 * d4 * d18 + wp.float64(2.0) * d0 * d3 * d19 + wp.float64(2.0) * d1 * d4 * d19 + wp.float64(2.0) * d0 * d1 * d21 + wp.float64(2.0) * d3 * d4 * d21 - (d0*d0) * d22 + (d1*d1) * d22 + (d3*d3) * d22 + wp.float64(3.0) * (d4*d4) * d22
    if lane == 20:
        output[row, 148] = -(d2*d2) * d4 + wp.float64(2.0) * d1 * d2 * d5 + d4 * (d5*d5) - d4 * (d6*d6) + wp.float64(2.0) * d3 * d6 * d7 + d4 * (d7*d7)
    if lane == 21:
        output[row, 149] = wp.float64(2.0) * d0 * d1 * d3 - (d0*d0) * d4 + (d1*d1) * d4 + (d3*d3) * d4 + (d4*d4*d4)
    if lane == 22:
        output[row, 150] = d14 * (d17*d17)
    if lane == 23:
        output[row, 151] = (d17*d17) * d23 + wp.float64(2.0) * d14 * d17 * d26
    if lane == 24:
        output[row, 152] = wp.float64(2.0) * d17 * d23 * d26 + d14 * (d26*d26)
    if lane == 25:
        output[row, 153] = d23 * (d26*d26)
    if lane == 26:
        output[row, 154] = (d11*d11) * d14 + (d14*d14*d14) - d14 * (d15*d15) - d14 * (d16*d16) + wp.float64(2.0) * d12 * d15 * d17 + wp.float64(2.0) * d13 * d16 * d17
    if lane == 27:
        output[row, 155] = wp.float64(2.0) * d11 * d14 * d20 + wp.float64(2.0) * d15 * d17 * d21 + wp.float64(2.0) * d16 * d17 * d22 + (d11*d11) * d23 + wp.float64(3.0) * (d14*d14) * d23 - (d15*d15) * d23 - (d16*d16) * d23 - wp.float64(2.0) * d14 * d15 * d24 + wp.float64(2.0) * d12 * d17 * d24 - wp.float64(2.0) * d14 * d16 * d25 + wp.float64(2.0) * d13 * d17 * d25 + wp.float64(2.0) * d12 * d15 * d26 + wp.float64(2.0) * d13 * d16 * d26
    if lane == 28:
        output[row, 156] = d14 * (d20*d20) + wp.float64(2.0) * d11 * d20 * d23 + wp.float64(3.0) * d14 * (d23*d23) + wp.float64(2.0) * d17 * d21 * d24 - wp.float64(2.0) * d15 * d23 * d24 - d14 * (d24*d24) + wp.float64(2.0) * d17 * d22 * d25 - wp.float64(2.0) * d16 * d23 * d25 - d14 * (d25*d25) + wp.float64(2.0) * d15 * d21 * d26 + wp.float64(2.0) * d16 * d22 * d26 + wp.float64(2.0) * d12 * d24 * d26 + wp.float64(2.0) * d13 * d25 * d26
    if lane == 29:
        output[row, 157] = (d20*d20) * d23 + (d23*d23*d23) - d23 * (d24*d24) - d23 * (d25*d25) + wp.float64(2.0) * d21 * d24 * d26 + wp.float64(2.0) * d22 * d25 * d26
    if lane == 30:
        output[row, 158] = wp.float64(2.0) * d8 * d14 * d17 + d5 * (d17*d17)
    if lane == 31:
        output[row, 159] = wp.float64(2.0) * d8 * d17 * d23 + wp.float64(2.0) * d8 * d14 * d26 + wp.float64(2.0) * d5 * d17 * d26
    if lane == 0:
        output[row, 160] = wp.float64(2.0) * d8 * d23 * d26 + d5 * (d26*d26)
    if lane == 1:
        output[row, 161] = wp.float64(2.0) * d9 * d11 * d12 + wp.float64(2.0) * d10 * d11 * d13 - (d9*d9) * d14 - (d10*d10) * d14 + (d12*d12) * d14 + (d13*d13) * d14
    if lane == 2:
        output[row, 162] = wp.float64(2.0) * d11 * d12 * d18 - wp.float64(2.0) * d9 * d14 * d18 + wp.float64(2.0) * d11 * d13 * d19 - wp.float64(2.0) * d10 * d14 * d19 + wp.float64(2.0) * d9 * d12 * d20 + wp.float64(2.0) * d10 * d13 * d20 + wp.float64(2.0) * d9 * d11 * d21 + wp.float64(2.0) * d12 * d14 * d21 + wp.float64(2.0) * d10 * d11 * d22 + wp.float64(2.0) * d13 * d14 * d22 - (d9*d9) * d23 - (d10*d10) * d23 + (d12*d12) * d23 + (d13*d13) * d23
    if lane == 3:
        output[row, 163] = -d14 * (d18*d18) - d14 * (d19*d19) + wp.float64(2.0) * d12 * d18 * d20 + wp.float64(2.0) * d13 * d19 * d20 + wp.float64(2.0) * d11 * d18 * d21 + wp.float64(2.0) * d9 * d20 * d21 + d14 * (d21*d21) + wp.float64(2.0) * d11 * d19 * d22 + wp.float64(2.0) * d10 * d20 * d22 + d14 * (d22*d22) - wp.float64(2.0) * d9 * d18 * d23 - wp.float64(2.0) * d10 * d19 * d23 + wp.float64(2.0) * d12 * d21 * d23 + wp.float64(2.0) * d13 * d22 * d23
    if lane == 4:
        output[row, 164] = wp.float64(2.0) * d18 * d20 * d21 + wp.float64(2.0) * d19 * d20 * d22 - (d18*d18) * d23 - (d19*d19) * d23 + (d21*d21) * d23 + (d22*d22) * d23
    if lane == 5:
        output[row, 165] = d5 * (d11*d11) + wp.float64(2.0) * d2 * d11 * d14 + wp.float64(3.0) * d5 * (d14*d14) + wp.float64(2.0) * d8 * d12 * d15 - wp.float64(2.0) * d6 * d14 * d15 - d5 * (d15*d15) + wp.float64(2.0) * d8 * d13 * d16 - wp.float64(2.0) * d7 * d14 * d16 - d5 * (d16*d16) + wp.float64(2.0) * d6 * d12 * d17 + wp.float64(2.0) * d7 * d13 * d17 + wp.float64(2.0) * d3 * d15 * d17 + wp.float64(2.0) * d4 * d16 * d17
    if lane == 6:
        output[row, 166] = wp.float64(2.0) * d5 * d11 * d20 + wp.float64(2.0) * d2 * d14 * d20 + wp.float64(2.0) * d8 * d15 * d21 + wp.float64(2.0) * d6 * d17 * d21 + wp.float64(2.0) * d8 * d16 * d22 + wp.float64(2.0) * d7 * d17 * d22 + wp.float64(2.0) * d2 * d11 * d23 + wp.float64(6.0) * d5 * d14 * d23 - wp.float64(2.0) * d6 * d15 * d23 - wp.float64(2.0) * d7 * d16 * d23 + wp.float64(2.0) * d8 * d12 * d24 - wp.float64(2.0) * d6 * d14 * d24 - wp.float64(2.0) * d5 * d15 * d24 + wp.float64(2.0) * d3 * d17 * d24 + wp.float64(2.0) * d8 * d13 * d25 - wp.float64(2.0) * d7 * d14 * d25 - wp.float64(2.0) * d5 * d16 * d25 + wp.float64(2.0) * d4 * d17 * d25 + wp.float64(2.0) * d6 * d12 * d26 + wp.float64(2.0) * d7 * d13 * d26 + wp.float64(2.0) * d3 * d15 * d26 + wp.float64(2.0) * d4 * d16 * d26
    if lane == 7:
        output[row, 167] = d5 * (d20*d20) + wp.float64(2.0) * d2 * d20 * d23 + wp.float64(3.0) * d5 * (d23*d23) + wp.float64(2.0) * d8 * d21 * d24 - wp.float64(2.0) * d6 * d23 * d24 - d5 * (d24*d24) + wp.float64(2.0) * d8 * d22 * d25 - wp.float64(2.0) * d7 * d23 * d25 - d5 * (d25*d25) + wp.float64(2.0) * d6 * d21 * d26 + wp.float64(2.0) * d7 * d22 * d26 + wp.float64(2.0) * d3 * d24 * d26 + wp.float64(2.0) * d4 * d25 * d26
    if lane == 8:
        output[row, 168] = (d8*d8) * d14 + wp.float64(2.0) * d5 * d8 * d17
    if lane == 9:
        output[row, 169] = (d8*d8) * d23 + wp.float64(2.0) * d5 * d8 * d26
    if lane == 10:
        output[row, 170] = -d5 * (d9*d9) - d5 * (d10*d10) + wp.float64(2.0) * d3 * d9 * d11 + wp.float64(2.0) * d4 * d10 * d11 + wp.float64(2.0) * d2 * d9 * d12 + wp.float64(2.0) * d0 * d11 * d12 + d5 * (d12*d12) + wp.float64(2.0) * d2 * d10 * d13 + wp.float64(2.0) * d1 * d11 * d13 + d5 * (d13*d13) - wp.float64(2.0) * d0 * d9 * d14 - wp.float64(2.0) * d1 * d10 * d14 + wp.float64(2.0) * d3 * d12 * d14 + wp.float64(2.0) * d4 * d13 * d14
    if lane == 11:
        output[row, 171] = -wp.float64(2.0) * d5 * d9 * d18 + wp.float64(2.0) * d3 * d11 * d18 + wp.float64(2.0) * d2 * d12 * d18 - wp.float64(2.0) * d0 * d14 * d18 - wp.float64(2.0) * d5 * d10 * d19 + wp.float64(2.0) * d4 * d11 * d19 + wp.float64(2.0) * d2 * d13 * d19 - wp.float64(2.0) * d1 * d14 * d19 + wp.float64(2.0) * d3 * d9 * d20 + wp.float64(2.0) * d4 * d10 * d20 + wp.float64(2.0) * d0 * d12 * d20 + wp.float64(2.0) * d1 * d13 * d20 + wp.float64(2.0) * d2 * d9 * d21 + wp.float64(2.0) * d0 * d11 * d21 + wp.float64(2.0) * d5 * d12 * d21 + wp.float64(2.0) * d3 * d14 * d21 + wp.float64(2.0) * d2 * d10 * d22 + wp.float64(2.0) * d1 * d11 * d22 + wp.float64(2.0) * d5 * d13 * d22 + wp.float64(2.0) * d4 * d14 * d22 - wp.float64(2.0) * d0 * d9 * d23 - wp.float64(2.0) * d1 * d10 * d23 + wp.float64(2.0) * d3 * d12 * d23 + wp.float64(2.0) * d4 * d13 * d23
    if lane == 12:
        output[row, 172] = -d5 * (d18*d18) - d5 * (d19*d19) + wp.float64(2.0) * d3 * d18 * d20 + wp.float64(2.0) * d4 * d19 * d20 + wp.float64(2.0) * d2 * d18 * d21 + wp.float64(2.0) * d0 * d20 * d21 + d5 * (d21*d21) + wp.float64(2.0) * d2 * d19 * d22 + wp.float64(2.0) * d1 * d20 * d22 + d5 * (d22*d22) - wp.float64(2.0) * d0 * d18 * d23 - wp.float64(2.0) * d1 * d19 * d23 + wp.float64(2.0) * d3 * d21 * d23 + wp.float64(2.0) * d4 * d22 * d23
    if lane == 13:
        output[row, 173] = wp.float64(2.0) * d2 * d5 * d11 + wp.float64(2.0) * d6 * d8 * d12 + wp.float64(2.0) * d7 * d8 * d13 + (d2*d2) * d14 + wp.float64(3.0) * (d5*d5) * d14 - (d6*d6) * d14 - (d7*d7) * d14 - wp.float64(2.0) * d5 * d6 * d15 + wp.float64(2.0) * d3 * d8 * d15 - wp.float64(2.0) * d5 * d7 * d16 + wp.float64(2.0) * d4 * d8 * d16 + wp.float64(2.0) * d3 * d6 * d17 + wp.float64(2.0) * d4 * d7 * d17
    if lane == 14:
        output[row, 174] = wp.float64(2.0) * d2 * d5 * d20 + wp.float64(2.0) * d6 * d8 * d21 + wp.float64(2.0) * d7 * d8 * d22 + (d2*d2) * d23 + wp.float64(3.0) * (d5*d5) * d23 - (d6*d6) * d23 - (d7*d7) * d23 - wp.float64(2.0) * d5 * d6 * d24 + wp.float64(2.0) * d3 * d8 * d24 - wp.float64(2.0) * d5 * d7 * d25 + wp.float64(2.0) * d4 * d8 * d25 + wp.float64(2.0) * d3 * d6 * d26 + wp.float64(2.0) * d4 * d7 * d26
    if lane == 15:
        output[row, 175] = d5 * (d8*d8)
    if lane == 16:
        output[row, 176] = wp.float64(2.0) * d2 * d3 * d9 - wp.float64(2.0) * d0 * d5 * d9 + wp.float64(2.0) * d2 * d4 * d10 - wp.float64(2.0) * d1 * d5 * d10 + wp.float64(2.0) * d0 * d3 * d11 + wp.float64(2.0) * d1 * d4 * d11 + wp.float64(2.0) * d0 * d2 * d12 + wp.float64(2.0) * d3 * d5 * d12 + wp.float64(2.0) * d1 * d2 * d13 + wp.float64(2.0) * d4 * d5 * d13 - (d0*d0) * d14 - (d1*d1) * d14 + (d3*d3) * d14 + (d4*d4) * d14
    if lane == 17:
        output[row, 177] = wp.float64(2.0) * d2 * d3 * d18 - wp.float64(2.0) * d0 * d5 * d18 + wp.float64(2.0) * d2 * d4 * d19 - wp.float64(2.0) * d1 * d5 * d19 + wp.float64(2.0) * d0 * d3 * d20 + wp.float64(2.0) * d1 * d4 * d20 + wp.float64(2.0) * d0 * d2 * d21 + wp.float64(2.0) * d3 * d5 * d21 + wp.float64(2.0) * d1 * d2 * d22 + wp.float64(2.0) * d4 * d5 * d22 - (d0*d0) * d23 - (d1*d1) * d23 + (d3*d3) * d23 + (d4*d4) * d23
    if lane == 18:
        output[row, 178] = (d2*d2) * d5 + (d5*d5*d5) - d5 * (d6*d6) - d5 * (d7*d7) + wp.float64(2.0) * d3 * d6 * d8 + wp.float64(2.0) * d4 * d7 * d8
    if lane == 19:
        output[row, 179] = wp.float64(2.0) * d0 * d2 * d3 + wp.float64(2.0) * d1 * d2 * d4 - (d0*d0) * d5 - (d1*d1) * d5 + (d3*d3) * d5 + (d4*d4) * d5
    if lane == 20:
        output[row, 180] = d15 * (d17*d17)
    if lane == 21:
        output[row, 181] = (d17*d17) * d24 + wp.float64(2.0) * d15 * d17 * d26
    if lane == 22:
        output[row, 182] = wp.float64(2.0) * d17 * d24 * d26 + d15 * (d26*d26)
    if lane == 23:
        output[row, 183] = d24 * (d26*d26)
    if lane == 24:
        output[row, 184] = -(d11*d11) * d15 - (d14*d14) * d15 + (d15*d15*d15) + d15 * (d16*d16) + wp.float64(2.0) * d9 * d11 * d17 + wp.float64(2.0) * d12 * d14 * d17
    if lane == 25:
        output[row, 185] = wp.float64(2.0) * d11 * d17 * d18 - wp.float64(2.0) * d11 * d15 * d20 + wp.float64(2.0) * d9 * d17 * d20 + wp.float64(2.0) * d14 * d17 * d21 - wp.float64(2.0) * d14 * d15 * d23 + wp.float64(2.0) * d12 * d17 * d23 - (d11*d11) * d24 - (d14*d14) * d24 + wp.float64(3.0) * (d15*d15) * d24 + (d16*d16) * d24 + wp.float64(2.0) * d15 * d16 * d25 + wp.float64(2.0) * d9 * d11 * d26 + wp.float64(2.0) * d12 * d14 * d26
    if lane == 26:
        output[row, 186] = wp.float64(2.0) * d17 * d18 * d20 - d15 * (d20*d20) + wp.float64(2.0) * d17 * d21 * d23 - d15 * (d23*d23) - wp.float64(2.0) * d11 * d20 * d24 - wp.float64(2.0) * d14 * d23 * d24 + wp.float64(3.0) * d15 * (d24*d24) + wp.float64(2.0) * d16 * d24 * d25 + d15 * (d25*d25) + wp.float64(2.0) * d11 * d18 * d26 + wp.float64(2.0) * d9 * d20 * d26 + wp.float64(2.0) * d14 * d21 * d26 + wp.float64(2.0) * d12 * d23 * d26
    if lane == 27:
        output[row, 187] = -(d20*d20) * d24 - (d23*d23) * d24 + (d24*d24*d24) + d24 * (d25*d25) + wp.float64(2.0) * d18 * d20 * d26 + wp.float64(2.0) * d21 * d23 * d26
    if lane == 28:
        output[row, 188] = wp.float64(2.0) * d8 * d15 * d17 + d6 * (d17*d17)
    if lane == 29:
        output[row, 189] = wp.float64(2.0) * d8 * d17 * d24 + wp.float64(2.0) * d8 * d15 * d26 + wp.float64(2.0) * d6 * d17 * d26
    if lane == 30:
        output[row, 190] = wp.float64(2.0) * d8 * d24 * d26 + d6 * (d26*d26)
    if lane == 31:
        output[row, 191] = (d9*d9) * d15 - (d10*d10) * d15 + (d12*d12) * d15 - (d13*d13) * d15 + wp.float64(2.0) * d9 * d10 * d16 + wp.float64(2.0) * d12 * d13 * d16
    if lane == 0:
        output[row, 192] = wp.float64(2.0) * d9 * d15 * d18 + wp.float64(2.0) * d10 * d16 * d18 - wp.float64(2.0) * d10 * d15 * d19 + wp.float64(2.0) * d9 * d16 * d19 + wp.float64(2.0) * d12 * d15 * d21 + wp.float64(2.0) * d13 * d16 * d21 - wp.float64(2.0) * d13 * d15 * d22 + wp.float64(2.0) * d12 * d16 * d22 + (d9*d9) * d24 - (d10*d10) * d24 + (d12*d12) * d24 - (d13*d13) * d24 + wp.float64(2.0) * d9 * d10 * d25 + wp.float64(2.0) * d12 * d13 * d25
    if lane == 1:
        output[row, 193] = d15 * (d18*d18) + wp.float64(2.0) * d16 * d18 * d19 - d15 * (d19*d19) + d15 * (d21*d21) + wp.float64(2.0) * d16 * d21 * d22 - d15 * (d22*d22) + wp.float64(2.0) * d9 * d18 * d24 - wp.float64(2.0) * d10 * d19 * d24 + wp.float64(2.0) * d12 * d21 * d24 - wp.float64(2.0) * d13 * d22 * d24 + wp.float64(2.0) * d10 * d18 * d25 + wp.float64(2.0) * d9 * d19 * d25 + wp.float64(2.0) * d13 * d21 * d25 + wp.float64(2.0) * d12 * d22 * d25
    if lane == 2:
        output[row, 194] = (d18*d18) * d24 - (d19*d19) * d24 + (d21*d21) * d24 - (d22*d22) * d24 + wp.float64(2.0) * d18 * d19 * d25 + wp.float64(2.0) * d21 * d22 * d25
    if lane == 3:
        output[row, 195] = wp.float64(2.0) * d8 * d9 * d11 - d6 * (d11*d11) + wp.float64(2.0) * d8 * d12 * d14 - d6 * (d14*d14) - wp.float64(2.0) * d2 * d11 * d15 - wp.float64(2.0) * d5 * d14 * d15 + wp.float64(3.0) * d6 * (d15*d15) + wp.float64(2.0) * d7 * d15 * d16 + d6 * (d16*d16) + wp.float64(2.0) * d2 * d9 * d17 + wp.float64(2.0) * d0 * d11 * d17 + wp.float64(2.0) * d5 * d12 * d17 + wp.float64(2.0) * d3 * d14 * d17
    if lane == 4:
        output[row, 196] = wp.float64(2.0) * d8 * d11 * d18 + wp.float64(2.0) * d2 * d17 * d18 + wp.float64(2.0) * d8 * d9 * d20 - wp.float64(2.0) * d6 * d11 * d20 - wp.float64(2.0) * d2 * d15 * d20 + wp.float64(2.0) * d0 * d17 * d20 + wp.float64(2.0) * d8 * d14 * d21 + wp.float64(2.0) * d5 * d17 * d21 + wp.float64(2.0) * d8 * d12 * d23 - wp.float64(2.0) * d6 * d14 * d23 - wp.float64(2.0) * d5 * d15 * d23 + wp.float64(2.0) * d3 * d17 * d23 - wp.float64(2.0) * d2 * d11 * d24 - wp.float64(2.0) * d5 * d14 * d24 + wp.float64(6.0) * d6 * d15 * d24 + wp.float64(2.0) * d7 * d16 * d24 + wp.float64(2.0) * d7 * d15 * d25 + wp.float64(2.0) * d6 * d16 * d25 + wp.float64(2.0) * d2 * d9 * d26 + wp.float64(2.0) * d0 * d11 * d26 + wp.float64(2.0) * d5 * d12 * d26 + wp.float64(2.0) * d3 * d14 * d26
    if lane == 5:
        output[row, 197] = wp.float64(2.0) * d8 * d18 * d20 - d6 * (d20*d20) + wp.float64(2.0) * d8 * d21 * d23 - d6 * (d23*d23) - wp.float64(2.0) * d2 * d20 * d24 - wp.float64(2.0) * d5 * d23 * d24 + wp.float64(3.0) * d6 * (d24*d24) + wp.float64(2.0) * d7 * d24 * d25 + d6 * (d25*d25) + wp.float64(2.0) * d2 * d18 * d26 + wp.float64(2.0) * d0 * d20 * d26 + wp.float64(2.0) * d5 * d21 * d26 + wp.float64(2.0) * d3 * d23 * d26
    if lane == 6:
        output[row, 198] = (d8*d8) * d15 + wp.float64(2.0) * d6 * d8 * d17
    if lane == 7:
        output[row, 199] = (d8*d8) * d24 + wp.float64(2.0) * d6 * d8 * d26
    if lane == 8:
        output[row, 200] = d6 * (d9*d9) + wp.float64(2.0) * d7 * d9 * d10 - d6 * (d10*d10) + d6 * (d12*d12) + wp.float64(2.0) * d7 * d12 * d13 - d6 * (d13*d13) + wp.float64(2.0) * d0 * d9 * d15 - wp.float64(2.0) * d1 * d10 * d15 + wp.float64(2.0) * d3 * d12 * d15 - wp.float64(2.0) * d4 * d13 * d15 + wp.float64(2.0) * d1 * d9 * d16 + wp.float64(2.0) * d0 * d10 * d16 + wp.float64(2.0) * d4 * d12 * d16 + wp.float64(2.0) * d3 * d13 * d16
    if lane == 9:
        output[row, 201] = wp.float64(2.0) * d6 * d9 * d18 + wp.float64(2.0) * d7 * d10 * d18 + wp.float64(2.0) * d0 * d15 * d18 + wp.float64(2.0) * d1 * d16 * d18 + wp.float64(2.0) * d7 * d9 * d19 - wp.float64(2.0) * d6 * d10 * d19 - wp.float64(2.0) * d1 * d15 * d19 + wp.float64(2.0) * d0 * d16 * d19 + wp.float64(2.0) * d6 * d12 * d21 + wp.float64(2.0) * d7 * d13 * d21 + wp.float64(2.0) * d3 * d15 * d21 + wp.float64(2.0) * d4 * d16 * d21 + wp.float64(2.0) * d7 * d12 * d22 - wp.float64(2.0) * d6 * d13 * d22 - wp.float64(2.0) * d4 * d15 * d22 + wp.float64(2.0) * d3 * d16 * d22 + wp.float64(2.0) * d0 * d9 * d24 - wp.float64(2.0) * d1 * d10 * d24 + wp.float64(2.0) * d3 * d12 * d24 - wp.float64(2.0) * d4 * d13 * d24 + wp.float64(2.0) * d1 * d9 * d25 + wp.float64(2.0) * d0 * d10 * d25 + wp.float64(2.0) * d4 * d12 * d25 + wp.float64(2.0) * d3 * d13 * d25
    if lane == 10:
        output[row, 202] = d6 * (d18*d18) + wp.float64(2.0) * d7 * d18 * d19 - d6 * (d19*d19) + d6 * (d21*d21) + wp.float64(2.0) * d7 * d21 * d22 - d6 * (d22*d22) + wp.float64(2.0) * d0 * d18 * d24 - wp.float64(2.0) * d1 * d19 * d24 + wp.float64(2.0) * d3 * d21 * d24 - wp.float64(2.0) * d4 * d22 * d24 + wp.float64(2.0) * d1 * d18 * d25 + wp.float64(2.0) * d0 * d19 * d25 + wp.float64(2.0) * d4 * d21 * d25 + wp.float64(2.0) * d3 * d22 * d25
    if lane == 11:
        output[row, 203] = wp.float64(2.0) * d2 * d8 * d9 - wp.float64(2.0) * d2 * d6 * d11 + wp.float64(2.0) * d0 * d8 * d11 + wp.float64(2.0) * d5 * d8 * d12 - wp.float64(2.0) * d5 * d6 * d14 + wp.float64(2.0) * d3 * d8 * d14 - (d2*d2) * d15 - (d5*d5) * d15 + wp.float64(3.0) * (d6*d6) * d15 + (d7*d7) * d15 + wp.float64(2.0) * d6 * d7 * d16 + wp.float64(2.0) * d0 * d2 * d17 + wp.float64(2.0) * d3 * d5 * d17
    if lane == 12:
        output[row, 204] = wp.float64(2.0) * d2 * d8 * d18 - wp.float64(2.0) * d2 * d6 * d20 + wp.float64(2.0) * d0 * d8 * d20 + wp.float64(2.0) * d5 * d8 * d21 - wp.float64(2.0) * d5 * d6 * d23 + wp.float64(2.0) * d3 * d8 * d23 - (d2*d2) * d24 - (d5*d5) * d24 + wp.float64(3.0) * (d6*d6) * d24 + (d7*d7) * d24 + wp.float64(2.0) * d6 * d7 * d25 + wp.float64(2.0) * d0 * d2 * d26 + wp.float64(2.0) * d3 * d5 * d26
    if lane == 13:
        output[row, 205] = d6 * (d8*d8)
    if lane == 14:
        output[row, 206] = wp.float64(2.0) * d0 * d6 * d9 + wp.float64(2.0) * d1 * d7 * d9 - wp.float64(2.0) * d1 * d6 * d10 + wp.float64(2.0) * d0 * d7 * d10 + wp.float64(2.0) * d3 * d6 * d12 + wp.float64(2.0) * d4 * d7 * d12 - wp.float64(2.0) * d4 * d6 * d13 + wp.float64(2.0) * d3 * d7 * d13 + (d0*d0) * d15 - (d1*d1) * d15 + (d3*d3) * d15 - (d4*d4) * d15 + wp.float64(2.0) * d0 * d1 * d16 + wp.float64(2.0) * d3 * d4 * d16
    if lane == 15:
        output[row, 207] = wp.float64(2.0) * d0 * d6 * d18 + wp.float64(2.0) * d1 * d7 * d18 - wp.float64(2.0) * d1 * d6 * d19 + wp.float64(2.0) * d0 * d7 * d19 + wp.float64(2.0) * d3 * d6 * d21 + wp.float64(2.0) * d4 * d7 * d21 - wp.float64(2.0) * d4 * d6 * d22 + wp.float64(2.0) * d3 * d7 * d22 + (d0*d0) * d24 - (d1*d1) * d24 + (d3*d3) * d24 - (d4*d4) * d24 + wp.float64(2.0) * d0 * d1 * d25 + wp.float64(2.0) * d3 * d4 * d25
    if lane == 16:
        output[row, 208] = -(d2*d2) * d6 - (d5*d5) * d6 + (d6*d6*d6) + d6 * (d7*d7) + wp.float64(2.0) * d0 * d2 * d8 + wp.float64(2.0) * d3 * d5 * d8
    if lane == 17:
        output[row, 209] = (d0*d0) * d6 - (d1*d1) * d6 + (d3*d3) * d6 - (d4*d4) * d6 + wp.float64(2.0) * d0 * d1 * d7 + wp.float64(2.0) * d3 * d4 * d7
    if lane == 18:
        output[row, 210] = d16 * (d17*d17)
    if lane == 19:
        output[row, 211] = (d17*d17) * d25 + wp.float64(2.0) * d16 * d17 * d26
    if lane == 20:
        output[row, 212] = wp.float64(2.0) * d17 * d25 * d26 + d16 * (d26*d26)
    if lane == 21:
        output[row, 213] = d25 * (d26*d26)
    if lane == 22:
        output[row, 214] = -(d11*d11) * d16 - (d14*d14) * d16 + (d15*d15) * d16 + (d16*d16*d16) + wp.float64(2.0) * d10 * d11 * d17 + wp.float64(2.0) * d13 * d14 * d17
    if lane == 23:
        output[row, 215] = wp.float64(2.0) * d11 * d17 * d19 - wp.float64(2.0) * d11 * d16 * d20 + wp.float64(2.0) * d10 * d17 * d20 + wp.float64(2.0) * d14 * d17 * d22 - wp.float64(2.0) * d14 * d16 * d23 + wp.float64(2.0) * d13 * d17 * d23 + wp.float64(2.0) * d15 * d16 * d24 - (d11*d11) * d25 - (d14*d14) * d25 + (d15*d15) * d25 + wp.float64(3.0) * (d16*d16) * d25 + wp.float64(2.0) * d10 * d11 * d26 + wp.float64(2.0) * d13 * d14 * d26
    if lane == 24:
        output[row, 216] = wp.float64(2.0) * d17 * d19 * d20 - d16 * (d20*d20) + wp.float64(2.0) * d17 * d22 * d23 - d16 * (d23*d23) + d16 * (d24*d24) - wp.float64(2.0) * d11 * d20 * d25 - wp.float64(2.0) * d14 * d23 * d25 + wp.float64(2.0) * d15 * d24 * d25 + wp.float64(3.0) * d16 * (d25*d25) + wp.float64(2.0) * d11 * d19 * d26 + wp.float64(2.0) * d10 * d20 * d26 + wp.float64(2.0) * d14 * d22 * d26 + wp.float64(2.0) * d13 * d23 * d26
    if lane == 25:
        output[row, 217] = -(d20*d20) * d25 - (d23*d23) * d25 + (d24*d24) * d25 + (d25*d25*d25) + wp.float64(2.0) * d19 * d20 * d26 + wp.float64(2.0) * d22 * d23 * d26
    if lane == 26:
        output[row, 218] = wp.float64(2.0) * d8 * d16 * d17 + d7 * (d17*d17)
    if lane == 27:
        output[row, 219] = wp.float64(2.0) * d8 * d17 * d25 + wp.float64(2.0) * d8 * d16 * d26 + wp.float64(2.0) * d7 * d17 * d26
    if lane == 28:
        output[row, 220] = wp.float64(2.0) * d8 * d25 * d26 + d7 * (d26*d26)
    if lane == 29:
        output[row, 221] = wp.float64(2.0) * d9 * d10 * d15 + wp.float64(2.0) * d12 * d13 * d15 - (d9*d9) * d16 + (d10*d10) * d16 - (d12*d12) * d16 + (d13*d13) * d16
    if lane == 30:
        output[row, 222] = wp.float64(2.0) * d10 * d15 * d18 - wp.float64(2.0) * d9 * d16 * d18 + wp.float64(2.0) * d9 * d15 * d19 + wp.float64(2.0) * d10 * d16 * d19 + wp.float64(2.0) * d13 * d15 * d21 - wp.float64(2.0) * d12 * d16 * d21 + wp.float64(2.0) * d12 * d15 * d22 + wp.float64(2.0) * d13 * d16 * d22 + wp.float64(2.0) * d9 * d10 * d24 + wp.float64(2.0) * d12 * d13 * d24 - (d9*d9) * d25 + (d10*d10) * d25 - (d12*d12) * d25 + (d13*d13) * d25
    if lane == 31:
        output[row, 223] = -d16 * (d18*d18) + wp.float64(2.0) * d15 * d18 * d19 + d16 * (d19*d19) - d16 * (d21*d21) + wp.float64(2.0) * d15 * d21 * d22 + d16 * (d22*d22) + wp.float64(2.0) * d10 * d18 * d24 + wp.float64(2.0) * d9 * d19 * d24 + wp.float64(2.0) * d13 * d21 * d24 + wp.float64(2.0) * d12 * d22 * d24 - wp.float64(2.0) * d9 * d18 * d25 + wp.float64(2.0) * d10 * d19 * d25 - wp.float64(2.0) * d12 * d21 * d25 + wp.float64(2.0) * d13 * d22 * d25
    if lane == 0:
        output[row, 224] = wp.float64(2.0) * d18 * d19 * d24 + wp.float64(2.0) * d21 * d22 * d24 - (d18*d18) * d25 + (d19*d19) * d25 - (d21*d21) * d25 + (d22*d22) * d25
    if lane == 1:
        output[row, 225] = wp.float64(2.0) * d8 * d10 * d11 - d7 * (d11*d11) + wp.float64(2.0) * d8 * d13 * d14 - d7 * (d14*d14) + d7 * (d15*d15) - wp.float64(2.0) * d2 * d11 * d16 - wp.float64(2.0) * d5 * d14 * d16 + wp.float64(2.0) * d6 * d15 * d16 + wp.float64(3.0) * d7 * (d16*d16) + wp.float64(2.0) * d2 * d10 * d17 + wp.float64(2.0) * d1 * d11 * d17 + wp.float64(2.0) * d5 * d13 * d17 + wp.float64(2.0) * d4 * d14 * d17
    if lane == 2:
        output[row, 226] = wp.float64(2.0) * d8 * d11 * d19 + wp.float64(2.0) * d2 * d17 * d19 + wp.float64(2.0) * d8 * d10 * d20 - wp.float64(2.0) * d7 * d11 * d20 - wp.float64(2.0) * d2 * d16 * d20 + wp.float64(2.0) * d1 * d17 * d20 + wp.float64(2.0) * d8 * d14 * d22 + wp.float64(2.0) * d5 * d17 * d22 + wp.float64(2.0) * d8 * d13 * d23 - wp.float64(2.0) * d7 * d14 * d23 - wp.float64(2.0) * d5 * d16 * d23 + wp.float64(2.0) * d4 * d17 * d23 + wp.float64(2.0) * d7 * d15 * d24 + wp.float64(2.0) * d6 * d16 * d24 - wp.float64(2.0) * d2 * d11 * d25 - wp.float64(2.0) * d5 * d14 * d25 + wp.float64(2.0) * d6 * d15 * d25 + wp.float64(6.0) * d7 * d16 * d25 + wp.float64(2.0) * d2 * d10 * d26 + wp.float64(2.0) * d1 * d11 * d26 + wp.float64(2.0) * d5 * d13 * d26 + wp.float64(2.0) * d4 * d14 * d26
    if lane == 3:
        output[row, 227] = wp.float64(2.0) * d8 * d19 * d20 - d7 * (d20*d20) + wp.float64(2.0) * d8 * d22 * d23 - d7 * (d23*d23) + d7 * (d24*d24) - wp.float64(2.0) * d2 * d20 * d25 - wp.float64(2.0) * d5 * d23 * d25 + wp.float64(2.0) * d6 * d24 * d25 + wp.float64(3.0) * d7 * (d25*d25) + wp.float64(2.0) * d2 * d19 * d26 + wp.float64(2.0) * d1 * d20 * d26 + wp.float64(2.0) * d5 * d22 * d26 + wp.float64(2.0) * d4 * d23 * d26
    if lane == 4:
        output[row, 228] = (d8*d8) * d16 + wp.float64(2.0) * d7 * d8 * d17
    if lane == 5:
        output[row, 229] = (d8*d8) * d25 + wp.float64(2.0) * d7 * d8 * d26
    if lane == 6:
        output[row, 230] = -d7 * (d9*d9) + wp.float64(2.0) * d6 * d9 * d10 + d7 * (d10*d10) - d7 * (d12*d12) + wp.float64(2.0) * d6 * d12 * d13 + d7 * (d13*d13) + wp.float64(2.0) * d1 * d9 * d15 + wp.float64(2.0) * d0 * d10 * d15 + wp.float64(2.0) * d4 * d12 * d15 + wp.float64(2.0) * d3 * d13 * d15 - wp.float64(2.0) * d0 * d9 * d16 + wp.float64(2.0) * d1 * d10 * d16 - wp.float64(2.0) * d3 * d12 * d16 + wp.float64(2.0) * d4 * d13 * d16
    if lane == 7:
        output[row, 231] = -wp.float64(2.0) * d7 * d9 * d18 + wp.float64(2.0) * d6 * d10 * d18 + wp.float64(2.0) * d1 * d15 * d18 - wp.float64(2.0) * d0 * d16 * d18 + wp.float64(2.0) * d6 * d9 * d19 + wp.float64(2.0) * d7 * d10 * d19 + wp.float64(2.0) * d0 * d15 * d19 + wp.float64(2.0) * d1 * d16 * d19 - wp.float64(2.0) * d7 * d12 * d21 + wp.float64(2.0) * d6 * d13 * d21 + wp.float64(2.0) * d4 * d15 * d21 - wp.float64(2.0) * d3 * d16 * d21 + wp.float64(2.0) * d6 * d12 * d22 + wp.float64(2.0) * d7 * d13 * d22 + wp.float64(2.0) * d3 * d15 * d22 + wp.float64(2.0) * d4 * d16 * d22 + wp.float64(2.0) * d1 * d9 * d24 + wp.float64(2.0) * d0 * d10 * d24 + wp.float64(2.0) * d4 * d12 * d24 + wp.float64(2.0) * d3 * d13 * d24 - wp.float64(2.0) * d0 * d9 * d25 + wp.float64(2.0) * d1 * d10 * d25 - wp.float64(2.0) * d3 * d12 * d25 + wp.float64(2.0) * d4 * d13 * d25
    if lane == 8:
        output[row, 232] = -d7 * (d18*d18) + wp.float64(2.0) * d6 * d18 * d19 + d7 * (d19*d19) - d7 * (d21*d21) + wp.float64(2.0) * d6 * d21 * d22 + d7 * (d22*d22) + wp.float64(2.0) * d1 * d18 * d24 + wp.float64(2.0) * d0 * d19 * d24 + wp.float64(2.0) * d4 * d21 * d24 + wp.float64(2.0) * d3 * d22 * d24 - wp.float64(2.0) * d0 * d18 * d25 + wp.float64(2.0) * d1 * d19 * d25 - wp.float64(2.0) * d3 * d21 * d25 + wp.float64(2.0) * d4 * d22 * d25
    if lane == 9:
        output[row, 233] = wp.float64(2.0) * d2 * d8 * d10 - wp.float64(2.0) * d2 * d7 * d11 + wp.float64(2.0) * d1 * d8 * d11 + wp.float64(2.0) * d5 * d8 * d13 - wp.float64(2.0) * d5 * d7 * d14 + wp.float64(2.0) * d4 * d8 * d14 + wp.float64(2.0) * d6 * d7 * d15 - (d2*d2) * d16 - (d5*d5) * d16 + (d6*d6) * d16 + wp.float64(3.0) * (d7*d7) * d16 + wp.float64(2.0) * d1 * d2 * d17 + wp.float64(2.0) * d4 * d5 * d17
    if lane == 10:
        output[row, 234] = wp.float64(2.0) * d2 * d8 * d19 - wp.float64(2.0) * d2 * d7 * d20 + wp.float64(2.0) * d1 * d8 * d20 + wp.float64(2.0) * d5 * d8 * d22 - wp.float64(2.0) * d5 * d7 * d23 + wp.float64(2.0) * d4 * d8 * d23 + wp.float64(2.0) * d6 * d7 * d24 - (d2*d2) * d25 - (d5*d5) * d25 + (d6*d6) * d25 + wp.float64(3.0) * (d7*d7) * d25 + wp.float64(2.0) * d1 * d2 * d26 + wp.float64(2.0) * d4 * d5 * d26
    if lane == 11:
        output[row, 235] = d7 * (d8*d8)
    if lane == 12:
        output[row, 236] = wp.float64(2.0) * d1 * d6 * d9 - wp.float64(2.0) * d0 * d7 * d9 + wp.float64(2.0) * d0 * d6 * d10 + wp.float64(2.0) * d1 * d7 * d10 + wp.float64(2.0) * d4 * d6 * d12 - wp.float64(2.0) * d3 * d7 * d12 + wp.float64(2.0) * d3 * d6 * d13 + wp.float64(2.0) * d4 * d7 * d13 + wp.float64(2.0) * d0 * d1 * d15 + wp.float64(2.0) * d3 * d4 * d15 - (d0*d0) * d16 + (d1*d1) * d16 - (d3*d3) * d16 + (d4*d4) * d16
    if lane == 13:
        output[row, 237] = wp.float64(2.0) * d1 * d6 * d18 - wp.float64(2.0) * d0 * d7 * d18 + wp.float64(2.0) * d0 * d6 * d19 + wp.float64(2.0) * d1 * d7 * d19 + wp.float64(2.0) * d4 * d6 * d21 - wp.float64(2.0) * d3 * d7 * d21 + wp.float64(2.0) * d3 * d6 * d22 + wp.float64(2.0) * d4 * d7 * d22 + wp.float64(2.0) * d0 * d1 * d24 + wp.float64(2.0) * d3 * d4 * d24 - (d0*d0) * d25 + (d1*d1) * d25 - (d3*d3) * d25 + (d4*d4) * d25
    if lane == 14:
        output[row, 238] = -(d2*d2) * d7 - (d5*d5) * d7 + (d6*d6) * d7 + (d7*d7*d7) + wp.float64(2.0) * d1 * d2 * d8 + wp.float64(2.0) * d4 * d5 * d8
    if lane == 15:
        output[row, 239] = wp.float64(2.0) * d0 * d1 * d6 + wp.float64(2.0) * d3 * d4 * d6 - (d0*d0) * d7 + (d1*d1) * d7 - (d3*d3) * d7 + (d4*d4) * d7
    if lane == 16:
        output[row, 240] = (d17*d17*d17)
    if lane == 17:
        output[row, 241] = wp.float64(3.0) * (d17*d17) * d26
    if lane == 18:
        output[row, 242] = wp.float64(3.0) * d17 * (d26*d26)
    if lane == 19:
        output[row, 243] = (d26*d26*d26)
    if lane == 20:
        output[row, 244] = (d11*d11) * d17 + (d14*d14) * d17 + (d15*d15) * d17 + (d16*d16) * d17
    if lane == 21:
        output[row, 245] = wp.float64(2.0) * d11 * d17 * d20 + wp.float64(2.0) * d14 * d17 * d23 + wp.float64(2.0) * d15 * d17 * d24 + wp.float64(2.0) * d16 * d17 * d25 + (d11*d11) * d26 + (d14*d14) * d26 + (d15*d15) * d26 + (d16*d16) * d26
    if lane == 22:
        output[row, 246] = d17 * (d20*d20) + d17 * (d23*d23) + d17 * (d24*d24) + d17 * (d25*d25) + wp.float64(2.0) * d11 * d20 * d26 + wp.float64(2.0) * d14 * d23 * d26 + wp.float64(2.0) * d15 * d24 * d26 + wp.float64(2.0) * d16 * d25 * d26
    if lane == 23:
        output[row, 247] = (d20*d20) * d26 + (d23*d23) * d26 + (d24*d24) * d26 + (d25*d25) * d26
    if lane == 24:
        output[row, 248] = wp.float64(3.0) * d8 * (d17*d17)
    if lane == 25:
        output[row, 249] = wp.float64(6.0) * d8 * d17 * d26
    if lane == 26:
        output[row, 250] = wp.float64(3.0) * d8 * (d26*d26)
    if lane == 27:
        output[row, 251] = wp.float64(2.0) * d9 * d11 * d15 + wp.float64(2.0) * d12 * d14 * d15 + wp.float64(2.0) * d10 * d11 * d16 + wp.float64(2.0) * d13 * d14 * d16 - (d9*d9) * d17 - (d10*d10) * d17 - (d12*d12) * d17 - (d13*d13) * d17
    if lane == 28:
        output[row, 252] = wp.float64(2.0) * d11 * d15 * d18 - wp.float64(2.0) * d9 * d17 * d18 + wp.float64(2.0) * d11 * d16 * d19 - wp.float64(2.0) * d10 * d17 * d19 + wp.float64(2.0) * d9 * d15 * d20 + wp.float64(2.0) * d10 * d16 * d20 + wp.float64(2.0) * d14 * d15 * d21 - wp.float64(2.0) * d12 * d17 * d21 + wp.float64(2.0) * d14 * d16 * d22 - wp.float64(2.0) * d13 * d17 * d22 + wp.float64(2.0) * d12 * d15 * d23 + wp.float64(2.0) * d13 * d16 * d23 + wp.float64(2.0) * d9 * d11 * d24 + wp.float64(2.0) * d12 * d14 * d24 + wp.float64(2.0) * d10 * d11 * d25 + wp.float64(2.0) * d13 * d14 * d25 - (d9*d9) * d26 - (d10*d10) * d26 - (d12*d12) * d26 - (d13*d13) * d26
    if lane == 29:
        output[row, 253] = -d17 * (d18*d18) - d17 * (d19*d19) + wp.float64(2.0) * d15 * d18 * d20 + wp.float64(2.0) * d16 * d19 * d20 - d17 * (d21*d21) - d17 * (d22*d22) + wp.float64(2.0) * d15 * d21 * d23 + wp.float64(2.0) * d16 * d22 * d23 + wp.float64(2.0) * d11 * d18 * d24 + wp.float64(2.0) * d9 * d20 * d24 + wp.float64(2.0) * d14 * d21 * d24 + wp.float64(2.0) * d12 * d23 * d24 + wp.float64(2.0) * d11 * d19 * d25 + wp.float64(2.0) * d10 * d20 * d25 + wp.float64(2.0) * d14 * d22 * d25 + wp.float64(2.0) * d13 * d23 * d25 - wp.float64(2.0) * d9 * d18 * d26 - wp.float64(2.0) * d10 * d19 * d26 - wp.float64(2.0) * d12 * d21 * d26 - wp.float64(2.0) * d13 * d22 * d26
    if lane == 30:
        output[row, 254] = wp.float64(2.0) * d18 * d20 * d24 + wp.float64(2.0) * d21 * d23 * d24 + wp.float64(2.0) * d19 * d20 * d25 + wp.float64(2.0) * d22 * d23 * d25 - (d18*d18) * d26 - (d19*d19) * d26 - (d21*d21) * d26 - (d22*d22) * d26
    if lane == 31:
        output[row, 255] = d8 * (d11*d11) + d8 * (d14*d14) + d8 * (d15*d15) + d8 * (d16*d16) + wp.float64(2.0) * d2 * d11 * d17 + wp.float64(2.0) * d5 * d14 * d17 + wp.float64(2.0) * d6 * d15 * d17 + wp.float64(2.0) * d7 * d16 * d17
    if lane == 0:
        output[row, 256] = wp.float64(2.0) * d8 * d11 * d20 + wp.float64(2.0) * d2 * d17 * d20 + wp.float64(2.0) * d8 * d14 * d23 + wp.float64(2.0) * d5 * d17 * d23 + wp.float64(2.0) * d8 * d15 * d24 + wp.float64(2.0) * d6 * d17 * d24 + wp.float64(2.0) * d8 * d16 * d25 + wp.float64(2.0) * d7 * d17 * d25 + wp.float64(2.0) * d2 * d11 * d26 + wp.float64(2.0) * d5 * d14 * d26 + wp.float64(2.0) * d6 * d15 * d26 + wp.float64(2.0) * d7 * d16 * d26
    if lane == 1:
        output[row, 257] = d8 * (d20*d20) + d8 * (d23*d23) + d8 * (d24*d24) + d8 * (d25*d25) + wp.float64(2.0) * d2 * d20 * d26 + wp.float64(2.0) * d5 * d23 * d26 + wp.float64(2.0) * d6 * d24 * d26 + wp.float64(2.0) * d7 * d25 * d26
    if lane == 2:
        output[row, 258] = wp.float64(3.0) * (d8*d8) * d17
    if lane == 3:
        output[row, 259] = wp.float64(3.0) * (d8*d8) * d26
    if lane == 4:
        output[row, 260] = -d8 * (d9*d9) - d8 * (d10*d10) + wp.float64(2.0) * d6 * d9 * d11 + wp.float64(2.0) * d7 * d10 * d11 - d8 * (d12*d12) - d8 * (d13*d13) + wp.float64(2.0) * d6 * d12 * d14 + wp.float64(2.0) * d7 * d13 * d14 + wp.float64(2.0) * d2 * d9 * d15 + wp.float64(2.0) * d0 * d11 * d15 + wp.float64(2.0) * d5 * d12 * d15 + wp.float64(2.0) * d3 * d14 * d15 + wp.float64(2.0) * d2 * d10 * d16 + wp.float64(2.0) * d1 * d11 * d16 + wp.float64(2.0) * d5 * d13 * d16 + wp.float64(2.0) * d4 * d14 * d16 - wp.float64(2.0) * d0 * d9 * d17 - wp.float64(2.0) * d1 * d10 * d17 - wp.float64(2.0) * d3 * d12 * d17 - wp.float64(2.0) * d4 * d13 * d17
    if lane == 5:
        output[row, 261] = -wp.float64(2.0) * d8 * d9 * d18 + wp.float64(2.0) * d6 * d11 * d18 + wp.float64(2.0) * d2 * d15 * d18 - wp.float64(2.0) * d0 * d17 * d18 - wp.float64(2.0) * d8 * d10 * d19 + wp.float64(2.0) * d7 * d11 * d19 + wp.float64(2.0) * d2 * d16 * d19 - wp.float64(2.0) * d1 * d17 * d19 + wp.float64(2.0) * d6 * d9 * d20 + wp.float64(2.0) * d7 * d10 * d20 + wp.float64(2.0) * d0 * d15 * d20 + wp.float64(2.0) * d1 * d16 * d20 - wp.float64(2.0) * d8 * d12 * d21 + wp.float64(2.0) * d6 * d14 * d21 + wp.float64(2.0) * d5 * d15 * d21 - wp.float64(2.0) * d3 * d17 * d21 - wp.float64(2.0) * d8 * d13 * d22 + wp.float64(2.0) * d7 * d14 * d22 + wp.float64(2.0) * d5 * d16 * d22 - wp.float64(2.0) * d4 * d17 * d22 + wp.float64(2.0) * d6 * d12 * d23 + wp.float64(2.0) * d7 * d13 * d23 + wp.float64(2.0) * d3 * d15 * d23 + wp.float64(2.0) * d4 * d16 * d23 + wp.float64(2.0) * d2 * d9 * d24 + wp.float64(2.0) * d0 * d11 * d24 + wp.float64(2.0) * d5 * d12 * d24 + wp.float64(2.0) * d3 * d14 * d24 + wp.float64(2.0) * d2 * d10 * d25 + wp.float64(2.0) * d1 * d11 * d25 + wp.float64(2.0) * d5 * d13 * d25 + wp.float64(2.0) * d4 * d14 * d25 - wp.float64(2.0) * d0 * d9 * d26 - wp.float64(2.0) * d1 * d10 * d26 - wp.float64(2.0) * d3 * d12 * d26 - wp.float64(2.0) * d4 * d13 * d26
    if lane == 6:
        output[row, 262] = -d8 * (d18*d18) - d8 * (d19*d19) + wp.float64(2.0) * d6 * d18 * d20 + wp.float64(2.0) * d7 * d19 * d20 - d8 * (d21*d21) - d8 * (d22*d22) + wp.float64(2.0) * d6 * d21 * d23 + wp.float64(2.0) * d7 * d22 * d23 + wp.float64(2.0) * d2 * d18 * d24 + wp.float64(2.0) * d0 * d20 * d24 + wp.float64(2.0) * d5 * d21 * d24 + wp.float64(2.0) * d3 * d23 * d24 + wp.float64(2.0) * d2 * d19 * d25 + wp.float64(2.0) * d1 * d20 * d25 + wp.float64(2.0) * d5 * d22 * d25 + wp.float64(2.0) * d4 * d23 * d25 - wp.float64(2.0) * d0 * d18 * d26 - wp.float64(2.0) * d1 * d19 * d26 - wp.float64(2.0) * d3 * d21 * d26 - wp.float64(2.0) * d4 * d22 * d26
    if lane == 7:
        output[row, 263] = wp.float64(2.0) * d2 * d8 * d11 + wp.float64(2.0) * d5 * d8 * d14 + wp.float64(2.0) * d6 * d8 * d15 + wp.float64(2.0) * d7 * d8 * d16 + (d2*d2) * d17 + (d5*d5) * d17 + (d6*d6) * d17 + (d7*d7) * d17
    if lane == 8:
        output[row, 264] = wp.float64(2.0) * d2 * d8 * d20 + wp.float64(2.0) * d5 * d8 * d23 + wp.float64(2.0) * d6 * d8 * d24 + wp.float64(2.0) * d7 * d8 * d25 + (d2*d2) * d26 + (d5*d5) * d26 + (d6*d6) * d26 + (d7*d7) * d26
    if lane == 9:
        output[row, 265] = (d8*d8*d8)
    if lane == 10:
        output[row, 266] = wp.float64(2.0) * d2 * d6 * d9 - wp.float64(2.0) * d0 * d8 * d9 + wp.float64(2.0) * d2 * d7 * d10 - wp.float64(2.0) * d1 * d8 * d10 + wp.float64(2.0) * d0 * d6 * d11 + wp.float64(2.0) * d1 * d7 * d11 + wp.float64(2.0) * d5 * d6 * d12 - wp.float64(2.0) * d3 * d8 * d12 + wp.float64(2.0) * d5 * d7 * d13 - wp.float64(2.0) * d4 * d8 * d13 + wp.float64(2.0) * d3 * d6 * d14 + wp.float64(2.0) * d4 * d7 * d14 + wp.float64(2.0) * d0 * d2 * d15 + wp.float64(2.0) * d3 * d5 * d15 + wp.float64(2.0) * d1 * d2 * d16 + wp.float64(2.0) * d4 * d5 * d16 - (d0*d0) * d17 - (d1*d1) * d17 - (d3*d3) * d17 - (d4*d4) * d17
    if lane == 11:
        output[row, 267] = wp.float64(2.0) * d2 * d6 * d18 - wp.float64(2.0) * d0 * d8 * d18 + wp.float64(2.0) * d2 * d7 * d19 - wp.float64(2.0) * d1 * d8 * d19 + wp.float64(2.0) * d0 * d6 * d20 + wp.float64(2.0) * d1 * d7 * d20 + wp.float64(2.0) * d5 * d6 * d21 - wp.float64(2.0) * d3 * d8 * d21 + wp.float64(2.0) * d5 * d7 * d22 - wp.float64(2.0) * d4 * d8 * d22 + wp.float64(2.0) * d3 * d6 * d23 + wp.float64(2.0) * d4 * d7 * d23 + wp.float64(2.0) * d0 * d2 * d24 + wp.float64(2.0) * d3 * d5 * d24 + wp.float64(2.0) * d1 * d2 * d25 + wp.float64(2.0) * d4 * d5 * d25 - (d0*d0) * d26 - (d1*d1) * d26 - (d3*d3) * d26 - (d4*d4) * d26
    if lane == 12:
        output[row, 268] = (d2*d2) * d8 + (d5*d5) * d8 + (d6*d6) * d8 + (d7*d7) * d8
    if lane == 13:
        output[row, 269] = wp.float64(2.0) * d0 * d2 * d6 + wp.float64(2.0) * d3 * d5 * d6 + wp.float64(2.0) * d1 * d2 * d7 + wp.float64(2.0) * d4 * d5 * d7 - (d0*d0) * d8 - (d1*d1) * d8 - (d3*d3) * d8 - (d4*d4) * d8
    if lane == 14:
        output[row, 270] = -d11 * d13 * d15 + d10 * d14 * d15 + d11 * d12 * d16 - d9 * d14 * d16 - d10 * d12 * d17 + d9 * d13 * d17
    if lane == 15:
        output[row, 271] = -d14 * d16 * d18 + d13 * d17 * d18 + d14 * d15 * d19 - d12 * d17 * d19 - d13 * d15 * d20 + d12 * d16 * d20 + d11 * d16 * d21 - d10 * d17 * d21 - d11 * d15 * d22 + d9 * d17 * d22 + d10 * d15 * d23 - d9 * d16 * d23 - d11 * d13 * d24 + d10 * d14 * d24 + d11 * d12 * d25 - d9 * d14 * d25 - d10 * d12 * d26 + d9 * d13 * d26
    if lane == 16:
        output[row, 272] = -d17 * d19 * d21 + d16 * d20 * d21 + d17 * d18 * d22 - d15 * d20 * d22 - d16 * d18 * d23 + d15 * d19 * d23 + d14 * d19 * d24 - d13 * d20 * d24 - d11 * d22 * d24 + d10 * d23 * d24 - d14 * d18 * d25 + d12 * d20 * d25 + d11 * d21 * d25 - d9 * d23 * d25 + d13 * d18 * d26 - d12 * d19 * d26 - d10 * d21 * d26 + d9 * d22 * d26
    if lane == 17:
        output[row, 273] = -d20 * d22 * d24 + d19 * d23 * d24 + d20 * d21 * d25 - d18 * d23 * d25 - d19 * d21 * d26 + d18 * d22 * d26
    if lane == 18:
        output[row, 274] = -d8 * d10 * d12 + d7 * d11 * d12 + d8 * d9 * d13 - d6 * d11 * d13 - d7 * d9 * d14 + d6 * d10 * d14 + d5 * d10 * d15 - d4 * d11 * d15 - d2 * d13 * d15 + d1 * d14 * d15 - d5 * d9 * d16 + d3 * d11 * d16 + d2 * d12 * d16 - d0 * d14 * d16 + d4 * d9 * d17 - d3 * d10 * d17 - d1 * d12 * d17 + d0 * d13 * d17
    if lane == 19:
        output[row, 275] = d8 * d13 * d18 - d7 * d14 * d18 - d5 * d16 * d18 + d4 * d17 * d18 - d8 * d12 * d19 + d6 * d14 * d19 + d5 * d15 * d19 - d3 * d17 * d19 + d7 * d12 * d20 - d6 * d13 * d20 - d4 * d15 * d20 + d3 * d16 * d20 - d8 * d10 * d21 + d7 * d11 * d21 + d2 * d16 * d21 - d1 * d17 * d21 + d8 * d9 * d22 - d6 * d11 * d22 - d2 * d15 * d22 + d0 * d17 * d22 - d7 * d9 * d23 + d6 * d10 * d23 + d1 * d15 * d23 - d0 * d16 * d23 + d5 * d10 * d24 - d4 * d11 * d24 - d2 * d13 * d24 + d1 * d14 * d24 - d5 * d9 * d25 + d3 * d11 * d25 + d2 * d12 * d25 - d0 * d14 * d25 + d4 * d9 * d26 - d3 * d10 * d26 - d1 * d12 * d26 + d0 * d13 * d26
    if lane == 20:
        output[row, 276] = -d8 * d19 * d21 + d7 * d20 * d21 + d8 * d18 * d22 - d6 * d20 * d22 - d7 * d18 * d23 + d6 * d19 * d23 + d5 * d19 * d24 - d4 * d20 * d24 - d2 * d22 * d24 + d1 * d23 * d24 - d5 * d18 * d25 + d3 * d20 * d25 + d2 * d21 * d25 - d0 * d23 * d25 + d4 * d18 * d26 - d3 * d19 * d26 - d1 * d21 * d26 + d0 * d22 * d26
    if lane == 21:
        output[row, 277] = -d5 * d7 * d9 + d4 * d8 * d9 + d5 * d6 * d10 - d3 * d8 * d10 - d4 * d6 * d11 + d3 * d7 * d11 + d2 * d7 * d12 - d1 * d8 * d12 - d2 * d6 * d13 + d0 * d8 * d13 + d1 * d6 * d14 - d0 * d7 * d14 - d2 * d4 * d15 + d1 * d5 * d15 + d2 * d3 * d16 - d0 * d5 * d16 - d1 * d3 * d17 + d0 * d4 * d17
    if lane == 22:
        output[row, 278] = -d5 * d7 * d18 + d4 * d8 * d18 + d5 * d6 * d19 - d3 * d8 * d19 - d4 * d6 * d20 + d3 * d7 * d20 + d2 * d7 * d21 - d1 * d8 * d21 - d2 * d6 * d22 + d0 * d8 * d22 + d1 * d6 * d23 - d0 * d7 * d23 - d2 * d4 * d24 + d1 * d5 * d24 + d2 * d3 * d25 - d0 * d5 * d25 - d1 * d3 * d26 + d0 * d4 * d26
    if lane == 23:
        output[row, 279] = -d2 * d4 * d6 + d1 * d5 * d6 + d2 * d3 * d7 - d0 * d5 * d7 - d1 * d3 * d8 + d0 * d4 * d8
