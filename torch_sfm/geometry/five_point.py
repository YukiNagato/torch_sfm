# Elimination layout adapted from PoseLib/solvers/relpose_5pt.cc (BSD-3-Clause).
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

"""Nister elimination and back substitution; real roots use a bounded CUDA kernel."""

import torch
import torch.nn.functional as F

from .two_view import _MONOMIALS


_ORDER = [(3, 0, 0), (0, 3, 0), (2, 1, 0), (1, 2, 0), (2, 0, 1), (2, 0, 0),
          (0, 2, 1), (0, 2, 0), (1, 1, 1), (1, 1, 0), (1, 0, 2), (1, 0, 1),
          (1, 0, 0), (0, 1, 2), (0, 1, 1), (0, 1, 0), (0, 0, 3), (0, 0, 2), (0, 0, 1), (0, 0, 0)]
_COLUMNS = [_MONOMIALS.index(m) for m in _ORDER]


def essential_from_constraints(points0, points1, coefficients, basis, singular, return_owners):
    from .warp_roots import determinant_roots
    from .two_view import essential_five_point
    reordered = coefficients[:, :, _COLUMNS]
    reduction, info = torch.linalg.solve_ex(reordered[:, :, :10], reordered[:, :, 10:])
    usable = (info == 0) & (singular[:, -1] > 1e-10 * singular[:, 0])
    usable &= torch.isfinite(reduction).all(dim=(-2, -1))
    owners = torch.arange(len(points0), device=points0.device)
    # Failed numerical reductions use the original elimination ordering.
    fallback = ~usable
    reduction = reduction[usable]
    samples = owners[usable]
    # Eliminate degree-three monomials, then subtract z times the neighboring
    # relation. Each row becomes x*bx(z) + y*by(z) + b0(z) = 0; det B(z)=0.
    rows0, rows1 = reduction[:, 4:10:2], reduction[:, 5:10:2]
    zero = reduction.new_zeros((len(reduction), 3, 1))
    blocks = []
    for lo, hi in ((0, 3), (3, 6), (6, 10)):
        blocks.append(torch.cat((zero, rows0[:, :, lo:hi]), -1)
                      - torch.cat((rows1[:, :, lo:hi], zero), -1))
    a = torch.cat(blocks, -1)
    roots, counts = determinant_roots(a)
    good_sample = counts >= 0
    fallback[samples[~good_sample]] = True
    valid = good_sample[:, None] & (torch.arange(10, device=points0.device)[None] < counts[:, None])
    sample_ids = torch.arange(len(samples), device=points0.device)[:, None].expand_as(valid)[valid]
    z = roots[valid]
    selected = a[sample_ids]
    bx = ((selected[:, :, 0] * z[:, None] + selected[:, :, 1]) * z[:, None] + selected[:, :, 2]) * z[:, None] + selected[:, :, 3]
    by = ((selected[:, :, 4] * z[:, None] + selected[:, :, 5]) * z[:, None] + selected[:, :, 6]) * z[:, None] + selected[:, :, 7]
    rhs = -((((selected[:, :, 8] * z[:, None] + selected[:, :, 9]) * z[:, None] + selected[:, :, 10])
            * z[:, None] + selected[:, :, 11]) * z[:, None] + selected[:, :, 12])
    b = torch.stack((bx, by), -1)
    row_pairs = torch.tensor([[0, 1], [0, 2], [1, 2]], device=points0.device)
    systems = b[:, row_pairs]
    determinants = systems[..., 0, 0] * systems[..., 1, 1] - systems[..., 0, 1] * systems[..., 1, 0]
    best = determinants.abs().argmax(-1)
    ids = torch.arange(len(z), device=points0.device)
    xy, status = torch.linalg.solve_ex(systems[ids, best], rhs[:, row_pairs][ids, best])
    coordinates = torch.cat((xy, z[:, None], torch.ones_like(z[:, None])), -1)
    original_ids = samples[sample_ids]
    models = torch.einsum('bijp,bp->bij', basis[original_ids], coordinates)
    models = F.normalize(models.flatten(-2), dim=-1).reshape(-1, 3, 3)
    eet = models @ models.transpose(-1, -2)
    constraint = 2 * eet @ models - eet.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None] * models
    fit = (b @ xy[..., None]).squeeze(-1) - rhs
    relative_error = fit.norm(dim=-1) / (b.norm(dim=(-2, -1)) * xy.norm(dim=-1) + rhs.norm(dim=-1)).clamp_min(1e-30)
    good = (status == 0) & torch.isfinite(models).all(dim=(-2, -1))
    good &= (constraint.norm(dim=(-2, -1)) < 1e-5) & (relative_error < 1e-7)
    fallback[original_ids[~good]] = True
    keep = good & ~fallback[original_ids]
    models, original_ids = models[keep], original_ids[keep]
    failed = torch.where(fallback)[0]
    if len(failed):
        reference, reference_owners = essential_five_point(points0[failed], points1[failed],
                                                          return_owners=True, backend='torch')
        models = torch.cat((models, reference))
        original_ids = torch.cat((original_ids, failed[reference_owners]))
    return (models, original_ids) if return_owners else models
