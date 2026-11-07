"""Float64 tall-skinny Householder QR for nine-column local-refit systems.

No normal equations, precision retries, or CPU transfers. QR reduces independent
128-row panels before recursively reducing their stacked upper triangles.
"""
import torch
import warp as wp

wp.init()
wp.set_module_options({'enable_backward': False, 'fast_math': False,
                       'fuse_fp': False, 'max_unroll': 1})
BLOCK = 128
vec9d = wp.types.vector(length=9, dtype=wp.float64)


@wp.kernel
def _qr_panel(source: wp.array3d(dtype=wp.float64),
              target: wp.array3d(dtype=wp.float64)):
    sample, panel, lane = wp.tid()
    row = panel * BLOCK + lane
    a = vec9d()
    if row < source.shape[1]:
        for j in range(9):
            a[j] = source[sample, row, j]
    for k in range(9):
        x = wp.float64(0.0)
        pivot = wp.float64(0.0)
        if lane >= k:
            x = a[k]
        if lane == k:
            pivot = x
        norm = wp.sqrt(wp.tile_extract(wp.tile_sum(wp.tile(x*x)), 0))
        pivot = wp.tile_extract(wp.tile_sum(wp.tile(pivot)), 0)
        alpha = -norm
        if pivot < wp.float64(0.0):
            alpha = norm
        v = wp.float64(0.0)
        tau = wp.float64(0.0)
        if norm > wp.float64(0.0):
            tau = (alpha-pivot)/alpha
            if lane == k:
                v = wp.float64(1.0)
            elif lane > k:
                v = x/(pivot-alpha)
        for j in range(k+1, 9):
            dot = wp.tile_extract(wp.tile_sum(wp.tile(v*a[j])), 0)
            a[j] -= tau*v*dot
        if lane == k:
            a[k] = alpha
        elif lane > k:
            a[k] = wp.float64(0.0)
    if lane < 9:
        for j in range(9):
            target[sample, panel*9+lane, j] = a[j]


def reduced_qr(design):
    """Return a 9x9 R; inputs with fewer rows are implicitly zero-padded.

    This forward-only helper is used on normalized E/F/H design matrices.
    The caller retains the SVD and singular-value rank test.
    """
    if design.ndim != 3 or design.shape[-1] != 9:
        raise ValueError('Expected a (batch, rows, 9) design matrix')
    if not design.is_cuda or design.dtype != torch.float64:
        raise ValueError('Warp refit QR requires CUDA float64')
    if torch.is_grad_enabled() and design.requires_grad:
        raise ValueError('Warp refit QR does not support autograd')
    stream = wp.stream_from_torch(torch.cuda.current_stream(design.device))
    current = design
    while True:
        panels = max(1, (current.shape[1]+BLOCK-1)//BLOCK)
        target = design.new_empty((len(design),panels*9,9))
        if len(design):
            wp.launch(_qr_panel, dim=(len(design),panels,BLOCK), block_dim=BLOCK,
                      inputs=[wp.from_torch(current),wp.from_torch(target)],stream=stream)
        current = target
        if panels == 1:
            return current
