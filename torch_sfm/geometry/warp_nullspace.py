"""Guarded right null spaces of full-row-rank 7x9/8x9 systems in float64.

Householder QR of A.T avoids a normal matrix. A conservative singular-value
ratio bound selects the fast path; ambiguous rank decisions use Torch SVD.
"""
import torch
import warp as wp

wp.init()
wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False,
                       'max_unroll': 1})
mat99d = wp.types.matrix(shape=(9,9), dtype=wp.float64)
vec9d = wp.types.vector(length=9, dtype=wp.float64)
vec8d = wp.types.vector(length=8, dtype=wp.float64)


@wp.kernel
def _nullspace(design: wp.array3d(dtype=wp.float64), vectors: wp.array3d(dtype=wp.float64),
               safe: wp.array(dtype=wp.bool), rows: int):
    sample = wp.tid()
    scale = wp.float64(0.0)
    finite = wp.bool(True)
    for i in range(rows):
        for j in range(9):
            value = design[sample,i,j]
            scale = wp.max(scale,wp.abs(value))
            finite = finite and wp.isfinite(value)
    safe[sample] = False
    if not finite or scale == wp.float64(0.0):
        return
    # Upper triangle stores R; below it are Householder tails with implicit 1.
    a = mat99d()
    tau = vec8d()
    for i in range(9):
        for j in range(rows):
            a[i,j] = design[sample,j,i]/scale
    for k in range(rows):
        norm2 = wp.float64(0.0)
        for i in range(k,9):
            norm2 += a[i,k]*a[i,k]
        norm = wp.sqrt(norm2)
        if norm < wp.float64(1e-14):
            return
        alpha = -norm
        if a[k,k] < wp.float64(0.0):
            alpha = norm
        denominator = a[k,k]-alpha
        tau[k] = (alpha-a[k,k])/alpha
        for i in range(k+1,9):
            a[i,k] /= denominator
        a[k,k] = alpha
        for j in range(k+1,rows):
            dot = a[k,j]
            for i in range(k+1,9):
                dot += a[i,k]*a[i,j]
            dot *= tau[k]
            a[k,j] -= dot
            for i in range(k+1,9):
                a[i,j] -= a[i,k]*dot
    norm_r2 = wp.float64(0.0)
    norm_inverse2 = wp.float64(0.0)
    for j in range(rows):
        column = vec8d()
        for step in range(rows):
            i = rows-1-step
            value = wp.float64(0.0)
            if i == j:
                value = wp.float64(1.0)
            for k in range(i+1,rows):
                value -= a[i,k]*column[k]
            column[i] = value/a[i,i]
            norm_inverse2 += column[i]*column[i]
        for i in range(j+1):
            norm_r2 += a[i,j]*a[i,j]
    # 1/(||R||F ||inv(R)||F) <= sigma_min(A)/sigma_max(A).
    # Tenfold margin above the existing 1e-10 rank threshold. All uncertain
    # cases (including singular input) are deferred to the exact old test.
    ratio_bound = wp.float64(1.0)/wp.sqrt(norm_r2*norm_inverse2)
    if not wp.isfinite(ratio_bound) or ratio_bound <= wp.float64(1e-9):
        return
    for slot in range(9-rows):
        q = vec9d()
        q[rows+slot] = wp.float64(1.0)
        for step in range(rows):
            k = rows-1-step
            dot = q[k]
            for i in range(k+1,9):
                dot += a[i,k]*q[i]
            dot *= tau[k]
            q[k] -= dot
            for i in range(k+1,9):
                q[i] -= a[i,k]*dot
        norm2 = wp.float64(0.0)
        for j in range(9):
            norm2 += q[j]*q[j]
        norm = wp.sqrt(norm2)
        residual2 = wp.float64(0.0)
        for i in range(rows):
            value = wp.float64(0.0)
            for j in range(9):
                value += design[sample,i,j]/scale*q[j]
            residual2 += value*value
        if not wp.isfinite(norm) or wp.abs(norm-wp.float64(1.0)) > wp.float64(1e-12):
            return
        if residual2 > wp.float64(1e-24)*norm_r2:
            return
        for j in range(9):
            vectors[sample,j,slot] = q[j]/norm
    safe[sample] = True


def right_nullspace(design):
    """Return orthonormal columns and the original SVD rank-test decisions."""
    if design.ndim != 3 or design.shape[-1] != 9 or design.shape[-2] not in (7,8):
        raise ValueError('Expected a batch of 7x9 or 8x9 matrices')
    if not design.is_cuda or design.dtype != torch.float64:
        raise ValueError('Warp nullspace requires CUDA float64')
    rows = design.shape[-2]
    vectors = design.new_empty((len(design),9,9-rows))
    good = torch.empty(len(design),device=design.device,dtype=torch.bool)
    if not len(design):
        return vectors,good
    wp.launch(_nullspace,dim=len(design),inputs=[wp.from_torch(design),
              wp.from_torch(vectors),wp.from_torch(good),rows],
              stream=wp.stream_from_torch(torch.cuda.current_stream(design.device)))
    fallback = torch.where(~good)[0]
    if len(fallback):
        _,singular,vh = torch.linalg.svd(design[fallback],full_matrices=True)
        vectors[fallback] = vh[:,rows:].transpose(-1,-2)
        good[fallback] = singular[:,-1] > 1e-10*singular[:,0]
    return vectors,good


def right_null_vector8(design):
    if design.shape[-2:] != (8,9):
        raise ValueError('Expected a batch of 8x9 matrices')
    vectors,good = right_nullspace(design)
    return vectors[:,:,0],good
