"""Fused four-pose voting for local shared-focal refinement.

Keep cuSOLVER's SVD and its candidate ordering, including the first-candidate
tie rule. One cooperative block evaluates all four poses without allocating
candidate-by-point intermediate tensors. This is midpoint angular voting,
not the all-five test used by the minimal essential solver.
"""
import torch
import warp as wp
from .relative_pose import _matrix

wp.set_module_options({'enable_backward': False, 'fast_math': False,
                       'fuse_fp': False, 'max_unroll': 1})
BLOCK = 128


@wp.kernel
def _choose(u: wp.array3d(dtype=wp.float64), vh: wp.array3d(dtype=wp.float64),
            rays0: wp.array3d(dtype=wp.float64), rays1: wp.array3d(dtype=wp.float64),
            mask: wp.array2d(dtype=wp.bool), rotation: wp.array3d(dtype=wp.float64),
            translation: wp.array2d(dtype=wp.float64)):
    i, lane = wp.tid()
    left, right = _matrix(u, i), _matrix(vh, i)
    du, dv = wp.determinant(left), wp.determinant(right)
    for k in range(3):
        left[k, 2] *= du
        right[2, k] *= dv
    w = wp.mat33d(wp.float64(0.0), wp.float64(-1.0), wp.float64(0.0),
                 wp.float64(1.0), wp.float64(0.0), wp.float64(0.0),
                 wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
    chosen_r, chosen_t = wp.mat33d(), wp.vec3d()
    best_count, best_cost = int(-1), wp.float64(0.0)
    for candidate in range(4):
        middle = w
        if candidate >= 2:
            middle = wp.transpose(w)
        r = left * middle * right
        t = wp.vec3d(left[0, 2], left[1, 2], left[2, 2])
        if candidate % 2 == 1:
            t = -t
        center = -(wp.transpose(r) * t)
        count, cost = int(0), wp.float64(0.0)
        for j in range(lane, rays0.shape[1], BLOCK):
            a = wp.vec3d(rays0[i,j,0], rays0[i,j,1], rays0[i,j,2])
            b = wp.vec3d(rays1[i,j,0], rays1[i,j,1], rays1[i,j,2])
            a = a / wp.max(wp.length(a), wp.float64(1e-12))
            b = wp.transpose(r) * (b / wp.max(wp.length(b), wp.float64(1e-12)))
            cosine = wp.dot(a,b)
            denominator = wp.max(wp.float64(1.0)-cosine*cosine, wp.float64(1e-15))
            ac, bc = wp.dot(a,center), wp.dot(b,center)
            d0, d1 = (ac-cosine*bc)/denominator, (cosine*ac-bc)/denominator
            point = (d0*a + center + d1*b)*wp.float64(.5)
            positive = (mask[i,j] and d0>wp.float64(1e-8) and d1>wp.float64(1e-8)
                        and wp.isfinite(point[0]) and wp.isfinite(point[1]) and wp.isfinite(point[2]))
            error = point/wp.max(wp.length(point),wp.float64(1e-12))-a
            count += int(positive)
            # Preserve NaN propagation of the reference's errors * positive.
            cost += wp.dot(error,error)*wp.float64(int(positive))
        total = wp.tile_extract(wp.tile_sum(wp.tile(count)),0)
        score = wp.tile_extract(wp.tile_sum(wp.tile(cost)),0)
        if total>best_count or (total==best_count and
                (score<best_cost or (wp.isnan(score) and not wp.isnan(best_cost)))):
            best_count, best_cost, chosen_r, chosen_t = total, score, r, t
    if lane == 0:
        for row in range(3):
            translation[i,row] = chosen_t[row]
            for col in range(3):
                rotation[i,row,col] = chosen_r[row,col]


def choose_pose(essential, rays0, rays1, mask):
    u, _, vh = torch.linalg.svd(essential)
    rotation = torch.empty_like(essential)
    translation = essential.new_empty((len(essential),3))
    if len(essential):
        wp.launch(_choose, dim=(len(essential),BLOCK), block_dim=BLOCK,
                  inputs=[wp.from_torch(v) for v in (u,vh,rays0,rays1,mask,rotation,translation)],
                  stream=wp.stream_from_torch(torch.cuda.current_stream(essential.device)))
    return rotation, translation
