"""Fused cheirality voting and tangent Sampson scoring on Torch's CUDA stream."""
import torch
import warp as wp

wp.init()


@wp.kernel
def _vote(owners: wp.array(dtype=wp.int64),
          rays0: wp.array3d(dtype=wp.float64), rays1: wp.array3d(dtype=wp.float64),
          valid: wp.array2d(dtype=wp.bool), rotations: wp.array4d(dtype=wp.float64),
          translation: wp.array2d(dtype=wp.float64),
          flags: wp.array2d(dtype=wp.int32), counts: wp.array2d(dtype=wp.int32)):
    model, point = wp.tid()
    pair = owners[model]
    bits = int(0)
    if valid[pair, point]:
        b = wp.vec3d(rays1[pair, point, 0], rays1[pair, point, 1], rays1[pair, point, 2])
        t = wp.vec3d(translation[model, 0], translation[model, 1], translation[model, 2])
        second = wp.dot(b, t)
        for pose in range(2):
            rotated = wp.vec3d(wp.float64(0.0))
            for row in range(3):
                value = wp.float64(0.0)
                for col in range(3):
                    value += rotations[model, pose, row, col] * rays0[pair, point, col]
                rotated[row] = value
            cosine = -wp.dot(rotated, b)
            first = -wp.dot(rotated, t)
            depth0, depth1 = first - cosine * second, second - cosine * first
            if depth0 > wp.float64(0.0) and depth1 > wp.float64(0.0):
                bits = bits | (1 << pose)
                wp.atomic_add(counts, model, pose, 1)
            if depth0 < wp.float64(0.0) and depth1 < wp.float64(0.0):
                bits = bits | (1 << (pose + 2))
                wp.atomic_add(counts, model, pose + 2, 1)
    flags[model, point] = bits


@wp.kernel
def _score(models: wp.array3d(dtype=wp.float64), owners: wp.array(dtype=wp.int64),
           rays0: wp.array3d(dtype=wp.float64), rays1: wp.array3d(dtype=wp.float64),
           jac0: wp.array4d(dtype=wp.float64), jac1: wp.array4d(dtype=wp.float64),
           valid: wp.array2d(dtype=wp.bool), flags: wp.array2d(dtype=wp.int32),
           best: wp.array(dtype=wp.int64), errors: wp.array2d(dtype=wp.float64)):
    model, point = wp.tid()
    pair = owners[model]
    error = wp.float64(0.0)
    if valid[pair, point]:
        error = wp.float64(wp.inf)
        if (flags[model, point] & (1 << int(best[model]))) != 0:
            line0, line1 = wp.vec3d(wp.float64(0.0)), wp.vec3d(wp.float64(0.0))
            for row in range(3):
                for col in range(3):
                    line0[row] += models[model, col, row] * rays1[pair, point, col]
                    line1[row] += models[model, row, col] * rays0[pair, point, col]
            numerator, denominator = wp.float64(0.0), wp.float64(0.0)
            for row in range(3):
                numerator += rays1[pair, point, row] * line1[row]
            for axis in range(2):
                a, b = wp.float64(0.0), wp.float64(0.0)
                for row in range(3):
                    a += jac0[pair, point, row, axis] * line0[row]
                    b += jac1[pair, point, row, axis] * line1[row]
                denominator += a * a + b * b
            if denominator > wp.float64(0.0):
                value = numerator * numerator / denominator
                if wp.isfinite(value):
                    error = value
    errors[model, point] = error


def score(models, owners, rays0, rays1, jac0, jac1, valid, rotations, translation,
          *, return_ambiguous=False):
    shape = (len(models), valid.shape[1])
    flags = torch.empty(shape, device=models.device, dtype=torch.int32)
    counts = torch.zeros((len(models), 4), device=models.device, dtype=torch.int32)
    stream = wp.stream_from_torch(torch.cuda.current_stream(models.device))
    wp.launch(_vote, dim=shape,
        inputs=[wp.from_torch(x) for x in (owners, rays0, rays1, valid, rotations, translation)],
        outputs=[wp.from_torch(flags), wp.from_torch(counts)], stream=stream)
    best = (counts * 4 + torch.arange(4, device=models.device)).argmax(-1)
    errors = models.new_empty(shape)
    wp.launch(_score, dim=shape,
        inputs=[wp.from_torch(x) for x in (models, owners, rays0, rays1, jac0, jac1, valid, flags, best)],
        outputs=[wp.from_torch(errors)], stream=stream)
    if return_ambiguous:
        ambiguous = (counts == counts.amax(-1, keepdim=True)).sum(-1) > 1
        return errors, ambiguous
    return errors
