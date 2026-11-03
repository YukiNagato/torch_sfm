"""Optional fused hypothesis scoring. Warp arrays alias Torch on its current stream."""

import torch
import warp as wp

wp.init()


@wp.kernel
def _score(models: wp.array3d(dtype=wp.float64),
           owners: wp.array(dtype=wp.int64),
           points0: wp.array3d(dtype=wp.float64), points1: wp.array3d(dtype=wp.float64),
           valid: wp.array2d(dtype=wp.bool), homography: int,
           errors: wp.array2d(dtype=wp.float64)):
    model, point = wp.tid()
    pair = owners[model]
    error = wp.float64(0.0)
    if valid[pair, point]:
        x, y = points0[pair, point, 0], points0[pair, point, 1]
        u, v = points1[pair, point, 0], points1[pair, point, 1]
        a = models[model, 0, 0] * x + models[model, 0, 1] * y + models[model, 0, 2]
        b = models[model, 1, 0] * x + models[model, 1, 1] * y + models[model, 1, 2]
        c = models[model, 2, 0] * x + models[model, 2, 1] * y + models[model, 2, 2]
        if homography != 0:
            error = wp.float64(wp.inf)
            if wp.abs(c) > wp.float64(1e-12):
                dx, dy = a / c - u, b / c - v
                error = dx * dx + dy * dy
        else:
            d = models[model, 0, 0] * u + models[model, 1, 0] * v + models[model, 2, 0]
            e = models[model, 0, 1] * u + models[model, 1, 1] * v + models[model, 2, 1]
            numerator = u * a + v * b + c
            error = numerator * numerator / wp.max(a * a + b * b + d * d + e * e, wp.float64(1e-30))
        if not wp.isfinite(error):
            error = wp.float64(wp.inf)
    errors[model, point] = error


def score_models(models, owners, points0, points1, valid, homography=False):
    errors = models.new_empty((len(models), points0.shape[1]))
    wp.launch(_score, dim=errors.shape,
        inputs=[wp.from_torch(models), wp.from_torch(owners), wp.from_torch(points0),
                wp.from_torch(points1), wp.from_torch(valid), int(homography)],
        outputs=[wp.from_torch(errors)], stream=wp.stream_from_torch(torch.cuda.current_stream(models.device)))
    return errors
