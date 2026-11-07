"""Optional fused hypothesis scoring. Warp arrays alias Torch on its current stream."""

import torch
import warp as wp

wp.init()
wp.set_module_options({'enable_backward': False})
BLOCK = 128


@wp.func
def _error(models: wp.array3d(dtype=wp.float64),
           owners: wp.array(dtype=wp.int64),
           points0: wp.array3d(dtype=wp.float64), points1: wp.array3d(dtype=wp.float64),
           valid: wp.array2d(dtype=wp.bool), homography: int,
           model: int, point: int):
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
    return error


@wp.kernel
def _score(models: wp.array3d(dtype=wp.float64), owners: wp.array(dtype=wp.int64),
           points0: wp.array3d(dtype=wp.float64), points1: wp.array3d(dtype=wp.float64),
           valid: wp.array2d(dtype=wp.bool), homography: int,
           errors: wp.array2d(dtype=wp.float64)):
    model, point = wp.tid()
    errors[model, point] = _error(models, owners, points0, points1, valid, homography, model, point)


@wp.kernel
def _score_cost(models: wp.array3d(dtype=wp.float64), owners: wp.array(dtype=wp.int64),
                points0: wp.array3d(dtype=wp.float64), points1: wp.array3d(dtype=wp.float64),
                valid: wp.array2d(dtype=wp.bool), homography: int, threshold: wp.float64,
                errors: wp.array2d(dtype=wp.float64), costs: wp.array(dtype=wp.float64)):
    model, lane = wp.tid()
    total = wp.float64(0.0)
    for point in range(lane, points0.shape[1], BLOCK):
        error = _error(models, owners, points0, points1, valid, homography, model, point)
        errors[model, point] = error
        total += wp.min(error, threshold)
    value = wp.tile_extract(wp.tile_sum(wp.tile(total)), 0)
    if lane == 0:
        costs[model] = value


class ModelScorer:
    """Per-RANSAC immutable observation views; owners index the original batch.

Outputs have independent storage, so later local fits cannot overwrite a
previous candidate's errors. No global cache or cross-stream mutable workspace.
"""

    def __init__(self, points0, points1, valid, threshold, homography=False):
        self.references = (points0, points1, valid)
        self.points = points0.shape[1]
        self.observations = [wp.from_torch(v) for v in (points0, points1, valid)]
        self.homography = int(homography)
        self.threshold = wp.float64(threshold)
        self.stream = wp.stream_from_torch(torch.cuda.current_stream(points0.device))

    def __call__(self, models, owners):
        errors = models.new_empty((len(models), self.points))
        costs = models.new_empty(len(models))
        if len(models):
            wp.launch(_score_cost, dim=(len(models), BLOCK), block_dim=BLOCK,
                inputs=[wp.from_torch(models), wp.from_torch(owners), *self.observations,
                        self.homography, self.threshold],
                outputs=[wp.from_torch(errors), wp.from_torch(costs)], stream=self.stream)
        return errors, costs


def score_models(models, owners, points0, points1, valid, homography=False):
    errors = models.new_empty((len(models), points0.shape[1]))
    wp.launch(_score, dim=errors.shape,
        inputs=[wp.from_torch(models), wp.from_torch(owners), wp.from_torch(points0),
                wp.from_torch(points1), wp.from_torch(valid), int(homography)],
        outputs=[wp.from_torch(errors)], stream=wp.stream_from_torch(torch.cuda.current_stream(models.device)))
    return errors
