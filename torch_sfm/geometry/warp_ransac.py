"""Fused sampling and LO-RANSAC state updates on the current Torch stream."""

import math

import torch
import warp as wp

wp.init()
wp.set_module_options({'enable_backward': False})
BLOCK = 128


@wp.kernel
def _hash(counts: wp.array(dtype=wp.int64), valid: wp.array2d(dtype=wp.bool),
          trial: wp.int64, bias: wp.uint32, values: wp.array3d(dtype=wp.float64)):
    pair, hypothesis, point = wp.tid()
    value = wp.float64(-1.0)
    if valid[pair, point]:
        counter = (trial + wp.int64(hypothesis)) * counts[pair] + wp.int64(point)
        bits = wp.uint32(counter) + bias
        bits = (bits ^ (bits >> wp.uint32(16))) * wp.uint32(2246822519)
        bits = (bits ^ (bits >> wp.uint32(13))) * wp.uint32(3266489917)
        bits = bits ^ (bits >> wp.uint32(16))
        value = wp.float64(bits)
    values[pair, hypothesis, point] = value


def sample_indices(counts, valid, trial, batch, size, seed):
    values = torch.empty((len(counts), batch, valid.shape[1]), device=valid.device, dtype=torch.float64)
    wp.launch(_hash, dim=values.shape,
        inputs=[wp.from_torch(counts), wp.from_torch(valid), wp.int64(trial),
                wp.uint32(((seed % 2**32) * 104729 + 374761393) & 0xffffffff)],
        outputs=[wp.from_torch(values)], stream=wp.stream_from_torch(torch.cuda.current_stream(valid.device)))
    return values.topk(size, -1).indices


@wp.kernel
def _select(costs: wp.array(dtype=wp.float64), owners: wp.array(dtype=wp.int64),
            candidates: wp.array3d(dtype=wp.float64), active: wp.array(dtype=wp.int64),
            models: wp.array3d(dtype=wp.float64), best_score: wp.array(dtype=wp.float64),
            found: wp.array(dtype=wp.bool), sorted_owners: int,
            selected: wp.array(dtype=wp.int64), improved: wp.array(dtype=wp.bool),
            errors: wp.array2d(dtype=wp.float64), valid: wp.array2d(dtype=wp.bool),
            threshold: wp.float64, masks: wp.array2d(dtype=wp.bool)):
    pair, lane = wp.tid()
    first, last = int(0), costs.shape[0]
    if sorted_owners != 0:
        # Existing packed minimal solvers already produce contiguous owner
        # segments. Locate each segment without another allocation or launch.
        lo, hi = int(0), costs.shape[0]
        while lo < hi:
            mid = (lo + hi) // 2
            if owners[mid] < wp.int64(pair):
                lo = mid + 1
            else:
                hi = mid
        first = lo
        hi = costs.shape[0]
        while lo < hi:
            mid = (lo + hi) // 2
            if owners[mid] <= wp.int64(pair):
                lo = mid + 1
            else:
                hi = mid
        last = lo
    local_cost = wp.float64(wp.inf)
    local_index = wp.int64(costs.shape[0])
    for index in range(first + lane, last, BLOCK):
        if owners[index] == wp.int64(pair):
            cost = costs[index]
            if cost < local_cost or (cost == local_cost and wp.int64(index) < local_index):
                local_cost = cost
                local_index = wp.int64(index)
    minimum = wp.tile_extract(wp.tile_min(wp.tile(local_cost)), 0)
    if local_cost != minimum:
        local_index = wp.int64(costs.shape[0])
    winner = wp.tile_extract(wp.tile_min(wp.tile(local_index)), 0)
    target = active[pair]
    flag = int(0)
    if lane == 0:
        flag = int(winner < wp.int64(costs.shape[0]) and minimum < best_score[target])
    # All lanes observe acceptance before lane 0 updates the incumbent.
    accept = wp.tile_extract(wp.tile_sum(wp.tile(flag)), 0) != 0
    if lane == 0:
        selected[pair] = winner
        improved[pair] = accept
        found[target] = found[target] or accept
        best_score[target] = wp.min(minimum, best_score[target])
        if accept:
            for row in range(3):
                for col in range(3):
                    models[target, row, col] = candidates[winner, row, col]
    if accept:
        for point in range(lane, masks.shape[1], BLOCK):
            masks[target, point] = valid[target, point] and errors[winner, point] <= threshold


@wp.kernel
def _local(costs: wp.array(dtype=wp.float64), good: wp.array(dtype=wp.bool),
           candidates: wp.array3d(dtype=wp.float64), pair_ids: wp.array(dtype=wp.int64),
           survivors: wp.array(dtype=wp.int64), models: wp.array3d(dtype=wp.float64),
           best_score: wp.array(dtype=wp.float64), improved: wp.array(dtype=wp.bool),
           accepted: wp.array(dtype=wp.bool), errors: wp.array2d(dtype=wp.float64),
           valid: wp.array2d(dtype=wp.bool), threshold: wp.float64,
           masks: wp.array2d(dtype=wp.bool)):
    pair, lane = wp.tid()
    target = pair_ids[pair]
    flag = int(0)
    if lane == 0:
        flag = int(good[pair] and costs[pair] < best_score[target])
    accept = wp.tile_extract(wp.tile_sum(wp.tile(flag)), 0) != 0
    if lane == 0:
        accepted[pair] = accept
        improved[survivors[pair]] = accept
        if accept:
            best_score[target] = costs[pair]
            for row in range(3):
                for col in range(3):
                    models[target, row, col] = candidates[pair, row, col]
    if accept:
        for point in range(lane, masks.shape[1], BLOCK):
            masks[target, point] = valid[target, point] and errors[pair, point] <= threshold


@wp.func_native('return log1pf(x);')
def _log1p(x: wp.float32) -> wp.float32: ...


@wp.kernel
def _budget(active: wp.array(dtype=wp.int64), masks: wp.array2d(dtype=wp.bool),
            sizes: wp.array(dtype=wp.int64), limits: wp.array(dtype=wp.int64),
            required: wp.array(dtype=wp.int64), sample_size: int, log_failure: wp.float32,
            min_trials: wp.int64, max_trials: wp.int64):
    pair, lane = wp.tid()
    target = active[pair]
    count = wp.int64(0)
    for point in range(lane, masks.shape[1], BLOCK):
        if masks[target, point]:
            count += wp.int64(1)
    total = wp.tile_extract(wp.tile_sum(wp.tile(count)), 0)
    if lane == 0:
        # Torch integer / integer division promotes to its default float32.
        # Keep float32 pow/log1p/division so ceil sees the same budget boundary.
        ratio = wp.float32(total) / wp.float32(sizes[target])
        probability = wp.pow(ratio, wp.float32(sample_size))
        estimate = max_trials
        if probability >= wp.float32(1.0):
            estimate = min_trials
        elif probability > wp.float32(0.0):
            value = wp.ceil(log_failure / _log1p(-probability))
            if wp.isfinite(value):
                estimate = wp.int64(wp.clamp(value, wp.float32(min_trials), wp.float32(max_trials)))
        required[target] = wp.min(estimate, limits[target])


class State:
    """Per-call views of Torch state; independent calls share no mutable buffers."""

    def __init__(self, models, masks, best_score, found, valid, sizes, limits, required, cfg):
        self.reference = best_score
        self.capacity = masks.shape[1]
        self.models, self.masks, self.best_score, self.found, self.valid, self.sizes, self.limits, self.required = [
            wp.from_torch(t) for t in (models, masks, best_score, found, valid, sizes, limits, required)]
        self.stream = wp.stream_from_torch(torch.cuda.current_stream(models.device))
        self.threshold = wp.float64(cfg.max_error ** 2)
        self.log_failure = wp.float32(math.log1p(-cfg.confidence))
        self.min_trials, self.max_trials = wp.int64(cfg.min_trials), wp.int64(cfg.max_trials)
        # Prefixes are active for one round at a time. Masks and model outputs
        # keep their independent storage; only internal transient flags reuse.
        self.selected = torch.empty_like(sizes)
        self.improved = torch.empty_like(found)
        self.accepted = torch.empty_like(found)
        self.winner_view, self.improved_view, self.accepted_view = [
            wp.from_torch(t) for t in (self.selected, self.improved, self.accepted)]

    def select(self, candidates, owners, costs, active, errors, *, sorted_owners=False):
        """Return transient flags, valid until the next select on this state.

        sorted_owners is a producer guarantee, never inferred by a GPU readback.
        False handles arbitrary owner order, including legacy fallback solvers.
        """
        ids, winner, flags = wp.from_torch(active), self.winner_view, self.improved_view
        wp.launch(_select, dim=(len(active), BLOCK), block_dim=BLOCK,
            inputs=[wp.from_torch(costs), wp.from_torch(owners), wp.from_torch(candidates), ids,
                    self.models, self.best_score, self.found, int(sorted_owners), winner, flags,
                    wp.from_torch(errors), self.valid, self.threshold, self.masks], stream=self.stream)
        return self.improved[:len(active)]

    def local(self, refined, good, costs, pair_ids, survivors, improved, errors):
        ids, flags = wp.from_torch(pair_ids), self.accepted_view
        wp.launch(_local, dim=(len(pair_ids), BLOCK), block_dim=BLOCK,
            inputs=[wp.from_torch(costs), wp.from_torch(good),
            wp.from_torch(refined), ids, wp.from_torch(survivors), self.models, self.best_score,
            wp.from_torch(improved), flags, wp.from_torch(errors), self.valid, self.threshold,
            self.masks], stream=self.stream)

    def update_budget(self, active, sample_size):
        wp.launch(_budget, dim=(len(active), BLOCK), block_dim=BLOCK,
            inputs=[wp.from_torch(active), self.masks, self.sizes, self.limits, self.required,
                    sample_size, self.log_failure, self.min_trials, self.max_trials], stream=self.stream)
