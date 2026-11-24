"""Schedule independent image pairs for reuse of their GPU feature tensors."""
from heapq import heapify, heappop, heappush

import numpy as np


def tile_pair_order(pairs, capacity):
    """Group by the unordered pair of tiles, retaining input order within a tile pair."""
    # At most two tiles are needed for any group. Visit adjacent rows in
    # opposite directions so successive groups can share one tile.
    first, second = (np.asarray(pairs) // (capacity // 2)).T
    left, right = np.minimum(first, second), np.maximum(first, second)
    return np.lexsort((np.where(left % 2 == 0, right, -right), left))


def belady_evictions(pairs, capacity):
    """Yield feature IDs to evict before each pair, starting with an empty cache."""
    end = len(pairs)
    next_uses = [None] * end
    last = {}
    for index in range(end - 1, -1, -1):
        left, right = pairs[index]
        next_uses[index] = last.get(left, end), last.get(right, end)
        last[left] = last[right] = index

    resident = {}
    priorities = []
    for (left, right), (next_left, next_right) in zip(pairs, next_uses):
        evicted = []
        protected = []
        for frame, future in ((left, next_left), (right, next_right)):
            if frame not in resident and len(resident) == capacity:
                while priorities:
                    priority, victim = heappop(priorities)
                    if resident.get(victim) != -priority:
                        continue
                    if victim in (left, right):
                        protected.append((priority, victim))
                        continue
                    del resident[victim]
                    evicted.append(victim)
                    break
            resident[frame] = future
            heappush(priorities, (-future, frame))
        for entry in protected:
            heappush(priorities, entry)
        # Bound the number of entries left behind by priority updates.
        if len(priorities) > 4 * capacity:
            priorities = [(-future, frame) for frame, future in resident.items()]
            heapify(priorities)
        yield evicted
