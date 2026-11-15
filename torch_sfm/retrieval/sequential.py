"""Temporal windows plus additional global retrieval, as used for LaMAR."""
import json
from pathlib import Path

import numpy as np
import torch


def load_sequence_ids(path, num_images):
    """Read ordered start/end ranges aligned with images.txt (end exclusive)."""
    records = json.loads(Path(path).read_text())
    ids = np.full(num_images, -1, dtype=np.int64)
    for label, record in enumerate(records):
        start, end = record['start'], record['end']
        if (not isinstance(start, int) or not isinstance(end, int)
                or not 0 <= start < end <= num_images or (ids[start:end] >= 0).any()
                or record.get('count', end-start) != end-start):
            raise ValueError('Invalid or overlapping sequence ranges')
        ids[start:end] = label
    if (ids < 0).any():
        raise ValueError('Sequence ranges must cover every input image')
    return ids


def temporal_neighbors(sequence_ids, num_neighbors=30):
    """Centered windows in input order, extended at sequence boundaries."""
    if isinstance(num_neighbors, bool) or not isinstance(num_neighbors, int) or num_neighbors < 0:
        raise ValueError('num_neighbors must be a nonnegative integer')
    ids = np.asarray(sequence_ids)
    if ids.ndim != 1:
        raise ValueError('sequence_ids must be one-dimensional')
    neighbors = [np.empty(0, dtype=np.int64) for _ in ids]
    for label in np.unique(ids):
        members = np.flatnonzero(ids == label)
        count = min(num_neighbors, len(members)-1)
        for local, center in enumerate(members):
            left = max(0, local-(count+1)//2)
            left = min(left, len(members)-count-1)
            window = members[left:left+count+1]
            neighbors[center] = window[window != center]
    return neighbors


def _pairs(rows):
    pairs = {(min(i, int(j)), max(i, int(j))) for i, row in enumerate(rows) for j in row if i != j}
    return np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)


@torch.no_grad()
def retrieve_sequence_pairs(similarities, sequence_ids, *, global_neighbors=100, sequential_neighbors=30):
    """Return canonical pairs and temporal edges. Global neighbors exclude the local window.

    A global neighbor may belong to the same recording or another recording.
    Input order within each sequence must be temporal order. The score matrix
    is not modified; pass None when global_neighbors=0.
    """
    if isinstance(global_neighbors, bool) or not isinstance(global_neighbors, int) or global_neighbors < 0:
        raise ValueError('global_neighbors must be a nonnegative integer')
    local = temporal_neighbors(sequence_ids, sequential_neighbors)
    temporal = _pairs(local)
    n = len(local)
    if global_neighbors == 0 or n < 2:
        return temporal, temporal.copy()
    if similarities is None or similarities.shape != (n, n) or not similarities.is_floating_point():
        raise ValueError('Expected a floating-point N x N similarity matrix')
    if not torch.isfinite(similarities).all():
        raise ValueError('Retrieval similarities must be finite')
    scores = similarities.clone()
    scores.fill_diagonal_(-torch.inf)
    for i, neighbors in enumerate(local):
        scores[i, torch.as_tensor(neighbors, device=scores.device)] = -torch.inf
    values, indices = scores.topk(min(global_neighbors, n-1), dim=1)
    values, indices = values.cpu(), indices.cpu()
    rows = [np.concatenate((local[i], indices[i, torch.isfinite(values[i])].numpy())) for i in range(n)]
    return _pairs(rows), temporal
