"""COLMAP positioning tracks: parallel connected components and exact conflicts.

Fixed observation slots, monotone root hooking and double-buffered pointer jumping.
No geometric triangulation or greedy merge arbitration participates in this policy.
"""
import torch
import warp as wp

from torch_sfm.core.storage import ObservationIndex

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})


@wp.kernel
def _hook(edges: wp.array2d(dtype=wp.int32), labels: wp.array(dtype=wp.int32),
          parent: wp.array(dtype=wp.int32), changed: wp.array(dtype=wp.int32)):
    i = wp.tid()
    a, b = labels[edges[i, 0]], labels[edges[i, 1]]
    if a != b:
        wp.atomic_min(parent, wp.max(a, b), wp.min(a, b))
        wp.atomic_max(changed, 0, 1)


@wp.kernel
def _jump(parent: wp.array(dtype=wp.int32), output: wp.array(dtype=wp.int32),
          changed: wp.array(dtype=wp.int32)):
    i = wp.tid()
    p = parent[i]
    q = parent[p]
    output[i] = q
    if p != q:
        wp.atomic_max(changed, 0, 1)


@wp.kernel
def _conflicts(order: wp.array(dtype=wp.int64), labels: wp.array(dtype=wp.int32),
               group: wp.array(dtype=wp.int64), ends: wp.array(dtype=wp.int64),
               xy: wp.array2d(dtype=wp.float64), bad: wp.array(dtype=wp.int32),
               threshold2: wp.float64):
    i = wp.tid()
    a = order[i]
    root = labels[a]
    if bad[root] != 0:
        return
    j = wp.int64(i + 1)
    end = ends[group[i]]
    while j < end:
        b = order[j]
        dx = xy[a, 0] - xy[b, 0]
        dy = xy[a, 1] - xy[b, 1]
        if dx * dx + dy * dy > threshold2:
            wp.atomic_max(bad, root, 1)
            break
        j += wp.int64(1)


@torch.no_grad()
def connected_components(edges, num_nodes):
    """Return the minimum node ID of each component, including isolated nodes."""
    if edges.device.type != 'cuda' or edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError('edges must be a CUDA (E, 2) tensor')
    if not 0 <= num_nodes < 2**31:
        raise ValueError('Warp connected components require fewer than 2**31 nodes')
    if edges.dtype not in (torch.int32, torch.int64):
        raise ValueError('edges must have integer indices')
    if edges.numel() and bool(((edges < 0) | (edges >= num_nodes)).any()):
        raise ValueError('edge node index out of range')
    labels = torch.arange(num_nodes, device=edges.device, dtype=torch.int32)
    if not edges.numel():
        return labels
    wp.init()
    edges = edges.to(torch.int32).contiguous()
    stream = wp.stream_from_torch(torch.cuda.current_stream(edges.device))
    changed = labels.new_zeros(1)
    parent, scratch = torch.empty_like(labels), torch.empty_like(labels)
    we, wc = wp.from_torch(edges), wp.from_torch(changed)
    # Roots decrease strictly. Separate buffers avoid concurrent parent reads/writes.
    while True:
        parent.copy_(labels)
        changed.zero_()
        wp.launch(_hook, dim=len(edges), inputs=[we, wp.from_torch(labels),
                  wp.from_torch(parent), wc], stream=stream)
        if not changed.item():
            return labels
        while True:
            changed.zero_()
            wp.launch(_jump, dim=num_nodes, inputs=[wp.from_torch(parent),
                      wp.from_torch(scratch), wc], stream=stream)
            parent, scratch = scratch, parent
            if not changed.item():
                break
        labels, parent = parent, labels


@torch.no_grad()
def build_colmap_tracks(geometries, active_frames, frames, min_views, threshold):
    index = ObservationIndex.from_frames(frames, dtype=torch.float64)
    xy, frame, feature = index.xy, index.frame_id, index.feature_id
    if xy.device.type != 'cuda':
        raise ValueError('Warp track construction requires CUDA keypoints')
    empty = torch.empty((0, 3), device=xy.device, dtype=torch.long)
    active = set(active_frames)
    arrays = []
    for (i, j), g in geometries.items():
        if i in active and j in active and len(g.matches.indices):
            local = g.matches.indices.to(device=xy.device, dtype=torch.long)
            arrays.append(local + local.new_tensor([index.frame_offsets[i], index.frame_offsets[j]]))
    if not arrays:
        return empty
    edges = torch.cat(arrays)
    n = len(xy)
    labels = connected_components(edges, n)
    # Stable sorting preserves (frame, feature) order within each component.
    order = torch.argsort(labels, stable=True)
    roots = labels[order].long()
    counts = torch.bincount(roots, minlength=n)
    group_keys = roots * len(frames) + frame[order]
    _, group, group_counts = torch.unique_consecutive(group_keys, return_inverse=True, return_counts=True)
    ends = group_counts.cumsum(0)
    starts = ends - group_counts
    group_roots = roots[starts]
    views = torch.bincount(group_roots, minlength=n)
    bad = torch.zeros(n, dtype=torch.int32, device=xy.device)
    stream = wp.stream_from_torch(torch.cuda.current_stream(xy.device))
    wp.launch(_conflicts, dim=n, inputs=[wp.from_torch(order), wp.from_torch(labels),
              wp.from_torch(group), wp.from_torch(ends), wp.from_torch(xy),
              wp.from_torch(bad), threshold * threshold], stream=stream)
    retained = ((views >= min_views) & (bad == 0)).nonzero().flatten()
    # CPU sorts by (-track length, lexicographic track). Distinct components have
    # distinct minimum observations, so root order resolves equal-size ties exactly.
    retained = retained[torch.argsort(counts[retained], descending=True, stable=True)]
    track = torch.full((n,), -1, dtype=torch.long, device=xy.device)
    track[retained] = torch.arange(len(retained), device=xy.device)
    ids = track[labels.long()]
    selected = (ids >= 0).nonzero().flatten()
    selected = selected[torch.argsort(ids[selected], stable=True)]
    return torch.stack((ids[selected], frame[selected], feature[selected]), -1)
