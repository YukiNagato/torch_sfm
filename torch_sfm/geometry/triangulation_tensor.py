"""Experimental fixed-slot tensor triangulation.

All authoritative observation/point state lives in ReconstructionStorage on the
reconstruction device. Geometry is gathered by index and evaluated in batches;
Python orchestrates batches, never individual projections or tiny matrix solves.
Parallel completion and disjoint merging have deterministic arbitration but do
not reproduce the sequential backend's mutation order. Creation selects the best
raw consensus before local refinement, instead of refining each improving pair.
Both older implementations remain selectable for policy/quality comparisons.
"""

import math

import torch
import torch.nn.functional as F
from PIL import Image

from torch_sfm.core.storage import ReconstructionStorage

from .triangulation import TriangulationConfig
from .two_view import camera_matrix


class TensorTriangulator:
    def __init__(self, frames, geometries, registered, config=None):
        self.config = config or TriangulationConfig(implementation="tensor")
        self.frames = frames
        self.registered = registered
        device = frames[0].keypoints_2d.device if frames else registered.device
        self.storage = ReconstructionStorage.from_frames(
            frames, device=device, dtype=torch.float64
        )
        self.candidate_cache = None  # compatible with pipeline diagnostic interface
        self.re_trials = {}
        self.pairs = {}
        self.sizes = []
        for frame in frames:
            size = getattr(frame, "image_size", None)
            if size is None and getattr(frame, "image", None) is not None:
                size = frame.image.shape[-2:]
            if size is None and getattr(frame, "frame_path", None) is not None:
                with Image.open(frame.frame_path) as im:
                    size = (im.height, im.width)
            self.sizes.append(size)
        edges = []
        for (i, j), geometry in sorted(geometries.items()):
            indices = geometry.matches.indices.to(device=device, dtype=torch.long)
            a = self.storage.global_ids(
                torch.full_like(indices[:, 0], i), indices[:, 0]
            )
            b = self.storage.global_ids(
                torch.full_like(indices[:, 1], j), indices[:, 1]
            )
            self.pairs[i, j] = torch.stack((a, b), -1)
            edges.extend((torch.stack((a, b), -1), torch.stack((b, a), -1)))
        edges = (
            torch.unique(torch.cat(edges), dim=0, sorted=True)
            if edges
            else torch.empty(0, 2, dtype=torch.long, device=device)
        )
        n = self.storage.capacity
        degree = torch.bincount(edges[:, 0], minlength=n)
        # Graph preprocessing for non-default transitivity happens only once.
        # Its final representation still has one fixed-width row per observation.
        if self.config.max_transitivity > 1:
            adjacency = [[] for _ in range(n)]
            for a, b in edges.cpu().tolist():
                adjacency[a].append(b)
            expanded = []
            for a, adjacent in enumerate(adjacency):
                if not adjacent:
                    continue
                seen, frontier = {a}, [a]
                for _ in range(self.config.max_transitivity):
                    following = []
                    for u in frontier:
                        for v in adjacency[u]:
                            if v not in seen:
                                seen.add(v)
                                following.append(v)
                                expanded.append((a, v))
                    frontier = following
            expanded = torch.tensor(expanded, dtype=torch.long, device=device).reshape(
                -1, 2
            )
        else:
            expanded = edges
        self.neighbors = self._pack_edges(edges, n)
        self.local_neighbors = (
            self.neighbors if expanded is edges else self._pack_edges(expanded, n)
        )
        self.degree = degree
        self.pure_two_view = degree == 1
        if self.neighbors.shape[1]:
            self.pure_two_view &= degree[self.neighbors[:, 0].clamp_min(0)] == 1

    @staticmethod
    def _pack_edges(edges, n):
        if not len(edges):
            return torch.empty(n, 0, dtype=torch.long, device=edges.device)
        edges = edges[torch.argsort(edges[:, 0], stable=True)]
        degree = torch.bincount(edges[:, 0], minlength=n)
        width = int(degree.max())
        neighbors = torch.full((n, width), -1, dtype=torch.long, device=edges.device)
        start = degree.cumsum(0) - degree
        neighbors[
            edges[:, 0],
            torch.arange(len(edges), device=edges.device) - start[edges[:, 0]],
        ] = edges[:, 1]
        return neighbors

    def update_geometry(self, rotations, centers, intrinsics, *, registered=None):
        """Refresh geometry and optionally camera registration, reusing topology."""
        if registered is not None:
            self.registered = registered
        s = self.storage
        if centers.device != s.xy.device:
            raise ValueError(
                "Tensor triangulation requires frames and geometry on the same device"
            )
        self.dtype = centers.dtype
        self.rotations = rotations.detach().double()
        self.centers = centers.detach().double()
        k = camera_matrix(intrinsics.detach().double())
        if k.ndim == 2:
            k = k.expand(len(self.frames), 3, 3)
        self.k = k
        self.poses = torch.cat(
            (self.rotations, -(self.rotations @ self.centers[..., None])), -1
        )
        self.calibration = k[:, [0, 1, 0, 1], [0, 1, 2, 2]]
        sizes = k.new_tensor(
            [
                (w, h) if size is not None else (0, 0)
                for size in self.sizes
                for h, w in [size or (0, 0)]
            ]
        ).reshape(-1, 2)
        sizes = torch.where(
            sizes > 0, sizes, (2 * self.calibration[:, 2:]).clamp_min(1)
        )
        extent = sizes.max(-1).values
        ratios = self.calibration[:, :2] / extent[:, None]
        valid = (
            self.registered.to(s.xy.device)
            & torch.isfinite(self.calibration).all(-1)
            & torch.isfinite(self.rotations).all(dim=(-1, -2))
            & torch.isfinite(self.centers).all(-1)
            & (self.calibration[:, 2:] >= 0).all(-1)
            & (self.calibration[:, 2:] <= sizes).all(-1)
            & (ratios >= self.config.min_focal_length_ratio).all(-1)
            & (ratios <= self.config.max_focal_length_ratio).all(-1)
        )
        self.valid_cameras = valid
        obs_valid = valid[s.frame_id] & torch.isfinite(s.xy).all(-1)
        s.obs_version[s.obs_valid != obs_valid] += 1
        s.obs_valid.copy_(obs_valid)
        calibration = self.calibration[s.frame_id]
        xy = (s.xy - calibration[:, 2:]) / calibration[:, :2]
        s.ray.copy_(
            F.normalize(
                torch.nan_to_num(torch.cat((xy, torch.ones_like(xy[:, :1])), -1)),
                dim=-1,
            )
        )
        s.geometry_version += 1
        s.point_dirty.copy_(s.point_alive)
        s.clear_work()

    def load(self, points, observations):
        self.storage.load(points, observations)

    def export(self):
        xyz, observations = self.storage.export()
        return xyz.to(dtype=self.dtype), observations

    def _errors(self, xyz, nodes, angular=False):
        """Broadcastable xyz (...,3) and observation indices (...)."""
        s = self.storage
        safe = nodes.clamp_min(0)
        fid = s.frame_id[safe]
        q = (self.rotations[fid] @ (xyz - self.centers[fid])[..., None]).squeeze(-1)
        if angular:
            ray = s.ray[safe]
            error = ((q * ray).sum(-1) / q.norm(dim=-1)).clamp(-1, 1).acos()
        else:
            k = self.calibration[fid]
            error = (
                q[..., :2] / q[..., 2:] * k[..., :2] + k[..., 2:] - s.xy[safe]
            ).norm(dim=-1)
        valid = (
            (nodes >= 0)
            & s.obs_valid[safe]
            & torch.isfinite(q).all(-1)
            & (q[..., 2] >= 1e-12)
        )
        return torch.where(
            valid & torch.isfinite(error), error, torch.full_like(error, math.inf)
        )

    def _stable(self, xyz, nodes, mask):
        safe = nodes.clamp_min(0)
        offset = xyz[:, None] - self.centers[self.storage.frame_id[safe]]
        directions = F.normalize(offset, dim=-1)
        cosine = (directions @ directions.transpose(-1, -2)).abs()
        pair_mask = torch.triu(mask[:, :, None] & mask[:, None, :], diagonal=1)
        wide = (
            (cosine <= math.cos(math.radians(self.config.min_angle))) & pair_mask
        ).any(dim=(-1, -2))
        return (
            torch.isfinite(xyz).all(-1)
            & (torch.isfinite(self._errors(xyz[:, None], nodes)) | ~mask).all(-1)
            & wide
        )

    def _hypotheses(self, nodes, pair):
        """Evaluate one bounded batch of pair hypotheses for each track."""
        s = self.storage
        mask = (nodes >= 0) & s.obs_valid[nodes.clamp_min(0)]
        b, _width = nodes.shape
        if not len(pair):
            return (
                s.xy.new_zeros(b, 3),
                torch.zeros_like(mask),
                torch.zeros(b, dtype=torch.long, device=nodes.device),
                s.xy.new_full((b,), math.inf),
            )
        pn = nodes[:, pair]
        safe = pn.clamp_min(0)
        poses = self.poses[s.frame_id[safe]]
        rays = s.ray[safe]
        xy = rays[..., :2] / rays[..., 2:]
        a = (xy[..., None] * poses[..., 2:3, :] - poses[..., :2, :]).reshape(
            b, len(pair), 4, 4
        )
        # Invalid padding is masked after solving a finite placeholder system.
        a = torch.nan_to_num(a)
        _, _, v = torch.linalg.svd(a)
        h = v[..., -1, :]
        denom = torch.where(
            h[..., 3:].abs() > 1e-300, h[..., 3:], torch.ones_like(h[..., 3:])
        )
        xyz = h[..., :3] / denom
        stable = self._stable(
            xyz.reshape(-1, 3), pn.reshape(-1, 2), (pn >= 0).reshape(-1, 2)
        ).reshape(b, -1)
        stable &= (pn >= 0).all(-1) & (h[..., 3].abs() > 1e-300)
        error = self._errors(xyz[:, :, None], nodes[:, None, :], angular=True)
        inside = (
            (error <= math.radians(self.config.create_max_angle_error))
            & mask[:, None]
            & stable[..., None]
        )
        counts = inside.sum(-1)
        squared = torch.where(inside, error.square(), 0.0).sum(-1)
        max_count = counts.max(-1, keepdim=True).values
        best = torch.where(counts == max_count, squared, math.inf).argmin(-1)
        rows = torch.arange(b, device=nodes.device)
        xyz, inside = xyz[rows, best], inside[rows, best]
        best_count, best_error = counts[rows, best], squared[rows, best]
        return xyz, inside, best_count, best_error

    def _estimate(self, nodes, *, lengths=None):
        backend = self.config.tensor_create_backend
        if backend != "torch" and self.storage.xy.is_cuda:
            from importlib.util import find_spec

            if backend == "warp" or find_spec("warp") is not None:
                from .warp_triangulation_create import estimate

                return estimate(
                    self,
                    nodes,
                    lengths=lengths,
                    mixed=lengths is not None
                    and self.config.tensor_create_bucketing == "none",
                )
        elif backend == "warp":
            raise ValueError("Warp track creation requires CUDA storage")
        return self._estimate_torch(nodes)

    def _estimate_torch(self, nodes):
        """Adaptive pair hypotheses and batched local multi-view refinement."""
        s = self.storage
        mask = (nodes >= 0) & s.obs_valid[nodes.clamp_min(0)]
        b, width = nodes.shape
        pairs = torch.triu_indices(width, width, offset=1, device=nodes.device).T[
            : self.config.max_num_trials
        ]
        step = (
            len(pairs)
            if width <= self.config.exhaustive_max_track_length
            else self.config.tensor_hypothesis_batch
        )
        xyz = s.xy.new_zeros(b, 3)
        inside = torch.zeros_like(mask)
        best_count = torch.zeros(b, dtype=torch.long, device=nodes.device)
        best_error = s.xy.new_full((b,), math.inf)
        needed = torch.full_like(best_count, len(pairs))
        total = mask.sum(-1)
        for start in range(0, len(pairs), max(1, step)):
            ids = torch.where(needed > start)[0]
            if not len(ids):
                break
            proposed, support, nc, ne = self._hypotheses(
                nodes[ids], pairs[start : start + step]
            )
            better = (nc > best_count[ids]) | (
                (nc == best_count[ids]) & (ne < best_error[ids])
            )
            winners = ids[better]
            xyz[winners], inside[winners] = proposed[better], support[better]
            best_count[winners], best_error[winners] = nc[better], ne[better]
            if width > self.config.exhaustive_max_track_length:
                probability = (
                    best_count.double()
                    * (best_count - 1)
                    / (total * (total - 1)).clamp_min(1)
                )
                trials = (
                    3
                    * math.log1p(-self.config.confidence)
                    / torch.log1p(-probability.clamp(1e-12, 1 - 1e-12))
                ).ceil()
                needed = trials.clamp(1, len(pairs)).long()
        poses = self.poses[s.frame_id[nodes.clamp_min(0)]]
        rays = s.ray[nodes.clamp_min(0)]
        a = poses - rays[..., :, None] * (rays[..., None, :] @ poses)
        a = torch.nan_to_num(a)
        normal = a.transpose(-1, -2) @ a
        active = best_count > 2
        for _ in range(self.config.local_iterations):
            if not bool(active.any()):
                break
            ids = torch.where(active)[0]
            matrix = (normal[ids] * inside[ids, :, None, None]).sum(1)
            _, vectors = torch.linalg.eigh(matrix)
            h = vectors[..., 0]
            safe_denom = torch.where(
                h[:, 3:].abs() > 1e-300, h[:, 3:], torch.ones_like(h[:, 3:])
            )
            refined = h[:, :3] / safe_denom
            errors = self._errors(refined[:, None], nodes[ids], angular=True)
            updated = (
                errors <= math.radians(self.config.create_max_angle_error)
            ) & mask[ids]
            nc = updated.sum(-1)
            ne = torch.where(updated, errors.square(), 0.0).sum(-1)
            stable = self._stable(refined, nodes[ids], inside[ids]) & (
                h[:, 3].abs() > 1e-300
            )
            better = stable & (
                (nc > best_count[ids])
                | ((nc == best_count[ids]) & (ne < best_error[ids]))
            )
            keep_running = better & (nc != best_count[ids]) & (nc > 2)
            winners = ids[better]
            xyz[winners], inside[winners] = refined[better], updated[better]
            best_count[winners], best_error[winners] = nc[better], ne[better]
            active.zero_()
            active[ids[keep_running]] = True
        inside &= (best_count >= 2)[:, None]
        return xyz, inside

    def _create(self, groups):
        s = self.storage
        if not groups.numel():
            return 0
        # Canonicalize/deduplicate groups before solving identical hypotheses.
        groups = torch.where(
            (groups >= 0)
            & (s.point_id[groups.clamp_min(0)] == -1)
            & s.obs_valid[groups.clamp_min(0)],
            groups,
            -1,
        )
        groups = torch.sort(groups, dim=-1).values
        groups = torch.unique(groups, dim=0, sorted=True)
        count = (groups >= 0).sum(-1)
        valid = count >= 2
        if self.config.ignore_two_view_tracks:
            first = groups.max(-1).values.clamp_min(0)
            valid &= ~((count == 2) & self.pure_two_view[first])
        groups, count = groups[valid], count[valid]
        added = 0
        widths = torch.unique(count).cpu().tolist()
        ranged = self.config.tensor_create_bucketing != "exact" and s.xy.is_cuda
        if ranged:
            from importlib.util import find_spec

            backend = self.config.tensor_create_backend
            ranged = backend == "warp" or (
                backend == "auto" and find_spec("warp") is not None
            )
        buckets = [(w - 1, w) for w in widths]
        unbucketed = ranged and self.config.tensor_create_bucketing == "none"
        if ranged and widths:
            threshold = self.config.exhaustive_max_track_length
            bounds = sorted({min(4, threshold), min(8, threshold), threshold})
            upper = max(32, 1 << threshold.bit_length())
            while upper < widths[-1]:
                bounds.append(upper)
                upper *= 2
            bounds.append(upper)
            buckets = []
            lower = 1
            for upper in bounds:
                present = [w for w in widths if lower < w <= upper]
                if present:
                    buckets.append((lower, max(present)))
                lower = upper
            if unbucketed:
                buckets = [(1, widths[-1])]
            # Keep shorter tracks ahead of longer ones, as in exact buckets.
            order = torch.argsort(count, stable=True)
            groups, count = groups[order], count[order]
        for lower, width in buckets:
            if unbucketed:
                selected, lengths = groups[:, -width:], count
            else:
                selection = (
                    ((count > lower) & (count <= width)) if ranged else count == width
                )
                selected = groups[selection, -width:]
                lengths = count[selection] if ranged else None
            hypotheses = min(width * (width - 1) // 2, self.config.max_num_trials)
            if width > self.config.exhaustive_max_track_length:
                hypotheses = min(hypotheses, self.config.tensor_hypothesis_batch)
            work = max(1, hypotheses * width)
            batch = max(
                1,
                min(
                    self.config.batch_lookahead,
                    self.config.batch_max_observations // work,
                ),
            )
            if unbucketed:
                # One candidate batch per Create call; bucket work caps do not apply.
                batch = len(selected)
            for start in range(0, len(selected), batch):
                nodes = selected[start : start + batch]
                # A previous batch may have claimed observations in this one.
                if lengths is None:
                    nodes = torch.where(s.point_id[nodes] == -1, nodes, -1)
                    xyz, inliers = self._estimate(nodes)
                else:
                    nodes = torch.where(
                        (nodes >= 0) & (s.point_id[nodes.clamp_min(0)] == -1), nodes, -1
                    )
                    xyz, inliers = self._estimate(
                        nodes, lengths=lengths[start : start + batch]
                    )
                rows, cols = torch.where(inliers)
                winner = torch.full_like(s.point_id, len(nodes))
                winner.scatter_reduce_(
                    0, nodes[rows, cols], rows, reduce="amin", include_self=True
                )
                # No partial commits: conflicting jobs can be reconsidered by
                # the next image pass with the then-current ownership snapshot.
                accepted = (
                    (
                        winner[nodes.clamp_min(0)]
                        == torch.arange(len(nodes), device=nodes.device)[:, None]
                    )
                    | ~inliers
                ).all(-1)
                accepted &= inliers.sum(-1) >= 2
                s.create(xyz[accepted], nodes[accepted], inliers[accepted])
                added += int(inliers[accepted].sum())
        return added

    def _continue(self, observations, neighbors, threshold):
        s = self.storage
        if not len(observations) or not neighbors.shape[1]:
            return 0
        owners = s.point_id[neighbors.clamp_min(0)]
        valid = (
            (neighbors >= 0)
            & (owners >= 0)
            & (s.point_id[observations, None] == -1)
            & s.obs_valid[observations, None]
        )
        error = self._errors(
            s.point_xyz[owners.clamp_min(0)], observations[:, None], angular=True
        )
        error = error.masked_fill(~valid, math.inf)
        minimum, best = error.min(-1)
        accepted = minimum <= math.radians(threshold)
        selected = owners.gather(1, best[:, None]).squeeze(-1)
        ids = observations[accepted]
        s.assign(ids, selected[accepted])
        return len(ids)

    def triangulate(self, frame_indices=None):
        """Process each image once; Create bounds its geometry work internally."""
        s = self.storage
        added = 0
        images = torch.where(self.valid_cameras)[0].cpu().tolist()
        if frame_indices is not None:
            requested = set(frame_indices)
            images = [image for image in images if image in requested]
        for image in images:
            lo, hi = s.frame_offsets[image : image + 2]
            ids = torch.arange(lo, hi, device=s.xy.device)
            ids = ids[(self.degree[ids] > 0) & s.obs_valid[ids]]
            if not len(ids):
                continue
            neighbors = self.local_neighbors[ids]
            added += self._continue(
                ids, neighbors, self.config.continue_max_angle_error
            )
            added += self._create(torch.cat((neighbors, ids[:, None]), -1))
        s.rebuild_tracks()
        return added

    def _edge_batches(self, sources=None):
        s = self.storage
        if not self.neighbors.shape[1]:
            return
        if sources is None:
            sources = torch.where((self.degree > 0) & s.obs_valid)[0]
        batch = max(1, self.config.batch_max_observations // self.neighbors.shape[1])
        for start in range(0, len(sources), batch):
            src = sources[start : start + batch]
            dst = self.neighbors[src]
            valid = (dst >= 0) & s.obs_valid[dst.clamp_min(0)]
            row, col = torch.where(valid)
            yield src[row], dst[row, col]

    def _complete(self):
        s = self.storage
        s.frontier_mask.copy_((s.point_id >= 0) & s.obs_valid)
        total = 0
        for _ in range(self.config.complete_max_transitivity):
            s.proposal_point.fill_(s.capacity)
            frontier = torch.where(s.frontier_mask)[0]
            for src, dst in self._edge_batches(frontier):
                unassigned = s.point_id[dst] == -1
                src, dst = src[unassigned], dst[unassigned]
                owners = s.point_id[src]
                error = self._errors(s.point_xyz[owners], dst)
                good = error <= self.config.complete_max_reproj_error
                s.proposal_point.scatter_reduce_(
                    0, dst[good], owners[good], reduce="amin", include_self=True
                )
            accepted = (
                (s.proposal_point < s.capacity) & (s.point_id == -1) & s.obs_valid
            )
            s.proposal_valid.copy_(accepted)
            s.next_frontier_mask.copy_(accepted)
            obs = torch.where(accepted)[0]
            if not len(obs):
                break
            s.assign(obs, s.proposal_point[obs])
            total += len(obs)
            s.frontier_mask.copy_(s.next_frontier_mask)
        s.rebuild_tracks()
        return total

    def _merge(self):
        backend = self.config.tensor_merge_backend
        if backend != "torch" and self.storage.xy.is_cuda:
            from importlib.util import find_spec

            if backend == "warp" or find_spec("warp") is not None:
                from .warp_triangulation import merge_tracks

                return merge_tracks(self)
        elif backend == "warp":
            raise ValueError("Warp track merging requires CUDA storage")
        return self._merge_torch()

    def _merge_torch(self):
        s = self.storage
        merged = 0
        while True:
            candidates = []
            for src, dst in self._edge_batches():
                a, b = s.point_id[src], s.point_id[dst]
                valid = (a >= 0) & (b >= 0) & (a != b)
                candidates.append(
                    torch.sort(torch.stack((a[valid], b[valid]), -1), dim=-1).values
                )
            if not candidates:
                break
            pairs = torch.unique(torch.cat(candidates), dim=0, sorted=True)
            if not len(pairs):
                break
            width = int(s.point_num_obs[pairs].max())
            batch = max(1, self.config.batch_max_observations // max(2 * width, 1))
            accepted_pairs, accepted_xyz = [], []
            index = torch.arange(width, device=s.xy.device)
            for start in range(0, len(pairs), batch):
                pair = pairs[start : start + batch]
                counts = s.point_num_obs[pair]
                positions = s.track_start[pair][..., None] + index
                mask = index < counts[..., None]
                nodes = s.track_obs[positions.clamp(0, max(s.capacity - 1, 0))]
                nodes = torch.where(mask, nodes, -1).flatten(1)
                weights = counts.to(s.xy.dtype)
                xyz = (s.point_xyz[pair] * weights[..., None]).sum(1) / weights.sum(
                    -1, keepdim=True
                )
                errors = self._errors(xyz[:, None], nodes)
                good = (
                    (errors <= self.config.merge_max_reproj_error) | (nodes < 0)
                ).all(-1)
                accepted_pairs.append(pair[good])
                accepted_xyz.append(xyz[good])
            pairs, xyz = torch.cat(accepted_pairs), torch.cat(accepted_xyz)
            if not len(pairs):
                break
            priority = torch.arange(len(pairs), device=s.xy.device)
            winner = torch.full_like(s.point_id, len(pairs))
            winner.scatter_reduce_(
                0,
                pairs.flatten(),
                priority[:, None].expand(-1, 2).flatten(),
                reduce="amin",
                include_self=True,
            )
            selected = (winner[pairs] == priority[:, None]).all(-1)
            pairs, xyz = pairs[selected], xyz[selected]
            merged += int(s.point_num_obs[pairs].sum())
            s.merge(pairs, xyz)
        return merged

    def complete_and_merge(self):
        return {"completed": self._complete(), "merged": self._merge()}

    def retriangulate(self):
        s = self.storage
        added = 0
        for pair, matches in self.pairs.items():
            if (
                not len(matches)
                or self.re_trials.get(pair, 0) >= self.config.re_max_trials
            ):
                continue
            if not bool(self.valid_cameras[list(pair)].all()):
                continue
            owners = s.point_id[matches]
            ratio = (
                ((owners[:, 0] >= 0) & (owners[:, 0] == owners[:, 1])).double().mean()
            )
            if float(ratio) >= self.config.re_min_ratio:
                continue
            self.re_trials[pair] = self.re_trials.get(pair, 0) + 1
            # Deduplicate anchors: each observation gets at most one assignment.
            for reverse in (False, True):
                current = matches.flip(-1) if reverse else matches
                unique = torch.unique(current[:, 0], sorted=True)
                neighbors = self.local_neighbors[unique]
                # Restrict continuation to this image pair's direct matches.
                target_image = pair[0] if reverse else pair[1]
                neighbors = torch.where(
                    s.frame_id[neighbors.clamp_min(0)] == target_image, neighbors, -1
                )
                added += self._continue(
                    unique, neighbors, self.config.re_max_angle_error
                )
            added += self._create(matches)
        return added
