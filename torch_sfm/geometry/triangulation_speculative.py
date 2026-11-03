"""Experimental batched evidence with sequential observation ownership.

The reference implementation remains in triangulation.py. This backend retains
its traversal, creation and merge policy (including completion during the image
pass). Speculation may be discarded; it never decides observation ownership.
Geometry caches are bounded by batch_max_observations and cleared on load/update.
"""

import math

import torch
import torch.nn.functional as F

from .triangulation import IncrementalTriangulator
from .triangulation_batch import CandidateCache
from .two_view import camera_matrix


class SpeculativeTriangulator(IncrementalTriangulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._static_key = None
        self._errors = {}
        self._completed = {}
        self._gpu = None
        self.speculation_stats = {
            "batches": 0,
            "queries": 0,
            "hits": 0,
            "misses": 0,
            "completion_skips": 0,
            "merge_hits": 0,
        }

    def _reset_evidence(self):
        self._errors.clear()
        self._completed.clear()
        self._gpu = None

    def load(self, points, observations):
        super().load(points, observations)
        self._reset_evidence()

    def update_geometry(self, rotations, centers, intrinsics):
        self._reset_evidence()
        self.device, self.dtype = centers.device, centers.dtype
        self.rotations = rotations.detach().cpu().double()
        self.centers = centers.detach().cpu().double()
        k = camera_matrix(intrinsics.detach().cpu().double())
        if k.ndim == 2:
            k = k.expand(len(self.frames), 3, 3)
        self.k = k
        self.poses = torch.cat(
            (self.rotations, -(self.rotations @ self.centers[..., None])), -1
        )
        self.r, self.c = self.rotations.tolist(), self.centers.tolist()
        self.calibration = k[:, [0, 1, 0, 1], [0, 1, 2, 2]].tolist()
        self.valid_cameras = set()
        for i in self.active:
            fx, fy, cx, cy = self.calibration[i]
            w, h = self.sizes[i]
            w, h = w or max(1.0, 2 * cx), h or max(1.0, 2 * cy)
            if (
                all(math.isfinite(v) for v in (fx, fy, cx, cy))
                and 0 <= cx <= w
                and 0 <= cy <= h
                and all(
                    self.config.min_focal_length_ratio
                    <= ratio
                    <= self.config.max_focal_length_ratio
                    for ratio in (fx / max(w, h), fy / max(w, h))
                )
            ):
                self.valid_cameras.add(i)
        # Features/matching topology are immutable between load/update calls.
        # Include tensor identity/version so replacing or editing pixels refreshes
        # the cache. load() can append isolated observation nodes.
        key = (
            len(self.nodes),
            tuple(
                (
                    id(f.keypoints_2d),
                    object()
                    if torch.is_inference(f.keypoints_2d)
                    else f.keypoints_2d._version,
                )
                for f in self.frames
            ),
        )
        if key != self._static_key:
            pixels = [f.keypoints_2d.detach().cpu().double() for f in self.frames]
            self.frame_ids = torch.tensor([i for i, _ in self.nodes], dtype=torch.long)
            self.xy = torch.empty((len(self.nodes), 2), dtype=torch.float64)
            # Vectorized gather per image instead of one Torch op per feature.
            groups = {}
            for n, (i, j) in enumerate(self.nodes):
                groups.setdefault(i, []).append((n, j))
            for i, entries in groups.items():
                dst, src = torch.tensor(entries, dtype=torch.long).unbind(1)
                self.xy[dst] = pixels[i][src]
            self.pixel_list = self.xy.tolist()
            self._neighbors = [tuple(sorted(adj)) for adj in self.adj]
            self._static_key = key
        calibrated = (self.xy - k[self.frame_ids, :2, 2]) / k[self.frame_ids][
            :, [0, 1], [0, 1]
        ]
        self.rays = F.normalize(
            torch.cat((calibrated, torch.ones(len(calibrated), 1)), -1), dim=-1
        )
        enabled = self.config.numerical_backend == "batched" or (
            self.config.numerical_backend == "auto" and self.device.type == "cuda"
        )
        self.candidate_cache = CandidateCache(self) if enabled else None

    def local(self, n):
        seen, frontier, result = {n}, [n], []
        for _ in range(self.config.max_transitivity):
            following = []
            for u in frontier:
                for v in self._neighbors[u]:
                    if v not in seen:
                        seen.add(v)
                        following.append(v)
                        if self.nodes[v][0] in self.valid_cameras:
                            result.append(v)
            frontier = following
        return result

    def error(self, n, xyz, angular=False):
        key = (tuple(xyz), n, angular)
        value = self._errors.get(key)
        if value is not None:
            self.speculation_stats["hits"] += 1
            return value
        self.speculation_stats["misses"] += 1
        return super().error(n, xyz, angular)

    def _prepare_errors(self, jobs, angular=False):
        """Speculate only geometry. Borderline predicates use reference arithmetic."""
        if self.config.numerical_backend == "scalar":
            return
        limit = self.config.batch_max_observations
        jobs = list(
            dict.fromkeys(
                (xyz, n, angular)
                for xyz, n in jobs
                if (xyz, n, angular) not in self._errors
            )
        )
        jobs = jobs[: max(0, limit - len(self._errors))]
        if len(jobs) < self.config.speculative_min_queries:
            return
        rotations, centers, frame_ids, pixels, calibration = self._device_data()
        xyz = torch.tensor(
            [j[0] for j in jobs], dtype=torch.float64, device=self.device
        )
        nodes = torch.tensor([j[1] for j in jobs], dtype=torch.long, device=self.device)
        fid = frame_ids[nodes]
        q = (rotations[fid] @ (xyz - centers[fid])[..., None]).squeeze(-1)
        k, xy = calibration[fid], pixels[nodes]
        if angular:
            ray = torch.cat(((xy - k[:, 2:]) / k[:, :2], torch.ones_like(q[:, :1])), -1)
            cosine = (q * ray).sum(-1) / (
                q.square().sum(-1) * ray.square().sum(-1)
            ).sqrt()
            error = cosine.clamp(-1, 1).acos()
            thresholds = [
                math.radians(v)
                for v in (
                    self.config.continue_max_angle_error,
                    self.config.re_max_angle_error,
                    self.config.create_max_angle_error,
                )
            ]
            epsilon = 5e-8
        else:
            error = (q[:, :2] / q[:, 2:] * k[:, :2] + k[:, 2:] - xy).norm(dim=-1)
            thresholds = [
                self.config.complete_max_reproj_error,
                self.config.merge_max_reproj_error,
            ]
            epsilon = 1e-7
        invalid = ~torch.isfinite(q).all(-1) | (q[:, 2] < 1e-12)
        error = error.masked_fill(invalid, math.inf)
        ambiguous = (q[:, 2] - 1e-12).abs() <= 1e-12
        for threshold in thresholds:
            ambiguous |= (error - threshold).abs() <= epsilon
        values = torch.stack((error, ambiguous.to(error.dtype)), -1).cpu().tolist()
        for job, (value, fallback) in zip(jobs, values, strict=True):
            if fallback or math.isnan(value):
                value = super().error(job[1], job[0], angular)
            self._errors[job] = value
        self.speculation_stats["batches"] += 1
        self.speculation_stats["queries"] += len(jobs)

    def _device_data(self):
        if self._gpu is None:
            self._gpu = tuple(
                t.to(self.device)
                for t in (
                    self.rotations,
                    self.centers,
                    self.frame_ids,
                    self.xy,
                    self.k[:, [0, 1, 0, 1], [0, 1, 2, 2]],
                )
            )
        return self._gpu

    def _prepare_completion(self, ids):
        """Bounded first-frontier lookahead; later frontiers fall back safely.

        No closure of the correspondence graph is materialized: bad components can be
        huge. Most already reconstructed tracks only need this immediate frontier.
        """
        jobs = []
        limit = self.config.batch_max_observations
        for p in dict.fromkeys(ids):
            if p not in self.points or self._completed.get(p) == len(self.tracks[p]):
                continue
            xyz = self.points[p]
            seen = set()
            for u in self.tracks[p]:
                for n in self._neighbors[u]:
                    if (
                        n not in seen
                        and self.assignment[n] == -1
                        and self.nodes[n][0] in self.valid_cameras
                    ):
                        seen.add(n)
                        jobs.append((xyz, n))
                        if len(jobs) >= limit:
                            self._prepare_errors(jobs)
                            return
        self._prepare_errors(jobs)

    def complete(self, p):
        if p not in self.points:
            return 0
        if self._completed.get(p) == len(self.tracks[p]):
            self.speculation_stats["completion_skips"] += 1
            return 0
        visited = set(self.tracks[p])
        frontier = sorted(visited)
        count = 0
        for _ in range(self.config.complete_max_transitivity):
            following = []
            for u in frontier:
                for v in self._neighbors[u]:
                    if v in visited:
                        continue
                    visited.add(v)
                    if (
                        self.nodes[v][0] not in self.valid_cameras
                        or self.assignment[v] != -1
                    ):
                        continue
                    if (
                        self.error(v, self.points[p])
                        <= self.config.complete_max_reproj_error
                    ):
                        self.assignment[v] = p
                        self.tracks[p].add(v)
                        following.append(v)
                        count += 1
            frontier = following
            if not frontier:
                # Saturated at this geometry. Future assignment can only claim
                # nodes, never free them. A new root changes track length.
                self._completed[p] = len(self.tracks[p])
                break
        return count

    def triangulate(self):
        added = 0
        order = [
            n for i in sorted(self.valid_cameras) for n in self.by_image.get(i, [])
        ]
        size = self.config.batch_lookahead
        for start in range(0, len(order), size):
            window = order[start : start + size]
            self._errors.clear()
            self._prepare_completion(
                self.assignment[n] for n in window if self.assignment[n] != -1
            )
            local, tracks, jobs = {}, [], []
            for n in window:
                if self.assignment[n] != -1 or (
                    self.config.ignore_two_view_tracks and self.pure_two_view(n)
                ):
                    continue
                local[n] = self.local(n)
                for p in {self.assignment[v] for v in local[n]} - {-1}:
                    jobs.append((self.points[p], n))
                    if len(jobs) >= self.config.batch_max_observations:
                        break
                nodes = list(
                    dict.fromkeys(v for v in [*local[n], n] if self.assignment[v] == -1)
                )
                if 2 <= len(nodes) <= self.config.exhaustive_max_track_length:
                    tracks.append(nodes)
            self._prepare_errors(
                jobs[: self.config.batch_max_observations], angular=True
            )
            if self.candidate_cache:
                self.candidate_cache.prepare(tracks)
            for n in window:
                if self.assignment[n] != -1:
                    added += self.complete(self.assignment[n])
                    continue
                if self.config.ignore_two_view_tracks and self.pure_two_view(n):
                    continue
                neighbors = local[n] if n in local else self.local(n)
                if neighbors:
                    added += self.continue_track(n, neighbors)
                    added += self.create([*neighbors, n])
        return added

    def _merge_candidates(self, p):
        return sorted(
            {
                self.assignment[v]
                for n in self.tracks[p]
                for v in self._neighbors[n]
                if self.assignment[v] not in (-1, p)
            }
        )

    def _merge_geometry(self, p, q):
        a, b = len(self.tracks[p]), len(self.tracks[q])
        xyz = tuple(
            (a * x + b * y) / (a + b)
            for x, y in zip(self.points[p], self.points[q], strict=True)
        )
        return xyz, self.tracks[p] | self.tracks[q]

    def _prepare_merges(self, ids):
        pending, count = {}, 0
        limit = self.config.batch_max_observations
        for p in ids:
            if p not in self.points:
                continue
            for q in self._merge_candidates(p):
                pair = tuple(sorted((p, q)))
                if pair in pending:
                    continue
                xyz, nodes = self._merge_geometry(p, q)
                if count + len(nodes) > limit:
                    return self._evaluate_merges(pending, count)
                pending[pair] = (xyz, nodes, None)
                count += len(nodes)
        return self._evaluate_merges(pending, count)

    def _evaluate_merges(self, pending, count):
        """Reduce on device: transfer one decision per pair, not every error."""
        if (
            count < self.config.speculative_min_queries
            or self.config.numerical_backend == "scalar"
        ):
            return pending
        entries = list(pending.values())
        xyz = torch.tensor(
            [entry[0] for entry in entries], dtype=torch.float64, device=self.device
        )
        lengths = torch.tensor([len(entry[1]) for entry in entries], device=self.device)
        model = torch.repeat_interleave(
            torch.arange(len(entries), device=self.device), lengths, output_size=count
        )
        nodes = torch.tensor(
            [n for entry in entries for n in entry[1]],
            dtype=torch.long,
            device=self.device,
        )
        rotations, centers, frame_ids, pixels, calibration = self._device_data()
        fid = frame_ids[nodes]
        q = (rotations[fid] @ (xyz[model] - centers[fid])[..., None]).squeeze(-1)
        k = calibration[fid]
        error = (q[:, :2] / q[:, 2:] * k[:, :2] + k[:, 2:] - pixels[nodes]).norm(dim=-1)
        threshold = self.config.merge_max_reproj_error
        valid = torch.isfinite(q).all(-1) & (q[:, 2] >= 1e-12) & (error <= threshold)
        ambiguous = ((q[:, 2] - 1e-12).abs() <= 1e-12) | (
            (error - threshold).abs() <= 1e-7
        )
        # 0 = pass, 1 = reject, 2 = evaluate with reference arithmetic.
        decision = torch.where(ambiguous, 2, (~valid).to(torch.int64))
        verdict = torch.zeros(len(entries), dtype=torch.int64, device=self.device)
        verdict.scatter_reduce_(0, model, decision, reduce="amax", include_self=True)
        for pair, value in zip(pending, verdict.cpu().tolist(), strict=True):
            xyz, nodes, _ = pending[pair]
            pending[pair] = (xyz, nodes, None if value == 2 else value == 0)
        self.speculation_stats["batches"] += 1
        self.speculation_stats["queries"] += count
        return pending

    def complete_and_merge(self):
        ids = list(self.points)
        size = self.config.batch_lookahead
        completed = 0
        for start in range(0, len(ids), size):
            window = ids[start : start + size]
            self._errors.clear()
            self._prepare_completion(window)
            completed += sum(self.complete(p) for p in window)
        merged, tried = 0, set()
        ids = list(self.points)
        for start in range(0, len(ids), size):
            window = ids[start : start + size]
            self._errors.clear()
            pending = self._prepare_merges(window)
            for initial in window:
                p, chain_size = initial, 0
                while p in self.points:
                    success = False
                    for q in self._merge_candidates(p):
                        pair = tuple(sorted((p, q)))
                        if pair in tried:
                            continue
                        tried.add(pair)
                        # Merging deletes both IDs and allocates a fresh ID.
                        # Existing IDs never change during this merge pass, so
                        # their identity is also their geometry/track version.
                        if pair in pending:
                            xyz, nodes, valid = pending[pair]
                            self.speculation_stats["merge_hits"] += 1
                        else:
                            xyz, nodes = self._merge_geometry(p, q)
                            valid = None
                        if valid is None:
                            valid = all(
                                self.error(n, xyz) <= self.config.merge_max_reproj_error
                                for n in nodes
                            )
                        if not valid:
                            continue
                        del (
                            self.points[p],
                            self.points[q],
                            self.tracks[p],
                            self.tracks[q],
                        )
                        p = self.add(xyz, nodes)
                        chain_size, success = len(nodes), True
                        break
                    if not success:
                        break
                merged += chain_size
        return {"completed": completed, "merged": merged}

    def retriangulate(self):
        count = 0
        for pair, matches in self.pairs.items():
            if not set(pair) <= self.valid_cameras or not matches:
                continue
            common = sum(
                self.assignment[a] != -1 and self.assignment[a] == self.assignment[b]
                for a, b in matches
            )
            if (
                common / len(matches) >= self.config.re_min_ratio
                or self.re_trials.get(pair, 0) >= self.config.re_max_trials
            ):
                continue
            self.re_trials[pair] = self.re_trials.get(pair, 0) + 1
            size = self.config.batch_lookahead
            for start in range(0, len(matches), size):
                window = matches[start : start + size]
                self._errors.clear()
                tracks, jobs = [], []
                for a, b in window:
                    pa, pb = self.assignment[a], self.assignment[b]
                    if pa == -1 and pb == -1:
                        tracks.append([a, b])
                    elif pa == -1:
                        jobs.append((self.points[pb], a))
                    elif pb == -1:
                        jobs.append((self.points[pa], b))
                self._prepare_errors(jobs, angular=True)
                if self.candidate_cache:
                    self.candidate_cache.prepare(tracks)
                for a, b in window:
                    pa, pb = self.assignment[a], self.assignment[b]
                    if pa != -1 and pb != -1:
                        continue
                    if pa == -1 and pb == -1:
                        count += self.create([a, b])
                    elif pa == -1:
                        count += self.continue_track(
                            a, [b], self.config.re_max_angle_error
                        )
                    else:
                        count += self.continue_track(
                            b, [a], self.config.re_max_angle_error
                        )
        return count
