# Copyright (c), ETH Zurich and UNC Chapel Hill. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of ETH Zurich and UNC Chapel Hill nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Python/Torch incremental triangulation following COLMAP's track policies.

Topology is managed in Python on the CPU. CUDA inputs use bounded look-ahead
to batch initial DLT candidates and geometric evidence across tracks. Local
refinement and assignment decisions retain the sequential CPU policy.
Camera/feature traversal and sampling order are deterministic here and need
not match COLMAP's hash-container traversal or C++ sampler bit for bit.
"""

import math
from dataclasses import dataclass
from itertools import combinations, islice

import torch
import torch.nn.functional as F
from PIL import Image

from .triangulation_batch import CandidateCache
from .two_view import camera_matrix


@dataclass
class TriangulationConfig:
    max_transitivity: int = 1
    create_max_angle_error: float = 2.0
    continue_max_angle_error: float = 2.0
    merge_max_reproj_error: float = 15.0
    complete_max_reproj_error: float = 15.0
    complete_max_transitivity: int = 5
    re_max_angle_error: float = 5.0
    re_min_ratio: float = 0.2
    re_max_trials: int = 1
    min_angle: float = 1.0
    ignore_two_view_tracks: bool = True
    min_focal_length_ratio: float = 0.1
    max_focal_length_ratio: float = 10.0
    max_num_trials: int = 10000
    confidence: float = 0.9999
    local_iterations: int = 10
    exhaustive_max_track_length: int = 15
    numerical_backend: str = "auto"
    batch_lookahead: int = 1024
    batch_max_observations: int = 65536
    implementation: str = "sequential"
    speculative_min_queries: int = 128
    tensor_hypothesis_batch: int = 32
    tensor_create_backend: str = "auto"
    tensor_merge_backend: str = "auto"
    tensor_merge_matching: str = "maximal"
    tensor_merge_cache: bool = True

    def __post_init__(self):
        if self.implementation not in ("sequential", "speculative", "tensor"):
            raise ValueError("implementation must be sequential, speculative or tensor")
        if self.speculative_min_queries < 1:
            raise ValueError("speculative_min_queries must be positive")
        if self.tensor_hypothesis_batch < 1:
            raise ValueError("tensor_hypothesis_batch must be positive")
        if self.tensor_create_backend not in ("auto", "torch", "warp"):
            raise ValueError("tensor_create_backend must be auto, torch or warp")
        if self.tensor_merge_backend not in ("auto", "torch", "warp"):
            raise ValueError("tensor_merge_backend must be auto, torch or warp")
        if self.tensor_merge_matching not in ("mutual", "maximal"):
            raise ValueError("tensor_merge_matching must be mutual or maximal")
        if self.numerical_backend not in ("auto", "scalar", "batched"):
            raise ValueError("numerical_backend must be auto, scalar or batched")
        if self.batch_lookahead < 1 or self.batch_max_observations < 1:
            raise ValueError("Triangulation batch limits must be positive")
        if (
            min(
                self.max_transitivity,
                self.complete_max_transitivity,
                self.max_num_trials,
            )
            < 1
        ):
            raise ValueError(
                "Triangulation traversal and trial limits must be positive"
            )
        if self.re_max_trials < 0 or self.local_iterations < 0:
            raise ValueError("Triangulation iteration limits must be nonnegative")
        if self.exhaustive_max_track_length < 2:
            raise ValueError("exhaustive_max_track_length must be at least two")
        for value in (
            self.create_max_angle_error,
            self.continue_max_angle_error,
            self.re_max_angle_error,
            self.merge_max_reproj_error,
            self.complete_max_reproj_error,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "Triangulation error thresholds must be positive and finite"
                )
        if not 0 <= self.min_angle < 90 or not 0 < self.confidence < 1:
            raise ValueError("Invalid triangulation angle or confidence")
        if not 0 <= self.re_min_ratio <= 1:
            raise ValueError("re_min_ratio must be in [0, 1]")
        if not 0 < self.min_focal_length_ratio <= self.max_focal_length_ratio:
            raise ValueError("Invalid focal length ratio bounds")


class IncrementalTriangulator:
    def __init__(self, frames, geometries, registered, config=None):
        self.config = config or TriangulationConfig()
        self.frames = frames
        self.active = set(torch.where(registered)[0].tolist())
        self.nodes, self.lookup, self.adj = [], {}, []
        self.pairs = {}
        for (i, j), geometry in sorted(geometries.items()):
            matches = []
            for a, b in geometry.matches.indices.cpu().tolist():
                u, v = self.node(i, a), self.node(j, b)
                self.adj[u].add(v)
                self.adj[v].add(u)
                matches.append((u, v))
            self.pairs[i, j] = matches
        self.by_image = {}
        for n, (image, _) in enumerate(self.nodes):
            self.by_image.setdefault(image, []).append(n)
        for image in self.by_image:
            self.by_image[image].sort(key=lambda n: self.nodes[n][1])
        self.assignment = [-1] * len(self.nodes)
        self.points, self.tracks = {}, {}
        self.next_id = 0
        self.re_trials = {}
        self.candidate_cache = None
        self.sizes = []
        for frame in frames:
            if getattr(frame, "image_size", None) is not None:
                h, w = frame.image_size
            elif getattr(frame, "image", None) is not None:
                h, w = frame.image.shape[-2:]
            elif getattr(frame, "frame_path", None) is not None:
                with Image.open(frame.frame_path) as image:
                    w, h = image.size
            else:
                h = w = None
            self.sizes.append((w, h))

    def node(self, image, feature):
        key = int(image), int(feature)
        if key not in self.lookup:
            self.lookup[key] = len(self.nodes)
            self.nodes.append(key)
            self.adj.append(set())
        return self.lookup[key]

    def update_geometry(self, rotations, centers, intrinsics):
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
        self.valid_cameras = set()
        self.r = self.rotations.tolist()
        self.c = self.centers.tolist()
        self.calibration = k[:, [0, 1, 0, 1], [0, 1, 2, 2]].tolist()
        for i in self.active:
            fx, fy, cx, cy = self.calibration[i]
            w, h = self.sizes[i]
            w = w or max(1.0, 2 * cx)
            h = h or max(1.0, 2 * cy)
            ratios = (fx / max(w, h), fy / max(w, h))
            if (
                all(math.isfinite(v) for v in (fx, fy, cx, cy))
                and 0 <= cx <= w
                and 0 <= cy <= h
                and all(
                    self.config.min_focal_length_ratio
                    <= ratio
                    <= self.config.max_focal_length_ratio
                    for ratio in ratios
                )
            ):
                self.valid_cameras.add(i)
        pixels = [frame.keypoints_2d.detach().cpu().double() for frame in self.frames]
        self.xy = (
            torch.stack([pixels[i][j] for i, j in self.nodes])
            if self.nodes
            else k.new_empty(0, 2)
        )
        self.frame_ids = torch.tensor([i for i, _ in self.nodes], dtype=torch.long)
        calibrated = self.xy - k[self.frame_ids, :2, 2]
        calibrated = calibrated / k[self.frame_ids][:, [0, 1], [0, 1]]
        self.rays = F.normalize(
            torch.cat((calibrated, torch.ones(len(calibrated), 1)), -1), dim=-1
        )
        self.pixel_list = self.xy.tolist()
        enabled = self.config.numerical_backend == "batched" or (
            self.config.numerical_backend == "auto" and self.device.type == "cuda"
        )
        self.candidate_cache = CandidateCache(self) if enabled else None

    def load(self, points, observations):
        self.candidate_cache = None
        # Original matches normally include every observation. Add isolated
        # input observations as well so topology never silently drops them.
        rows = observations.detach().cpu().tolist()
        for _, i, j in rows:
            self.node(i, j)
        self.assignment = [-1] * len(self.nodes)
        self.points = dict(
            enumerate(map(tuple, points.detach().cpu().double().tolist()))
        )
        self.tracks = {i: set() for i in self.points}
        for p, i, j in rows:
            n = self.lookup[i, j]
            if self.assignment[n] != -1:
                raise ValueError("One image feature cannot belong to two 3D points")
            self.assignment[n] = p
            self.tracks[p].add(n)
        self.next_id = len(self.points)

    def export(self):
        ids = sorted(p for p in self.points if len(self.tracks[p]) >= 2)
        xyz = torch.tensor(
            [self.points[p] for p in ids], dtype=self.dtype, device=self.device
        ).reshape(-1, 3)
        rows = [
            (local, *self.nodes[n])
            for local, p in enumerate(ids)
            for n in sorted(self.tracks[p], key=lambda n: self.nodes[n])
        ]
        obs = torch.tensor(rows, dtype=torch.long, device=self.device).reshape(-1, 3)
        return xyz, obs

    def pure_two_view(self, n):
        return len(self.adj[n]) == 1 and len(self.adj[next(iter(self.adj[n]))]) == 1

    def local(self, n):
        seen, frontier = {n}, [n]
        result = []
        for _ in range(self.config.max_transitivity):
            following = []
            for u in frontier:
                for v in sorted(self.adj[u]):
                    if v not in seen:
                        seen.add(v)
                        following.append(v)
                        if self.nodes[v][0] in self.valid_cameras:
                            result.append(v)
            frontier = following
        return result

    def error(self, n, xyz, angular=False):
        i, _ = self.nodes[n]
        offset = tuple(x - c for x, c in zip(xyz, self.c[i], strict=True))
        q = [sum(a * b for a, b in zip(row, offset, strict=True)) for row in self.r[i]]
        if not all(math.isfinite(v) for v in q) or q[2] < 1e-12:
            return math.inf
        fx, fy, cx, cy = self.calibration[i]
        x, y = self.pixel_list[n]
        if angular:
            ray = ((x - cx) / fx, (y - cy) / fy, 1.0)
            cosine = sum(a * b for a, b in zip(ray, q, strict=True)) / math.sqrt(
                sum(v * v for v in ray) * sum(v * v for v in q)
            )
            return math.acos(max(-1.0, min(1.0, cosine)))
        return math.hypot(fx * q[0] / q[2] + cx - x, fy * q[1] / q[2] + cy - y)

    def stable(self, xyz, nodes):
        if not all(math.isfinite(x) for x in xyz):
            return False
        directions = []
        for n in nodes:
            if not math.isfinite(self.error(n, xyz)):
                return False
            d = [x - c for x, c in zip(xyz, self.c[self.nodes[n][0]], strict=True)]
            length = math.sqrt(sum(x * x for x in d))
            if length == 0:
                return False
            directions.append([x / length for x in d])
        cosine = math.cos(math.radians(self.config.min_angle))
        return any(
            abs(sum(x * y for x, y in zip(a, b, strict=True))) <= cosine
            for a, b in combinations(directions, 2)
        )

    def dlt(self, nodes):
        idx = torch.tensor(nodes, dtype=torch.long)
        poses = self.poses[self.frame_ids[idx]]
        rays = self.rays[idx]
        if len(nodes) == 2:
            xy = rays[:, :2] / rays[:, 2:]
            a = (xy[..., None] * poses[:, 2:3] - poses[:, :2]).reshape(4, 4)
            _, _, v = torch.linalg.svd(a)
            homogeneous = v[-1]
        else:
            a = poses - rays[:, :, None] * (rays[:, None, :] @ poses)
            _, vectors = torch.linalg.eigh((a.transpose(-1, -2) @ a).sum(0))
            homogeneous = vectors[:, 0]
        if abs(float(homogeneous[3])) < torch.finfo(torch.float64).tiny:
            return None
        xyz = tuple((homogeneous[:3] / homogeneous[3]).tolist())
        return xyz if self.stable(xyz, nodes) else None

    def estimate(self, nodes):
        if len(nodes) < 2:
            return None, []
        threshold = math.radians(self.config.create_max_angle_error)
        best, best_nodes, best_score = None, [], (-1, -math.inf)
        budget = self.config.max_num_trials
        exhaustive = len(nodes) <= self.config.exhaustive_max_track_length
        pairs = islice(combinations(nodes, 2), self.config.max_num_trials)
        if exhaustive:
            # Native enumerates every pair for short tracks. Batch the small
            # DLT systems while preserving the same ordered candidate list.
            pairs = list(pairs)
            cached = (
                self.candidate_cache.get(nodes, pairs) if self.candidate_cache else None
            )
            if cached is not None:
                candidates = cached
            else:
                idx = torch.tensor(pairs, dtype=torch.long)
                poses = self.poses[self.frame_ids[idx]]
                rays = self.rays[idx]
                xy = rays[..., :2] / rays[..., 2:]
                a = (xy[..., None] * poses[:, :, 2:3] - poses[:, :, :2]).reshape(
                    -1, 4, 4
                )
                _, _, vectors = torch.linalg.svd(a)
                homogeneous = vectors[:, -1]
                models = (homogeneous[:, :3] / homogeneous[:, 3:]).tolist()
                candidates = (
                    (tuple(xyz) if self.stable(xyz, pair) else None, None)
                    for pair, xyz in zip(pairs, models, strict=True)
                )
        else:
            candidates = ((self.dlt(pair), None) for pair in pairs)
        for trial, (model, cached_errors) in enumerate(candidates):
            if model is None:
                continue

            def evaluate(xyz, errors=None):
                if errors is None:
                    errors = [self.error(n, xyz, angular=True) for n in nodes]
                inside = [
                    n for n, e in zip(nodes, errors, strict=True) if e <= threshold
                ]
                return inside, (
                    len(inside),
                    -sum(e * e for e in errors if e <= threshold),
                )

            inside, score = evaluate(model, cached_errors)
            if score <= best_score:
                if not exhaustive and trial >= budget:
                    break
                continue
            for _ in range(self.config.local_iterations):
                if len(inside) <= 2:
                    break
                refined = self.dlt(inside)
                if refined is None:
                    break
                updated, updated_score = evaluate(refined)
                if updated_score <= score:
                    break
                unchanged = len(updated) == len(inside)
                model, inside, score = refined, updated, updated_score
                if unchanged:
                    break
            best, best_nodes, best_score = model, inside, score
            # Combination sampling is without replacement, as in native RANSAC.
            probability = (
                len(inside) * (len(inside) - 1) / (len(nodes) * (len(nodes) - 1))
            )
            needed = (
                (
                    1
                    if probability >= 1
                    else math.ceil(
                        3
                        * math.log1p(-self.config.confidence)
                        / math.log1p(-probability)
                    )
                )
                if probability
                else budget
            )
            budget = min(budget, max(1, needed))
            if not exhaustive and trial >= budget:
                break
        return best, best_nodes

    def add(self, xyz, nodes):
        p = self.next_id
        self.next_id += 1
        self.points[p], self.tracks[p] = xyz, set(nodes)
        for n in nodes:
            self.assignment[n] = p
        return p

    def create(self, nodes):
        nodes = list(dict.fromkeys(n for n in nodes if self.assignment[n] == -1))
        if len(nodes) < 2 or (
            len(nodes) == 2
            and self.config.ignore_two_view_tracks
            and self.pure_two_view(nodes[0])
        ):
            return 0
        xyz, inside = self.estimate(nodes)
        if xyz is None or len(inside) < 2:
            return 0
        self.add(xyz, inside)
        remaining = [n for n in nodes if self.assignment[n] == -1]
        return len(inside) + (self.create(remaining) if len(remaining) >= 3 else 0)

    def continue_track(self, n, neighbors, threshold=None):
        if self.assignment[n] != -1:
            return 0
        candidates = {self.assignment[v] for v in neighbors if self.assignment[v] != -1}
        if not candidates:
            return 0
        threshold = (
            self.config.continue_max_angle_error if threshold is None else threshold
        )
        error, p = min(
            (self.error(n, self.points[p], angular=True), p) for p in candidates
        )
        if error > math.radians(threshold):
            return 0
        self.assignment[n] = p
        self.tracks[p].add(n)
        return 1

    def complete(self, p):
        if p not in self.points:
            return 0
        visited = set(self.tracks[p])
        frontier = sorted(visited)
        count = 0
        for _ in range(self.config.complete_max_transitivity):
            following = []
            for u in frontier:
                for v in sorted(self.adj[u]):
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
                break
        return count

    def triangulate(self):
        added = 0
        order = [
            n
            for image in sorted(self.valid_cameras)
            for n in self.by_image.get(image, [])
        ]
        size = self.config.batch_lookahead if self.candidate_cache else len(order) or 1
        for start in range(0, len(order), size):
            window = order[start : start + size]
            local = {}
            if self.candidate_cache:
                tracks = []
                for n in window:
                    if self.assignment[n] != -1 or (
                        self.config.ignore_two_view_tracks and self.pure_two_view(n)
                    ):
                        continue
                    local[n] = self.local(n)
                    nodes = list(
                        dict.fromkeys(
                            v for v in [*local[n], n] if self.assignment[v] == -1
                        )
                    )
                    if 2 <= len(nodes) <= self.config.exhaustive_max_track_length:
                        tracks.append(nodes)
                self.candidate_cache.prepare(tracks)
            for n in window:
                if self.assignment[n] != -1:
                    added += self.complete(self.assignment[n])
                    continue
                if self.config.ignore_two_view_tracks and self.pure_two_view(n):
                    continue
                neighbors = local[n] if n in local else self.local(n)
                if not neighbors:
                    continue
                added += self.continue_track(n, neighbors)
                added += self.create([*neighbors, n])
        return added

    def complete_and_merge(self):
        completed = sum(self.complete(p) for p in list(self.points))
        merged, tried = 0, set()
        for initial in list(self.points):
            p = initial
            chain_size = 0
            while p in self.points:
                candidates = sorted(
                    {
                        self.assignment[v]
                        for n in self.tracks[p]
                        for v in self.adj[n]
                        if self.assignment[v] not in (-1, p)
                    }
                )
                success = False
                for q in candidates:
                    pair = tuple(sorted((p, q)))
                    if pair in tried:
                        continue
                    tried.add(pair)
                    a, b = len(self.tracks[p]), len(self.tracks[q])
                    xyz = tuple(
                        (a * x + b * y) / (a + b)
                        for x, y in zip(self.points[p], self.points[q], strict=True)
                    )
                    nodes = self.tracks[p] | self.tracks[q]
                    if not all(
                        self.error(n, xyz) <= self.config.merge_max_reproj_error
                        for n in nodes
                    ):
                        continue
                    del self.points[p], self.points[q], self.tracks[p], self.tracks[q]
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
            for a, b in matches:
                pa, pb = self.assignment[a], self.assignment[b]
                if pa != -1 and pb != -1:
                    continue
                if pa == -1 and pb == -1:
                    count += self.create([a, b])
                elif pa == -1:
                    count += self.continue_track(a, [b], self.config.re_max_angle_error)
                else:
                    count += self.continue_track(b, [a], self.config.re_max_angle_error)
        return count
