# Adapted algorithms from colmap/gluemap (BSD-3-Clause).
# Copyright (c), ETH Zurich.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# * Neither the name of ETH Zurich nor the names of its contributors
# may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Depth-derived virtual observations for GlueMap augmented bundle adjustment.

Algorithm reference: colmap/gluemap adc9e4b, virtual_tracks.py and
covisibility_extraction.py. Coordinates follow our prior's pixel-center convention.
Virtual measurements stay frozen after construction; their 3D points are optimized.
"""
from dataclasses import dataclass, fields
import math
from pathlib import Path

import torch
from .two_view import camera_matrix


@dataclass
class VirtualTrackConfig:
    enabled: bool = True
    grid_stride: int = 14
    tracks_per_star: int = 100
    min_tracks_per_view: int = 5
    depth_noise: float = .1
    global_ratio: float = .1
    imaging_plane_angle: float = 5.
    inconsistent_rotation_degrees: float = 1.
    inconsistent_translation_degrees: float = 5.
    pair_coverage: int = 512
    seed: int = 42

    def __post_init__(self):
        if min(self.grid_stride, self.tracks_per_star, self.min_tracks_per_view) < 1:
            raise ValueError('Virtual sampling counts must be positive')
        if not 0 <= self.global_ratio <= .5 or not 0 < self.imaging_plane_angle < 90:
            raise ValueError('Invalid virtual global ratio / imaging plane angle')
        if (not math.isfinite(self.depth_noise) or self.depth_noise < 0 or self.pair_coverage < 0
                or min(self.inconsistent_rotation_degrees, self.inconsistent_translation_degrees) <= 0):
            raise ValueError('Invalid virtual depth noise / coverage / consistency threshold')


@dataclass
class VirtualTracks:
    points: torch.Tensor
    observations: torch.Tensor  # (point_id, image_id); no real feature indices
    pixels: torch.Tensor
    negative: torch.Tensor  # immutable expected depth sign for each measurement
    source_stars: torch.Tensor  # one per point
    global_conditioned: torch.Tensor  # one per observation

    @classmethod
    def empty(cls, device='cpu'):
        return cls(torch.empty(0, 3, dtype=torch.float64, device=device),
            torch.empty(0, 2, dtype=torch.long, device=device),
            torch.empty(0, 2, dtype=torch.float64, device=device),
            torch.empty(0, dtype=torch.bool, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.bool, device=device))

    def to(self, device):
        return VirtualTracks(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    def subset(self, keep, min_views=2):
        counts = torch.bincount(self.observations[keep, 0], minlength=len(self.points))
        point_mask = counts >= min_views
        keep = keep & point_mask[self.observations[:, 0]]
        mapping = point_mask.long().cumsum(0) - 1
        obs = self.observations[keep].clone()
        obs[:, 0] = mapping[obs[:, 0]]
        return VirtualTracks(self.points[point_mask], obs, self.pixels[keep], self.negative[keep],
                             self.source_stars[point_mask], self.global_conditioned[keep])


def project_virtual(local, K, angle_degrees=5.):
    """Signed projection, retaining off-image/behind-camera points away from z=0."""
    z = local[..., 2]
    valid = (torch.isfinite(local).all(-1) & (local.norm(dim=-1) > 1e-12)
             & (z.abs() >= math.sin(math.radians(angle_degrees)) * local.norm(dim=-1)))
    # Upstream's projection helper clips |z| at 1e-3 before the angle mask.
    safe_z = torch.where(z < 0, -1., 1.) * z.abs().clamp_min(1e-3)
    pixels = (local @ K.transpose(-1, -2))[..., :2] / safe_z[..., None]
    return pixels, valid & torch.isfinite(pixels).all(-1), z < 0


def virtual_candidate_config(cfg):
    # Only parameters baked into the cached depth sampling. Selection/projection
    # thresholds remain adjustable when resuming refinement.
    return {key: getattr(cfg, key) for key in ('grid_stride', 'depth_noise', 'seed')}


def prepare_virtual_candidates(data, center, cfg):
    """Sample local 3D candidates while dense depth/masks are still available."""
    depth = data['depth']
    if depth is None:
        raise ValueError('Virtual tracks require saved candidates or legacy depth; regenerate stars with save_virtual_candidates=True')
    depth = depth[0].double()
    h, w = depth.shape
    generator = torch.Generator().manual_seed(cfg.seed + int(center))
    overlap = data['center_masks'][1:].any(0) & torch.isfinite(depth) & (depth > 0)
    values = depth[overlap]
    if not values.numel():
        # Upstream uses 1000 when no overlap supplies a reliable depth statistic.
        median = depth.new_tensor(1000.)
        valid_depth = torch.zeros_like(overlap)
    else:
        q1, median, q3 = torch.quantile(values, depth.new_tensor([.25, .5, .75]))
        valid_depth = overlap & (depth > q1) & (depth < q3)
    step = cfg.grid_stride
    y, x = torch.meshgrid(torch.arange(h // step) * step + step // 2,
                         torch.arange(w // step) * step + step // 2, indexing='ij')
    if not x.numel():
        raise ValueError('Prior depth map is smaller than virtual grid stride')
    offset = float(data.get('pixel_center_offset', .5))
    pixels = torch.stack((x.flatten()+offset, y.flatten()+offset, torch.ones(x.numel())), -1).double()
    z = torch.where(valid_depth, depth, median)[y, x].flatten()
    z = z * (1 + cfg.depth_noise * torch.randn(z.shape, generator=generator, dtype=z.dtype))
    points = (pixels @ torch.linalg.inv(data['intrinsics'][0].double()).T) * z[:, None]
    return dict(points=points, generator_state=generator.get_state(),
                config=virtual_candidate_config(cfg))


def sample_star(data, ids, R, C, K, inverse_scale, cfg, pair_support=None):
    """Return world points and frozen observations for one independently scaled star."""
    candidates = data.get('virtual_candidates')
    if candidates is None:
        candidates = prepare_virtual_candidates(data, int(ids[0]), cfg)
    elif candidates['config'] != virtual_candidate_config(cfg):
        raise ValueError('Virtual depth sampling settings differ from the compact star cache; '
                         'regenerate stars to change grid_stride, depth_noise or seed')
    generator = torch.Generator().set_state(candidates['generator_state'])
    points = candidates['points'] / inverse_scale
    poses = data['camera_to_world'].double()
    if not torch.allclose(poses[0], torch.eye(4, dtype=torch.float64), atol=1e-5):
        raise ValueError('Virtual tracks require a center-anchored local star')
    local = (points[None] - poses[:, None, :3, 3] / inverse_scale) @ poses[:, :3, :3]
    uv, valid, negative = project_virtual(local, K, cfg.imaging_plane_angle)
    valid &= data['valid_views'][:, None]
    valid &= torch.isfinite(R).all((-1, -2))[:, None] & torch.isfinite(C).all(-1)[:, None]
    # Cover each neighbor, then fill by the number of views seeing each candidate.
    selected = torch.zeros(len(points), dtype=torch.bool)
    for view in range(1, len(ids)):
        choices = torch.where(valid[view])[0]
        selected[choices[torch.randperm(len(choices), generator=generator)[:cfg.min_tracks_per_view]]] = True
    count = max(cfg.tracks_per_star, int(selected.sum()))
    weights = valid[1:].sum(0).double()
    weights[selected] = 0
    extra = min(count-int(selected.sum()), int((weights > 0).sum()))
    chosen = torch.where(selected)[0]
    if extra:
        chosen = torch.cat((chosen, torch.multinomial(weights, extra, generator=generator)))
    # A deterministic shuffle avoids biasing the global-conditioned 10% to early pixels.
    chosen = chosen[torch.randperm(len(chosen), generator=generator)]
    points, uv, valid, negative = points[chosen], uv[:, chosen], valid[:, chosen], negative[:, chosen]
    world = points @ R[0] + C[0]
    global_local = (world[None] - C[:, None]) @ R.transpose(-1, -2)
    guv, gvalid, gnegative = project_virtual(global_local, K, cfg.imaging_plane_angle)
    conditioned = torch.zeros_like(valid)
    amount = int(len(points)*cfg.global_ratio)
    conditioned[:, :amount] = True
    # Upstream drops inconsistent local edges and substitutes a second global
    # slice only if their real-track coverage is insufficient.
    relative_R = R @ R[0].T
    local_R = poses[:, :3, :3].transpose(-1, -2)
    rotation_error = torch.acos((((relative_R @ local_R.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1)).rad2deg()
    gt = (R @ (C[0]-C)[..., None]).squeeze(-1)
    lt = -(local_R @ poses[:, :3, 3, None]).squeeze(-1)
    direction = torch.nn.functional.cosine_similarity(gt, lt, dim=-1, eps=1e-6)
    inconsistent = (rotation_error > cfg.inconsistent_rotation_degrees) | (torch.acos(direction.clamp(-1, 1)).rad2deg() > cfg.inconsistent_translation_degrees)
    inconsistent[0] = False
    valid[inconsistent, amount:] = False
    for j in torch.where(inconsistent)[0].tolist():
        support = (pair_support or {}).get(tuple(sorted((int(ids[0]), int(ids[j])))), 0)
        if support < len(points):
            conditioned[j, :2*amount] = True
    uv = torch.where(conditioned[..., None], guv, uv)
    valid = torch.where(conditioned, gvalid, valid)
    valid &= data['valid_views'][:, None] & torch.isfinite(R).all((-1, -2))[:, None]
    negative = torch.where(conditioned, gnegative, negative)
    # Upstream initialize_world_points rejects near-zero center depth.
    valid &= points[:, 2].abs()[None] >= 1e-3
    view, point = torch.where(valid)
    tracks = VirtualTracks(world, torch.stack((point, ids[view]), -1), uv[view, point], negative[view, point],
        torch.full((len(points),), int(ids[0]), dtype=torch.long), conditioned[view, point])
    return tracks.subset(torch.ones(len(view), dtype=torch.bool))


def build_virtual_tracks(manifest, directory, initial, cfg, pair_support=None, device='cuda'):
    """Map one saved star at a time; never retain the dense depth cache in RAM."""
    if not cfg.enabled:
        return VirtualTracks.empty(device)
    K = camera_matrix(initial['intrinsics']).cpu().double()
    n = len(initial['registered'])
    if K.ndim == 2:
        K = K.expand(n, -1, -1)
    R, C = initial['rotations'].cpu().double(), initial['centers'].cpu().double()
    pieces, offset = [], 0
    for i, record in enumerate(manifest['stars']):
        if int(initial['star_centers'][i]) != record['center']:
            raise ValueError('Star manifest differs from fused scale ordering')
        scale = initial['inverse_star_scales'][i].item()
        if not math.isfinite(scale) or scale <= 0 or not initial['registered'][record['center']]:
            continue
        data = torch.load(Path(directory)/record['path'], weights_only=True, map_location='cpu', mmap=True)
        ids = torch.tensor(data['image_ids'], dtype=torch.long)
        if ids.tolist() != record['image_ids']:
            raise ValueError('Star image IDs differ from the manifest')
        # Only the local graph retained by consistency/fusion can supply priors.
        data['valid_views'] = data['valid_views'].clone() & initial['registered'][ids].cpu()
        rejected = {tuple(sorted(r['pair'])) for r in initial['fusion_report']['reverse_rejected']}
        for j in range(1, len(ids)):
            if tuple(sorted((int(ids[0]), int(ids[j])))) in rejected:
                data['valid_views'][j] = False
        tracks = sample_star(data, ids, R[ids], C[ids], K[ids], scale, cfg, pair_support)
        tracks.observations[:, 0] += offset
        offset += len(tracks.points)
        pieces.append(tracks)
    if not pieces:
        return VirtualTracks.empty(device)
    return VirtualTracks(**{f.name: torch.cat([getattr(p, f.name) for p in pieces]).to(device)
                            for f in fields(VirtualTracks)})
