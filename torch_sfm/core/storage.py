from dataclasses import dataclass, field, fields
from pathlib import Path

import torch


@dataclass
class FrameIndexedData:
    keypoints_2d: torch.Tensor | None = field(
        default=None,
        metadata={"index_by": "frame"},
    )
    keypoints_3d: torch.Tensor | None = field(
        default=None,
        metadata={"index_by": "frame"},
    )
    features: torch.Tensor | None = field(
        default=None,
        metadata={"index_by": "frame"},
    )


@dataclass
class SFMStorage(FrameIndexedData):
    num_frames: int | None = None
    num_keypoints_per_frame: int | None = None
    keypoint_counts: tuple[int, ...] | None = None
    frame_paths: tuple[Path | None, ...] | None = None
    image_sizes: tuple[tuple[int, int] | None, ...] | None = None
    reconstruction: "ReconstructionStorage | None" = None

    def create_reconstruction(self, *, device=None, dtype=None):
        self.reconstruction = ReconstructionStorage.from_frames(
            [Frame.from_storage(self, i) for i in range(self.num_frames or 0)],
            device=device,
            dtype=dtype,
        )
        return self.reconstruction

    @classmethod
    def from_frames(cls, frames, *, device=None, dtype=None):
        """Pack variable counts without changing keypoint indices; drop matching data.

        Padding is never exposed by Frame views. Images/descriptors are not
        copied: verification and reconstruction only need xy and image metadata.
        """
        from PIL import Image

        counts = tuple(len(frame.keypoints_2d) for frame in frames)
        capacity = max(counts, default=0)
        reference = frames[0].keypoints_2d if frames else torch.empty(0, 2)
        points = torch.zeros(
            (len(frames), capacity, 2),
            device=device or reference.device,
            dtype=dtype or reference.dtype,
        )
        sizes = []
        for index, frame in enumerate(frames):
            points[index, : counts[index]] = frame.keypoints_2d.to(points)
            size = frame.image_size
            if frame.image is not None:
                size = tuple(frame.image.shape[-2:])
            elif size is None and frame.frame_path is not None:
                with Image.open(frame.frame_path) as image:
                    size = (image.height, image.width)
            sizes.append(size)
        return cls(
            keypoints_2d=points,
            num_frames=len(frames),
            num_keypoints_per_frame=capacity,
            keypoint_counts=counts,
            frame_paths=tuple(frame.frame_path for frame in frames),
            image_sizes=tuple(sizes),
        )


@dataclass
class Frame(FrameIndexedData):
    frame_idx: int | None = None
    frame_path: Path | None = None
    image: torch.Tensor | None = None  # RGB, (3, H, W), float in [0, 1]
    image_size: tuple[int, int] | None = (
        None  # original (height, width), without retaining pixels
    )

    @classmethod
    def from_storage(cls, storage: SFMStorage, frame_idx: int) -> "Frame":
        kwargs = {
            "frame_idx": frame_idx,
        }

        for f in fields(storage):
            if f.metadata.get("index_by") == "frame":
                value = getattr(storage, f.name)

                if value is not None:
                    value = value[frame_idx]
                    if storage.keypoint_counts is not None:
                        value = value[: storage.keypoint_counts[frame_idx]]
                    kwargs[f.name] = value

        if storage.frame_paths is not None:
            kwargs["frame_path"] = storage.frame_paths[frame_idx]
        if storage.image_sizes is not None:
            kwargs["image_size"] = storage.image_sizes[frame_idx]

        return cls(**kwargs)


@dataclass
class ReconstructionStorage:
    """Fixed N-slot observation, point and work arrays on one device.

    N includes every input keypoint, even unmatched features. Point slots are
    independent of observation ownership. No arrays are resized when tracks
    change. Compact tensors exist only at API boundaries / as temporary work.
    Mutation methods operate on batches; no per-observation CUDA scalar reads.
    """

    xy: torch.Tensor
    frame_id: torch.Tensor
    feature_id: torch.Tensor
    ray: torch.Tensor
    obs_valid: torch.Tensor
    point_id: torch.Tensor
    obs_version: torch.Tensor
    point_xyz: torch.Tensor
    point_alive: torch.Tensor
    point_parent: torch.Tensor
    point_num_obs: torch.Tensor
    point_version: torch.Tensor
    point_dirty: torch.Tensor
    point_order: torch.Tensor
    track_start: torch.Tensor
    track_obs: torch.Tensor
    proposal_point: torch.Tensor
    proposal_error: torch.Tensor
    proposal_valid: torch.Tensor
    frontier_mask: torch.Tensor
    next_frontier_mask: torch.Tensor
    frame_offsets: tuple[int, ...]
    geometry_version: int = 0
    next_order: int = 0

    @classmethod
    def from_frames(cls, frames, *, device=None, dtype=None):
        reference = frames[0].keypoints_2d if frames else torch.empty(0, 2)
        device = device or reference.device
        dtype = dtype or reference.dtype
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise ValueError("Reconstruction coordinates must be floating point")
        counts = [len(f.keypoints_2d) for f in frames]
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        n = offsets[-1]
        xy = (
            torch.cat(
                [f.keypoints_2d.detach().to(device=device, dtype=dtype) for f in frames]
            )
            if frames
            else torch.empty(0, 2, device=device, dtype=dtype)
        )
        if xy.shape != (n, 2):
            raise ValueError("Expected keypoint coordinates of shape (count, 2)")

        def integer(value=0):
            return torch.full((n,), value, dtype=torch.long, device=device)

        def boolean(value=False):
            return torch.full((n,), value, dtype=torch.bool, device=device)

        def floating(width=None, value=0.0):
            return torch.full(
                (n,) if width is None else (n, width), value, dtype=dtype, device=device
            )

        ids = torch.arange(n, device=device)
        frame_id = torch.repeat_interleave(
            torch.arange(len(counts), device=device),
            torch.tensor(counts, device=device, dtype=torch.long),
            output_size=n,
        )
        starts = torch.tensor(offsets[:-1], device=device, dtype=torch.long)
        feature_id = ids - starts[frame_id]
        return cls(
            xy,
            frame_id,
            feature_id,
            floating(3),
            boolean(True),
            integer(-1),
            integer(),
            floating(3),
            boolean(),
            ids.clone(),
            integer(),
            integer(),
            boolean(),
            integer(-1),
            integer(-1),
            integer(-1),
            integer(-1),
            floating(value=float("inf")),
            boolean(),
            boolean(),
            boolean(),
            tuple(offsets),
        )

    @property
    def capacity(self):
        return len(self.xy)

    def global_ids(self, image_ids, feature_ids):
        image_ids = torch.as_tensor(image_ids, device=self.xy.device, dtype=torch.long)
        feature_ids = torch.as_tensor(
            feature_ids, device=self.xy.device, dtype=torch.long
        )
        offsets = torch.tensor(self.frame_offsets, device=self.xy.device)
        if image_ids.shape != feature_ids.shape:
            raise ValueError("Image and feature index shapes differ")
        if image_ids.numel() and bool(
            ((image_ids < 0) | (image_ids >= len(self.frame_offsets) - 1)).any()
        ):
            raise ValueError("Image index out of range")
        if feature_ids.numel() and bool(
            (
                (feature_ids < 0)
                | (feature_ids >= offsets[image_ids + 1] - offsets[image_ids])
            ).any()
        ):
            raise ValueError("Feature index out of range")
        return offsets[image_ids] + feature_ids

    def reset(self):
        self.point_id.fill_(-1)
        self.obs_version.add_(1)
        self.point_xyz.zero_()
        self.point_alive.zero_()
        self.point_parent.copy_(torch.arange(self.capacity, device=self.xy.device))
        self.point_num_obs.zero_()
        self.point_version.add_(1)
        self.point_dirty.zero_()
        self.point_order.fill_(-1)
        self.track_start.fill_(-1)
        self.track_obs.fill_(-1)
        self.clear_work()
        self.next_order = 0

    def clear_work(self):
        self.proposal_point.fill_(-1)
        self.proposal_error.fill_(float("inf"))
        self.proposal_valid.zero_()
        self.frontier_mask.zero_()
        self.next_frontier_mask.zero_()

    def rebuild_tracks(self):
        """Build point -> observation ranges without a Python track dictionary."""
        assigned = torch.where(self.point_id >= 0)[0]
        ordered = assigned[torch.argsort(self.point_id[assigned], stable=True)]
        owners = self.point_id[ordered]
        self.point_num_obs.zero_()
        self.point_num_obs.scatter_add_(0, owners, torch.ones_like(owners))
        self.track_obs.fill_(-1)
        self.track_obs[: len(ordered)] = ordered
        self.track_start.copy_(self.point_num_obs.cumsum(0) - self.point_num_obs)
        self.track_start.masked_fill_(self.point_num_obs == 0, -1)

    def load(self, points, observations):
        points = points.detach().to(self.point_xyz)
        observations = observations.to(device=self.xy.device, dtype=torch.long)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or observations.ndim != 2
            or observations.shape[1] != 3
        ):
            raise ValueError("Expected points (M,3) and observations (L,3)")
        if len(points) > self.capacity:
            raise ValueError("Point count exceeds fixed storage capacity")
        if observations.numel() and bool(
            ((observations[:, 0] < 0) | (observations[:, 0] >= len(points))).any()
        ):
            raise ValueError("Point index out of range")
        ids = self.global_ids(observations[:, 1], observations[:, 2])
        if len(torch.unique(ids)) != len(ids):
            raise ValueError("One observation cannot belong to multiple points")
        # Validate first: invalid input must not destroy a valid reconstruction.
        self.reset()
        count = len(points)
        self.point_xyz[:count] = points
        self.point_id[ids] = observations[:, 0]
        self.rebuild_tracks()
        self.point_alive.copy_(self.point_num_obs > 0)
        self.point_dirty.copy_(self.point_alive)
        self.point_order[:count] = torch.arange(count, device=self.xy.device)
        self.next_order = count

    def create(self, xyz, nodes, inliers):
        """Commit disjoint new tracks. Caller arbitrates conflicts before commit."""
        count = len(xyz)
        if not count:
            return
        free = torch.where(~self.point_alive)[0][:count]
        if len(free) != count:
            raise RuntimeError("Fixed point storage exhausted")
        rows, cols = torch.where(inliers)
        obs = nodes[rows, cols]
        self.point_xyz[free] = xyz
        self.point_alive[free] = True
        self.point_parent[free] = free
        self.point_version[free] += 1
        self.point_dirty[free] = True
        self.point_order[free] = torch.arange(
            self.next_order, self.next_order + count, device=self.xy.device
        )
        self.next_order += count
        self.point_id[obs] = free[rows]
        self.obs_version[obs] += 1
        self.point_num_obs[free] = inliers.sum(-1)

    def assign(self, observations, points):
        self.point_id[observations] = points
        self.obs_version[observations] += 1
        counts = torch.bincount(points, minlength=self.capacity)
        self.point_num_obs.add_(counts)
        changed = counts > 0
        self.point_version[changed] += 1
        self.point_dirty[changed] = True

    def merge(self, pairs, xyz):
        """Commit disjoint pairs and canonicalize owners before any slot reuse."""
        if not len(pairs):
            return
        keep, remove = pairs.unbind(-1)
        self.point_xyz[keep] = xyz
        self.point_parent[remove] = keep
        self.point_alive[remove] = False
        self.point_version[keep] += 1
        self.point_version[remove] += 1
        self.point_dirty[keep] = True
        self.point_dirty[remove] = False
        self.point_order[keep] = torch.arange(
            self.next_order, self.next_order + len(pairs), device=self.xy.device
        )
        self.next_order += len(pairs)
        assigned = torch.where(self.point_id >= 0)[0]
        previous = self.point_id[assigned]
        canonical = self.point_parent[previous]
        self.point_id[assigned] = canonical
        self.obs_version[assigned[canonical != previous]] += 1
        # Owners now directly reference live roots; no dangling references to
        # a freed slot, even when a later creation reuses that slot.
        self.point_parent.copy_(torch.arange(self.capacity, device=self.xy.device))
        self.rebuild_tracks()

    def export(self):
        self.rebuild_tracks()
        slots = torch.where(self.point_alive & (self.point_num_obs >= 2))[0]
        slots = slots[torch.argsort(self.point_order[slots], stable=True)]
        compact = torch.full_like(self.point_id, -1)
        compact[slots] = torch.arange(len(slots), device=self.xy.device)
        obs = torch.where(self.point_id >= 0)[0]
        local = compact[self.point_id[obs]]
        obs, local = obs[local >= 0], local[local >= 0]
        order = torch.argsort(local, stable=True)
        obs, local = obs[order], local[order]
        return self.point_xyz[slots], torch.stack(
            (local, self.frame_id[obs], self.feature_id[obs]), -1
        )
