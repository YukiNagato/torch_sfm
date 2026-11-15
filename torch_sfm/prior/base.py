"""Pose priors for a local group of overlapping images."""
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass
class PosePrior:
    """Local reconstructions, in input order and with arbitrary per-group scale.

    The first image is the reference camera (identity pose). Separate groups
    have independent gauges and must be aligned before global reconstruction.
    Intrinsics refer to the processed images at image_size, NOT original files.
    Camera axes are OpenCV: x right, y down, z forward.
    Optional depth is camera z-depth in the same arbitrary scale as translations;
    its pixel centers are (x+.5,y+.5). Confidence is a model score, not probability.
    A batch adds a leading B dimension and leaves image_ids=None; the pipeline
    owns the mapping from batch positions to images.
    """

    image_ids: tuple[int, ...] | None
    camera_to_world: torch.Tensor  # (N,4,4) or (B,N,4,4), CPU float32
    intrinsics: torch.Tensor  # (N,3,3) or (B,N,3,3), CPU float32; predicted per view
    image_size: tuple[int, int]  # processed (height, width)
    depth: torch.Tensor | None = None  # (...,N,H,W), CPU float32, camera z-depth
    depth_conf: torch.Tensor | None = None  # same shape; NOT a probability

    @property
    def rotations(self) -> torch.Tensor:
        """World-to-camera rotations, as used by torch_sfm."""
        return self.camera_to_world[..., :3, :3].transpose(-1, -2)

    @property
    def centers(self) -> torch.Tensor:
        """Camera centers in the local world frame, as used by torch_sfm."""
        return self.camera_to_world[..., :3, 3]

    @property
    def world_to_camera(self) -> torch.Tensor:
        result = torch.zeros_like(self.camera_to_world)
        result[..., :3, :3] = self.rotations
        result[..., :3, 3] = -(self.rotations @ self.centers[..., None]).squeeze(-1)
        result[..., 3, 3] = 1
        return result


class PriorModel(ABC):
    @abstractmethod
    def predict(self, images: torch.Tensor, *, image_ids: Sequence[int] | None = None,
                return_depth: bool = False) -> PosePrior:
        """Infer one connected local group; RGB (N,3,H,W) floats in [0,1].

        The caller selects a center followed by its topology neighbors (or an
        overlapping temporal window). Edges are used for selection, not as an
        attention mask. Model-specific preprocessing is the caller's responsibility.
        """
        ...
