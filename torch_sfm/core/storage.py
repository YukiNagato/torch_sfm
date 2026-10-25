import torch
from dataclasses import dataclass, field, fields
from pathlib import Path


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


@dataclass
class Frame(FrameIndexedData):
    frame_idx: int | None = None
    frame_path: Path | None = None
    image: torch.Tensor | None = None  # RGB, (3, H, W), float in [0, 1]

    @classmethod
    def from_storage(cls, storage: SFMStorage, frame_idx: int) -> "Frame":
        kwargs = {
            "frame_idx": frame_idx,
        }

        for f in fields(storage):
            if f.metadata.get("index_by") == "frame":
                value = getattr(storage, f.name)

                if value is not None:
                    kwargs[f.name] = value[frame_idx]

        return cls(**kwargs)
