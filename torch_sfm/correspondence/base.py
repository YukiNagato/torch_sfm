from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from ..core.storage import Frame


@dataclass
class Matches:
    indices: torch.Tensor  # (M, 2), int64; indices into each frame's keypoints
    scores: torch.Tensor  # (M,)


class Extractor(ABC):
    @abstractmethod
    def extract(self, frame: Frame) -> Frame:
        """Fill keypoints_2d (N, 2) in pixel xy and features (N, D); return frame."""
        ...


class Matcher(ABC):
    @abstractmethod
    def match(self, frame0: Frame, frame1: Frame) -> Matches:
        """Match injected frame data without modifying either frame."""
        ...
