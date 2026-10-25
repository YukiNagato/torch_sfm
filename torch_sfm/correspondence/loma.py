from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from ..core.storage import Frame
from .base import Extractor, Matcher, Matches

if TYPE_CHECKING:
    from loma import LoMa


def create_loma(
    cfg: LoMa.Cfg | None = None,
    *,
    weights_path: str | Path | None = None,
) -> LoMa:
    """Load a LoMa state_dict locally, or use official weights when omitted.

    DeDoDe-G may still download its DINOv2 backbone during initialization.
    """
    from loma import LoMa, LoMaB, LoMaL

    cfg = LoMaB() if cfg is None else cfg
    if weights_path is None:
        return LoMa(cfg)

    weights = torch.load(Path(weights_path).expanduser(), map_location="cpu", weights_only=True)
    model = LoMa(replace(cfg, weights_url=None))
    weights = {
        key: value
        for key, value in weights.items()
        if not model._is_unexpected_extra_layer_key(key, cfg.n_layers)
    }
    model.load_state_dict(weights, strict=True)
    return model.eval()


class LoMaExtractor(Extractor):
    def __init__(self, model: LoMa):
        self.model = model.eval()

    @torch.inference_mode()
    def extract(self, frame: Frame) -> Frame:
        height, width = frame.image.shape[-2:]
        image = frame.image.unsqueeze(0)
        if height % 14 or width % 14:
            size = ((height + 13) // 14 * 14, (width + 13) // 14 * 14)
            image = F.interpolate(image, size=size, mode='bilinear', align_corners=False)
        keypoints, descriptors, _, _ = self.model.detect_and_describe(image)
        # Normalized coordinates map back to the original image dimensions.
        scale = keypoints.new_tensor([width, height])
        frame.keypoints_2d = (keypoints[0] + 1) * scale / 2
        frame.features = descriptors[0]
        return frame


class LoMaMatcher(Matcher):
    def __init__(self, model: LoMa):
        self.model = model.eval()

    @staticmethod
    def _inputs(frame: Frame) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = frame.image.shape[-2:]
        keypoints = frame.keypoints_2d
        scale = keypoints.new_tensor([width, height])
        return (
            (2 * keypoints / scale - 1).unsqueeze(0),
            frame.features.unsqueeze(0),
        )

    @torch.inference_mode()
    def match(self, frame0: Frame, frame1: Frame) -> Matches:
        from loma.loma import filter_matches

        keypoints0, descriptors0 = self._inputs(frame0)
        keypoints1, descriptors1 = self._inputs(frame1)
        if keypoints0.shape[1] == 0 or keypoints1.shape[1] == 0:
            return Matches(
                torch.empty((0, 2), dtype=torch.long, device=keypoints0.device),
                keypoints0.new_empty((0,)),
            )

        scores = self.model(
            keypoints0, keypoints1, descriptors0, descriptors1
        )["scores"]
        matches0, _, scores0, _ = filter_matches(
            scores, self.model.cfg.filter_threshold
        )
        indices0 = torch.where(matches0[0] >= 0)[0]
        indices = torch.stack((indices0, matches0[0, indices0]), dim=-1)
        return Matches(indices, scores0[0, indices0])







