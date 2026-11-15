"""Optional VGGSfM v2 point tracker, loaded from a GlueMap checkout.

Only the tracker is loaded: no VGGSfM camera/depth prediction or BA. Upstream
model code and its license remain in the external checkout. Images are padded
and resized using GlueMap's 1024-pixel tracker convention.
"""
from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import sys
import types

import torch
import torch.nn.functional as F


@dataclass
class VGGSfMTrackConfig:
    source: str | None = None  # GlueMap checkout containing thirdparty/vggsfm
    checkpoint: str | None = None
    image_size: int = 1024
    queries_per_star: int = 1024
    visibility_threshold: float = .05  # GlueMap's real-track admission threshold
    snapping_threshold: float = 1.  # scaled by (original H+W)/image_size
    fine_chunk: int = 4096  # only batches independent fine-refinement patches
    image_cache_gib: float = 8.
    seed: int = 42

    def __post_init__(self):
        if self.image_size < 64 or self.image_size % 8:
            raise ValueError('Tracker image_size must be a multiple of 8, at least 64')
        if min(self.queries_per_star, self.fine_chunk) < 1:
            raise ValueError('Tracker query and fine chunk counts must be positive')
        if not 0 <= self.visibility_threshold < 1 or self.snapping_threshold < 0:
            raise ValueError('Invalid tracker visibility/snapping threshold')
        if self.image_cache_gib <= 0:
            raise ValueError('Tracker CPU image cache budget must be positive')


def preprocess_image(image, size=1024):
    """Return padded RGB tensor and original-to-tracker (sx,sy,ox,oy)."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError('Expected RGB image (3,H,W)')
    h, w = image.shape[-2:]
    nh, nw = round(h * size / max(h, w)), round(w * size / max(h, w))
    resized = F.interpolate(image[None].float(), size=(nh, nw), mode='bilinear', align_corners=False)[0]
    left, top = (size-nw)//2, (size-nh)//2
    padded = F.pad(resized, (left, size-nw-left, top, size-nh-top), value=1.)
    return padded, torch.tensor([nw/w, nh/h, left, top], dtype=torch.float32)


def transform_pixels(points, transform, *, inverse=False):
    transform = transform.to(points)
    return ((points-transform[..., 2:])/transform[..., :2] if inverse
            else points*transform[..., :2]+transform[..., 2:])


class VGGSfMTracker:
    def __init__(self, config=None, device='cuda'):
        self.config = config or VGGSfMTrackConfig()
        self.device = torch.device(device)
        self.model = None

    def load(self):
        if self.model is not None:
            return self.model
        cfg = self.config
        if cfg.source is None or cfg.checkpoint is None:
            raise ValueError('Set tracker.source to a GlueMap checkout and tracker.checkpoint to VGGSfM v2 tracker weights')
        package = Path(cfg.source).resolve() / 'thirdparty/vggsfm'
        if not (package/'vggsfm_tracker.py').is_file():
            raise FileNotFoundError(package/'vggsfm_tracker.py')
        name = '_torch_sfm_vggsfm_' + hashlib.sha256(str(package).encode()).hexdigest()[:12]
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(package)]
            sys.modules[name] = module
        cls = importlib.import_module(name+'.vggsfm_tracker').TrackerPredictor
        model = cls()
        model.load_state_dict(torch.load(cfg.checkpoint, map_location='cpu', weights_only=True), strict=True)
        self.model = model.eval().to(self.device)
        return self.model

    @torch.inference_mode()
    def predict(self, images, query_points, *, transforms=None):
        """Track center-image queries through one (S,3,H,W) RGB group.

        Coordinates refer to original images when transforms (S,4) are supplied;
        otherwise they refer to the input image grid. Returned tensors are on CPU.
        """
        if images.ndim != 4 or images.shape[1] != 3 or len(images) < 2:
            raise ValueError('Expected at least two RGB images (S,3,H,W)')
        if query_points.ndim != 2 or query_points.shape[1] != 2 or len(query_points) == 0:
            raise ValueError('Expected nonempty queries (Q,2)')
        if transforms is not None and transforms.shape != (len(images), 4):
            raise ValueError('Expected one image transform per view')
        queries = query_points.float().to(self.device)
        if transforms is not None:
            queries = transform_pixels(queries, transforms[0])
        tracks, _, vis, score = self.load()(images[None].to(self.device), queries[None],
            fine_chunk=self.config.fine_chunk)
        tracks = tracks[0].float().cpu()
        if transforms is not None:
            tracks = transform_pixels(tracks, transforms[:, None].cpu(), inverse=True)
        return dict(tracks=tracks, visibility=vis[0].float().cpu(), confidence=score[0].float().cpu())

    def close(self):
        self.model = None
