"""RaCo detections and cached FlexGlue dense features, with fixed observation IDs."""
from collections import OrderedDict
from dataclasses import asdict, dataclass

import torch

from .base import Extractor, Matcher, Matches


@dataclass
class FlexGlueConfig:
    max_keypoints: int = 4096
    detection_resize: int = 1024
    feature_resize: int = 512  # square backbone input, independent of detection
    n_layers: int = 9
    min_covisibility: float = 0.9  # zero disables pair rejection
    early_stop_decay: float | None = None
    filter_threshold: float = 0.1
    max_distance_px: float = 3.0  # in the resized matcher image
    gpu_cache_images: int = 32
    weights: str | None = None  # None selects the official versioned URL
    detector_weights: str | None = None

    def __post_init__(self):
        if self.max_keypoints < 1 or self.detection_resize < 32:
            raise ValueError('Positive keypoint count and detection_resize >= 32 required')
        if self.feature_resize < 32 or self.feature_resize % 32:
            raise ValueError('feature_resize must be a positive multiple of 32')
        if not 0 <= self.min_covisibility <= 1 or not 1 <= self.n_layers <= 9:
            raise ValueError('Require covisibility in [0,1] and 1..9 transformer layers')
        if self.early_stop_decay is not None and (
                not 0 <= self.early_stop_decay < 1 or self.min_covisibility == 0):
            raise ValueError('Early stopping needs decay in [0,1) and positive covisibility')
        if self.gpu_cache_images < 2:
            raise ValueError('gpu_cache_images must retain at least two images')

    def feature_options(self):
        return dict(version=1, detector='raco-v1.0.0', descriptor='flexglue-v0.1',
                    **{k: getattr(self, k) for k in ('max_keypoints', 'detection_resize',
                       'feature_resize', 'weights', 'detector_weights')})

    def matching_options(self):
        options = asdict(self)
        options.pop('gpu_cache_images')
        return dict(version=1, **options)


class FlexGlueExtractor(Extractor):
    """Keep dense maps on CPU; the matcher manages a bounded GPU working set."""

    def __init__(self, config, device='cuda'):
        self.config, self.device = config, device
        self.detector = self.descriptor = None

    @torch.inference_mode()
    def extract(self, frame):
        if self.detector is None:
            from raco import RaCo
            from flexglue import DINOv3Descriptor
            detector_kw = {} if self.config.detector_weights is None else {'weights': self.config.detector_weights}
            descriptor_kw = {} if self.config.weights is None else {'weights': self.config.weights}
            self.detector = RaCo(max_num_keypoints=self.config.max_keypoints,
                                 **detector_kw).eval().to(self.device)
            self.descriptor = DINOv3Descriptor(
                resize=(self.config.feature_resize,) * 2, describe=False,
                **descriptor_kw).eval().to(self.device)
        image = frame.image.to(self.device)
        detections = self.detector.extract(image, resize=self.config.detection_resize)
        prediction = self.descriptor({'image': image[None]})
        frame.keypoints_2d = detections['keypoints'][0].cpu()
        frame.features = None
        frame.dense_features = prediction['features'][0].cpu()
        frame.image_size = tuple(image.shape[-2:])
        return frame

    def release(self):
        self.detector = self.descriptor = None


class FlexGlueMatcher(Matcher):
    requires_images = False

    def __init__(self, config, device='cuda'):
        self.config, self.device = config, device
        self.model = None
        self._images = OrderedDict()

    def _input(self, frame):
        # Keep the source object in the entry to protect against changed frame data.
        key = id(frame)
        entry = self._images.pop(key, None)
        if entry is None or entry[0] is not frame.dense_features:
            if frame.dense_features is None or frame.image_size is None:
                raise ValueError('FlexGlue requires cached dense_features and image_size')
            height, width = frame.image_size
            data = dict(features=frame.dense_features[None].to(self.device),
                        keypoints=frame.keypoints_2d[None].to(self.device),
                        image_size=torch.tensor([[width, height]], device=self.device,
                                                dtype=torch.float32))
            entry = (frame.dense_features, data)
        self._images[key] = entry
        while len(self._images) > self.config.gpu_cache_images:
            self._images.popitem(last=False)
        return entry[1]

    @torch.inference_mode()
    def match(self, frame0, frame1):
        if min(len(frame0.keypoints_2d), len(frame1.keypoints_2d)) == 0:
            return Matches(torch.empty((0, 2), dtype=torch.long), torch.empty(0))
        if self.model is None:
            from flexglue import FlexGlue
            kw = {} if self.config.weights is None else {'weights': self.config.weights}
            self.model = FlexGlue(describe=False, refine=False,
                n_layers=self.config.n_layers, min_covisibility=self.config.min_covisibility,
                early_stop_decay=self.config.early_stop_decay,
                filter_threshold=self.config.filter_threshold,
                max_distance_px=self.config.max_distance_px, **kw).eval().to(self.device)
        pred = self.model.match(self._input(frame0), self._input(frame1), sparse=True)
        score = float(pred['covisibility_score'][0])
        rejected = score < self.config.min_covisibility
        matches = pred['matches0'][0]
        source = torch.where(matches >= 0)[0]
        indices = torch.stack((source, matches[source]), -1)
        scores = pred['matching_scores0'][0, source]
        if rejected:
            indices, scores = indices[:0], scores[:0]
        return Matches(indices.cpu(), scores.cpu(), dict(covisibility_score=score,
            rejected=rejected, stop_layer=int(pred['stop_layer'])))


def create_flexglue(config=None):
    config = FlexGlueConfig() if config is None else config
    if isinstance(config, dict):
        config = FlexGlueConfig(**config)
    return FlexGlueExtractor(config), FlexGlueMatcher(config)
