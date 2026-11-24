"""RaCo detections and cached FlexGlue dense features, with fixed observation IDs."""
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Literal

import torch

from .base import Extractor, Matcher, Matches
from .matching_schedule import belady_evictions, tile_pair_order


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
    backend: Literal['float', 'fp8'] = 'float'
    quantized_weights: str | None = None  # None selects the default FP8 release
    matching_schedule: Literal['original', 'tiled_belady'] = 'original'

    def __post_init__(self):
        if self.backend not in ('float', 'fp8'):
            raise ValueError('FlexGlue backend must be float or fp8')
        if self.backend == 'float' and self.quantized_weights is not None:
            raise ValueError('Set backend=fp8 when supplying quantized_weights')
        if self.quantized_weights == '':
            raise ValueError('quantized_weights must be a nonempty checkpoint path or None')
        if self.max_keypoints < 1 or self.detection_resize < 32:
            raise ValueError('Positive keypoint count and detection_resize >= 32 required')
        if self.feature_resize < 32 or self.feature_resize % 32:
            raise ValueError('feature_resize must be a positive multiple of 32')
        if not 0 <= self.min_covisibility <= 1 or not 1 <= self.n_layers <= 9:
            raise ValueError('Require covisibility in [0,1] and 1..9 transformer layers')
        if self.early_stop_decay is not None and not 0 <= self.early_stop_decay < 1:
            raise ValueError('Early stopping needs decay in [0,1)')
        if self.min_covisibility == 0:
            self.early_stop_decay = None
        if self.gpu_cache_images < 2:
            raise ValueError('gpu_cache_images must retain at least two images')
        if self.matching_schedule not in ('original', 'tiled_belady'):
            raise ValueError('FlexGlue matching_schedule must be original or tiled_belady')

    def feature_options(self):
        return dict(version=1, detector='raco-v1.0.0', descriptor='flexglue-v0.1',
                    **{k: getattr(self, k) for k in ('max_keypoints', 'detection_resize',
                       'feature_resize', 'weights', 'detector_weights')})

    def matching_options(self):
        options = asdict(self)
        options.pop('gpu_cache_images')
        options.pop('matching_schedule')
        # Preserve existing float cache identities; both backends reuse the
        # same detector/descriptor features but never each other's matches.
        if self.backend == 'float':
            options.pop('backend')
            options.pop('quantized_weights')
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

    def match_tiled(self, frames, pairs):
        """Yield (i, j, matches) in tile order with future-aware feature eviction."""
        if len(pairs) == 0:
            return
        capacity = self.config.gpu_cache_images
        ordered = [pairs[index] for index in tile_pair_order(pairs, capacity)]
        nonempty = [frame.keypoints_2d is not None and len(frame.keypoints_2d) > 0 for frame in frames]
        active = [(i, j) for i, j in ordered if nonempty[i] and nonempty[j]]
        evictions = belady_evictions(active, capacity)
        self._images.clear()
        try:
            for i, j in ordered:
                if nonempty[i] and nonempty[j]:
                    for frame in next(evictions):
                        self._images.pop(id(frames[frame]), None)
                yield i, j, self.match(frames[i], frames[j])
        finally:
            self._images.clear()

    def _input(self, frame):
        # Keep the source object in the entry to protect against changed frame data.
        key = id(frame)
        entry = self._images.pop(key, None)
        source = frame.dense_features
        if entry is None or entry[0] is not source:
            if source is None or frame.image_size is None:
                raise ValueError('FlexGlue requires cached dense_features and image_size')
            height, width = frame.image_size
            features = source() if callable(source) else source
            data = dict(features=features[None].to(self.device),
                        keypoints=frame.keypoints_2d[None].to(self.device),
                        image_size=torch.tensor([[width, height]], device=self.device,
                                                dtype=torch.float32))
            entry = (source, data)
        self._images[key] = entry
        while len(self._images) > self.config.gpu_cache_images:
            self._images.popitem(last=False)
        return entry[1]

    @torch.inference_mode()
    def match(self, frame0, frame1):
        if min(len(frame0.keypoints_2d), len(frame1.keypoints_2d)) == 0:
            return Matches(torch.empty((0, 2), dtype=torch.long), torch.empty(0))
        if self.model is None:
            if self.config.backend == 'fp8':
                try:
                    from flexglue_quantize import FlexGlue
                except ModuleNotFoundError as error:
                    if error.name != 'flexglue_quantize':
                        raise
                    raise ImportError('FP8 matching requires the flexglue-quantize package; '
                                      'install the TorchSfM flexglue-fp8 extra') from error
                kw = dict(weights=self.config.quantized_weights)
            else:
                from flexglue import FlexGlue
                kw = {} if self.config.weights is None else {'weights': self.config.weights}
                kw['n_layers'] = self.config.n_layers
            self.model = FlexGlue(describe=False, refine=False,
                min_covisibility=self.config.min_covisibility,
                early_stop_decay=self.config.early_stop_decay,
                filter_threshold=self.config.filter_threshold,
                max_distance_px=self.config.max_distance_px, **kw).eval().to(self.device)
        # A packed checkpoint retains all layers; truncate at inference time.
        depth = {'n_layers': self.config.n_layers} if self.config.backend == 'fp8' else {}
        pred = self.model.match(self._input(frame0), self._input(frame1), sparse=True, **depth)
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
