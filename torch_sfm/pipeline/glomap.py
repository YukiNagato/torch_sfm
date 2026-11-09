from dataclasses import dataclass, field, asdict, replace
from torch_sfm.core import Frame, SFMStorage
from torch_sfm.utils import Timer
from torch_sfm.cache import create_cache, copy_cache
import numpy as np
import torch
from pathlib import Path
from torch_sfm.retrieval import get_retrieval, RetrievalResult
from torch_sfm.correspondence import get_extractor_and_matcher, Matches, FlexGlueConfig
from torch_sfm.geometry import (
    TwoViewConfig, TwoViewGeometry, RotationConfig, PositionConfig,
    camera_matrix, verify_two_view_batch, rotation_averaging, translation_averaging,
    BundleAdjustmentConfig, bundle_adjustment,
    ViewGraphCalibrationConfig, calibrate_view_graph,
)
from PIL import Image
from typing import List
from tqdm import tqdm


timer = Timer()


def _calibration_cache_options(config):
    if config is None:
        return None
    options = dict(config) if isinstance(config, dict) else asdict(config)
    # Execution choices retain the objective and convergence policy. Excluding
    # new fields also preserves the fingerprints of existing calibration caches.
    options.pop('shared_analytic_jacobian', None)
    options.pop('shared_cpu_max_pairs', None)
    return options


def _geometry_cache_options(config):
    if config is None:
        return None
    options = dict(config) if isinstance(config, dict) else asdict(config)
    # New controllers have a separate identity. Never reuse a geometry entry
    # from the removed mixed implementation; features/matches remain reusable.
    options['verification_revision'] = options.get('verification_revision', 2) if isinstance(config,dict) else 3
    for key in ('shared_focal_pair_batch_size','shared_focal_max_padded_matches'):
        options.pop(key,None)
    if options.get('calibration') == 'uncalibrated':
        options.pop('essential_cheirality',None)
    return options


def _verification_batches(pending, config):
    """Bound shared-focal batches by pairs and padded matching storage.

    Adjacent size buckets can share a batch, avoiding separate RANSAC tails.
    A pair larger than the storage budget is processed alone. Other model
    families keep their existing bucket grouping and pair limit.
    """
    if (config.calibration!='uncalibrated' or config.uncalibrated_model!='shared_focal'
            or config.shared_focal_pair_batch_size==0):
        for bucket in pending.values():
            for start in range(0,len(bucket),config.pair_batch_size):
                yield bucket[start:start+config.pair_batch_size]
        return
    chunk=[];capacity=0
    for key in sorted(pending):
        for entry in pending[key]:
            size=max(8,len(entry[4].indices))
            if chunk and (len(chunk)>=config.verification_pair_batch_size or
                          (len(chunk)+1)*max(capacity,size)>config.shared_focal_max_padded_matches):
                yield chunk
                chunk=[];capacity=0
            chunk.append(entry);capacity=max(capacity,size)
    if chunk:yield chunk


def load_frame_image(frame: Frame):
    if frame.image is None:
        image = np.array(Image.open(frame.frame_path))
        image = torch.from_numpy(image).cuda().float().permute((2, 0, 1)) / 255
        return image
    return frame.image


@dataclass
class GlomapConfig:
    retrieval_method: str = 'salad'
    correspondence_method: str = 'loma'
    flexglue: FlexGlueConfig = field(default_factory=FlexGlueConfig)
    retrieval_top_k: int = 10
    retrieval_threshold: float = 0.5

    cache_frame_images: bool = True
    use_cache: bool = True
    cache_backend: str = 'filesystem'  # filesystem or colmap
    database_path: Path | None = None  # default: workspace/database.db
    update_cache_for_retrieval: bool = False
    update_cache_for_keypoints: bool = False
    update_cache_for_correspondence: bool = False
    update_cache_for_two_view: bool = False
    geometry_device: str | None = None
    two_view: TwoViewConfig = field(default_factory=TwoViewConfig)
    rotation: RotationConfig = field(default_factory=RotationConfig)
    position: PositionConfig = field(default_factory=PositionConfig)
    bundle_adjustment: BundleAdjustmentConfig = field(default_factory=BundleAdjustmentConfig)
    export_colmap: bool = True
    calibration: ViewGraphCalibrationConfig = field(default_factory=ViewGraphCalibrationConfig)

    def __post_init__(self):
        if isinstance(self.flexglue, dict):
            self.flexglue = FlexGlueConfig(**self.flexglue)
        if not isinstance(self.flexglue, FlexGlueConfig):
            raise TypeError('flexglue must be FlexGlueConfig or a dictionary')
        if self.correspondence_method == 'flexglue' and self.cache_backend != 'filesystem':
            raise ValueError('FlexGlue dense feature caches require cache_backend="filesystem"')
        if isinstance(self.calibration, dict):
            self.calibration = ViewGraphCalibrationConfig(**self.calibration)
        if not isinstance(self.calibration, ViewGraphCalibrationConfig):
            raise TypeError('calibration must be ViewGraphCalibrationConfig or a dictionary')
        if self.cache_backend not in ('filesystem', 'colmap'):
            raise ValueError('cache_backend must be filesystem or colmap')


@dataclass
class GlomapResult:
    rotations: torch.Tensor  # (N, 3, 3), world-to-camera
    centers: torch.Tensor  # (N, 3), camera centers in world coordinates
    translations: torch.Tensor  # (N, 3), t = -R C
    registered: torch.Tensor  # (N,), cameras with both rotation and position
    rotation_registered: torch.Tensor
    points3d: torch.Tensor
    observations: torch.Tensor  # (M, 3), [point, frame-list index, keypoint]
    two_view_geometries: dict
    positioning_method: str
    solver_summaries: list
    intrinsics: torch.Tensor | None = None
    reprojection_errors: torch.Tensor | None = None
    bundle_adjustment_stats: dict = field(default_factory=dict)
    
    
class GlomapPipeline:
    def __init__(self, cfg: GlomapConfig, workspace: Path):
        self.timer = timer
        self.cfg = cfg
        self.workspace = workspace
        self.cache_root = self.workspace / 'cache'
        self.cache = create_cache(cfg.cache_backend, self.workspace, cfg.correspondence_method, cfg.database_path)
        self._cache_prepared = False
        self.image_txt_path = self.workspace / 'images.txt'
        self.intrinsic_path = self.workspace / 'intrinsics.txt'
        with open(self.image_txt_path) as f:
            self.image_list = [Path(line.strip()) for line in f.readlines()]
        self.frames = []
        self.extractor = None
        self.matcher = None
        self._frontend_options = None
        self.storage = None

    def _prepare_frontend_config(self):
        """Do not reuse loaded networks after changing frontend settings."""
        options = (asdict(self.cfg.flexglue)
                   if self.cfg.correspondence_method == 'flexglue' else None)
        if options != self._frontend_options:
            self.extractor = self.matcher = None
            self._frontend_options = options

    @timer
    def prepare_verification(self):
        """Retain packed keypoint views and metadata, release pixels/features/models."""
        device = self.cfg.geometry_device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.storage = SFMStorage.from_frames(self.frames, device=device, dtype=torch.float64)
        for index, frame in enumerate(self.frames):
            view = Frame.from_storage(self.storage, index)
            frame.keypoints_2d, frame.image_size = view.keypoints_2d, view.image_size
            frame.image, frame.features, frame.dense_features = None, None, None
        self.extractor = self.matcher = None
        return self.storage

    def _prepare_cache(self, frames, intrinsic=None):
        if self.cfg.use_cache and (intrinsic is not None or not self._cache_prepared):
            if self.cfg.cache_backend == 'colmap':
                trusted = self.cfg.calibration.mode == 'trusted' or getattr(self, '_calibration_ready', False)
                # The pipeline validates exact K, matched pixels and options
                # per cached pair. Preserve both stages across initial/new K.
                self.cache.prepare(frames, self.get_intrinsics() if intrinsic is None else intrinsic,
                                   preserve_geometry_cache=True, prior_focal_length=trusted)
            self._cache_prepared = True

    def close(self):
        self.cache.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @timer
    def convert_cache(self, backend, database_path=None):
        """Copy cached stages without extracting/matching or changing this pipeline's backend."""
        frames = self.frames or [Frame(frame_idx=i, frame_path=path)
                                 for i, path in enumerate(self.image_list)]
        intrinsic = self.get_intrinsics()
        trusted = self.cfg.calibration.mode == 'trusted'
        export_pairs = None if trusted else set()
        state_path = self.workspace / 'view_graph_calibration.pt'
        if not trusted and state_path.exists():
            state = torch.load(state_path, map_location='cpu', weights_only=True)
            if (state.get('version') == 1 and _calibration_cache_options(state.get('config'))
                    == _calibration_cache_options(self.cfg.calibration)
                    and torch.equal(state['input_intrinsics'], intrinsic.cpu())):
                intrinsic = state['calibrated_intrinsics']
                trusted = True
                export_pairs = (set(map(tuple, state['accepted_pairs']))
                                if state.get('verification_complete')
                                and _geometry_cache_options(state.get('two_view_config'))
                                == _geometry_cache_options(self.cfg.two_view) else set())
        with create_cache(backend, self.workspace, self.cfg.correspondence_method, database_path) as target:
            if target.path.resolve() == self.cache.path.resolve():
                raise ValueError('Source and target caches must be different')
            if self.cfg.cache_backend == 'colmap':
                self.cache.prepare(frames, intrinsic, preserve_geometry_cache=True, prior_focal_length=trusted)
            else:
                self.cache.prepare(frames, intrinsic)
            if backend == 'colmap':
                target.prepare(frames, intrinsic, prior_focal_length=trusted)
            else:
                target.prepare(frames, intrinsic)
            return copy_cache(self.cache, target, frames, intrinsics=intrinsic, two_view_pairs=export_pairs)
    
    @timer
    def get_intrinsics(self):
        if not self.intrinsic_path.exists():
            from torch_sfm.calibration import AnyCalibCalibration
            intrinsic = AnyCalibCalibration('pinhole').estimate(self.image_list[:10])
            params = intrinsic.params
            np.savetxt(self.intrinsic_path, params)
        intrinsic = np.loadtxt(self.intrinsic_path)
        return torch.from_numpy(intrinsic)
    
    @timer
    @torch.inference_mode()
    def get_retrieval_pairs(self):
        cached = (self.cache.load('retrieval') if self.cfg.use_cache
                  and not self.cfg.update_cache_for_retrieval else None)
        if cached is not None:
            retrieval_pairs = cached
        else:
            retrieval = get_retrieval(self.cfg.retrieval_method)
            # extract features
            features = []

            for frame in self.frames:
                feature = retrieval.extract(load_frame_image(frame))
                features.append(feature)

            retrieval_top_k = retrieval.retrieval_top_k(features, self.cfg.retrieval_top_k)

            retrieval_pairs = set()

            for i, result in enumerate(retrieval_top_k):
                for j in result.indices.cpu().numpy():
                    j = int(j)
                    if i == j:
                        continue
                    if (i, j) in retrieval_pairs or (j, i) in retrieval_pairs:
                        continue
                    retrieval_pairs.add((i, j))
            retrieval_pairs = list(retrieval_pairs)
            retrieval_pairs = np.array(retrieval_pairs, dtype=np.int32).reshape(-1, 2)
            if self.cfg.use_cache:
                self.cache.save('retrieval', retrieval_pairs)

        return retrieval_pairs
    
    @timer
    @torch.inference_mode()
    def get_keypoints(self, frames: List[Frame]):
        """Fill each frame's keypoints_2d and features, and return the frames."""
        self._prepare_cache(frames)
        self._prepare_frontend_config()

        feature_options = (self.cfg.flexglue.feature_options()
                           if self.cfg.correspondence_method == 'flexglue' else None)
        for frame in tqdm(frames, desc='Extract keypoints'):
            data = (self.cache.load('keypoints', frame.frame_idx) if self.cfg.use_cache
                    and not self.cfg.update_cache_for_keypoints else None)
            if data is not None and feature_options is not None and data.get('options') != feature_options:
                data = None
            if data is not None:
                device = frame.image.device if frame.image is not None else 'cpu'
                frame.keypoints_2d = data['keypoints_2d'].to(device)
                frame.features = data['features'].to(device) if data['features'] is not None else None
                frame.dense_features = data.get('dense_features')
                frame.image_size = data.get('image_size', frame.image_size)
            else:
                if self.extractor is None:
                    self.extractor, self.matcher = get_extractor_and_matcher(self.cfg.correspondence_method, self.cfg.flexglue)
                image = frame.image
                frame.image = load_frame_image(frame)
                self.extractor.extract(frame)
                if not self.cfg.cache_frame_images:
                    frame.image = image
                if self.cfg.use_cache:
                    self.cache.save('keypoints', {
                        'keypoints_2d': frame.keypoints_2d.cpu(),
                        'features': frame.features.cpu() if frame.features is not None else None,
                        'dense_features': frame.dense_features,
                        'image_size': frame.image_size,
                        'options': feature_options,
                    }, frame.frame_idx)

        if hasattr(self.extractor, 'release'):
            self.extractor.release()
        return frames

    @timer
    @torch.inference_mode()
    def get_correspondences(self, frames: List[Frame], retrieval_pairs: np.ndarray):
        """Return {(i, j): Matches} on CPU; pairs index the supplied frames."""
        self._prepare_cache(frames)
        self._prepare_frontend_config()

        match_options = (self.cfg.flexglue.matching_options()
                         if self.cfg.correspondence_method == 'flexglue' else None)
        correspondences = {}
        for i, j in tqdm(retrieval_pairs, desc='Match pairs'):
            i, j = int(i), int(j)
            frame0, frame1 = frames[i], frames[j]
            key = (frame0.frame_idx, frame1.frame_idx)
            data = (self.cache.load('correspondences', key) if self.cfg.use_cache
                    and not self.cfg.update_cache_for_correspondence
                    and not self.cfg.update_cache_for_keypoints else None)
            if data is not None and match_options is not None and data.get('options') != match_options:
                data = None
            if data is not None:
                matches = Matches(data['indices'], data['scores'], data.get('metadata', {}))
            else:
                if self.matcher is None:
                    self.extractor, self.matcher = get_extractor_and_matcher(self.cfg.correspondence_method, self.cfg.flexglue)
                image0, image1 = frame0.image, frame1.image
                if getattr(self.matcher, 'requires_images', True):
                    frame0.image = load_frame_image(frame0)
                    frame1.image = load_frame_image(frame1)
                matches = self.matcher.match(frame0, frame1)
                matches = Matches(matches.indices.cpu(), matches.scores.cpu(), matches.metadata)
                if not self.cfg.cache_frame_images:
                    frame0.image, frame1.image = image0, image1
                if self.cfg.use_cache:
                    self.cache.save('correspondences', {
                        'indices': matches.indices,
                        'scores': matches.scores,
                        'metadata': matches.metadata,
                        'options': match_options,
                    }, key)
            correspondences[i, j] = matches

        self.matching_stats = {
            'pairs': len(correspondences),
            'rejected_pairs': sum(bool(m.metadata.get('rejected')) for m in correspondences.values()),
            'empty_pairs': sum(len(m.indices) == 0 for m in correspondences.values()),
            'below_geometry_minimum': sum(len(m.indices) < self.cfg.two_view.min_num_inliers
                                          for m in correspondences.values()),
            'matches': sum(len(m.indices) for m in correspondences.values()),
        }
        return correspondences

    @timer
    def get_two_view_geometries(self, frames: List[Frame], correspondences: dict, intrinsic: torch.Tensor,
                                *, config=None, cache_kind='two_view'):
        config = self.cfg.two_view if config is None else config
        device = torch.device(self.cfg.geometry_device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        matrices = camera_matrix(intrinsic.to(device=device, dtype=torch.float64))
        if matrices.ndim == 2:
            matrices = matrices.expand(len(frames), 3, 3)
        if matrices.shape != (len(frames), 3, 3):
            raise ValueError('Provide shared pinhole intrinsics or one camera matrix per frame')
        self._prepare_cache(frames, intrinsic)
        sizes = []
        for frame in frames:
            if frame.image is not None:
                sizes.append(tuple(frame.image.shape[-2:]))
            elif frame.image_size is not None:
                sizes.append(frame.image_size)
            else:
                with Image.open(frame.frame_path) as image:
                    sizes.append((image.height, image.width))

        geometries, pending = {}, {}
        refresh = (self.cfg.update_cache_for_two_view or self.cfg.update_cache_for_keypoints
                   or self.cfg.update_cache_for_correspondence)
        from torch_sfm.geometry.batched_two_view import resolve_backend
        config = replace(config,backend=resolve_backend(config,device))
        options = _geometry_cache_options(config)
        for (i, j), matches in tqdm(correspondences.items(), desc='Verify pairs'):
            # Rejected/undersized pairs never enter padding, CUDA transfers or RANSAC.
            if len(matches.indices) < max(4, config.min_num_inliers):
                geometries[i, j] = TwoViewGeometry(Matches(
                    matches.indices[:0].to(device), matches.scores[:0].to(device)))
                continue
            frame0, frame1 = frames[i], frames[j]
            key = (frame0.frame_idx, frame1.frame_idx)
            # Geometry also depends on calibration, thresholds and matched pixels.
            inputs = {
                'intrinsics': matrices[[i, j]].cpu(),
                'indices': matches.indices.cpu(),
                'scores': matches.scores.cpu(),
                'points0': frame0.keypoints_2d[matches.indices[:, 0].to(frame0.keypoints_2d.device)].cpu(),
                'points1': frame1.keypoints_2d[matches.indices[:, 1].to(frame1.keypoints_2d.device)].cpu(),
            }
            data = None
            cached = self.cache.load(cache_kind, key) if self.cfg.use_cache and not refresh else None
            if cached is not None:
                if (cached.get('version') == 3 and _geometry_cache_options(cached['options']) == options
                        and cached['sizes'] == (sizes[i], sizes[j])
                        and all(torch.equal(value, cached['inputs'][key]) for key, value in inputs.items())):
                    data = cached['geometry']
            if data is not None:
                data = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                        for key, value in data.items()}
                data['matches'] = Matches(data.pop('indices'), data.pop('scores'))
                geometry = TwoViewGeometry(**data)
                geometries[i, j] = geometry
            else:
                capacity = 1 << max(3, (len(matches.indices) - 1).bit_length())
                pending.setdefault(capacity, []).append((i, j, key, inputs, matches))
        for chunk in tqdm(_verification_batches(pending, config), desc='Verify batches'):
            pair0, pair1 = [p[0] for p in chunk], [p[1] for p in chunk]
            results = verify_two_view_batch(
                [frames[i].keypoints_2d.to(device) for i in pair0],
                [frames[j].keypoints_2d.to(device) for j in pair1], [p[4] for p in chunk],
                matrices[pair0], matrices[pair1], config,
                [(sizes[i], sizes[j]) for i, j, *_ in chunk])
            for (i, j, key, inputs, _), geometry in zip(chunk, results):
                geometries[i, j] = geometry
                if self.cfg.use_cache:
                    data = {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                            for key, value in vars(geometry).items() if key != 'matches'}
                    data['indices'] = geometry.matches.indices.cpu()
                    data['scores'] = geometry.matches.scores.cpu()
                    self.cache.save(cache_kind, {'version': 3, 'options': options,
                        'sizes': (sizes[i], sizes[j]), 'inputs': inputs, 'geometry': data}, key)
        from torch_sfm.geometry.two_view_poses import complete_fundamental_poses
        complete_fundamental_poses(geometries, frames, intrinsic, config)
        return {pair: geometries[pair] for pair in correspondences}

    @timer
    def get_calibrated_view_graph(self, frames, correspondences, intrinsic):
        """Calibrate before any rotation/position averaging; preserve raw matches."""
        import json
        cfg = self.cfg.calibration
        if cfg.mode == 'trusted':
            self.calibration_stats = {'mode': 'trusted', 'optimized': False}
            geometries = self.get_two_view_geometries(frames, correspondences, intrinsic,
                config=replace(self.cfg.two_view, calibration='calibrated'))
            (self.workspace / 'view_graph_calibration.json').write_text(
                json.dumps(self.calibration_stats, indent=2) + '\n')
            return intrinsic, geometries
        shared_focal = cfg.shared_focal_ransac and camera_matrix(intrinsic).ndim == 2
        first = self.get_two_view_geometries(frames, correspondences, intrinsic,
            config=replace(self.cfg.two_view, calibration='uncalibrated', refine_relative_pose=False,
                uncalibrated_model='shared_focal' if shared_focal else 'fundamental'),
            cache_kind='two_view_uncalibrated')
        device = self.cfg.geometry_device or ('cuda' if torch.cuda.is_available() else 'cpu')
        import hashlib
        digest = hashlib.sha256(json.dumps(_calibration_cache_options(cfg), sort_keys=True).encode())
        digest.update(str((len(frames), tuple(intrinsic.shape))).encode())
        digest.update(intrinsic.detach().cpu().double().numpy().tobytes())
        ordered = sorted(first)
        digest.update(json.dumps([(p, first[p].config, first[p].F is not None) for p in ordered]).encode())
        fundamentals = [first[p].F for p in ordered if first[p].F is not None]
        if fundamentals:
            digest.update(torch.stack(fundamentals).detach().cpu().double().numpy().tobytes())
        fingerprint = digest.hexdigest()
        state_path = self.workspace / 'view_graph_calibration.pt'
        refresh = (self.cfg.update_cache_for_two_view or self.cfg.update_cache_for_keypoints
                   or self.cfg.update_cache_for_correspondence)
        cached = (torch.load(state_path, map_location='cpu', weights_only=True)
                  if self.cfg.use_cache and not refresh and state_path.exists() else None)
        if cached is not None and cached.get('version') == 1 and cached.get('fingerprint') == fingerprint:
            updated = cached['calibrated_intrinsics'].to(device=device, dtype=intrinsic.dtype)
            accepted = set(map(tuple, cached['accepted_pairs']))
            stats = dict(cached['stats'], cache_hit=True)
        else:
            calibration_geometry = first
            if cfg.refine_fundamental and not shared_focal:
                from torch_sfm.geometry.fundamental_refinement import refine_view_graph_fundamentals
                calibration_geometry = refine_view_graph_fundamentals(frames,first,device=device,
                    batch_size=self.cfg.two_view.pair_batch_size, max_iterations=cfg.fundamental_refine_iterations,
                    intrinsics=intrinsic,model=cfg.fundamental_refine_model,
                    focal_ratio_bounds=(cfg.min_focal_ratio,cfg.max_focal_ratio))
            updated, accepted, stats = calibrate_view_graph(calibration_geometry, intrinsic.to(device), len(frames), cfg)
            stats['cache_hit'] = False
        self._calibration_ready = True
        if self.cfg.use_cache:
            self.cache.invalidate_two_view(set(correspondences)-accepted)
        def save_calibration(verification_complete=False):
            # Retain a completed calibration even if later verification is interrupted.
            self.calibration_stats = stats
            (self.workspace / 'view_graph_calibration.json').write_text(json.dumps(stats, indent=2) + '\n')
            temporary = state_path.with_suffix('.pt.tmp')
            torch.save({'version': 1, 'fingerprint': fingerprint, 'accepted_pairs': sorted(accepted),
                        'verification_complete': verification_complete,
                        'two_view_config': _geometry_cache_options(self.cfg.two_view),
                        'input_intrinsics': intrinsic.cpu(), 'calibrated_intrinsics': updated.cpu(),
                        'config': asdict(cfg), 'stats': stats}, temporary)
            temporary.replace(state_path)
        save_calibration()
        second_config = replace(self.cfg.two_view, calibration='calibrated',
            max_error=cfg.relpose_max_error, min_num_inliers=cfg.relpose_min_num_inliers,
            min_inlier_ratio=cfg.relpose_min_inlier_ratio, refine_relative_pose=True)
        # COLMAP re-estimates epipolar pairs only. A planar/panoramic H remains
        # valid in pixel space when the focal changes; decompose it with new K.
        planar = {pair: replace(first[pair]) for pair in correspondences
                  if pair in accepted and first[pair].config in ('PLANAR', 'PANORAMIC')}
        verified = self.get_two_view_geometries(frames,
            {pair: matches for pair, matches in correspondences.items() if pair in accepted and pair not in planar},
            updated, config=second_config)
        stats['reverified_pairs'] = len(verified)
        from torch_sfm.geometry.two_view_poses import complete_homography_poses
        complete_homography_poses(planar, frames, updated, second_config)
        stats['retained_homography_pairs'] = len(planar)
        # Publish the final geometry for cache conversion with its actual K
        # and distinct provenance, so it cannot masquerade as 1px E/F RANSAC.
        if self.cfg.use_cache:
            matrices = camera_matrix(updated).detach().cpu()
            if matrices.ndim == 2:
                matrices = matrices.expand(len(frames), -1, -1)
            for (i, j), geometry in planar.items():
                key = (frames[i].frame_idx, frames[j].frame_idx)
                source = self.cache.load('two_view_uncalibrated', key)
                if source is None:
                    continue
                data = {name: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                        for name, value in vars(geometry).items() if name != 'matches'}
                data.update(indices=geometry.matches.indices.cpu(), scores=geometry.matches.scores.cpu())
                inputs = dict(source['inputs'], intrinsics=matrices[[i, j]])
                self.cache.save('two_view', dict(source, geometry=data, inputs=inputs,
                    options=dict(source['options'], pose_source='retained_homography')), key)
        verified.update(planar)
        verified = {pair: verified[pair] for pair in correspondences if pair in verified}
        stats['relative_pose_pairs'] = sum(g.R is not None for g in verified.values())
        save_calibration(verification_complete=True)
        return updated, verified

    @timer
    def get_rotations(self, geometries: dict):
        return rotation_averaging(len(self.frames), geometries, self.cfg.rotation)

    @timer
    def get_positions(self, geometries: dict, rotations: torch.Tensor, intrinsic: torch.Tensor):
        return translation_averaging(self.frames, geometries, rotations, intrinsic, self.cfg.position)

    @timer
    def get_bundle_adjustment(self, rotations, positions, intrinsic):
        return bundle_adjustment(self.frames, rotations.rotations, positions.centers, positions.points3d,
            positions.observations, intrinsic, positions.registered,
            self.cfg.bundle_adjustment, rotations.edges)

    @timer
    def visualize(self, output_path: Path | None = None, *, max_points=100000, camera_scale=None, view_up='auto'):
        from torch_sfm.visualization import visualize_reconstruction
        source = self.result if hasattr(self, 'result') else self.workspace / 'reconstruction.pt'
        image_paths = [frame.frame_path for frame in self.frames] or self.image_list
        return visualize_reconstruction(source, output_path or self.workspace / 'reconstruction.html',
            image_paths=image_paths, colmap_path=self.workspace / 'sparse' / '0',
            max_points=max_points, camera_scale=camera_scale, view_up=view_up)

    @timer
    def load_frames(self):
        self.frames = []
        for idx, image_path in enumerate(tqdm(self.image_list)):
            frame = Frame(frame_idx=idx, frame_path=image_path)
            if self.cfg.cache_frame_images:
                frame.image = load_frame_image(frame)
            self.frames.append(frame)

    @timer
    def run(self):
        self._calibration_ready = False
        intrinsic = self.get_intrinsics()
        self.load_frames()
        self._prepare_cache(self.frames, intrinsic)
        retrieval_pairs = self.get_retrieval_pairs()
        self.get_keypoints(self.frames)
        correspondences = self.get_correspondences(self.frames, retrieval_pairs)
        self.prepare_verification()
        input_intrinsic = intrinsic.clone()
        intrinsic, geometries = self.get_calibrated_view_graph(self.frames, correspondences, intrinsic)
        rotations = self.get_rotations(geometries)
        positions = self.get_positions(rotations.edges, rotations.rotations, intrinsic)
        initial = {
            'rotations': rotations.rotations.cpu(), 'centers': positions.centers.cpu(),
            'translations': -(rotations.rotations @ positions.centers[..., None]).squeeze(-1).cpu(),
            'registered': positions.registered.cpu(), 'rotation_registered': rotations.registered.cpu(),
            'points3d': positions.points3d.cpu(), 'observations': positions.observations.cpu(),
            'intrinsics': intrinsic.cpu(), 'image_paths': [str(frame.frame_path) for frame in self.frames],
            'input_intrinsics': input_intrinsic.cpu(), 'calibration': self.calibration_stats,
            'positioning_method': positions.method, 'stage': 'global_positioning',
            'solver_summaries': [asdict(s) for s in rotations.summaries + positions.summaries],
        }
        torch.save(initial, self.workspace / 'reconstruction_initial.pt')
        refined = self.get_bundle_adjustment(rotations, positions, intrinsic)
        translations = -(refined.rotations @ refined.centers[..., None]).squeeze(-1)
        self.result = GlomapResult(
            refined.rotations, refined.centers, translations, refined.registered,
            rotations.registered, refined.points3d, refined.observations, geometries,
            positions.method, rotations.summaries + positions.summaries + refined.summaries,
            refined.intrinsics, refined.reprojection_errors, refined.stats,
        )
        torch.save({
            'rotations': self.result.rotations.cpu(),
            'centers': self.result.centers.cpu(),
            'translations': self.result.translations.cpu(),
            'registered': self.result.registered.cpu(),
            'rotation_registered': self.result.rotation_registered.cpu(),
            'points3d': self.result.points3d.cpu(),
            'observations': self.result.observations.cpu(),
            'intrinsics': self.result.intrinsics.cpu(),
            'initial_intrinsics': intrinsic.cpu(),
            'input_intrinsics': input_intrinsic.cpu(), 'calibration': self.calibration_stats,
            'image_paths': [str(frame.frame_path) for frame in self.frames],
            'positioning_method': self.result.positioning_method,
            'solver_summaries': [asdict(summary) for summary in self.result.solver_summaries],
            'reprojection_errors': self.result.reprojection_errors.cpu(),
            'bundle_adjustment_stats': self.result.bundle_adjustment_stats,
            'stage': 'bundle_adjustment' if refined.summaries else 'global_positioning',
        }, self.workspace / 'reconstruction.pt')
        if self.cfg.export_colmap:
            from torch_sfm.io.torch_model import export_colmap
            export_colmap(self.frames, self.result, self.workspace / 'sparse' / '0', self.workspace / 'images')
        return self.result


if __name__ == '__main__':
    cfg = GlomapConfig(
        retrieval_top_k=10
    )
    data_path = Path('/root/autodl-tmp/data/sfm_data/south-building-1920x1080')
    pipeline = GlomapPipeline(cfg, data_path)

    result = pipeline.run()
    pipeline.visualize()
    print(f'{int(result.registered.sum())}/{len(result.registered)} registered images, '
          f'{len(result.points3d)} points')
    timer.report()



        
        
    

