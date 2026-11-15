"""GlueMap-style reconstruction: FlexGlue -> stars -> consistency -> global poses.

Frontend code/cache semantics are shared with GlomapPipeline. Local stars are
kept for reuse; global fusion solves rotations, centers and per-star scales.
Refinement geometrically verifies real FlexGlue matches before triangulation and BA.
"""
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import json
import hashlib
import time

import numpy as np
import torch
from tqdm import tqdm

from torch_sfm.correspondence import FlexGlueConfig
from torch_sfm.geometry.depth_consistency import check_depth_consistency
from torch_sfm.geometry.star_cache import compact_star_cache
from torch_sfm.geometry.star_fusion import StarFusionConfig, compact_star, fuse_stars
from torch_sfm.geometry.track_refinement import TrackRefinementConfig, refine_tracks
from torch_sfm.core import Frame
from torch_sfm.correspondence import Matches
from torch_sfm.geometry.two_view import TwoViewConfig, TwoViewGeometry, camera_matrix
from torch_sfm.cache import FileSystemCache
from torch_sfm.prior import Solv3RPrior
from .glomap import GlomapConfig, GlomapPipeline
from .stars import build_stars, graph_components


@dataclass
class GluemapConfig:
    flexglue: FlexGlueConfig = field(default_factory=FlexGlueConfig)
    retrieval_top_k: int = 40
    retrieval_sequences: Path | None = None
    retrieval_sequential_neighbors: int = 30
    use_cache: bool = True
    max_neighbors: int = 25
    min_matches: int = 15  # same minimum as GLOMAP's TwoViewConfig
    checkpoint: str | None = None
    calls: int = 16
    device: str = 'cuda'
    reprojection_threshold: float = 4.  # pixels at Solv3R's processed resolution
    min_overlap: float = .05
    pair_batch_size: int = 16  # depth checks, not matcher or prior batching
    image_cache_gib: float = 20.  # hard CPU image-cache budget
    save_virtual_candidates: bool = True
    save_depth: bool | None = None  # legacy config alias; dense maps are never saved
    output: Path | None = None
    fuse_global: bool = True
    fusion: StarFusionConfig = field(default_factory=StarFusionConfig)
    intrinsics_source: str = 'solv3r'  # fixed reads workspace/intrinsics.txt in original pixels
    intrinsics_path: Path | None = None
    refine: bool = True
    verify_matches: bool = True
    two_view: TwoViewConfig = field(default_factory=TwoViewConfig)
    refinement: TrackRefinementConfig = field(default_factory=TrackRefinementConfig)

    def __post_init__(self):
        if self.save_depth is not None:
            self.save_virtual_candidates = self.save_depth
            self.save_depth = None
        if min(self.max_neighbors, self.min_matches, self.calls, self.pair_batch_size) < 1:
            raise ValueError('Counts must be positive')
        if self.reprojection_threshold <= 0 or not 0 <= self.min_overlap <= 1 or self.image_cache_gib <= 0:
            raise ValueError('Invalid consistency thresholds or image cache budget')
        if (self.fuse_global and self.refine and self.refinement.strategy == 'augmented'
                and self.refinement.virtual_tracks.enabled and not self.save_virtual_candidates):
            raise ValueError('Augmented BA requires save_virtual_candidates=True for virtual tracks')
        if self.two_view.calibration != 'calibrated':
            raise ValueError('GlueMap match verification uses the fused camera intrinsics; use calibrated mode')
        if self.intrinsics_source not in ('solv3r', 'fixed'):
            raise ValueError('Intrinsics source must be solv3r or fixed')
        if self.intrinsics_path is not None and self.intrinsics_source != 'fixed':
            raise ValueError('Set intrinsics_source=fixed when specifying intrinsics_path')

    def refinement_config(self, initial):
        """A saved fixed calibration stays fixed during refine-only resumes."""
        fixed = self.intrinsics_source == 'fixed' or initial.get('intrinsics_policy', {}).get('fixed', False)
        if not fixed:
            return self.refinement
        return replace(self.refinement, bundle_adjustment=replace(self.refinement.bundle_adjustment,
                       refine_focal_length=False, refine_principal_point=False))


class GluemapPipeline:
    def __init__(self, cfg: GluemapConfig, workspace: Path):
        self.cfg, self.workspace = cfg, Path(workspace)
        self.frontend = GlomapPipeline(GlomapConfig(
            correspondence_method='flexglue', flexglue=cfg.flexglue,
            retrieval_top_k=cfg.retrieval_top_k, cache_backend='filesystem',
            retrieval_sequences=cfg.retrieval_sequences,
            retrieval_sequential_neighbors=cfg.retrieval_sequential_neighbors,
            cache_frame_images=False, use_cache=cfg.use_cache,
            geometry_device=cfg.device, two_view=cfg.two_view), self.workspace)
        self.prior = Solv3RPrior(checkpoint=cfg.checkpoint, calls=cfg.calls, device=cfg.device)
        self.output = Path(cfg.output) if cfg.output is not None else self.workspace / 'gluemap'
        self.timings = {}
        self._refinement_inputs = None

    def close(self):
        self.frontend.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _sync(self):
        if torch.device(self.cfg.device).type == 'cuda':
            torch.cuda.synchronize(self.cfg.device)

    def _stage(self, name, function, *args, **kwargs):
        self._sync()
        start = time.perf_counter()
        result = function(*args, **kwargs)
        self._sync()
        self.timings[name] = self.timings.get(name, 0.) + time.perf_counter() - start
        return result

    def _load_images(self):
        # Pipeline owns the CPU cache; prior receives actual tensors, never IDs.
        from solv3r import load_images
        paths = self.frontend.image_list
        first = load_images([str(paths[0])])
        needed = len(paths) * first.numel() * first.element_size()
        if needed > self.cfg.image_cache_gib * 1024 ** 3:
            raise MemoryError(f'CPU image cache needs {needed / 1024**3:.2f} GiB; '
                              'increase image_cache_gib or reduce the input image list')
        images = torch.empty((len(paths), *first.shape[1:]), dtype=torch.float32)
        images[0] = first[0]
        for i in tqdm(range(1, len(paths)), desc='Cache prior images on CPU'):
            images[i] = load_images([str(paths[i])], size=first.shape[-2:])[0]
        return images

    def _resolve_intrinsics(self, initial):
        if self.cfg.intrinsics_source != 'fixed':
            return initial
        path = Path(self.cfg.intrinsics_path) if self.cfg.intrinsics_path is not None else self.frontend.intrinsic_path
        # Unlike auto calibration, a requested fixed input must actually exist.
        params = torch.from_numpy(np.loadtxt(path)).double()
        n = len(initial['registered'])
        if params.shape == (4,):
            groups = torch.zeros(n, dtype=torch.long)
        elif params.shape == (n, 4):
            _, groups = torch.unique(params, dim=0, return_inverse=True)
        else:
            raise ValueError('Fixed intrinsics need fx fy cx cy: one shared row or one row per image')
        matrix = camera_matrix(params)
        return dict(initial, intrinsics=matrix, camera_groups=groups,
                    intrinsics_policy=dict(source='file', fixed=True, path=str(path.resolve()),
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), parameters=params.tolist()))

    @torch.inference_mode(False)
    def fuse_saved_stars(self):
        """Resume from summary.json without loading frontend networks or Solv3R.

        Local star caches are never overwritten. The manifest's exact image
        order must match this workspace. Compact caches reuse saved parallax;
        legacy dense caches are mapped one star at a time.
        """
        start = time.perf_counter()
        manifest = json.loads((self.output / 'summary.json').read_text())
        if [Path(p).resolve() for p in manifest['image_paths']] != [p.resolve() for p in self.frontend.image_list]:
            raise ValueError('Saved star image list differs from the workspace')
        stars = []
        for record in manifest['stars']:
            data = torch.load(self.output / record['path'], map_location='cpu', weights_only=True, mmap=True)
            if tuple(record['image_ids']) != tuple(data['image_ids']):
                raise ValueError('Star IDs differ from the saved manifest')
            stars.append(compact_star(data, self.cfg.fusion))
            del data
        load_time = time.perf_counter() - start
        result = fuse_stars(stars, manifest['original_image_sizes'], self.cfg.fusion, self.cfg.device,
                            sequential_edges=manifest.get('sequential_edges', ()))
        result = self._resolve_intrinsics(result)
        result['image_paths'] = manifest['image_paths']
        result['image_sizes'] = manifest['original_image_sizes']
        report = result['fusion_report']
        report['config'] = asdict(self.cfg.fusion)
        report['intrinsics_policy'] = result.get('intrinsics_policy', dict(source='solv3r', fixed=False))
        report['timings_seconds']['load_and_parallax'] = load_time
        report['timings_seconds']['total'] = time.perf_counter() - start
        result['stage'] = 'global_poses_before_triangulation'
        torch.save(result, self.output / 'global_poses.pt')
        (self.output / 'global_summary.json').write_text(json.dumps(report, indent=2) + '\n')
        from torch_sfm.visualization.reconstruction import visualize_reconstruction
        visualize_reconstruction(result, self.output / 'global_poses.html',
                                 image_sizes=manifest['original_image_sizes'])
        return report

    def _load_refinement_inputs(self, manifest, *, features_only=False):
        """Read feature identities and matches using the original frontend settings.

        Resume is deliberately read-only: missing/incompatible caches raise an
        error rather than regenerating matches with another set of keypoint IDs.
        """
        if self._refinement_inputs is not None:
            frames, matches = self._refinement_inputs
            return frames, {} if features_only else matches
        from torch_sfm.correspondence import FlexGlueConfig
        original_config = FlexGlueConfig(**manifest['config']['flexglue'])
        frames = []
        for i,path in enumerate(self.frontend.image_list):
            data = self.frontend.cache.load('keypoints', i)
            if data is None or data.get('options') != original_config.feature_options():
                raise ValueError(f'Missing/incompatible FlexGlue feature cache for image {i}')
            frames.append(Frame(frame_idx=i, frame_path=path,
                keypoints_2d=data['keypoints_2d'], image_size=data['image_size']))
        if features_only:
            return frames, {}
        matches = {}
        for i,j in manifest['accepted_pairs']:
            data = self.frontend.cache.load('correspondences',(i,j))
            reverse = data is None
            if reverse:
                data = self.frontend.cache.load('correspondences',(j,i))
            if data is None or data.get('options') != original_config.matching_options():
                raise ValueError(f'Missing/incompatible FlexGlue match cache for pair {(i,j)}')
            indices = data['indices'].flip(-1) if reverse else data['indices']
            matches[i,j] = Matches(indices,data['scores'],data.get('metadata',{}))
        return frames,matches

    def _verify_refinement_matches(self, frames, matches, intrinsics):
        """Use GLOMAP's E/F/H inliers, preserving original feature identities.

        Only the correspondence subsets enter refinement: estimated pair poses
        do not replace the already fused cameras. Geometry lives under this
        output, separate from the frontend workspace's differently calibrated K.
        """
        report = dict(enabled=self.cfg.verify_matches, input_pairs=len(matches),
                      input_matches=sum(len(m.indices) for m in matches.values()))
        if self.cfg.verify_matches:
            K = intrinsics.detach().clone()
            if K.ndim == 3:
                # Unused camera groups may have no intrinsic estimate. Their K
                # never enters verification, but the shared verifier validates N matrices.
                used = torch.zeros(len(frames), dtype=torch.bool, device=K.device)
                for i,j in matches:
                    used[i] = used[j] = True
                K[~used] = torch.eye(3, dtype=K.dtype, device=K.device)
            geometries = self.frontend.get_two_view_geometries(frames, matches, K,
                config=self.cfg.two_view, cache=FileSystemCache(self.output / 'cache', 'flexglue'))
            report['config'] = asdict(self.cfg.two_view)
            report['classification_counts'] = {
                label: sum(g.config == label for g in geometries.values())
                for label in sorted({g.config for g in geometries.values()})}
            edges = {pair: g for pair,g in geometries.items()
                     if g.config not in ('DEGENERATE','WATERMARK') and g.R is not None
                     and len(g.matches.indices) >= self.cfg.two_view.min_num_inliers}
        else:
            edges = {pair: TwoViewGeometry(m) for pair,m in matches.items() if len(m.indices)}
        report.update(output_pairs=len(edges),
                      output_matches=sum(len(g.matches.indices) for g in edges.values()))
        return edges, report

    @torch.inference_mode(False)
    def refine_saved_poses(self):
        """Refine saved global poses with configured real tracks and optional virtuals."""
        start = time.perf_counter()
        manifest = json.loads((self.output / 'summary.json').read_text())
        initial = torch.load(self.output / 'global_poses.pt',map_location='cpu',weights_only=True)
        expected = [p.resolve() for p in self.frontend.image_list]
        for paths in (manifest['image_paths'],initial['image_paths']):
            if [Path(p).resolve() for p in paths] != expected:
                raise ValueError('Saved image order differs from the workspace')
        initial = self._resolve_intrinsics(initial)
        refinement = self.cfg.refinement_config(initial)
        # Preserve the actual K consumed by verification and the seed map even
        # when resuming from a global_poses.pt written with predicted intrinsics.
        torch.save(initial, self.output / 'refinement_initial.pt')
        learned = refinement.track_source in ('vggsfm', 'sift_vggsfm')
        frames,matches = (self._load_refinement_inputs(manifest, features_only=True) if learned
                          else self._load_refinement_inputs(manifest))
        tracking = sift = None
        if learned:
            from torch_sfm.geometry.learned_tracks import track_stars
            tracked_kwargs = {}
            if refinement.track_source == 'sift_vggsfm':
                from torch_sfm.correspondence.sift import prepare_sift
                rejected = {tuple(sorted(r['pair'])) for r in initial['fusion_report']['reverse_rejected']}
                pairs = {tuple(sorted(p)) for p in manifest['accepted_pairs']} - rejected
                pairs = {p for p in pairs if bool(initial['registered'][p[0]] & initial['registered'][p[1]])}
                sift_frames,sift_matches,sift = self._stage('sift', prepare_sift,
                    frames, pairs, refinement.sift, self.output / 'cache/sift')
                tracked_kwargs = dict(snap_frames=sift_frames, base_matches=sift_matches)
            frames,matches,tracking = self._stage('tracking', track_stars, frames, manifest, initial,
                refinement.tracker, self.output / 'cache/vggsfm', self.cfg.device, **tracked_kwargs)
            if sift is not None:
                initial['sift_keypoint_counts'] = torch.tensor(tracking['base_keypoint_counts'],dtype=torch.long)
            del tracked_kwargs
        for i,frame in enumerate(frames):
            if tuple(frame.image_size) != tuple(manifest['original_image_sizes'][i]):
                raise ValueError('Feature image size differs from the local-star input')
            frame.keypoints_2d = frame.keypoints_2d.to(device=self.cfg.device,dtype=torch.float64).clone()
        rejected = {tuple(sorted(r['pair'])) for r in initial['fusion_report']['reverse_rejected']}
        accepted = {tuple(p) for p in manifest['accepted_pairs']}
        candidates = {}
        for (i,j),m in matches.items():
            pair = (min(i,j),max(i,j))
            if (pair not in accepted or pair in rejected or m.metadata.get('rejected')
                    or not bool(initial['registered'][i] & initial['registered'][j])):
                continue
            indices = m.indices.cpu()
            if indices.ndim != 2 or indices.shape[1] != 2 or indices.is_floating_point():
                raise ValueError(f'Invalid match indices for {(i,j)}')
            if (indices < 0).any() or (indices[:,0] >= len(frames[i].keypoints_2d)).any() or (indices[:,1] >= len(frames[j].keypoints_2d)).any():
                raise ValueError(f'Out-of-range match indices for {(i,j)}')
            if len(indices):
                candidates[i,j] = Matches(indices,m.scores,m.metadata)
        load_seconds = (time.perf_counter()-start - (tracking['total_seconds'] if tracking else 0.)
                        - (sift['total_seconds'] if sift else 0.))
        verification_start = time.perf_counter()
        edges, verification = self._stage('match_verification', self._verify_refinement_matches,
                                         frames, candidates, initial['intrinsics'])
        verification['seconds'] = time.perf_counter()-verification_start
        (self.output / 'match_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
        del candidates, matches
        result = refine_tracks(frames,edges,initial,refinement,self.cfg.device,
            star_manifest=manifest, star_directory=self.output)
        report = result['refinement_report']
        if tracking is not None:
            report['tracking'] = tracking
            report['timings_seconds']['tracking'] = tracking['total_seconds']
        if sift is not None:
            report['sift'] = sift
            report['timings_seconds']['sift'] = sift['total_seconds']
            result['sift_keypoint_counts'] = initial['sift_keypoint_counts']
        result['intrinsics_policy'] = initial.get('intrinsics_policy', dict(source='solv3r', fixed=False))
        report['intrinsics_policy'] = result['intrinsics_policy']
        report['match_verification'] = verification
        report['timings_seconds']['match_verification'] = verification['seconds']
        report['timings_seconds']['load_inputs'] = load_seconds
        report['timings_seconds']['total_with_loading'] = time.perf_counter()-start
        torch.save(result,self.output / 'reconstruction.pt')
        from types import SimpleNamespace
        if self.cfg.refinement.export_colmap:
            from torch_sfm.io.torch_model import export_colmap
            export_colmap(frames,SimpleNamespace(**result),self.output / 'sparse' / '0')
        from torch_sfm.visualization import visualize_reconstruction
        visualize_reconstruction(result,self.output / 'reconstruction.html',
            image_sizes=manifest['original_image_sizes'],
            colmap_path=self.output / 'sparse' / '0' if self.cfg.refinement.export_colmap else None)
        report['timings_seconds']['total_with_export'] = time.perf_counter()-start
        (self.output / 'refinement_summary.json').write_text(json.dumps(report,indent=2)+'\n')
        self._refinement_inputs = None
        return report

    @torch.inference_mode()
    def run(self):
        self.timings = {}
        self._sync()
        start = time.perf_counter()
        frontend = self.frontend
        if len(frontend.image_list) < 2:
            raise ValueError('At least two input images are needed')
        frontend.load_frames()
        pairs = self._stage('retrieval', frontend.get_retrieval_pairs)
        self._stage('features', frontend.get_keypoints, frontend.frames)
        matches = self._stage('matching', frontend.get_correspondences, frontend.frames, pairs)
        topology = self._stage('star_topology', build_stars, len(frontend.frames), matches,
                               max_neighbors=self.cfg.max_neighbors, min_matches=self.cfg.min_matches,
                               sequential_edges=frontend.sequential_edges)
        star_edges = {(min(s[0], j), max(s[0], j)) for s in topology.stars for j in s[1:]}
        sequential_edges = sorted(frontend.sequential_edges & star_edges)
        frontend_models_loaded = frontend.extractor is not None or frontend.matcher is not None
        original_sizes = [tuple(f.image_size) for f in frontend.frames]
        if self.cfg.refine and self.cfg.fuse_global:
            self._refinement_inputs = ([Frame(frame_idx=f.frame_idx,frame_path=f.frame_path,
                keypoints_2d=f.keypoints_2d.cpu(),image_size=f.image_size) for f in frontend.frames],matches)
        for frame in frontend.frames:
            frame.image = frame.features = frame.dense_features = frame.keypoints_2d = None
        frontend.extractor = frontend.matcher = None
        del matches
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / 'stars').mkdir(exist_ok=True)
        images = self._stage('image_preprocessing', self._load_images) if topology.stars else None
        records, accepted_edges = [], set()
        for ids in tqdm(topology.stars, desc='Solv3R stars + depth checks'):
            # Native Solv3R handles one group at a time. B=1 avoids padding or
            # mixing independent stars in its cross-view attention.
            batch = self._stage('image_gather', lambda: images[list(ids)].unsqueeze(0))
            prior = self._stage('prior', self.prior.predict_batch, batch, return_depth=True)
            consistency = self._stage('depth_consistency', lambda: check_depth_consistency(
                prior.camera_to_world[0].to(self.cfg.device),
                prior.intrinsics[0].to(self.cfg.device), prior.depth[0].to(self.cfg.device),
                reprojection_threshold=self.cfg.reprojection_threshold,
                min_overlap=self.cfg.min_overlap, pair_batch_size=self.cfg.pair_batch_size))
            def save_star():
                checked = {k: v.cpu() for k, v in vars(consistency).items()}
                valid_ids = [j for j, keep in zip(ids[1:], checked['valid_views'][1:].tolist()) if keep]
                accepted_edges.update((min(ids[0], j), max(ids[0], j)) for j in valid_ids)
                relative_path = f'stars/{ids[0]:06d}.pt'
                data = dict(
                    version=1, image_ids=ids, camera_to_world=prior.camera_to_world[0],
                    intrinsics=prior.intrinsics[0], image_size=prior.image_size,
                    pixel_center_offset=.5, depth=prior.depth[0], **checked)
                saved = compact_star_cache(data, self.cfg.fusion, self.cfg.refinement.virtual_tracks,
                    save_virtual_candidates=self.cfg.save_virtual_candidates)
                torch.save(saved, self.output / relative_path)
                return dict(center=ids[0], image_ids=list(ids), valid_neighbors=valid_ids,
                            center_scores=checked['center_scores'].tolist(), path=relative_path)
            records.append(self._stage('save_stars', save_star))
            del prior, consistency, batch
        del images
        # The local prior is no longer needed; leave GPU memory for global BA.
        self.prior.model = None
        self.timings['total'] = time.perf_counter() - start
        summary = dict(
            version=1, stage='local_depth_consistency', config=asdict(self.cfg),
            image_paths=[str(p) for p in frontend.image_list], original_image_sizes=original_sizes,
            frontend_stats=frontend.matching_stats, frontend_models_loaded=frontend_models_loaded,
            num_images=len(frontend.frames), num_stars=len(topology.stars),
            num_frontend_edges=len(topology.edge_scores),
            num_star_edges=len({(min(s[0], j), max(s[0], j)) for s in topology.stars for j in s[1:]}),
            num_accepted_edges=len(accepted_edges),
            frontend_components=topology.components,
            checked_components=graph_components(len(frontend.frames), accepted_edges),
            restored_edges=topology.restored_edges,
            sequential_edges=sequential_edges,
            # Local-stage union, preserved independently from global filtering.
            accepted_pairs=sorted(accepted_edges), stars=records, timings_seconds=self.timings)
        (self.output / 'summary.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
        if self.cfg.fuse_global:
            summary['global_fusion'] = self._stage('global_fusion', self.fuse_saved_stars)
            summary['stage'] = 'global_pose_fusion'
            if self.cfg.refine:
                summary['refinement'] = self._stage('refinement',self.refine_saved_poses)
                summary['stage'] = ('augmented_bundle_adjustment' if self.cfg.refinement.strategy == 'augmented'
                                    else 'flexglue_tracks_bundle_adjustment')
            self.timings['total'] = time.perf_counter() - start
            (self.output / 'summary.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
        return summary
