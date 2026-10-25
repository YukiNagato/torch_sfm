from dataclasses import dataclass, field, asdict
from torch_sfm.core import Frame
from torch_sfm.utils import Timer
import numpy as np
import torch
from pathlib import Path
from torch_sfm.retrieval import get_retrieval, RetrievalResult
from torch_sfm.correspondence import get_extractor_and_matcher, Matches
from torch_sfm.geometry import (
    TwoViewConfig, TwoViewGeometry, RotationConfig, PositionConfig,
    camera_matrix, verify_two_view, rotation_averaging, translation_averaging,
    BundleAdjustmentConfig, bundle_adjustment,
)
from PIL import Image
from typing import List
from tqdm import tqdm


timer = Timer()


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
    retrieval_top_k: int = 10
    retrieval_threshold: float = 0.5

    cache_frame_images: bool = True
    use_cache: bool = True
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
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.image_txt_path = self.workspace / 'images.txt'
        self.intrinsic_path = self.workspace / 'intrinsics.txt'
        with open(self.image_txt_path) as f:
            self.image_list = [Path(line.strip()) for line in f.readlines()]
        self.frames = []
        self.extractor = None
        self.matcher = None
    
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
        retrieval_cache_path = self.cache_root / 'retrieval.txt'
        if retrieval_cache_path.exists() and self.cfg.use_cache and not self.cfg.update_cache_for_retrieval:
            retrieval_pairs = np.loadtxt(retrieval_cache_path, dtype=np.int32).reshape(-1, 2)
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
                np.savetxt(retrieval_cache_path, retrieval_pairs, fmt='%d')

        return retrieval_pairs
    
    @timer
    @torch.inference_mode()
    def get_keypoints(self, frames: List[Frame]):
        """Fill each frame's keypoints_2d and features, and return the frames."""
        keypoints_cache_root = self.cache_root / 'keypoints' / self.cfg.correspondence_method
        if self.cfg.use_cache:
            keypoints_cache_root.mkdir(parents=True, exist_ok=True)

        for frame in tqdm(frames, desc='Extract keypoints'):
            keypoints_cache_path = keypoints_cache_root / f'{frame.frame_idx}.pt'
            if keypoints_cache_path.exists() and self.cfg.use_cache and not self.cfg.update_cache_for_keypoints:
                device = frame.image.device if frame.image is not None else 'cpu'
                data = torch.load(keypoints_cache_path, map_location=device, weights_only=True)
                frame.keypoints_2d = data['keypoints_2d']
                frame.features = data['features']
            else:
                if self.extractor is None:
                    self.extractor, self.matcher = get_extractor_and_matcher(self.cfg.correspondence_method)
                image = frame.image
                frame.image = load_frame_image(frame)
                self.extractor.extract(frame)
                if not self.cfg.cache_frame_images:
                    frame.image = image
                if self.cfg.use_cache:
                    torch.save({
                        'keypoints_2d': frame.keypoints_2d.cpu(),
                        'features': frame.features.cpu(),
                    }, keypoints_cache_path)

        return frames

    @timer
    @torch.inference_mode()
    def get_correspondences(self, frames: List[Frame], retrieval_pairs: np.ndarray):
        """Return {(i, j): Matches} on CPU; pairs index the supplied frames."""
        correspondence_cache_root = self.cache_root / 'correspondences' / self.cfg.correspondence_method
        if self.cfg.use_cache:
            correspondence_cache_root.mkdir(parents=True, exist_ok=True)

        correspondences = {}
        for i, j in tqdm(retrieval_pairs, desc='Match pairs'):
            i, j = int(i), int(j)
            frame0, frame1 = frames[i], frames[j]
            correspondence_cache_path = correspondence_cache_root / f'{frame0.frame_idx}_{frame1.frame_idx}.pt'
            if (correspondence_cache_path.exists() and self.cfg.use_cache
                    and not self.cfg.update_cache_for_correspondence
                    and not self.cfg.update_cache_for_keypoints):
                data = torch.load(correspondence_cache_path, map_location='cpu', weights_only=True)
                matches = Matches(data['indices'], data['scores'])
            else:
                if self.matcher is None:
                    self.extractor, self.matcher = get_extractor_and_matcher(self.cfg.correspondence_method)
                image0, image1 = frame0.image, frame1.image
                frame0.image = load_frame_image(frame0)
                frame1.image = load_frame_image(frame1)
                matches = self.matcher.match(frame0, frame1)
                matches = Matches(matches.indices.cpu(), matches.scores.cpu())
                if not self.cfg.cache_frame_images:
                    frame0.image, frame1.image = image0, image1
                if self.cfg.use_cache:
                    torch.save({
                        'indices': matches.indices,
                        'scores': matches.scores,
                    }, correspondence_cache_path)
            correspondences[i, j] = matches

        return correspondences

    @timer
    def get_two_view_geometries(self, frames: List[Frame], correspondences: dict, intrinsic: torch.Tensor):
        device = torch.device(self.cfg.geometry_device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        matrices = camera_matrix(intrinsic.to(device=device, dtype=torch.float64))
        if matrices.ndim == 2:
            matrices = matrices.expand(len(frames), 3, 3)
        if matrices.shape != (len(frames), 3, 3):
            raise ValueError('Provide shared pinhole intrinsics or one camera matrix per frame')
        cache_root = self.cache_root / 'two_view' / self.cfg.correspondence_method
        if self.cfg.use_cache:
            cache_root.mkdir(parents=True, exist_ok=True)
        sizes = []
        for frame in frames:
            if frame.image is not None:
                sizes.append(tuple(frame.image.shape[-2:]))
            else:
                with Image.open(frame.frame_path) as image:
                    sizes.append((image.height, image.width))

        geometries = {}
        refresh = (self.cfg.update_cache_for_two_view or self.cfg.update_cache_for_keypoints
                   or self.cfg.update_cache_for_correspondence)
        options = asdict(self.cfg.two_view)
        for (i, j), matches in tqdm(correspondences.items(), desc='Verify pairs'):
            frame0, frame1 = frames[i], frames[j]
            cache_path = cache_root / f'{frame0.frame_idx}_{frame1.frame_idx}.pt'
            # Geometry also depends on calibration, thresholds and matched pixels.
            inputs = {
                'intrinsics': matrices[[i, j]].cpu(),
                'indices': matches.indices.cpu(),
                'scores': matches.scores.cpu(),
                'points0': frame0.keypoints_2d[matches.indices[:, 0].to(frame0.keypoints_2d.device)].cpu(),
                'points1': frame1.keypoints_2d[matches.indices[:, 1].to(frame1.keypoints_2d.device)].cpu(),
            }
            data = None
            if cache_path.exists() and self.cfg.use_cache and not refresh:
                cached = torch.load(cache_path, map_location='cpu', weights_only=True)
                if (cached.get('version') == 2 and cached['options'] == options
                        and cached['sizes'] == (sizes[i], sizes[j])
                        and all(torch.equal(value, cached['inputs'][key]) for key, value in inputs.items())):
                    data = cached['geometry']
            if data is not None:
                data = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                        for key, value in data.items()}
                data['matches'] = Matches(data.pop('indices'), data.pop('scores'))
                geometry = TwoViewGeometry(**data)
            else:
                geometry = verify_two_view(frame0.keypoints_2d.to(device), frame1.keypoints_2d.to(device),
                    matches, matrices[i], matrices[j], self.cfg.two_view, (sizes[i], sizes[j]))
                if self.cfg.use_cache:
                    data = {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                            for key, value in vars(geometry).items() if key != 'matches'}
                    data['indices'] = geometry.matches.indices.cpu()
                    data['scores'] = geometry.matches.scores.cpu()
                    torch.save({'version': 2, 'options': options, 'sizes': (sizes[i], sizes[j]),
                                'inputs': inputs, 'geometry': data}, cache_path)
            geometries[i, j] = geometry
        return geometries

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
    def visualize(self, output_path: Path | None = None, *, max_points=100000, camera_scale=None):
        from torch_sfm.visualization import visualize_reconstruction
        source = self.result if hasattr(self, 'result') else self.workspace / 'reconstruction.pt'
        image_paths = [frame.frame_path for frame in self.frames] or self.image_list
        return visualize_reconstruction(source, output_path or self.workspace / 'reconstruction.html',
            image_paths=image_paths, colmap_path=self.workspace / 'sparse' / '0',
            max_points=max_points, camera_scale=camera_scale)

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
        intrinsic = self.get_intrinsics()
        self.load_frames()
        retrieval_pairs = self.get_retrieval_pairs()
        self.get_keypoints(self.frames)
        correspondences = self.get_correspondences(self.frames, retrieval_pairs)
        geometries = self.get_two_view_geometries(self.frames, correspondences, intrinsic)
        rotations = self.get_rotations(geometries)
        positions = self.get_positions(rotations.edges, rotations.rotations, intrinsic)
        initial = {
            'rotations': rotations.rotations.cpu(), 'centers': positions.centers.cpu(),
            'translations': -(rotations.rotations @ positions.centers[..., None]).squeeze(-1).cpu(),
            'registered': positions.registered.cpu(), 'rotation_registered': rotations.registered.cpu(),
            'points3d': positions.points3d.cpu(), 'observations': positions.observations.cpu(),
            'intrinsics': intrinsic.cpu(), 'image_paths': [str(frame.frame_path) for frame in self.frames],
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
        retrieval_top_k=5
    )
    data_path = Path('/root/autodl-tmp/torch_sfm/data/sfm')
    pipeline = GlomapPipeline(cfg, data_path)
    result = pipeline.run()
    print(f'{int(result.registered.sum())}/{len(result.registered)} registered images, '
          f'{len(result.points3d)} points')
    timer.report()



        
        
    

