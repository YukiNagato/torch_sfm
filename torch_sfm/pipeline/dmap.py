"""Refine initial camera poses and depth maps from a TorchSfM dataset manifest."""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from torch_sfm.geometry.dense_ba import refine_poses
from torch_sfm.geometry.view_graph import frustum_overlap, select_pairs
from torch_sfm.io.manifest import DatasetManifest
from torch_sfm.prior import vggt_omega


def read_depth(path, grid):
    if path.suffix.lower() == '.npy':
        depth = torch.from_numpy(np.load(path, allow_pickle=False))
    elif path.suffix.lower() == '.pt':
        depth = torch.load(path, weights_only=True, map_location='cpu')
    else:
        with Image.open(path) as image:
            depth = torch.from_numpy(np.array(image))
    if not isinstance(depth, torch.Tensor) or depth.ndim != 2:
        raise ValueError(f'{path} must contain a single H×W camera-Z depth map')
    depth = depth.double()
    valid = depth.isfinite() & (depth > 0)
    # Exclude invalid samples from bilinear interpolation at grid centers.
    values = torch.stack((depth.masked_fill(~valid, 0.), valid.to(depth)))[None]
    values, weights = F.interpolate(values, (grid, grid), mode='bilinear', align_corners=False)[0]
    return torch.where(weights > 0, values / weights.clamp_min(1e-12), torch.nan)


def align_prediction(poses, depths, known_poses, known_depths):
    """Map the prediction into the supplied camera frame and length units."""
    indices = list(known_poses)
    rotation = torch.eye(3, dtype=poses.dtype)
    translation = poses.new_zeros(3)
    scale = None
    if indices:
        supplied = torch.stack(list(known_poses.values()))
        rotations = supplied[:, :3, :3] @ poses[indices, :3, :3].mT
        u, _, vt = torch.linalg.svd(rotations.mean(0))
        sign = torch.linalg.det(u @ vt).sign()
        rotation = (u * torch.stack((sign.new_ones(()), sign.new_ones(()), sign))) @ vt
        before, after = torch.pdist(poses[indices, :3, 3]), torch.pdist(supplied[:, :3, 3])
        valid = (before > 1e-8) & (after > 1e-8)
        if valid.any():
            scale = (after[valid] / before[valid]).median()
    if scale is None:
        ratios = []
        for i, observed in known_depths.items():
            ratio = observed / depths[i]
            valid = ratio.isfinite() & (ratio > 0) & (observed > 0)
            if valid.any():
                ratios.append(ratio[valid].median())
        scale = torch.stack(ratios).median() if ratios else 1.
    if indices:
        translation = (supplied[:, :3, 3] - scale * (poses[indices, :3, 3] @ rotation.T)).mean(0)
    aligned = poses.clone()
    aligned[:, :3, :3] = rotation @ poses[:, :3, :3]
    aligned[:, :3, 3] = scale * (poses[:, :3, 3] @ rotation.T) + translation
    return aligned, depths * scale


def initialize(dataset, grid, checkpoint):
    images = dataset.images
    sizes = []
    for image in images:
        camera = dataset.cameras[image.cam_idx]
        with Image.open(image.image_path) as frame:
            if frame.size != (camera.width, camera.height):
                raise ValueError(f'{image.image_path} dimensions disagree with camera {image.cam_idx}')
            sizes.append(frame.size)
    sizes = torch.tensor(sizes, dtype=torch.float64)
    known_poses = {i: torch.tensor(im.pose, dtype=torch.float64) for i, im in enumerate(images) if im.pose is not None}
    known_depths = {i: read_depth(im.depth_path, grid) for i, im in enumerate(images) if im.depth_path is not None}
    groups = {}
    for i, im in enumerate(images):
        groups.setdefault(im.cam_idx, []).append(i)
    missing_geometry = min(len(known_poses), len(known_depths)) < len(images)
    if missing_geometry or any(dataset.cameras[key].params is None for key in groups):
        poses, depth, predicted_k = vggt_omega.predict(images, sizes, grid, checkpoint)
    if missing_geometry:
        poses, depth = align_prediction(poses, depth, known_poses, known_depths)
    else:
        poses = torch.empty((len(images), 4, 4), dtype=torch.float64)
        depth = torch.empty((len(images), grid, grid), dtype=torch.float64)
    intrinsics = torch.empty((len(images), 4), dtype=torch.float64)
    for key, indices in groups.items():
        params = dataset.cameras[key].params
        intrinsics[indices] = predicted_k[indices].median(0).values if params is None else intrinsics.new_tensor(params)
    for i, supplied in known_poses.items():
        poses[i] = supplied
    for i, supplied in known_depths.items():
        depth[i] = supplied
    return poses, depth, intrinsics, sizes


@dataclass
class DMapConfig:
    correspondence_method: Literal['romav2', 'flexglue'] = 'romav2'
    neighbors: int = 8
    omega_checkpoint: Path | None = None
    export_colmap: bool = True

    def __post_init__(self):
        if self.correspondence_method not in ('romav2', 'flexglue'):
            raise ValueError(f'Unknown dense matcher {self.correspondence_method!r}')
        if self.neighbors < 1:
            raise ValueError('The reference-view budget must be positive')


class DMapPipeline:
    def __init__(self, cfg: DMapConfig, workspace: Path, *, manifest: Path | None = None):
        self.cfg = cfg
        self.workspace = Path(workspace).expanduser().resolve()
        path = Path(manifest).expanduser() if manifest is not None else Path('dataset.json')
        self.dataset = DatasetManifest.read(self.workspace / path)
        if len(self.dataset.images) < 2:
            raise ValueError('Pose refinement requires at least two images')
        self.reconstruction_path = self.workspace / 'reconstruction.pt'
        self.colmap_dir = self.workspace

    def get_view_pairs(self, poses, depths, intrinsics, sizes):
        grid = depths.shape[-1]
        grid_k = intrinsics * (grid / sizes).repeat(1, 2)
        overlap = frustum_overlap(poses.numpy(), depths.numpy(), grid_k.numpy())
        return select_pairs(overlap, self.cfg.neighbors, seed=0)

    def get_correspondences(self, sizes, pairs, grid):
        from torch_sfm.correspondence.dense import dense_matches
        return dense_matches(self.dataset.images, sizes, pairs, grid, self.cfg.correspondence_method)

    def run(self):
        grid = 64
        poses, depths, intrinsics, sizes = initialize(self.dataset, grid, self.cfg.omega_checkpoint)
        pairs = self.get_view_pairs(poses, depths, intrinsics, sizes)
        matches = self.get_correspondences(sizes, pairs, grid)
        result = refine_poses(poses, depths, intrinsics, sizes, matches)
        images = self.dataset.images
        c2w = result['camera_to_world']
        result.update(image_paths=[str(im.image_path) for im in images], rotations=c2w[:, :3, :3].mT,
                      centers=c2w[:, :3, 3], registered=torch.ones(len(c2w), dtype=torch.bool),
                      intrinsics=intrinsics, image_sizes=sizes, cam_idx=[im.cam_idx for im in images])
        self.reconstruction_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, self.reconstruction_path)
        if self.cfg.export_colmap:
            from torch_sfm.io.dmap_export import export_dmap
            export_dmap(result, self.colmap_dir)
        self.result = result
        return result


def run(manifest: Path, output: Path, matcher: Literal['romav2', 'flexglue'] = 'romav2',
        omega_checkpoint: Path | None = None, neighbors: int = 8, colmap_dir: Path | None = None):
    """File-oriented Python entry point for existing experiment scripts."""
    manifest = Path(manifest).expanduser().resolve()
    config = DMapConfig(correspondence_method=matcher, neighbors=neighbors, omega_checkpoint=omega_checkpoint,
                        export_colmap=colmap_dir is not None)
    pipeline = DMapPipeline(config, manifest.parent, manifest=manifest)
    pipeline.reconstruction_path = Path(output)
    if colmap_dir is not None:
        pipeline.colmap_dir = Path(colmap_dir)
    return pipeline.run()


def main(workspace: Path, config: DMapConfig, manifest: Path | None = None):
    """Refine camera poses and depths from a workspace on CUDA.

    Args:
        workspace: Directory for dataset.json, reconstruction.pt and the COLMAP model.
        config: Dense matching, initialization and export settings.
        manifest: Optional input manifest, resolved relative to the workspace.
    """
    DMapPipeline(config, workspace, manifest=manifest).run()


if __name__ == '__main__':
    import tyro
    tyro.cli(main)
