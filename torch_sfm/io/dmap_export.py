"""Export a DMap depth reconstruction and its measured image observations."""
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .torch_model import export_colmap
from ..core.storage import Frame
from ..geometry.bundle_adjustment import BundleAdjustmentResult
from ..geometry.dense_ba import pixel_rays


def smooth_depth(depth):
    """Reject invalid samples and local depth steps on the 64×64 BA raster."""
    valid = depth.isfinite() & (depth > 0)
    z = depth.masked_fill(~valid, 0.)[:, None]
    high = F.max_pool2d(z, 3, 1, 1)
    low = -F.max_pool2d(-z, 3, 1, 1)
    mean = F.avg_pool2d(z, 3, 1, 1, count_include_pad=False)
    return ((low > 0) & (high - low < .10 * mean))[:, 0]


def depth_tracks(result):
    depth = result['depth']
    poses, sizes, k = result['camera_to_world'], result['image_sizes'], result['intrinsics']
    centers = poses[:, :3, 3]
    height, width = depth.shape[-2:]
    support = result['depth_support']
    anchor, target, xy = support['anchor'], support['target'], support['xy']
    keep = smooth_depth(depth).flatten()[anchor]
    target, xy = target[keep], xy[keep]
    cells, track = anchor[keep].unique(return_inverse=True)
    source = cells // (height * width)
    uv = torch.stack((cells % width, cells // width % height), -1)
    uv = (uv.to(depth) + .5) * (sizes[source] / depth.new_tensor([width, height]))
    ray = pixel_rays(uv, k[source]) * depth.flatten()[cells, None]
    xyz = (poses[source, :3, :3] @ ray[:, :, None]).squeeze(-1) + centers[source]

    # Even a small residual can leave depth unconstrained for a very short baseline.
    to_source = F.normalize(centers[source] - xyz, dim=-1)
    to_target = F.normalize(centers[target] - xyz[track], dim=-1)
    parallax = (to_source[track] * to_target).sum(-1).abs() <= math.cos(math.radians(1.))
    good = torch.zeros(len(cells), dtype=torch.bool)
    good[track[parallax]] = True
    retain = good[track]
    target, xy = target[retain], xy[retain]
    track = good.long().cumsum(0)[track[retain]] - 1
    source, uv, xyz = source[good], uv[good], xyz[good]

    # COLMAP stores the actual target measurement and the error in native pixels.
    local = (poses[target, :3, :3].mT @ (xyz[track] - centers[target])[:, :, None]).squeeze(-1)
    projected = k[target, :2] * (local[:, :2] / local[:, 2:]) + k[target, 2:]
    errors = torch.cat((depth.new_zeros(len(xyz)), (projected - xy).norm(dim=-1)))
    point_id = torch.cat((torch.arange(len(xyz)), track))
    view = torch.cat((source, target))
    order = view.argsort(stable=True)
    view = view[order]
    counts = view.bincount(minlength=len(poses))
    starts = counts.cumsum(0) - counts
    local_id = torch.arange(len(view)) - starts[view]
    observations = torch.stack((point_id[order], view, local_id), -1)
    keypoints = torch.cat((uv, xy))[order].split(counts.tolist())
    return xyz, observations, errors[order], keypoints


def write_ply(path, points):
    vertices = np.empty(len(points), dtype=[('xyz', '<f4', 3), ('rgb', 'u1', 3)])
    for i, p in enumerate(points):
        vertices[i] = p.xyz, p.rgb
    header = (f'ply\nformat binary_little_endian 1.0\nelement vertex {len(vertices)}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with path.open('wb') as file:
        file.write(header.encode('ascii'))
        vertices.tofile(file)


def export_dmap(result, output):
    """Write sparse/0, a colored PLY and image links from a CPU DMap result."""
    if 'depth_support' not in result:
        raise ValueError('This reconstruction predates saved BA depth support. Rerun DMap to export trusted points.')
    output = Path(output)
    paths = [Path(p).resolve(strict=True) for p in result['image_paths']]
    image_root = (output / 'images').resolve()
    if not all(path.is_relative_to(image_root) for path in paths):
        image_root = Path(os.path.commonpath([p.parent for p in paths]))
    xyz, observations, errors, keypoints = depth_tracks(result)
    frames = []
    for index, (path, cam_idx, xy) in enumerate(zip(paths, result['cam_idx'], keypoints, strict=True)):
        link = output / 'images' / path.relative_to(image_root)
        link.parent.mkdir(exist_ok=True, parents=True)
        if link.exists() or link.is_symlink():
            if link.resolve() != path:
                raise FileExistsError(f'Exported image name is already in use: {link}')
        else:
            link.symlink_to(os.path.relpath(path, link.parent))
        frames.append(Frame(frame_idx=index, frame_path=path, cam_idx=cam_idx, keypoints_2d=xy))
    model = BundleAdjustmentResult(result['rotations'], result['centers'], xyz, observations,
                                   result['registered'], result['intrinsics'], errors)
    recon = export_colmap(frames, model, output / 'sparse' / '0', image_root)
    write_ply(output / 'points3D.ply', recon.points3D.values())
    return recon


def main(reconstruction: Path, output: Path):
    result = torch.load(reconstruction, weights_only=True, map_location='cpu')
    export_dmap(result, output)


if __name__ == '__main__':
    import tyro
    tyro.cli(main)
