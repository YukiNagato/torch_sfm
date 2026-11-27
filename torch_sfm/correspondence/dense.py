"""RoMa v2 and FlexGlue measurements on a common, pixel-centered source grid."""
from functools import lru_cache

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

MATCH_SIZE = 512


def rgb_tensor(path):
    with Image.open(path) as image:
        return torch.from_numpy(np.array(image.convert('RGB'))).permute(2, 0, 1).cuda()


def roma_predictions(images, sizes, pairs, grid):
    from romav2 import RoMaV2

    model = RoMaV2(RoMaV2.Cfg(setting='fast'))
    model.bidirectional = True
    coord = (torch.arange(grid, device='cuda') + .5) * (2 / grid) - 1
    y, x = torch.meshgrid(coord, coord, indexing='ij')
    query = torch.stack((x, y), -1)[None]

    @lru_cache(maxsize=8)
    def image(index):
        return rgb_tensor(images[index].image_path)[None]

    def sample(field, xy):
        return F.grid_sample(field.movedim(-1, 1).float(), xy, align_corners=False).movedim(1, -1)

    for i, j in pairs:
        prediction = model.match(image(i), image(j))
        warps = {d: sample(prediction[f'warp_{d}'], query) for d in ('AB', 'BA')}
        for s, t, d, reverse in ((i, j, 'AB', 'BA'), (j, i, 'BA', 'AB')):
            xy = warps[d]
            score = sample(prediction[f'overlap_{d}'], query)[..., 0]
            error = (sample(warps[reverse], xy) - query).norm(dim=-1) * (MATCH_SIZE / 2)
            valid = (score >= .75) & (error < 3.) & (xy.abs() < 1).all(-1)
            yield s, t, (xy + 1) * (MATCH_SIZE / 2), valid, score


def flexglue_predictions(images, sizes, pairs, grid):
    from flexglue import DINOv3Descriptor, FlexGlue

    descriptor = DINOv3Descriptor(resize=(640, 640), describe=False).eval().cuda()
    features = []
    for image in images:
        rgb = rgb_tensor(image.image_path)[None].float() / 255
        features.append(descriptor({'image': rgb})['features'][0].cpu())
    del descriptor
    model = FlexGlue(describe=False, refine=False).eval().cuda()
    sizes = sizes.float().cuda()
    for start in range(0, len(pairs), 2):
        i, j = zip(*pairs[start:start + 2])
        data = [dict(features=torch.stack([features[k] for k in ids]).cuda(), image_size=sizes[list(ids)])
                for ids in (i, j)]
        prediction = model.match_dense(*data, coarse_stride=640 // grid, min_covisibility=.9,
                                       filter_threshold=.75, max_distance_px=3 * 640 / MATCH_SIZE)
        for d, (src, dst) in enumerate(((i, j), (j, i))):
            for row, (s, t) in enumerate(zip(src, dst)):
                xy = prediction[f'dense_p2d{d}_i{1-d}'][row] * (MATCH_SIZE / sizes[t])
                yield s, t, xy, prediction[f'dense_valid{d}'][row], prediction[f'dense_confidence{d}'][row]


def allowed_pixels(mask, xy, size):
    if mask is None:
        return np.ones(len(xy), dtype=bool)
    px = np.floor(xy * (size / MATCH_SIZE)).astype(np.int64)
    valid = ((px >= 0) & (px < size)).all(1)
    indices = px[valid]
    valid[valid] = mask[indices[:, 1], indices[:, 0]]
    return valid


@torch.inference_mode()
def dense_matches(images, sizes, edges, grid, matcher):
    from raco import RaCo
    from scipy.spatial import cKDTree

    if not edges:
        raise ValueError('Initial cameras and depths give no overlapping image pairs')
    detector = RaCo(max_num_keypoints=4096).eval().cuda()
    sizes_np = sizes.numpy()
    coord = (np.arange(grid) + .5) * MATCH_SIZE / grid
    grid_xy = np.stack(np.meshgrid(coord, coord), -1).reshape(-1, 2)
    trees, masks, source_allowed = [], [], []
    for image, size in zip(images, sizes_np, strict=True):
        mask = None
        if image.mask_path:
            with Image.open(image.mask_path) as raster:
                if raster.size != tuple(size):
                    raise ValueError(f'Mask {image.mask_path} must match the image dimensions')
                if raster.mode not in ('1', 'L', 'I', 'F') and not raster.mode.startswith('I;16'):
                    raster = raster.convert('L')
                mask = np.array(raster) != 0
        rgb = rgb_tensor(image.image_path).float() / 255
        points = detector.extract(rgb, resize=1024)['keypoints'][0].cpu().numpy()
        inside = np.isfinite(points).all(1) & ((points >= 0) & (points < size)).all(1)
        tree = cKDTree(points[inside] * (MATCH_SIZE / size))
        trees.append(tree)
        masks.append(mask)
        source_allowed.append(torch.from_numpy((tree.query(grid_xy)[0] <= 16) & allowed_pixels(mask, grid_xy, size)))
    del detector
    requested = set(edges)
    pairs = sorted({tuple(sorted(e)) for e in requested})
    predict = roma_predictions if matcher == 'romav2' else flexglue_predictions
    rows = dict(anchor=[], target=[], pixels=[], confidence=[])
    for s, t, xy, valid, score in predict(images, sizes, pairs, grid):
        if (s, t) not in requested:
            continue
        idx = (valid.flatten().cpu() & source_allowed[s]).nonzero().flatten()
        pixels = xy.reshape(-1, 2).cpu()[idx]
        keep = torch.from_numpy((trees[t].query(pixels.numpy())[0] <= 16)
                                & allowed_pixels(masks[t], pixels.numpy(), sizes_np[t]))
        idx = idx[keep]
        rows['anchor'].append(s * grid**2 + idx)
        rows['target'].append(torch.full_like(idx, t))
        rows['pixels'].append(pixels[keep])
        rows['confidence'].append(score.flatten().cpu()[idx])
    return {key: torch.cat(parts) for key, parts in rows.items()}
