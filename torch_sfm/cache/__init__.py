from pathlib import Path

from .base import Cache
from .filesystem import FileSystemCache
from .colmap import ColmapCache


def create_cache(backend, workspace, method='loma', database_path=None):
    workspace = Path(workspace)
    if backend == 'filesystem':
        return FileSystemCache(workspace / 'cache', method)
    if backend == 'colmap':
        path = Path(database_path) if database_path is not None else Path('database.db')
        return ColmapCache(path if path.is_absolute() else workspace / path,
                           method, workspace / 'images')
    raise ValueError(f'Unknown cache backend: {backend}')


def copy_cache(source, target, frames, *, intrinsics=None, two_view_pairs=None):
    """Copy available stages only. Prepare the target with initial intrinsics first."""
    counts = dict(retrieval=0, keypoints=0, correspondences=0, two_view=0, two_view_uncalibrated=0)
    if intrinsics is not None:
        import torch
        from ..geometry.two_view import camera_matrix
        matrices = camera_matrix(intrinsics.detach().cpu().double())
    pairs = source.load('retrieval')
    for frame in frames:
        data = source.load('keypoints', frame.frame_idx)
        if data is not None:
            target.save('keypoints', data, frame.frame_idx)
            counts['keypoints'] += 1
    if pairs is not None:
        target.save('retrieval', pairs)
        counts['retrieval'] = len(pairs)
        for i, j in pairs:
            key = (frames[int(i)].frame_idx, frames[int(j)].frame_idx)
            for kind in ('correspondences', 'two_view_uncalibrated', 'two_view'):
                if kind == 'two_view' and two_view_pairs is not None and key not in two_view_pairs:
                    continue
                data = source.load(kind, key)
                if data is not None:
                    if kind == 'two_view' and intrinsics is not None:
                        expected = matrices.expand(2,3,3) if matrices.ndim == 2 else matrices[[int(i),int(j)]]
                        actual = data.get('inputs', {}).get('intrinsics')
                        if actual is None or not torch.equal(actual.cpu().double(), expected):
                            continue  # Never export poses computed with a different K.
                    target.save(kind, data, key)
                    counts[kind] += 1
    return counts


__all__ = ['Cache', 'FileSystemCache', 'ColmapCache', 'create_cache', 'copy_cache']
