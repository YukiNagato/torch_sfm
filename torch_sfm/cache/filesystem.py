from pathlib import Path

import numpy as np
import torch

from .base import Cache


class FileSystemCache(Cache):
    """The original .txt/.pt layout, including existing cache files."""

    def __init__(self, root: Path, method='loma'):
        self.path = Path(root)
        self.method = method

    def _path(self, kind, key):
        if kind == 'retrieval':
            return self.path / 'retrieval.txt'
        if kind == 'keypoints':
            name = str(int(key))
        elif kind in ('correspondences', 'two_view', 'two_view_uncalibrated'):
            name = f'{int(key[0])}_{int(key[1])}'
        else:
            raise ValueError(f'Unknown cache kind: {kind}')
        return self.path / kind / self.method / f'{name}.pt'

    def load(self, kind, key=None):
        path = self._path(kind, key)
        if not path.exists():
            return None
        if kind == 'retrieval':
            return (np.loadtxt(path, dtype=np.int32).reshape(-1, 2) if path.stat().st_size
                    else np.empty((0, 2), dtype=np.int32))
        return torch.load(path, map_location='cpu', weights_only=True)

    def invalidate_two_view(self, keys):
        for key in keys:
            for pair in (key, key[::-1]):
                self._path('two_view', pair).unlink(missing_ok=True)

    def save(self, kind, data, key=None):
        path = self._path(kind, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Publish only complete payloads.
        temporary = path.with_suffix(path.suffix + '.tmp')
        if kind == 'retrieval':
            np.savetxt(temporary, data, fmt='%d')
        else:
            torch.save(data, temporary)
        temporary.replace(path)
        if kind == 'keypoints':
            for dependent in ('correspondences', 'two_view', 'two_view_uncalibrated'):
                for candidate in (self.path / dependent / self.method).glob('*.pt'):
                    if int(key) in tuple(map(int, candidate.stem.split('_'))):
                        candidate.unlink()
        elif kind == 'correspondences':
            for pair in (key, key[::-1]):
                self._path('two_view', pair).unlink(missing_ok=True)
                self._path('two_view_uncalibrated', pair).unlink(missing_ok=True)
