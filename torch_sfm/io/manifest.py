"""JSON image inputs and shared pinhole camera definitions."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class InputCamera:
    model: str
    width: int
    height: int
    params: tuple[float, ...] | None


@dataclass(frozen=True)
class InputImage:
    image_path: Path
    cam_idx: str
    mask_path: Path | None = None
    depth_path: Path | None = None
    pose: tuple[tuple[float, ...], ...] | None = None  # OpenCV camera to world


@dataclass(frozen=True)
class DatasetManifest:
    images: tuple[InputImage, ...]
    cameras: dict[str, InputCamera]

    @classmethod
    def read(cls, path: Path) -> 'DatasetManifest':
        path = Path(path).expanduser().resolve()
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or not isinstance(data.get('cameras'), dict):
            raise ValueError(f'{path} must define a cameras object')
        if not isinstance(data.get('images'), list) or not data['images']:
            raise ValueError(f'{path} must define a nonempty images array')

        cameras = {}
        for key, entry in data['cameras'].items():
            if not key.strip():
                raise ValueError('Camera IDs must be nonempty strings')
            if not isinstance(entry, dict) or entry.get('model') != 'PINHOLE':
                raise ValueError(f'Camera {key} must use the PINHOLE model')
            width, height = entry.get('width'), entry.get('height')
            if any(type(v) is not int or v <= 0 for v in (width, height)):
                raise ValueError(f'Camera {key} needs positive integer width and height')
            if 'params' not in entry:
                raise ValueError(f'Camera {key} needs params (use null to estimate intrinsics)')
            params = entry['params']
            if params is not None:
                if (not isinstance(params, list) or len(params) != 4
                        or any(type(v) not in (int, float) or not math.isfinite(v) for v in params)
                        or min(params[:2]) <= 0):
                    raise ValueError(f'Camera {key} needs finite [fx, fy, cx, cy] with positive focal lengths')
                params = tuple(map(float, params))
            cameras[key] = InputCamera(entry['model'], width, height, params)

        def resolve(value, field, index):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'Image {index} needs a nonempty {field} string')
            result = (path.parent / Path(value).expanduser()).resolve()
            if not result.is_file():
                raise FileNotFoundError(f'Image {index} {field} does not exist: {result}')
            return result

        images = []
        for index, entry in enumerate(data['images']):
            if not isinstance(entry, dict):
                raise ValueError(f'Image {index} must be an object')
            cam_idx = entry.get('cam_idx')
            if not isinstance(cam_idx, str):
                raise ValueError(f'Image {index} cam_idx must be a string')
            if cam_idx not in cameras:
                raise ValueError(f'Image {index} references an unknown cam_idx: {cam_idx!r}')
            image_path = resolve(entry.get('image_path'), 'image_path', index)
            optional = {field: resolve(entry[field], field, index)
                        for field in ('mask_path', 'depth_path') if entry.get(field) is not None}
            if entry.get('pose') is not None:
                pose = np.asarray(entry['pose'])
                if (pose.shape != (4, 4) or pose.dtype.kind not in 'fiu' or not np.isfinite(pose).all()
                        or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
                        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=2e-4)
                        or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=2e-4)):
                    raise ValueError(f'Image {index} pose must be a finite rigid 4x4 camera-to-world matrix')
                optional['pose'] = tuple(tuple(map(float, row)) for row in pose)
            images.append(InputImage(image_path, cam_idx, **optional))
        return cls(tuple(images), cameras)
