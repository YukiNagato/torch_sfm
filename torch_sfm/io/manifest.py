"""JSON image inputs and shared pinhole camera definitions."""

from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class InputCamera:
    model: str
    width: int
    height: int
    params: tuple[float, ...] | None


@dataclass(frozen=True)
class InputImage:
    image_path: Path
    cam_idx: int
    mask_path: Path | None = None
    depth_path: Path | None = None


@dataclass(frozen=True)
class DatasetManifest:
    images: tuple[InputImage, ...]
    cameras: dict[int, InputCamera]

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
            if not key.isascii() or not key.isdecimal() or str(int(key)) != key:
                raise ValueError(f'Invalid camera ID {key!r}. Use a nonnegative integer key.')
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
            cameras[int(key)] = InputCamera(entry['model'], width, height, params)

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
            if type(cam_idx) is not int or cam_idx not in cameras:
                raise ValueError(f'Image {index} references an unknown cam_idx: {cam_idx!r}')
            image_path = resolve(entry.get('image_path'), 'image_path', index)
            optional = {field: resolve(entry[field], field, index)
                        for field in ('mask_path', 'depth_path') if entry.get(field) is not None}
            images.append(InputImage(image_path, cam_idx, **optional))
        return cls(tuple(images), cameras)
