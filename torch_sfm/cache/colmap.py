from functools import partial
from io import BytesIO
from pathlib import Path
import math
import hashlib

import numpy as np
from PIL import Image
import torch

from .base import Cache
from ..io import (
    Database, CameraModel, Descriptors, FeatureType, GeometryConfig,
    TwoViewGeometry,
)
from ..io.database import MAX_IMAGE_ID, image_ids_to_pair_id
from ..io.torch_model import colmap_camera_ids, relative_image_name
from ..geometry.two_view import camera_matrix
from ..geometry.averaging import so3_log


class ColmapCache(Cache):
    """COLMAP core tables plus a torch_sfm_cache table for lossless extras."""

    def __init__(self, path: Path, method='loma', image_root: Path | None = None):
        self.path = Path(path)
        self.method = method
        self.image_root = None if image_root is None else Path(image_root).resolve()
        self._db = None

    @property
    def db(self):
        if self._db is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._db = Database(self.path)
            self._db.connection.execute('''CREATE TABLE IF NOT EXISTS torch_sfm_cache (
                kind TEXT NOT NULL, key TEXT NOT NULL, data BLOB NOT NULL,
                image_id1 INTEGER, image_id2 INTEGER, PRIMARY KEY(kind, key))''')
            self._db.commit()
        return self._db

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db = None

    def _key(self, kind, key):
        if kind in ('retrieval', 'metadata'):
            return ''
        if kind in ('keypoints', 'dense_features'):
            return str(int(key))
        if kind in ('correspondences', 'two_view', 'two_view_uncalibrated'):
            return '_'.join(map(str, sorted(map(int, key))))
        raise ValueError(f'Unknown cache kind: {kind}')

    def _extra(self, kind, key=None):
        row = self.db.connection.execute('SELECT data FROM torch_sfm_cache WHERE kind=? AND key=?',
                                         (kind, self._key(kind, key))).fetchone()
        return None if row is None else torch.load(BytesIO(row[0]), map_location='cpu', weights_only=True)

    def _write_extra(self, kind, data, key=None):
        if kind in ('keypoints', 'correspondences', 'two_view'):
            data = dict(data, signature=self._signature(kind, key))
        buffer = BytesIO()
        torch.save(data, buffer)
        image_id1 = image_id2 = None
        if kind in ('keypoints', 'dense_features'):
            image_id1 = int(key) + 1
        elif kind in ('correspondences', 'two_view', 'two_view_uncalibrated'):
            image_id1, image_id2 = sorted(int(index) + 1 for index in key)
        self.db.connection.execute('''INSERT INTO torch_sfm_cache VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(kind, key) DO UPDATE SET data=excluded.data,
            image_id1=excluded.image_id1, image_id2=excluded.image_id2''',
            (kind, self._key(kind, key), buffer.getvalue(), image_id1, image_id2))

    def _signature(self, kind, key):
        """Do not reuse Torch metadata after external tools edit the core tables."""
        if kind == 'keypoints':
            tables, column, identifier = ('keypoints', 'descriptors'), 'image_id', int(key) + 1
        else:
            tables = ('matches',) if kind == 'correspondences' else ('two_view_geometries',)
            column = 'pair_id'
            identifier = image_ids_to_pair_id(*(int(index) + 1 for index in key))
        digest = hashlib.sha256()
        for table in tables:
            names = 'rows, cols, data'
            if table == 'two_view_geometries':
                names += ', config, F, E, H, qvec, tvec'
            row = self.db.connection.execute(f'SELECT {names} FROM {table} WHERE {column}=?',
                                              (identifier,)).fetchone()
            for value in (() if row is None else row):
                value = value if isinstance(value, bytes) else repr(value).encode()
                digest.update(len(value).to_bytes(8, 'little'))
                digest.update(value)
        return digest.hexdigest()

    def prepare(self, frames, intrinsics, *, preserve_geometry_cache=False, prior_focal_length=False):
        matrices = camera_matrix(intrinsics.detach().cpu().double())
        shared = matrices.ndim == 2
        if not shared and matrices.shape != (len(frames), 3, 3):
            raise ValueError('Provide shared intrinsics or one camera matrix per frame')
        camera_ids = colmap_camera_ids(frames, shared)
        images, cameras = [], {}
        for index, frame in enumerate(frames):
            if frame.frame_idx != index:
                raise ValueError('COLMAP cache expects frame_idx to equal the frame list index')
            if frame.image is not None:
                height, width = frame.image.shape[-2:]
            else:
                with Image.open(frame.frame_path) as image:
                    width, height = image.size
            path = Path(frame.frame_path) if frame.frame_path is not None else Path(f'{index}.png')
            name = relative_image_name(path, self.image_root)
            camera_id = camera_ids[index]
            matrix = matrices if shared else matrices[index]
            params = matrix[[0, 1, 0, 1], [0, 1, 2, 2]].numpy()
            camera = (width, height, params)
            if camera_id in cameras:
                if cameras[camera_id][:2] != camera[:2]:
                    raise ValueError('Images sharing intrinsics must have the same resolution')
                if not np.array_equal(cameras[camera_id][2], params):
                    raise ValueError('Images sharing a camera must have the same intrinsic parameters')
            cameras[camera_id] = camera
            images.append((index + 1, name, camera_id))
        names = [image[1] for image in images]
        if len(set(names)) != len(names):
            raise ValueError('Duplicate COLMAP image names')
        metadata = {'method': self.method, 'images': images}
        previous = self._extra('metadata')
        if previous is not None and previous != metadata:
            raise ValueError('Database cache belongs to another image list or feature method; use a different database_path')
        with self.db.connection:
            calibration_changed = False
            for camera_id, (width, height, params) in cameras.items():
                try:
                    old = self.db.read_camera(camera_id)
                except KeyError:
                    self.db.add_camera(CameraModel.PINHOLE, width, height, params, camera_id=camera_id,
                                       prior_focal_length=prior_focal_length)
                else:
                    if old.model != CameraModel.PINHOLE or (old.width, old.height) != (width, height):
                        raise ValueError('Database camera model or image dimensions do not match the input')
                    if not np.array_equal(old.params, params):
                        calibration_changed = True
                        self.db.connection.execute('UPDATE cameras SET params=? WHERE camera_id=?',
                                                   (params.astype('<f8').tobytes(), camera_id))
                    if old.prior_focal_length != prior_focal_length:
                        self.db.connection.execute('UPDATE cameras SET prior_focal_length=? WHERE camera_id=?',
                                                   (int(prior_focal_length), camera_id))
            for image_id, name, camera_id in images:
                try:
                    old = self.db.read_image(image_id)
                except KeyError:
                    self.db.add_image(name, camera_id, image_id=image_id)
                else:
                    if (old.name, old.camera_id) != (name, camera_id):
                        raise ValueError('Database image IDs/names do not match the pipeline; use a different database_path')
            if calibration_changed and not preserve_geometry_cache:
                self.db.connection.execute('DELETE FROM two_view_geometries')
                self.db.connection.execute("DELETE FROM torch_sfm_cache WHERE kind='two_view'")
            self._write_extra('metadata', metadata)

    def load_keypoints(self, key):
        return self._load_keypoints(key, lazy_dense=True)

    def _load_dense_features(self, key):
        features = self._extra('dense_features', key)
        if features is None:
            raise KeyError(f'Missing FlexGlue dense features for frame {key}')
        return features

    def _load_keypoints(self, key, *, lazy_dense=False):
        if not self.path.exists():
            return None
        image_id = int(key) + 1
        try:
            xy = self.db.keypoints[image_id][:, :2]
        except KeyError:
            return None
        extra = self._extra('keypoints', key) or {}
        if extra and extra.get('signature') != self._signature('keypoints', key):
            return None
        result = {'keypoints_2d': torch.from_numpy(xy)}
        if self.method == 'flexglue':
            if lazy_dense:
                # Query the index without materializing the feature BLOB.
                row = self.db.connection.execute(
                    "SELECT 1 FROM torch_sfm_cache WHERE kind='dense_features' AND key=?",
                    (self._key('dense_features', key),)).fetchone()
                dense = None if row is None else partial(self._load_dense_features, key)
            else:
                dense = self._extra('dense_features', key)
            if dense is None or extra.get('image_size') is None:
                return None
            result.update(dense_features=dense, features=None, image_size=extra['image_size'],
                          options=extra.get('options'))
        else:
            try:
                descriptors = self.db.descriptors[image_id]
            except KeyError:
                return None
            if descriptors.type not in (FeatureType.LOMA_B, FeatureType.LOMA_B128):
                return None
            result['features'] = torch.from_numpy(descriptors.data)
        for name in ('keypoints_2d', 'features'):
            if extra.get(name) is not None:
                result[name] = extra[name]
            elif name + '_dtype' in extra:
                result[name] = result[name].to(extra[name + '_dtype'])
        if extra.get('feature_input') is not None:
            result['feature_input'] = extra['feature_input']
        return result

    def load(self, kind, key=None):
        if not self.path.exists():
            return None
        if kind == 'two_view_uncalibrated':
            extra = self._extra(kind, key)
            if extra is None or extra['direction'] != tuple(map(int, key)):
                return None
            return extra['payload']
        if kind == 'retrieval':
            pairs = self._extra(kind)
            return None if pairs is None else pairs.numpy()
        if kind == 'keypoints':
            return self._load_keypoints(key)
        if kind == 'correspondences':
            try:
                indices = self.db.matches[tuple(int(index) + 1 for index in key)]
            except KeyError:
                return None
            extra = self._extra(kind, key)
            if extra is not None and extra.get('signature') != self._signature(kind, key):
                return None
            scores = torch.ones(len(indices)) if extra is None else extra['scores']
            if len(scores) != len(indices):
                return None
            result = {'indices': torch.from_numpy(indices.astype(np.int64)), 'scores': scores}
            if extra is not None:
                for name in ('metadata', 'options'):
                    if name in extra:
                        result[name] = extra[name]
            return result
        if kind == 'two_view':
            extra = self._extra(kind, key)
            if extra is None or extra['direction'] != tuple(map(int, key)):
                return None  # External geometry lacks our verification inputs/options.
            if extra.get('signature') != self._signature(kind, key):
                return None
            return extra['payload']
        raise ValueError(f'Unknown cache kind: {kind}')

    def _invalidate_pairs(self, image_id):
        for table in ('matches', 'two_view_geometries'):
            self.db.connection.execute(f'DELETE FROM {table} WHERE pair_id / ? = ? OR pair_id % ? = ?',
                                       (MAX_IMAGE_ID, image_id, MAX_IMAGE_ID, image_id))
        self.db.connection.execute("""DELETE FROM torch_sfm_cache
            WHERE kind IN ('correspondences', 'two_view', 'two_view_uncalibrated') AND (image_id1=? OR image_id2=?)""",
            (image_id, image_id))

    def invalidate_two_view(self, keys):
        with self.db.connection:
            for key in keys:
                self.db.connection.execute('DELETE FROM two_view_geometries WHERE pair_id=?',
                    (image_ids_to_pair_id(*(int(i)+1 for i in key)),))
                self.db.connection.execute("DELETE FROM torch_sfm_cache WHERE kind='two_view' AND key=?",
                    (self._key('two_view', key),))

    def save(self, kind, data, key=None):
        if kind == 'two_view_uncalibrated':
            with self.db.connection:
                self._write_extra(kind, {'direction': tuple(map(int, key)), 'payload': data}, key)
            return
        with self.db.connection:
            if kind == 'retrieval':
                self._write_extra(kind, torch.as_tensor(data, dtype=torch.int32).cpu())
            elif kind == 'keypoints':
                image_id = int(key) + 1
                xy = data['keypoints_2d'].detach().cpu()
                self.db.keypoints[image_id] = xy.float().numpy()
                extra = {'feature_input': data.get('feature_input')}
                tensors = {'keypoints_2d': xy}
                if self.method == 'flexglue':
                    dense, image_size = data['dense_features'], data['image_size']
                    if not isinstance(dense, torch.Tensor) or image_size is None:
                        raise ValueError('FlexGlue requires dense_features and image_size')
                    self.db.connection.execute('DELETE FROM descriptors WHERE image_id=?', (image_id,))
                    self._write_extra('dense_features', dense.detach().cpu(), key)
                    extra.update(image_size=image_size, options=data.get('options'))
                elif self.method == 'loma':
                    features = data['features'].detach().cpu()
                    if len(xy) != len(features):
                        raise ValueError('Keypoints and descriptors must have the same number of rows')
                    if features.ndim != 2 or features.shape[1] not in (128, 256):
                        raise ValueError('LoMa descriptors must be 128- or 256-dimensional')
                    feature_type = FeatureType.LOMA_B if features.shape[1] == 256 else FeatureType.LOMA_B128
                    self.db.descriptors[image_id] = Descriptors(features.float().numpy(), feature_type)
                    tensors['features'] = features
                else:
                    raise ValueError(f'Unsupported feature method for COLMAP cache: {self.method}')
                for name, tensor in tensors.items():
                    extra[name + '_dtype'] = tensor.dtype
                    # COLMAP requires float32; keep double inputs losslessly for Torch callers.
                    if tensor.dtype == torch.float64:
                        extra[name] = tensor
                self._write_extra(kind, extra, key)
                self._invalidate_pairs(image_id)
            elif kind == 'correspondences':
                ids = tuple(int(index) + 1 for index in key)
                self.db.matches[ids] = data['indices'].detach().cpu().numpy()
                extra = {name: data[name] for name in ('metadata', 'options') if name in data}
                self._write_extra(kind, dict(extra, scores=data['scores'].detach().cpu()), key)
                self.db.connection.execute('DELETE FROM two_view_geometries WHERE pair_id=?',
                                           (image_ids_to_pair_id(*ids),))
                self.db.connection.execute("DELETE FROM torch_sfm_cache WHERE kind IN ('two_view','two_view_uncalibrated') AND key=?",
                                           (self._key(kind, key),))
            elif kind == 'two_view':
                geometry = data['geometry']
                rotation, translation = geometry['R'], geometry['t']
                qvec = None
                if rotation is not None and translation is not None:
                    vector = so3_log(rotation.detach().cpu().double())
                    angle = vector.norm()
                    qvec = torch.cat((torch.cos(angle / 2)[None],
                                      .5 * torch.sinc(angle / (2 * math.pi)) * vector)).numpy()
                def array(name):
                    value = geometry[name]
                    return None if value is None else value.detach().cpu().double().numpy()
                standard = TwoViewGeometry(
                    geometry['indices'].detach().cpu().numpy(), GeometryConfig[geometry['config']],
                    F=array('F'), E=array('E'), H=array('H'), qvec=qvec,
                    tvec=None if qvec is None else array('t'))
                self.db.write_two_view_geometry(*(int(index) + 1 for index in key), standard)
                self._write_extra(kind, {'direction': tuple(map(int, key)), 'payload': data}, key)
            else:
                raise ValueError(f'Unknown cache kind: {kind}')
