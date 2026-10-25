from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, replace
from enum import IntEnum
from pathlib import Path
import sqlite3

import numpy as np

from .model import Camera, CameraModel


MAX_IMAGE_ID = 2**31 - 1


class FeatureType(IntEnum):
    SIFT = 0
    ALIKED_N16ROT = 1
    ALIKED_N32 = 2
    LOMA_B = 3
    LOMA_B128 = 4


class GeometryConfig(IntEnum):
    UNDEFINED = 0
    DEGENERATE = 1
    CALIBRATED = 2
    UNCALIBRATED = 3
    PLANAR = 4
    PANORAMIC = 5
    PLANAR_OR_PANORAMIC = 6
    WATERMARK = 7
    MULTIPLE = 8
    CALIBRATED_RIG = 9


@dataclass
class DatabaseCamera(Camera):
    prior_focal_length: bool = False


@dataclass
class DatabaseImage:
    image_id: int
    name: str
    camera_id: int


@dataclass
class Descriptors:
    data: np.ndarray
    type: FeatureType = FeatureType.SIFT


@dataclass
class TwoViewGeometry:
    matches: np.ndarray  # (N, 2), inlier indices
    config: GeometryConfig = GeometryConfig.UNDEFINED
    F: np.ndarray | None = None
    E: np.ndarray | None = None
    H: np.ndarray | None = None
    qvec: np.ndarray | None = None  # (w, x, y, z), camera2_from_camera1
    tvec: np.ndarray | None = None
    camera1: bytes | None = None  # Optional COLMAP camera blobs, preserved verbatim
    camera2: bytes | None = None

    def inverse(self) -> TwoViewGeometry:
        qvec, tvec = self.qvec, self.tvec
        if (qvec is None) != (tvec is None):
            raise ValueError("qvec and tvec must be supplied together")
        if qvec is not None:
            q = np.asarray(qvec, dtype=np.float64)
            if q.shape != (4,) or not np.isclose(np.linalg.norm(q), 1):
                raise ValueError("qvec must be a unit quaternion of shape (4,)")
            qvec = q * [1, -1, -1, -1]
            v = qvec[1:]
            t = np.asarray(tvec, dtype=np.float64)
            tvec = -(t + 2 * np.cross(v, np.cross(v, t) + qvec[0] * t))
        return replace(self, matches=np.asarray(self.matches)[:, ::-1].copy(),
                       F=None if self.F is None else self.F.T.copy(),
                       E=None if self.E is None else self.E.T.copy(),
                       H=None if self.H is None else np.linalg.inv(self.H),
                       qvec=qvec, tvec=tvec, camera1=self.camera2, camera2=self.camera1)


def _image_id(value: int) -> int:
    if not isinstance(value, (int, np.integer)) or not 0 < value < MAX_IMAGE_ID:
        raise ValueError(f"Image ID must be an integer in (0, {MAX_IMAGE_ID})")
    return int(value)


def image_ids_to_pair_id(image_id1: int, image_id2: int) -> int:
    first, second = sorted((_image_id(image_id1), _image_id(image_id2)))
    if first == second:
        raise ValueError("An image pair must contain two different images")
    return first * MAX_IMAGE_ID + second


def pair_id_to_image_ids(pair_id: int) -> tuple[int, int]:
    return divmod(int(pair_id), MAX_IMAGE_ID)


def _matrix(value, dtype, cols: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] not in cols:
        raise ValueError(f"Expected a matrix with {cols} columns, got {array.shape}")
    dtype = np.dtype(dtype)
    if dtype.kind == "u" and array.size:
        if array.dtype.kind not in "iu" or np.any(array < 0) or np.any(array > np.iinfo(dtype).max):
            raise ValueError(f"Expected integers representable as {dtype}")
    return np.ascontiguousarray(array, dtype=dtype)


class _ArrayTable(MutableMapping):
    def __init__(self, db: Database, name: str):
        self.db, self.name = db, name
        self.key = "pair_id" if name == "matches" else "image_id"

    def _key(self, key):
        return image_ids_to_pair_id(*key) if self.name == "matches" else _image_id(key)

    def __getitem__(self, key):
        row = self.db.connection.execute(
            f"SELECT * FROM {self.name} WHERE {self.key} = ?", (self._key(key),)
        ).fetchone()
        if row is None:
            raise KeyError(key)
        dtype = "<f4" if self.name == "keypoints" else "<u4"
        cols = row["cols"]
        if self.name == "descriptors":
            kind = FeatureType(row["type"] if "type" in row.keys() else 0)
            dtype = "u1" if kind == FeatureType.SIFT else "<f4"
            if cols % np.dtype(dtype).itemsize:
                raise ValueError("Invalid descriptor byte width")
            cols //= np.dtype(dtype).itemsize
        array = np.frombuffer(row["data"] or b"", dtype=dtype).reshape(row["rows"], cols).copy()
        if self.name == "descriptors":
            return Descriptors(array, kind)
        return array[:, ::-1].copy() if self.name == "matches" and key[0] > key[1] else array

    def __setitem__(self, key, value):
        row = {self.key: self._key(key)}
        if self.name == "descriptors":
            kind = FeatureType(value.type)
            dtype, dim = ("u1", 128) if kind == FeatureType.SIFT else ("<f4", 256 if kind == FeatureType.LOMA_B else 128)
            array = _matrix(value.data, dtype, (dim,))
            if "type" in self.db._columns(self.name):
                row["type"] = int(kind)
            elif kind != FeatureType.SIFT:
                raise ValueError("This legacy database only supports SIFT descriptors")
            cols = array.shape[1] * array.dtype.itemsize
        else:
            array = _matrix(value, "<f4" if self.name == "keypoints" else "<u4",
                            (2, 4, 6) if self.name == "keypoints" else (2,))
            if self.name == "matches" and key[0] > key[1]:
                array = array[:, ::-1]
            cols = array.shape[1]
        row.update(rows=len(array), cols=cols, data=array.tobytes())
        self.db._upsert(self.name, self.key, row)

    def __delitem__(self, key):
        cursor = self.db.connection.execute(
            f"DELETE FROM {self.name} WHERE {self.key} = ?", (self._key(key),))
        if not cursor.rowcount:
            raise KeyError(key)

    def __iter__(self) -> Iterator:
        for row in self.db.connection.execute(f"SELECT {self.key} FROM {self.name} ORDER BY {self.key}"):
            yield pair_id_to_image_ids(row[0]) if self.name == "matches" else row[0]

    def __len__(self) -> int:
        return self.db.connection.execute(f"SELECT COUNT(*) FROM {self.name}").fetchone()[0]


class Database:
    """COLMAP core tables. Context exit commits/rolls back and closes the connection."""

    def __init__(self, path: str | Path = ":memory:", *, readonly: bool = False):
        target = Path(path).resolve().as_uri() + "?mode=ro" if readonly else str(path)
        self.connection = sqlite3.connect(target, uri=readonly)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        try:
            if not readonly and not self.connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1"
            ).fetchone():
                self._create_tables()
            self.keypoints = _ArrayTable(self, "keypoints")
            self.descriptors = _ArrayTable(self, "descriptors")
            self.matches = _ArrayTable(self, "matches")
        except Exception:
            self.close()
            raise

    def __enter__(self) -> Database:
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.commit() if exc_type is None else self.rollback()
        finally:
            self.close()

    def commit(self):
        self.connection.commit()

    def rollback(self):
        self.connection.rollback()

    def close(self):
        self.connection.close()

    def _columns(self, table: str) -> set[str]:
        return {row["name"] for row in self.connection.execute(f"PRAGMA table_info({table})")}

    def _upsert(self, table: str, key: str, row: dict):
        columns = ", ".join(row)
        placeholders = ", ".join("?" for _ in row)
        updates = ", ".join(f"{name}=excluded.{name}" for name in row if name != key)
        self.connection.execute(
            f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) "
            f"ON CONFLICT({key}) DO UPDATE SET {updates}", tuple(row.values()))

    def add_camera(self, model: CameraModel, width: int, height: int, params: np.ndarray,
                   *, camera_id: int | None = None, prior_focal_length: bool = False) -> int:
        params = np.asarray(params, dtype="<f8")
        if params.shape != (model.num_params,) or width <= 0 or height <= 0:
            raise ValueError("Invalid camera dimensions or parameter count")
        if camera_id is not None and not 0 < camera_id < 2**32:
            raise ValueError("camera_id must be a positive uint32")
        cursor = self.connection.execute(
            "INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model.model_id, int(width), int(height), params.tobytes(), int(prior_focal_length)))
        return cursor.lastrowid

    def read_camera(self, camera_id: int) -> DatabaseCamera:
        row = self.connection.execute("SELECT * FROM cameras WHERE camera_id=?", (int(camera_id),)).fetchone()
        if row is None:
            raise KeyError(camera_id)
        model = next(model for model in CameraModel if model.model_id == row["model"])
        return DatabaseCamera(row["camera_id"], model, row["width"], row["height"],
                              np.frombuffer(row["params"], dtype="<f8").copy(), bool(row["prior_focal_length"]))

    def add_image(self, name: str, camera_id: int, *, image_id: int | None = None) -> int:
        if not name or "\0" in name:
            raise ValueError("Image name must be nonempty and contain no NUL")
        cursor = self.connection.execute(
            "INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, ?)",
            (None if image_id is None else _image_id(image_id), name, int(camera_id)))
        return cursor.lastrowid

    def read_image(self, image: int | str) -> DatabaseImage:
        column = "name" if isinstance(image, str) else "image_id"
        row = self.connection.execute(
            f"SELECT image_id, name, camera_id FROM images WHERE {column}=?",
            (image if isinstance(image, str) else int(image),)).fetchone()
        if row is None:
            raise KeyError(image)
        return DatabaseImage(*row)

    def iter_cameras(self) -> Iterator[DatabaseCamera]:
        for row in self.connection.execute("SELECT camera_id FROM cameras ORDER BY camera_id"):
            yield self.read_camera(row[0])

    def iter_images(self) -> Iterator[DatabaseImage]:
        for row in self.connection.execute("SELECT image_id, name, camera_id FROM images ORDER BY image_id"):
            yield DatabaseImage(*row)

    def write_two_view_geometry(self, image_id1: int, image_id2: int, geometry: TwoViewGeometry):
        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        if image_id1 > image_id2:
            geometry = geometry.inverse()
        matches = _matrix(geometry.matches, "<u4", (2,))
        if (geometry.qvec is None) != (geometry.tvec is None):
            raise ValueError("qvec and tvec must be supplied together")
        row = dict(pair_id=pair_id, rows=len(matches), cols=2,
                   data=matches.tobytes(), config=int(GeometryConfig(geometry.config)))
        columns = self._columns("two_view_geometries")
        for name, shape in (("F", (3, 3)), ("E", (3, 3)), ("H", (3, 3)), ("qvec", (4,)), ("tvec", (3,))):
            value = getattr(geometry, name)
            if value is not None:
                value = np.asarray(value, dtype="<f8")
                if value.shape != shape:
                    raise ValueError(f"{name} must have shape {shape}")
                if name == "qvec" and not np.isclose(np.linalg.norm(value), 1):
                    raise ValueError("qvec must be a unit quaternion")
                value = value.tobytes()
            if name in columns:
                row[name] = value
            elif value is not None:
                raise ValueError(f"This database has no {name} column")
        for name in ("camera1", "camera2"):
            if name in columns:
                row[name] = getattr(geometry, name)
            elif getattr(geometry, name) is not None:
                raise ValueError(f"This database has no {name} column")
        self._upsert("two_view_geometries", "pair_id", row)

    def read_two_view_geometry(self, image_id1: int, image_id2: int) -> TwoViewGeometry:
        row = self.connection.execute("SELECT * FROM two_view_geometries WHERE pair_id=?",
                                      (image_ids_to_pair_id(image_id1, image_id2),)).fetchone()
        if row is None:
            raise KeyError((image_id1, image_id2))
        if row["cols"] != 2:
            raise ValueError("Two-view matches must have two columns")
        values = {}
        for name, shape in (("F", (3, 3)), ("E", (3, 3)), ("H", (3, 3)), ("qvec", (4,)), ("tvec", (3,))):
            value = row[name] if name in row.keys() else None
            values[name] = None if value is None else np.frombuffer(value, dtype="<f8").reshape(shape).copy()
        for name in ("camera1", "camera2"):
            values[name] = row[name] if name in row.keys() else None
        geometry = TwoViewGeometry(
            np.frombuffer(row["data"] or b"", dtype="<u4").reshape(row["rows"], 2).copy(),
            GeometryConfig(row["config"]), **values)
        return geometry.inverse() if image_id1 > image_id2 else geometry

    def iter_two_view_pairs(self) -> Iterator[tuple[int, int]]:
        for row in self.connection.execute("SELECT pair_id FROM two_view_geometries ORDER BY pair_id"):
            yield pair_id_to_image_ids(row[0])

    def _create_tables(self):
        self.connection.executescript("""
            CREATE TABLE cameras (
                camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                model INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
                params BLOB, prior_focal_length INTEGER NOT NULL);
            CREATE TABLE images (
                image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                name TEXT NOT NULL UNIQUE, camera_id INTEGER NOT NULL,
                CHECK(image_id > 0 AND image_id < 2147483647),
                FOREIGN KEY(camera_id) REFERENCES cameras(camera_id));
            CREATE UNIQUE INDEX index_name ON images(name);
            CREATE TABLE keypoints (
                image_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL,
                cols INTEGER NOT NULL, data BLOB,
                FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
            CREATE TABLE descriptors (
                image_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL,
                cols INTEGER NOT NULL, data BLOB, type INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
            CREATE TABLE matches (
                pair_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL,
                cols INTEGER NOT NULL, data BLOB);
            CREATE TABLE two_view_geometries (
                pair_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL,
                cols INTEGER NOT NULL, data BLOB, config INTEGER NOT NULL,
                F BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB, camera1 BLOB, camera2 BLOB);
            PRAGMA user_version = 3140002;
        """)
