from abc import ABC, abstractmethod


class Cache(ABC):
    """Stage payloads use CPU tensors; a missing entry returns None.

    Kinds: retrieval (no key), keypoints (frame index), correspondences and
    two_view (ordered pair of frame indices). prepare supplies camera metadata.
    """

    def prepare(self, frames, intrinsics):
        pass

    @abstractmethod
    def load(self, kind, key=None):
        ...

    def load_keypoints(self, key):
        """Load matching inputs, possibly with a lazy dense feature map."""
        return self.load('keypoints', key)

    @abstractmethod
    def save(self, kind, data, key=None):
        ...

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
