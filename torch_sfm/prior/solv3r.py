"""Optional Solv3R local pose/depth prior; no global mapping."""
from collections.abc import Sequence
import operator
from pathlib import Path

import torch

from .base import PosePrior, PriorModel


class Solv3RPrior(PriorModel):
    """Reuse one pretrained model across independent local image groups.

    predict() takes preprocessed tensors; predict_paths() uses Solv3R's official
    loader (long side 504, patch size 14, no crop). The first image sets the
    group's reference frame. Solv3R's native inference uses BF16 on CUDA.
    predict_batch() accepts groups already assembled by the pipeline.
    """

    def __init__(self, *, checkpoint: str | Path | None = None, calls: int = 16,
                 encode_chunk_size: int = 32, device: str | torch.device | None = None):
        for name, value in (("calls", calls), ("encode_chunk_size", encode_chunk_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.checkpoint = str(checkpoint) if checkpoint is not None else None
        self.calls = calls
        self.encode_chunk_size = encode_chunk_size
        self.device = torch.device(device if device is not None else
                                   ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None  # Importing torch_sfm.prior never imports/downloads Solv3R.

    @staticmethod
    def _backend():
        try:
            import solv3r
        except ModuleNotFoundError as error:
            if error.name != "solv3r":
                raise
            raise ImportError("Solv3R is optional: install torch-sfm[solv3r] with Python 3.11 or 3.12") from error
        return solv3r

    @staticmethod
    def _image_ids(count, image_ids):
        if image_ids is None:
            return tuple(range(count))
        values = tuple(image_ids)
        try:
            ids = tuple(operator.index(value) for value in values)
        except TypeError as error:
            raise ValueError("image_ids must be unique integers, one per image") from error
        if (len(ids) != count or len(set(ids)) != count
                or any(isinstance(value, bool) for value in values)):
            raise ValueError("image_ids must be unique integers, one per image")
        return ids

    @torch.inference_mode()
    def predict(self, images: torch.Tensor, *, image_ids: Sequence[int] | None = None,
                return_depth: bool = False) -> PosePrior:
        """Return CPU poses/K; H,W must be multiples of 14 (normally long side 504).

        Inputs and their order are preserved. This predicts a single local group,
        not a batch of independent groups or a globally aligned reconstruction.
        return_depth enables dense decoding; depths share the camera translation
        scale and use pixel centers (x+.5,y+.5). Invalid depths are left for the
        geometric checker to mask. Camera-only inference remains the default.
        """
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must be an RGB tensor of shape (N, 3, H, W)")
        count, _, height, width = images.shape
        if count < 2:
            raise ValueError("A multi-view pose prior needs at least two images")
        if min(height, width) < 14 or height % 14 or width % 14:
            raise ValueError("Image dimensions must be positive multiples of 14; use predict_paths for preprocessing")
        if not images.is_floating_point() or not bool(((images >= 0) & (images <= 1)).all()):
            raise ValueError("images must contain finite RGB floats in [0, 1]")
        ids = self._image_ids(count, image_ids)
        backend = self._backend()
        if self.model is None:
            self.model = backend.Solv3R.from_pretrained(checkpoint=self.checkpoint, device=str(self.device))
        # encode() transfers images in chunks; keep a CPU input on CPU here.
        context = self.model.encode(images.float(), reference_first=True, chunk=self.encode_chunk_size)
        state = self.model.solve(context, calls=self.calls)
        depth = confidence = None
        if return_depth:
            prediction = self.model.decode(state, context)
            encoding = prediction['pose_enc']
            depth = prediction['depth'].detach().float().cpu().squeeze(-1)
            confidence = prediction['depth_conf'].detach().float().cpu()
            if depth.shape != (count, height, width) or confidence.shape != depth.shape:
                raise RuntimeError('Solv3R returned unexpected depth shapes')
            del prediction
        else:
            encoding = self.model.decode_cameras(state, context)
        cameras, intrinsics = backend.cameras_from_pose_enc(encoding, (height, width))
        cameras = cameras.detach().to(device="cpu", dtype=torch.float32)
        intrinsics = intrinsics.detach().to(device="cpu", dtype=torch.float32)
        if cameras.shape != (count, 4, 4) or intrinsics.shape != (count, 3, 3):
            raise RuntimeError("Solv3R returned unexpected camera shapes")
        if not torch.isfinite(cameras).all() or not torch.isfinite(intrinsics).all():
            raise RuntimeError("Solv3R returned nonfinite cameras")
        # The learned reference pose need not be exactly identity. Reanchor all
        # cameras together, retaining relative translations and the group's scale.
        reference_rotation = cameras[0, :3, :3].T
        local = torch.zeros_like(cameras)
        local[:, :3, :3] = reference_rotation @ cameras[:, :3, :3]
        local[:, :3, 3] = (reference_rotation @ (cameras[:, :3, 3] - cameras[0, :3, 3])[..., None]).squeeze(-1)
        local[:, 3, 3] = 1
        local[0] = torch.eye(4)
        return PosePrior(ids, local, intrinsics, (height, width), depth, confidence)

    @torch.inference_mode()
    def predict_batch(self, images: torch.Tensor, *, return_depth: bool = False) -> PosePrior:
        """Infer preprocessed RGB images (B,N,3,H,W), floats in [0,1].

        B is the number of independent local groups; N is the views in each
        group. H,W must be multiples of 14 (normally long side 504). Grouping,
        preprocessing and image-cache ownership belong to the pipeline.

        CPU tensors can be passed directly. The upstream solver handles one
        group at a time, so reuse the model and transfer/solve each group in
        turn. Never flatten B into N: global attention and gauge normalization
        must remain independent across groups. This is a batched input/output
        interface, not concurrent GPU inference across groups.

        Returns CPU camera_to_world (B,N,4,4), intrinsics (B,N,3,3), and
        image_ids=None. Batch and view order are preserved; view 0 anchors each
        group. No input mutation, image decoding, caching or index selection.
        """
        if not isinstance(images, torch.Tensor) or images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("images must be an RGB tensor of shape (B, N, 3, H, W)")
        if images.shape[0] == 0:
            raise ValueError("The image batch must contain at least one group")
        # Indexing a CPU tensor gives views; only predict/encode transfers the
        # current group's image chunks. Requested outputs are accumulated on CPU.
        results = [self.predict(group, return_depth=True) if return_depth else self.predict(group)
                   for group in images]
        return PosePrior(
            image_ids=None,
            camera_to_world=torch.stack([result.camera_to_world for result in results]),
            intrinsics=torch.stack([result.intrinsics for result in results]),
            image_size=results[0].image_size,
            depth=torch.stack([r.depth for r in results]) if return_depth else None,
            depth_conf=torch.stack([r.depth_conf for r in results]) if return_depth else None,
        )

    def predict_paths(self, image_paths: Sequence[str | Path], *,
                      image_ids: Sequence[int] | None = None, return_depth: bool = False) -> PosePrior:
        """Load one ordered group using official resize/EXIF handling.

        All images take the first image's processed size. Returned K is for that
        size, not original pixels; differing aspect ratios are stretched by the
        official loader. No sorting, topology construction or neighbor pruning.
        """
        if isinstance(image_paths, (str, Path)):
            raise ValueError("Pass a sequence of image paths, not a directory or single path")
        paths = [str(path) for path in image_paths]
        if len(paths) < 2:
            raise ValueError("A multi-view pose prior needs at least two images")
        ids = self._image_ids(len(paths), image_ids)
        images = self._backend().load_images(paths)
        return self.predict(images, image_ids=ids, return_depth=return_depth)
