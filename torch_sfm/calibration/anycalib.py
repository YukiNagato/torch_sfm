from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import torch


@dataclass
class Intrinsics:
    cam_id: str
    width: int
    height: int
    params: np.ndarray
    source_path: Path
    inlier_paths: tuple[Path, ...]

    @property
    def K(self) -> np.ndarray:
        if self.cam_id.startswith("simple_"):
            f, cx, cy = self.params[:3]
            fx = fy = f
        else:
            fx, fy, cx, cy = self.params[:4]
        return np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]], dtype=np.float64)


class AnyCalibCalibration:
    def __init__(
        self,
        cam_id: str = "pinhole",
        *,
        model_id: str = "anycalib_pinhole",
        device: str | torch.device | None = None,
        **model_kwargs,
    ):
        from anycalib import AnyCalib
        from anycalib.cameras.factory import CameraFactory

        self.camera = CameraFactory.create_from_id(cam_id)
        self.cam_id = cam_id
        self.device = torch.device(device if device is not None else
                                   ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = AnyCalib(model_id=model_id, **model_kwargs).to(self.device).eval()

    @torch.inference_mode()
    def estimate(self, image_paths: list[str | Path], *, mad_threshold: float = 3.5) -> Intrinsics:
        """Select one consensus estimate for images sharing resolution and intrinsics."""
        from PIL import Image

        if not image_paths:
            raise ValueError("image_paths must not be empty")
        if not np.isfinite(mad_threshold) or mad_threshold <= 0:
            raise ValueError("mad_threshold must be finite and positive")
        paths = list(dict.fromkeys(Path(path).expanduser().resolve() for path in image_paths))
        size = None
        for path in paths:
            with Image.open(path) as image:
                if size is None:
                    size = image.size
                elif image.size != size:
                    raise ValueError(f"All images must have size {size}; {path} has {image.size}")

        estimates, valid_paths = [], []
        num_f = self.camera.NUM_F
        for path in paths:
            with Image.open(path) as image:
                rgb = np.array(image.convert("RGB"), dtype=np.float32)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device) / 255
            output = self.model.predict(tensor, cam_id=self.cam_id)
            params = output["intrinsics"].detach().cpu().double().numpy()
            self.camera.validate_params(output["intrinsics"])
            if (not bool(output.get("success", True)) or not np.isfinite(params).all()
                    or np.any(params[:num_f] <= 0)):
                warnings.warn(f"Skipping unsuccessful calibration: {path}", stacklevel=2)
                continue
            estimates.append(params)
            valid_paths.append(path)

        if not estimates:
            raise ValueError("AnyCalib produced no successful, finite calibration")
        if len(estimates) < 3:
            warnings.warn("Fewer than three valid estimates; outlier rejection is unreliable", stacklevel=2)

        params = np.stack(estimates)
        values = params.copy()
        values[:, :num_f] = np.log(values[:, :num_f])
        values[:, num_f:num_f + 2] /= np.array(size)
        center = np.median(values, axis=0)
        mad = np.median(np.abs(values - center), axis=0)
        scale = np.maximum(1.4826 * mad, 1e-6 * np.maximum(np.abs(center), 1.))
        inliers = np.all(np.abs(values - center) / scale <= mad_threshold, axis=1)
        if not inliers.any():
            raise ValueError("No consistent intrinsic estimates; check the images or mad_threshold")

        indices = np.flatnonzero(inliers)
        center = np.median(values[inliers], axis=0)
        distances = np.abs((values[inliers] - center) / scale).sum(axis=1)
        selected = indices[np.argmin(distances)]
        return Intrinsics(self.cam_id, *size, params[selected].copy(), valid_paths[selected],
                          tuple(valid_paths[index] for index in indices))
