from typing import Literal

import torch
import torch.nn.functional as F

from .base import Retrieval


class SAGERetrieval(Retrieval[torch.Tensor]):
    def __init__(
        self,
        variant: Literal["sage_vitb", "sage_vitl", "sage"] = "sage_vitb",
        *,
        device: str | torch.device | None = None,
    ):
        from torchvision import transforms

        if variant not in ("sage_vitb", "sage_vitl", "sage"):
            raise ValueError(f"Unknown SAGE variant: {variant}")
        self.device = torch.device(device if device is not None else
                                   ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = torch.hub.load(
            "chenshunpeng/SAGE", variant, pretrained=True, trust_repo=True
        ).to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Resize((322, 322), antialias=True),
        ])

    @torch.inference_mode()
    def extract(self, image: torch.Tensor) -> torch.Tensor:
        """Return one global descriptor (D,) from an RGB CHW float image in [0, 1]."""
        if image is None or image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("image must have shape (3, H, W)")
        if not image.is_floating_point():
            raise ValueError("image must be floating point in [0, 1]")
        image = self.transform(image.to(device=self.device, dtype=torch.float32))
        descriptor = self.model(image.unsqueeze(0))[0].float()
        if self.model.crossimage_encoder:
            descriptor = F.normalize(descriptor, dim=0)
        return descriptor

    @torch.inference_mode()
    def similarity(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Dot products of unit descriptors from extract(); returns (N, N)."""
        if not features:
            return torch.empty((0, 0), device=self.device)
        if any(feature.ndim != 1 or feature.shape != features[0].shape for feature in features):
            raise ValueError("SAGE features must be vectors of the same dimension")
        descriptors = torch.stack([
            feature.to(device=self.device, dtype=torch.float32) for feature in features
        ])
        if not torch.isfinite(descriptors).all():
            raise ValueError("SAGE features must be finite")
        return (descriptors @ descriptors.T).clamp(-1, 1)
