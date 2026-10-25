import torch

from .base import Retrieval


class SALADRetrieval(Retrieval[torch.Tensor]):
    def __init__(self, *, device: str | torch.device | None = None):
        from torchvision import transforms

        self.device = torch.device(device if device is not None else
                                   ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = torch.hub.load(
            "serizba/salad", "dinov2_salad", pretrained=True, trust_repo=True
        ).to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((322, 322), interpolation=transforms.InterpolationMode.BILINEAR,
                              antialias=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.inference_mode()
    def extract(self, image: torch.Tensor) -> torch.Tensor:
        """Return a unit descriptor (8448,) from an RGB CHW float image in [0, 1]."""
        if image is None or image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("image must have shape (3, H, W)")
        if not image.is_floating_point():
            raise ValueError("image must be floating point in [0, 1]")
        image = self.transform(image.to(device=self.device, dtype=torch.float32))
        return self.model(image.unsqueeze(0))[0].float()

    @torch.inference_mode()
    def similarity(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Dot products of unit descriptors from extract(); returns (N, N)."""
        if not features:
            return torch.empty((0, 0), device=self.device)
        if any(feature.ndim != 1 or feature.shape != features[0].shape for feature in features):
            raise ValueError("SALAD features must be vectors of the same dimension")
        descriptors = torch.stack([
            feature.to(device=self.device, dtype=torch.float32) for feature in features
        ])
        if not torch.isfinite(descriptors).all():
            raise ValueError("SALAD features must be finite")
        return (descriptors @ descriptors.T).clamp(-1, 1)
