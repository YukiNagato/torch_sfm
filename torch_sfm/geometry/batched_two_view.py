"""Public pair-batch adapter selecting one complete Torch or Warp implementation."""
from dataclasses import replace
import torch
from .two_view import TwoViewConfig, camera_matrix


def resolve_backend(cfg, device):
    """Resolve auto once, before entering a backend; never mix controllers."""
    cuda = torch.device(device).type == 'cuda'
    supported = cuda and (cfg.calibration == 'uncalibrated' or cfg.essential_cheirality)
    if cfg.backend == 'warp':
        if not supported:
            raise ValueError('Warp verification requires CUDA and calibrated essential cheirality')
        return 'warp'
    if cfg.backend == 'auto' and supported:
        return 'warp'
    return 'torch'


@torch.inference_mode(False)
@torch.no_grad()
def verify_two_view_batch(keypoints0,keypoints1,matches,intrinsics0,intrinsics1,cfg=None,image_sizes=None):
    """Verify independent pairs; Torch loops over the reference, Warp batches them.

    Intrinsics may be shared (4,) / (3,3), or per pair (B,4) / (B,3,3).
    Empty batches return an empty list. Output match indices refer to inputs.
    """
    cfg = TwoViewConfig() if cfg is None else cfg
    n = len(matches)
    if len(keypoints0)!=n or len(keypoints1)!=n or (image_sizes is not None and len(image_sizes)!=n):
        raise ValueError('Provide keypoints and optional image sizes for every image pair')
    if not n:
        return []
    if n>cfg.verification_pair_batch_size:
        raise ValueError('Split verification into chunks of verification_pair_batch_size')
    device=keypoints0[0].device
    backend=resolve_backend(cfg,device)
    cfg=replace(cfg,backend=backend)
    if backend=='warp':
        from .warp.verification import verify_two_view_batch as verify
        return verify(keypoints0,keypoints1,matches,intrinsics0,intrinsics1,cfg,image_sizes)
    from .torch_two_view import verify_two_view
    matrices=[]
    for intrinsics in (intrinsics0,intrinsics1):
        matrix=camera_matrix(intrinsics.detach().to(device=device,dtype=torch.float64))
        if matrix.ndim==2:
            matrix=matrix.expand(n,3,3)
        if matrix.shape!=(n,3,3):
            raise ValueError('Provide shared intrinsics or one matrix per image pair')
        matrices.append(matrix)
    return [verify_two_view(keypoints0[i],keypoints1[i],matches[i],matrices[0][i],matrices[1][i],
                            cfg,None if image_sizes is None else image_sizes[i]) for i in range(n)]
