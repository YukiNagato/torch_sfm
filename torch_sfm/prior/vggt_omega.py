"""VGGT-Omega pose, depth and pinhole intrinsics in original image coordinates."""
import math

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


@torch.inference_mode()
def predict(images, sizes, grid, checkpoint=None):
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.pose_enc import encoding_to_camera

    if checkpoint is None:
        from huggingface_hub import hf_hub_download
        checkpoint = hf_hub_download('facebook/VGGT-Omega', 'vggt_omega_1b_512.pt')
    model = VGGTOmega().eval()
    model.load_state_dict(torch.load(checkpoint, mmap=True, weights_only=True, map_location='cpu'))
    model.cuda()
    # Full-frame resize, using the official balanced patch count for the typical aspect ratio.
    ratio = float((sizes[:, 1] / sizes[:, 0]).median().clamp(.5, 2.))
    h = 16 * round(32 * math.sqrt(ratio))
    w = 16 * round(32 / math.sqrt(ratio))
    batch = []
    for image in images:
        with Image.open(image.image_path) as frame:
            rgb = frame.convert('RGB').resize((w, h), Image.Resampling.BICUBIC)
            batch.append(torch.from_numpy(np.array(rgb)).permute(2, 0, 1))
    prediction = model(torch.stack(batch).cuda().float().div_(255))
    w2c, k = encoding_to_camera(prediction['pose_enc'], (h, w))
    w2c, k = w2c[0].double(), k[0].double()
    poses = torch.eye(4, dtype=torch.float64).repeat(len(images), 1, 1)
    poses[:, :3, :3] = w2c[:, :3, :3].mT.cpu()
    poses[:, :3, 3] = -(poses[:, :3, :3] @ w2c[:, :3, 3:].cpu()).squeeze(-1)
    depth = F.interpolate(prediction['depth'][0].movedim(-1, 1).double(), (grid, grid),
                          mode='bilinear', align_corners=False)[:, 0].cpu()
    intrinsics = torch.stack((k[:, 0, 0], k[:, 1, 1], k[:, 0, 2], k[:, 1, 2]), 1).cpu()
    intrinsics *= (sizes / sizes.new_tensor([w, h])).repeat(1, 2)
    if not (poses.isfinite().all() and intrinsics.isfinite().all() and (intrinsics[:, :2] > 0).all()):
        raise ValueError('VGGT-Omega returned invalid cameras')
    return poses, depth, intrinsics
