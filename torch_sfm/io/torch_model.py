"""Convert Torch pipeline results to the existing COLMAP model I/O types."""

from pathlib import Path
import math

import torch
from PIL import Image as PILImage

from .model import Camera, CameraModel, Image, Point3D, Reconstruction
from ..geometry.averaging import so3_log
from ..geometry.two_view import camera_matrix


def relative_image_name(image_path: Path, image_root: Path | None = None):
    """Use identical portable image names in the database and sparse model."""
    image_path = Path(image_path)
    if image_root is not None:
        try:
            return image_path.resolve().relative_to(Path(image_root).resolve()).as_posix()
        except ValueError:
            pass
    return image_path.name


@torch.no_grad()
def export_colmap(frames, result, path: Path, image_root: Path | None = None):
    rotations = result.rotations.detach().cpu().double()
    centers = result.centers.detach().cpu().double()
    points = result.points3d.detach().cpu().double()
    observations = result.observations.detach().cpu()
    registered = result.registered.detach().cpu()
    matrices = camera_matrix(result.intrinsics.detach().cpu().double())
    shared = matrices.ndim == 2
    recon = Reconstruction()
    rgb_sum = torch.zeros((len(points), 3))
    counts = torch.bincount(observations[:, 0], minlength=len(points))
    error_sum = torch.zeros(len(points), dtype=torch.float64)
    if len(observations):
        error_sum.index_add_(0, observations[:, 0], result.reprojection_errors.detach().cpu().double())
    # Quaternion convention is Hamilton wxyz, world-to-camera.
    vectors = so3_log(rotations[registered])
    lengths = vectors.norm(dim=-1, keepdim=True)
    quaternions = torch.cat((torch.cos(lengths / 2),
                             .5 * torch.sinc(lengths / (2 * math.pi)) * vectors), -1)
    names = set()
    for local_index, frame_index in enumerate(torch.where(registered)[0].tolist()):
        frame = frames[frame_index]
        if frame.image is not None:
            image = frame.image.detach()
            height, width = image.shape[-2:]
        else:
            with PILImage.open(frame.frame_path) as source:
                source = source.convert('RGB')
                width, height = source.size
                image = torch.frombuffer(bytearray(source.tobytes()), dtype=torch.uint8)
                image = image.reshape(height, width, 3).permute(2, 0, 1)
        matrix = matrices if shared else matrices[frame_index]
        camera_id = 1 if shared else frame_index + 1
        params = torch.stack((matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]))
        if camera_id in recon.cameras:
            camera = recon.cameras[camera_id]
            if (camera.width, camera.height) != (width, height):
                raise ValueError('Images sharing intrinsics must have the same resolution')
        else:
            recon.cameras[camera_id] = Camera(camera_id, CameraModel.PINHOLE, width, height, params.numpy())
        frame_path = Path(frame.frame_path) if frame.frame_path is not None else Path(f'{frame_index}.png')
        name = relative_image_name(frame_path, image_root)
        if name in names:
            raise ValueError(f'Duplicate exported image name: {name}')
        names.add(name)
        xy = frame.keypoints_2d.detach().cpu().double()
        point_ids = torch.full((len(xy),), -1, dtype=torch.long)
        mask = observations[:, 1] == frame_index
        observed = observations[mask]
        point_ids[observed[:, 2]] = observed[:, 0] + 1
        if len(observed):
            coords = xy[observed[:, 2]].to(image.device).floor().long()
            x, y = coords[:, 0].clamp(0, width - 1), coords[:, 1].clamp(0, height - 1)
            colors = image[:, y, x].T.float()
            if image.is_floating_point():
                colors = colors.clamp(0, 1) * 255
            rgb_sum.index_add_(0, observed[:, 0], colors.cpu())
        translation = -rotations[frame_index] @ centers[frame_index]
        recon.images[frame_index + 1] = Image(frame_index + 1, camera_id, name,
            quaternions[local_index].numpy(), translation.numpy(), xy.numpy(), point_ids.numpy())
    colors = (rgb_sum / counts[:, None].clamp_min(1)).round().clamp(0, 255).to(torch.uint8)
    ordered = observations[observations[:, 0].argsort(stable=True)]
    offset = 0
    for index, length in enumerate(counts.tolist()):
        track = ordered[offset:offset + length, 1:].clone()
        track[:, 0] += 1
        offset += length
        recon.points3D[index + 1] = Point3D(index + 1, points[index].numpy(), colors[index].numpy(),
            float(error_sum[index] / max(length, 1)), track.numpy())
    recon.to_binary(path)
    recon.to_txt(path)
    return recon
