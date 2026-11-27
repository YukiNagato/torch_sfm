# TorchSfM

TorchSfM is a structure-from-motion package built with **PyTorch** and optimized with **NVIDIA Warp**. It aims to make deep learning methods easier to integrate into SfM while maintaining high performance.

## Installation

Install a CUDA-enabled PyTorch build appropriate for your system, then install from the repository root:

```bash
pip install -e '.[salad,loma]'      # SALAD retrieval + LoMa matching
# Or use RaCo + FlexGlue:
pip install -e '.[salad,flexglue]'
```

Core dependencies include [torchls](https://github.com/YukiNagato/torchls), installed directly from GitHub, and NVIDIA Warp. Model weights are downloaded when the corresponding frontend is first used. The `anycalib` extra provides optional learned intrinsics estimation.

## GLOMAP Pipeline

The project currently implements a complete global SfM [pipeline](torch_sfm/pipeline/glomap.py) following GLOMAP, from image retrieval and feature matching to geometric verification, rotation and position averaging, triangulation, and bundle adjustment. It also supports COLMAP model export and interactive reconstruction visualization.
On CUDA, GLOMAP defaults to Warp track construction, Tensor/Warp triangulation, and full-system Warp PCG for bundle adjustment.
See the [ETH3D and Mip-NeRF 360 benchmarks](docs/benchmarks/eth3d_mip360_glomap.md) for runtime and accuracy comparisons with COLMAP GLOMAP using Ceres and Caspar.

### Dataset input

Put `dataset.json` in the workspace. Image order defines image indices. Each `cam_idx` is a string key in `cameras`. Images with the same key share intrinsics through calibration, bundle adjustment, and COLMAP export.

```json
{
  "images": [
    {"image_path": "images/0001.jpg", "cam_idx": "main", "mask_path": "masks/0001.png"},
    {"image_path": "images/0002.jpg", "cam_idx": "main"}
  ],
  "cameras": {
    "main": {"model": "PINHOLE", "width": 1920, "height": 1080, "params": [1400, 1400, 960, 540]}
  }
}
```

Paths can be absolute or relative to the JSON file. Use undistorted images and intrinsics in original image pixels, ordered `[fx, fy, cx, cy]`. Set `params` to `null` to initialize a camera group with AnyCalib. An optional `mask_path` points to a grayscale mask at the image resolution. Black pixels (0) exclude keypoints and nonzero pixels keep them. An optional `depth_path` is recorded as per-image metadata.

### Try RaCo + FlexGlue

You can swap the correspondence method for **RaCo + FlexGlue**, combining learned keypoint detection and matching with covisibility-based pair filtering (which is pretty cool). The rest of the reconstruction pipeline stays the same.

The [FlexGlue example](examples/flexglue_glomap.py) shows how to configure the frontend and run reconstruction with known intrinsics:

```bash
python examples/flexglue_glomap.py /path/to/workspace
```

Optional frontend installation instructions are included in the example.

Pass `--cache-backend colmap` to cache FlexGlue features and matches in SQLite at `workspace/database.db`. The matcher reads dense features on demand.

See the [ETH3D FlexGlue experiment](docs/benchmarks/eth3d_flexglue.md) for runtime, pose accuracy, and pair-filtering results compared with the LoMa frontend in TorchSfM.

### Use the optional FP8 matcher

```bash
pip install -e '.[salad,flexglue,flexglue-fp8]'
python examples/flexglue_glomap.py /path/to/workspace --backend fp8
```

`FlexGlueConfig(backend="fp8")` downloads the [FP8 release checkpoint](https://github.com/YukiNagato/flexglue_quantize/releases/tag/v0.1) on the first uncached match. Pass `--quantized-weights /path/to/model_fp8.pt` to use a local matcher checkpoint.

## GlueMap Pipeline

The [GlueMap-style pipeline](torch_sfm/pipeline/gluemap.py) uses learned pose and depth priors on local image groups, checks depth consistency, fuses local poses into a global reconstruction, and refines it with triangulation and bundle adjustment.

### Try LaMAR

The [LaMAR example](examples/lamar_gluemap.py) includes data preparation, sequence-aware retrieval, and two-round refinement. Compared with standard GlueMap, this configuration uses **Solv3R** priors and **RaCo + FlexGlue** tracks, omits virtual tracks, and runs BA with **torchls / Warp PCG**. It fixes the supplied intrinsics and applies GLOMAP-style geometric verification before refinement. See the [LaMAR HGE results](docs/benchmarks/lamar_hge_gluemap.md).

## DMap Style Pose Refinement

Refine camera poses and depths with RoMa v2 or FlexGlue Using [DMap](https://openreview.net/forum?id=iikGG5Se3J) style pipeline. DMap selects up to eight overlapping targets per image, keeps dense matches near RaCo keypoints, and runs up to three BA rounds with fixed intrinsics. It uses the same `dataset.json` as GLOMAP. Each image may also specify its initial geometry:

```json
{
  "image_path": "images/0001.jpg", "cam_idx": "main",
  "depth_path": "depth/0001.npy",
  "pose": [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
}
```

`pose` is a 4×4 OpenCV camera-to-world matrix (+x right, +y down, +z forward). `depth_path` contains camera-Z depths in the same units as pose translations. Supply a 2D NumPy array, PyTorch tensor, or single-channel depth image covering the full undistorted RGB frame. Zero, negative and nonfinite depths are ignored. The optional GLOMAP image mask applies to both match endpoints.

Omit either prior per image to fill it with VGGT-Ω. Supplied poses and depths take precedence. The prediction is aligned using supplied camera orientations and centers, with depth ratios providing scale when a baseline is unavailable. For a camera with `params: null`, DMap uses the median predicted intrinsics of its images. Complete inputs with calibrated cameras skip VGGT-Ω.

```bash
pip install -e '.[romav2,omega]'  # or '.[flexglue,omega]'
python -m torch_sfm.pipeline.dmap --workspace /path/to/workspace
```

DMap reads `workspace/dataset.json`. By default it writes the COLMAP model to `workspace/sparse/0`, the colored cloud to `workspace/points3D.ply`, and the refined poses, intrinsics and 64×64 depth maps to `workspace/reconstruction.pt`. The model uses image names relative to `workspace/images/`.

Select FlexGlue with `--config.correspondence-method flexglue`, supply local VGGT-Ω weights with `--config.omega-checkpoint PATH`, or skip COLMAP export with `--config.no-export-colmap`. The same options are available in Python:

```python
from pathlib import Path
from torch_sfm.pipeline.dmap import DMapConfig, DMapPipeline

config = DMapConfig(correspondence_method="flexglue")
result = DMapPipeline(config, Path("workspace")).run()
```
