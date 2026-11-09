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
See the [ETH3D and Mip-NeRF 360 benchmarks](docs/benchmarks/eth3d_mip360_glomap.md) for runtime and accuracy comparisons with COLMAP GLOMAP using Ceres and Caspar.

## Try RaCo + FlexGlue

You can swap the correspondence method for **RaCo + FlexGlue**, combining learned keypoint detection and matching with covisibility-based pair filtering (which is pretty cool). The rest of the reconstruction pipeline stays the same.

The [FlexGlue example](examples/flexglue_glomap.py) shows how to configure the frontend and run reconstruction with known intrinsics:

```bash
python examples/flexglue_glomap.py /path/to/workspace
```

The workspace contains `images.txt` (one absolute image path per line) and `intrinsics.txt` (`fx fy cx cy`, either one shared row or one row per image). Optional frontend installation instructions are included in the example.

See the [ETH3D FlexGlue experiment](docs/benchmarks/eth3d_flexglue.md) for runtime, pose accuracy, and pair-filtering results compared with the LoMa frontend in TorchSfM.
