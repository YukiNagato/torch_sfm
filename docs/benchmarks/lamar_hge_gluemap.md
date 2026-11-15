# LaMAR HGE: RaCo + FlexGlue + Solv3R

TorchSfM reconstructs **7,540 / 7,553 images** of LaMAR HGE using a GlueMap-style pipeline. Joint pose **AUC@3° improves from 9.12% before BA to 57.56% after BA**, evaluated with the public GlueMap evaluator's conventions. The final reconstruction contains **4,181,148 points and 24,745,380 observations**.

This is a single-scene experiment with fixed official intrinsics, cached frontend inputs, and a checkpoint resume after an out-of-memory failure. It is **not an uninterrupted, from-scratch runtime benchmark**.

## Hardware and configuration

| Component | Setting |
| --- | --- |
| CPU | Intel Xeon Platinum 8474C; process affinity 0–15; PyTorch/OMP/MKL threads: 4 |
| GPU | NVIDIA GeForce RTX 4090 D, 24 GB VRAM |
| Host memory limit | 80 GiB |
| Software | PyTorch 2.11.0+cu128, NVIDIA Warp 1.17.0 |
| Dataset | LaMAR HGE, 17 iOS mapping sequences, 7,553 images, 1440×1920 portrait images |
| Intrinsics | Official per-frame PINHOLE calibration, fixed during geometric verification and BA |
| Retrieval | SALAD: 100 additional global neighbors + 30 temporal neighbors within each sequence |
| Detector / matcher | RaCo, up to 4,096 keypoints; detection resize 1,024; FlexGlue features at 512×512, 9 matcher layers |
| Pair filtering | FlexGlue covisibility threshold 0.9; early stopping disabled |
| Prior | Solv3R, 16 calls; maximum 25 neighbors per star before connectivity restoration |
| Global fusion | Rotation and star similarity averaging; temporal constraint weights multiplied by 2 |
| Refinement | Two rounds of triangulation, track completion/merging, retriangulation, and GlueMap-style BA/filtering |
| BA | torchls, FP64, joint rotation/center blocks, full-system Warp PCG, block-Jacobi preconditioning |
| Stopping settings | LM: at most 200 accepted iterations, cost tolerance `1e-6`; PCG: at most 300 iterations, relative tolerance `1e-5` |

The star selection preserves temporal coverage and reserves space for global neighbors. Retrieval produces **649,518 unique pairs**, including **115,335 temporal pairs**. Existing features and **200,017 matching pairs** are reused; **449,501 pairs** require new matching. The run builds **7,552 local stars**.

Local Solv3R inference and depth-consistency checks use the model's predicted intrinsics. Official intrinsics are applied to the global output, subsequent geometric verification, and BA. Ground-truth poses are used only for offline evaluation.

Before refinement, calibrated Warp E/F/H verification processes **105,003 candidate pairs / 89,519,571 matches** and retains **104,810 pairs / 80,659,731 inlier matches**. Its settings are 64 pairs per batch, 240 samples per sampling batch, a 4 px error threshold, and at least 15 inliers.

Refinement uses **only real FlexGlue tracks**, with Huber loss; no VGGSfM tracks or virtual tracks are used. It follows the GlueMap-style two-round controller rather than the GLOMAP BA controller. Each round performs one successful BA solve in this experiment, followed by observation filtering. No `torch.compile` is used.

## Accuracy and reconstruction size

The primary metric follows the public [GlueMap evaluator](https://github.com/colmap/gluemap/blob/adc9e4b/benchmark/evaluate.py): common registered images only, all directed image pairs including the diagonal, signed translation-direction errors, and joint error `max(rotation_error, translation_error)`. AUC is integrated up to 3° without pose alignment and reported as a percentage; higher is better.

| Metric | Before BA | After BA |
| --- | ---: | ---: |
| Registered images | 7,541 / 7,553 | 7,540 / 7,553 |
| AUC@3°, GlueMap evaluator conventions | 9.12% | **57.56%** |
| Evaluated directed pairs | 56,866,681 | 56,851,600 |
| Final 3D points | — | 4,181,148 |
| Final observations | — | 24,745,380 |
| Reprojection error, median | — | 2.686 px |
| Reprojection error, RMS | — | 5.144 px |


## Runtime

Times below are synchronized wall-clock measurements in seconds. Python imports and initial CUDA setup are excluded. Model loading, cache IO, and problem construction inside measured stages are included. Offline pose evaluation is excluded. Existing Warp compilation caches are available.

### Frontend and global initialization

| Stage | Seconds |
| --- | ---: |
| Retrieval | 222.73 |
| Feature cache loading | 2.72 |
| Matching | 15,894.22 |
| Star topology | 1.94 |
| Image preprocessing | 189.40 |
| Gathering image batches | 70.00 |
| Solv3R local inference | 5,292.80 |
| Depth consistency | 791.46 |
| Compact star serialization | 51.59 |
| Global fusion | 135.25 |
| **Measured wall time through saved global poses** | **22,660.76 (6 h 17 min 41 s)** |

The wall time also includes remaining orchestration, model loading, and visualization overhead, so it is not the exact sum of the listed stage timers. The feature row measures cache access, **not feature extraction**; matching is also partially cached.
