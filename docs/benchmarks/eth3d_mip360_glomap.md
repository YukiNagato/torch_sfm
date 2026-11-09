# ETH3D and Mip-NeRF 360: TorchSfM / COLMAP GLOMAP Runtime and Accuracy

All runs use the same **SALAD + LoMa** frontend inputs to compare TorchSfM with COLMAP global mapping (GLOMAP).

## Hardware and Software

| Component | Configuration |
| --- | --- |
| CPU | Intel Xeon Platinum 8474C; container CPU quota equivalent to 16 cores |
| GPU | NVIDIA GeForce RTX 4090 D, 24 GB VRAM |
| COLMAP CPU allocation | Affinity 0–15, 16 threads; identical for both BA backends |
| PyTorch | 2.11.0+cu128 |
| COLMAP | 4.3.0, commit `ed8080bcf42ef0e42d3f5d0cd21eeff0698dfde8`, built locally with Caspar enabled |

TorchSfM uses Warp geometric verification, Warp PCG, Tensor/Warp triangulation, and torchls BA. Position averaging and BA use full-system PCG. BA uses a block-Jacobi preconditioner with joint rotation/center blocks and adaptive restarts.

The Ceres configuration uses CPU BA. **The Caspar configuration follows COLMAP's native mixed pipeline**: joint BA uses Caspar on the GPU, fixed-rotation BA automatically uses Ceres, and rotation and position averaging remain on the CPU. The two COLMAP configurations were checked to match except for the BA backend and Caspar GPU index.

## Data and Methodology

### Images, Features, and Intrinsics

| Dataset | Scenes / images | Input resolution | Intrinsics / reference poses |
| --- | --- | --- | --- |
| ETH3D MVS DSLR | 25 / 898; 7–110 images per scene | Official undistorted images, resized directly to 1920×1080 | Official calibrated intrinsics and poses |
| Mip-NeRF 360 | 9 / 1,940; 125–311 images per scene | Outdoor `images_4` (~1.0 MP), indoor `images_2` (~1.6 MP) | COLMAP intrinsics and poses provided with the dataset |

ETH3D images are resized independently along each axis, without preserving the original aspect ratio; `fx/cx` and `fy/cy` are scaled accordingly. Mip360 intrinsics are adjusted to the actual dimensions of the selected images. All methods keep these intrinsics fixed, with no focal-length or principal-point optimization. Reference poses are used only for evaluation, never as solver inputs.

The shared frontend uses **SALAD retrieval top-k=40** (capped at N−1 neighbors for small scenes, similarity threshold=0.5) and **LoMa-B feature extraction and matching with 2,048 keypoints per image**. TorchSfM runs first and exports a COLMAP database; COLMAP then reuses the same retrieval pairs, raw keypoints, and raw matches.

COLMAP performs its own geometric verification, without reusing TorchSfM inliers or two-view poses. Both use calibrated verification with a 1 px error threshold, at least 30 inliers, a minimum inlier ratio of 0.25, and random seed 0. Different RANSAC implementations can still produce different models and inliers. TorchSfM uses a geometric verification pair batch size of 512 and a sampling batch size of 240 per round.

### Timing Scope and Comparability

**Comparable stage total = geometric verification + rotation averaging + position averaging (including track building) + triangulation and BA.**

Triangulation and BA includes iterative BA, retriangulation, and track refinement within that stage. The tables exclude retrieval, feature extraction, matching, process startup, Python imports, CUDA context initialization, offline pose evaluation, and visualization. Native mapper loading, graph preprocessing, coloring, and export are also excluded from the four-stage total. Problem construction and cache IO occurring within timed stages are included.

TorchSfM stage timings include CUDA synchronization and use an existing Warp compilation cache. Caspar initializes its CUDA context before timing; the first solve and each problem construction are included, with synchronization at the end of mapping. Results were not selected by repeatedly solving the same problem and reporting the fastest run.

Each solver retains its own stopping criteria and loss implementation. Final tracks, observations, and BA problems can differ.

### Accuracy Metrics

All registered camera centers participate in a single least-squares Sim(3) alignment, without trimming outlier cameras. Position error is defined as:

```text
Position NRMSE (%) = 100 × RMS(aligned estimated camera centers − reference camera centers)
                        / RMS(reference camera centers − their centroid)
```

Rotation error uses the rotation from the same alignment and is reported as the median angular error. Lower is better for both metrics. ETH3D uses official calibrated poses, but this is not the official ETH3D evaluation protocol. Mip360 references are the COLMAP reconstructions provided with the dataset, **not independent ground truth**; these metrics measure agreement with that reference. Full registration does not imply correct structure.

## ETH3D

Both methods register **898 / 898** images, producing one model per scene. Caspar has not been tested.

### Aggregate Stage Runtime

Times are in seconds, summed over all 25 scenes.

| Stage | TorchSfM | COLMAP Ceres CPU (16 threads) |
| --- | --- | --- |
| Geometric verification | 104.96 | 133.10 |
| Rotation averaging | 5.62 | 0.46 |
| Position averaging (including track building) | 79.43 | 56.23 |
| Triangulation + BA | 195.81 | 133.52 |
| Total | 385.82 | 323.31 |

### Per-Scene Runtime and Pose Error

Each cell lists **TorchSfM / Ceres CPU**. Runtime is the four-stage total in seconds; position NRMSE is a percentage. All scenes are included, including reconstructions with large deviations from the reference.

| Scene | Images | Comparable stage total (s) | Position NRMSE (%) |
| --- | --- | --- | --- |
| courtyard | 38 | 18.75 / 14.38 | 75.318 / 75.315 |
| delivery_area | 44 | 18.33 / 18.29 | 0.085 / 0.085 |
| electro | 45 | 19.73 / 15.02 | 0.129 / 0.127 |
| facade | 76 | 26.67 / 28.10 | 96.057 / 95.257 |
| kicker | 31 | 10.96 / 9.83 | 0.057 / 0.060 |
| meadow | 15 | 13.41 / 2.54 | 0.438 / 0.417 |
| office | 26 | 10.10 / 5.40 | 0.608 / 0.632 |
| pipes | 14 | 8.49 / 3.98 | 0.161 / 0.154 |
| playground | 38 | 14.40 / 12.03 | 0.129 / 0.125 |
| relief | 31 | 16.03 / 11.08 | 21.380 / 10.151 |
| relief_2 | 31 | 12.02 / 10.72 | 0.081 / 0.079 |
| terrace | 23 | 11.40 / 6.75 | 0.125 / 0.123 |
| terrains | 42 | 19.05 / 14.97 | 0.074 / 0.069 |
| botanical_garden | 30 | 15.70 / 10.01 | 0.135 / 0.151 |
| boulders | 26 | 13.31 / 7.83 | 0.157 / 0.165 |
| bridge | 110 | 40.83 / 57.02 | 93.829 / 93.836 |
| door | 7 | 4.87 / 1.86 | 0.034 / 0.034 |
| exhibition_hall | 68 | 27.88 / 28.85 | 97.407 / 95.544 |
| lecture_room | 23 | 11.79 / 6.78 | 0.194 / 78.105 |
| living_room | 65 | 20.39 / 18.07 | 55.798 / 54.683 |
| lounge | 10 | 9.77 / 2.30 | 47.208 / 45.330 |
| observatory | 27 | 9.54 / 8.62 | 0.475 / 0.482 |
| old_computer | 54 | 18.66 / 19.38 | 93.857 / 93.866 |
| statue | 11 | 6.87 / 4.64 | 0.024 / 0.027 |
| terrace_2 | 13 | 6.89 / 4.84 | 0.144 / 0.146 |

## Mip-NeRF 360

All three methods register **1,940 / 1,940** images.

### Aggregate Stage Runtime

Times are in seconds, summed over all 9 scenes.

| Stage | TorchSfM | COLMAP Ceres CPU (16 threads) | COLMAP Caspar |
| --- | --- | --- | --- |
| Geometric verification | 186.58 | 248.84 | 249.40 |
| Rotation averaging | 2.46 | 1.10 | 1.09 |
| Position averaging (including track building) | 38.16 | 31.49 | 31.14 |
| Triangulation + BA | 181.47 | 347.69 | 182.61 |
| Total | 408.67 | 629.13 | 464.24 |

### Per-Scene Runtime

Each cell lists **TorchSfM / Ceres CPU / Caspar**, in seconds. Triangulation + BA is already included in the comparable stage total.

| Scene | Images | Comparable stage total | Of which: triangulation + BA |
| --- | --- | --- | --- |
| bicycle | 194 | 37.17 / 54.79 / 40.58 | 18.89 / 28.75 / 14.17 |
| bonsai | 292 | 60.52 / 96.73 / 69.10 | 25.48 / 54.80 / 28.06 |
| counter | 240 | 58.47 / 104.95 / 80.78 | 24.73 / 42.82 / 18.28 |
| flowers | 173 | 35.52 / 46.46 / 40.99 | 14.74 / 19.17 / 13.79 |
| garden | 185 | 35.77 / 58.22 / 30.41 | 17.17 / 44.70 / 18.34 |
| kitchen | 279 | 58.68 / 91.65 / 62.36 | 28.12 / 59.75 / 30.21 |
| room | 311 | 65.72 / 101.27 / 74.90 | 28.75 / 63.22 / 36.07 |
| stump | 125 | 28.17 / 42.73 / 36.98 | 10.87 / 17.13 / 10.90 |
| treehill | 141 | 28.65 / 32.33 / 28.14 | 12.71 / 17.37 / 12.80 |

### Deviation from Reference Poses

Each cell lists **TorchSfM / Ceres CPU / Caspar**.

| Scene | Position NRMSE (%) | Median rotation error (°) |
| --- | --- | --- |
| bicycle | 0.157 / 0.178 / 0.197 | 0.026 / 0.026 / 0.035 |
| bonsai | 0.155 / 0.155 / 0.164 | 0.119 / 0.117 / 0.126 |
| counter | 0.030 / 0.027 / 0.079 | 0.031 / 0.029 / 0.039 |
| flowers | 0.226 / 0.227 / 0.231 | 0.098 / 0.094 / 0.086 |
| garden | 0.036 / 0.037 / 0.043 | 0.028 / 0.027 / 0.034 |
| kitchen | 0.085 / 0.081 / 0.098 | 0.057 / 0.056 / 0.066 |
| room | 0.082 / 0.082 / 0.103 | 0.061 / 0.051 / 0.066 |
| stump | 0.127 / 0.126 / 0.142 | 0.042 / 0.039 / 0.046 |
| treehill | 0.619 / 0.642 / 0.652 | 0.108 / 0.107 / 0.135 |


Caspar logs contain 81 solves: 34 reached the limit of 200 outer iterations, and 47 stopped at the damping limit. PCG allows at most 20 iterations per outer iteration, with initial damping 1.0; these defaults were retained. These exits do not establish convergence to a common accuracy target. Caspar has a higher median reprojection error than Ceres in every scene and slightly larger deviations from the reference poses:

| Scene | Ceres median reprojection error (px) | Caspar median reprojection error (px) |
| --- | --- | --- |
| bicycle | 0.708 | 0.924 |
| bonsai | 0.868 | 1.012 |
| counter | 0.695 | 0.855 |
| flowers | 0.842 | 1.113 |
| garden | 0.515 | 0.630 |
| kitchen | 0.825 | 1.005 |
| room | 0.749 | 0.966 |
| stump | 0.663 | 0.820 |
| treehill | 0.727 | 1.009 |

Final point counts are **195,225** for TorchSfM, **171,061** for Ceres, and **150,833** for Caspar; observation counts are **2,351,711 / 2,186,563 / 2,107,069**, respectively. Reprojection errors are measured over each method's retained observations, not an identical set of residuals.
