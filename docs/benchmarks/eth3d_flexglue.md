# ETH3D: RaCo + FlexGlue in TorchSfM

This experiment compares **RaCo + FlexGlue** with the previous **LoMa-B** frontend in TorchSfM on all 25 ETH3D MVS DSLR scenes (898 images). Both runs use TorchSfM reconstruction; no COLMAP or Caspar reconstruction was run with FlexGlue.

## Hardware and Configuration

| Component | Configuration |
| --- | --- |
| CPU | Intel Xeon Platinum 8474C; process affinity 0–3 |
| GPU | NVIDIA GeForce RTX 4090 D, 24 GB VRAM |
| PyTorch | 2.11.0+cu128 |
| Images | Official undistorted images, resized directly to 1920×1080 |
| Intrinsics | Official calibration, scaled independently along each image axis and kept fixed |
| Retrieval | Same SALAD top-40 pairs as the previous LoMa run |
| Reconstruction | Warp geometric verification, Warp PCG, Tensor/Warp triangulation, torchls BA |

The reconstruction backend settings are inherited from the [LoMa experiment](eth3d_mip360_glomap.md). Geometric verification uses pair batch size 512, sampling batch size 240, a 1 px error threshold, and at least 30 inliers. Official poses are used only for evaluation.

| Frontend setting | LoMa-B baseline | RaCo + FlexGlue |
| --- | --- | --- |
| Keypoints per image | 2,048 | 4,096 |
| Detector input | LoMa-B defaults | RaCo, 1,024 px on the long side |
| Matching features | LoMa-B | FlexGlue backbone at 512×512 |
| Matcher | LoMa-B | Sparse matching, 9 layers |
| Covisibility filter | None | Keep scores ≥ 0.9 |
| Early stopping | — | Disabled |
| Cache backend | COLMAP database | Filesystem |

FlexGlue caches dense float32 feature maps once per image and uses a GPU LRU cache of up to 32 images. Rejected pairs return zero matches and skip geometric verification. The run uses eager execution without `torch.compile`.

This compares frontend configurations within the pipeline. Keypoint counts, features, matching, pair filtering, and cache backends all differ; it is not an isolated matcher or filter ablation.

## Runtime

Times are in seconds, summed over all 25 scenes. The total includes **feature extraction, matching, geometric verification, rotation averaging, position averaging (including track building), and triangulation + BA**. It excludes retrieval, export, and visualization. Model loading and cache IO inside timed stages are included; Python imports and CUDA context initialization are excluded. Runs use an existing Warp compilation cache, with one run per scene.

This total includes extraction and matching, unlike the four-stage total in the COLMAP comparison.

| Stage | LoMa-B | RaCo + FlexGlue |
| --- | --- | --- |
| Feature extraction | 465.68 | 158.38 |
| Matching | 805.65 | 621.34 |
| Geometric verification | 104.96 | 48.47 |
| Rotation averaging | 5.62 | 6.88 |
| Position averaging (including track building) | 79.43 | 77.54 |
| Triangulation + BA | 195.81 | 172.27 |
| Total | 1657.15 | 1084.87 |

The six-stage total decreases from **1,657.15 s to 1,084.87 s (34.5%)**. FlexGlue rejects **7,557 / 17,264 pairs (43.8%)** by covisibility. Including those rejected pairs, **7,740 pairs** have fewer matches than the geometric verification minimum and are skipped.

## Accuracy and Per-Scene Results

All **898 / 898** images are registered in both runs. Position NRMSE uses one least-squares Sim(3) alignment of all registered camera centers to official ETH3D poses, without trimming outliers:

```text
Position NRMSE (%) = 100 × RMS(aligned estimated camera centers − reference camera centers)
                        / RMS(reference camera centers − their centroid)
```

Lower is better. This is a diagnostic metric, not the official ETH3D evaluation protocol. Each paired cell below lists **LoMa-B / RaCo + FlexGlue**; runtime follows the six-stage definition above.

| Scene | Registered images | Runtime (s) | Position NRMSE (%) | FlexGlue rejected pairs |
| --- | --- | --- | --- | --- |
| courtyard | 38/38 | 80.53 / 46.66 | 75.318 / 0.202 | 321/703 |
| delivery_area | 44/44 | 88.01 / 54.33 | 0.085 / 0.213 | 577/932 |
| electro | 45/45 | 98.42 / 54.22 | 0.129 / 0.488 | 499/944 |
| facade | 76/76 | 174.63 / 98.64 | 96.057 / 0.115 | 666/1977 |
| kicker | 31/31 | 48.81 / 34.12 | 0.057 / 0.119 | 111/465 |
| meadow | 15/15 | 32.21 / 21.53 | 0.438 / 0.314 | 26/105 |
| office | 26/26 | 40.96 / 28.51 | 0.608 / 0.930 | 110/325 |
| pipes | 14/14 | 22.45 / 16.59 | 0.161 / 0.276 | 16/91 |
| playground | 38/38 | 66.38 / 46.78 | 0.129 / 0.281 | 363/703 |
| relief | 31/31 | 53.33 / 31.42 | 21.380 / 0.127 | 151/465 |
| relief_2 | 31/31 | 52.22 / 32.34 | 0.081 / 0.211 | 174/465 |
| terrace | 23/23 | 38.94 / 23.12 | 0.125 / 0.161 | 76/253 |
| terrains | 42/42 | 75.79 / 49.93 | 0.074 / 0.144 | 385/859 |
| botanical_garden | 30/30 | 56.17 / 33.42 | 0.135 / 0.227 | 252/435 |
| boulders | 26/26 | 45.91 / 28.97 | 0.157 / 0.373 | 130/325 |
| bridge | 110/110 | 210.22 / 147.78 | 93.829 / 0.048 | 1446/3012 |
| door | 7/7 | 14.44 / 11.26 | 0.034 / 0.099 | 0/21 |
| exhibition_hall | 68/68 | 116.22 / 81.73 | 97.407 / 0.432 | 930/1601 |
| lecture_room | 23/23 | 36.89 / 25.01 | 0.194 / 0.290 | 88/253 |
| living_room | 65/65 | 109.33 / 79.99 | 55.798 / 0.169 | 570/1539 |
| lounge | 10/10 | 21.06 / 19.60 | 47.208 / 34.181 | 16/45 |
| observatory | 27/27 | 43.69 / 27.62 | 0.475 / 0.615 | 12/351 |
| old_computer | 54/54 | 89.10 / 64.29 | 93.857 / 0.291 | 638/1262 |
| statue | 11/11 | 19.00 / 12.89 | 0.024 / 0.042 | 0/55 |
| terrace_2 | 13/13 | 22.44 / 14.13 | 0.144 / 0.162 | 0/78 |

The number of scenes with position NRMSE below 1% increases from **17 / 25 to 24 / 25**. Seven scenes with large baseline errors fall below that threshold; **lounge remains incorrect**, with NRMSE 34.181%. The 1% threshold is a reporting indicator, not a solver convergence criterion. Several scenes that already had low errors with LoMa show slightly higher errors with FlexGlue, so the improvement is primarily fewer large reconstruction failures rather than uniformly better pose accuracy.

## Reconstruction Size

| Metric | LoMa-B | RaCo + FlexGlue |
| --- | --- | --- |
| Final 3D points | 184,047 | 348,837 |
| Final observations | 1,019,816 | 1,892,868 |

Despite filtering more image pairs, RaCo + FlexGlue produces more points and observations while reducing triangulation + BA time. The runtime improvement therefore does not simply come from a smaller final reconstruction. These measurements do not isolate the contribution of the detector, matcher, or covisibility filter.
