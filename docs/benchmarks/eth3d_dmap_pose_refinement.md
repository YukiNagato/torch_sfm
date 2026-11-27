# DMap-style pose refinement on ETH3D

## Hardware and evaluation

The experiments ran on one NVIDIA GeForce RTX 4090 D (24 GB) with an Intel Xeon Platinum 8474C, a 16-core CPU quota, and PyTorch 2.11.0 (CUDA 12.8).

All configurations cover the same 25 ETH3D DSLR scenes and all 898 images. The undistorted images were resized to 1920 × 1080, and the official intrinsics were scaled to that resolution and held fixed. Reference poses were used only for evaluation.

AUC@1° and AUC@3° are percentages, averaged equally across scenes. Within each scene, we evaluate all unordered image pairs. A pair's error is the larger of its relative rotation error and its sign-invariant translation-direction error. We integrate the recall curve up to each angular threshold and normalize by that threshold. Higher is better.

## Method

DMap uses features from a frozen multi-view backbone for both initialization and dense matching, while TorchSfM uses VGGT-Ω for the initial poses and depths, then obtains correspondences from a separate pairwise matcher like RoMA or Flexglue. A second difference is our three-round bundle adjustment (BA) schedule, which progressively filters observations on the same grid. [DMap paper](https://openreview.net/forum?id=iikGG5Se3J)

For each source image, we sample reference views using edge leverage scores on the predicted frustum-overlap graph (seed 0). Both matchers use a 64 × 64 source grid, bidirectional consistency, and a 16 px radius around RaCo keypoints at both match endpoints. FlexGlue uses 640 × 640 descriptor inputs, a 0.90 pair covisibility threshold and a 0.75 match confidence threshold. RoMa v2 uses its 512 × 512 fast setting with bidirectional prediction and a 0.75 overlap threshold. All pixel thresholds below use a common 512 × 512 coordinate system.

Each BA round jointly refines camera poses and per-cell log depths with confidence-weighted reprojection residuals, a 3 px Cauchy loss and a uniform log-depth prior. Each round allows up to 100 LM iterations. Later rounds start from the preceding solution and reuse the original matches.

## 1. Matcher choice

Both matchers receive the same 5,543 candidate image pairs, with up to eight references per source, and use the same three-round BA schedule. Pair counts are unique unordered image pairs summed across scenes. A matched pair has at least one retained correspondence.

| Matcher | Matched pairs | AUC@1° (%) | AUC@3° (%) |
| --- | ---: | ---: | ---: |
| FlexGlue | 5,215 | 85.60 | 94.72 |
| RoMa v2 fast | 5,371 | **87.51** | **95.52** |

Switching to RoMa v2 improves mean AUC@1° by **1.91 percentage points** and AUC@3° by **0.80 points**.

## 2. Number of BA rounds with RoMa v2

Each row evaluates the poses after that round of the same eight-reference run. The observation count is the total number of directed correspondences entering the round.

| BA round | Reprojection filter before BA | Observations | AUC@1° (%) | AUC@3° (%) |
| --- | --- | ---: | ---: | ---: |
| 1 | All retained matcher observations | 8,286,562 | 85.10 | 94.43 |
| 2 | ≤ 2 px after round 1 | 8,197,116 | 86.56 | 95.13 |
| 3 | ≤ 1 px after round 2 | 8,007,443 | **87.51** | **95.52** |

Filtering and refining twice more improves AUC@1° by **2.41 points** and AUC@3° by **1.09 points** over one round. The third round alone adds **0.95 points** at 1°.

## 3. Number of reference views with FlexGlue

The matcher, initialization and three-round BA schedule are fixed. Eight and sixteen are upper bounds on the number of references sampled per source image. Each budget samples its own graph, so individual pairs can change as the budget grows.

| Max references per source | Candidate pairs | Matched pairs | AUC@1° (%) | AUC@3° (%) |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 5,543 | 5,215 | 85.60 | 94.72 |
| 16 | 8,670 | 8,025 | **86.12** | **94.93** |

The larger budget evaluates **56% more candidate pairs** for a mean gain of **0.53 points** at 1° and **0.21 points** at 3°. With this pair-selection strategy, doubling the FlexGlue reference budget yields a smaller accuracy gain than switching matchers at the eight-reference budget.

## Per-scene results

Each accuracy cell gives **AUC@1° / AUC@3°** in percent. Differences are in percentage points and use unrounded scores. The last row sums counts and averages accuracy across scenes.

### Matcher choice at eight references and three BA rounds

| Scene | Images | FlexGlue | RoMa v2 fast | Δ RoMa − FlexGlue |
| --- | ---: | ---: | ---: | ---: |
| botanical_garden | 30 | 85.58 / 95.15 | 89.59 / 96.51 | +4.00 / +1.36 |
| boulders | 26 | 84.82 / 94.85 | 89.86 / 96.57 | +5.04 / +1.72 |
| bridge | 110 | 89.38 / 96.40 | 91.57 / 97.16 | +2.19 / +0.75 |
| courtyard | 38 | 89.27 / 96.42 | 91.55 / 97.18 | +2.28 / +0.76 |
| delivery_area | 44 | 88.20 / 96.05 | 92.30 / 97.43 | +4.11 / +1.38 |
| door | 7 | 87.26 / 92.58 | 86.81 / 95.93 | -0.45 / +3.36 |
| electro | 45 | 85.40 / 94.68 | 89.62 / 96.27 | +4.22 / +1.59 |
| exhibition_hall | 68 | 82.44 / 92.98 | 83.41 / 93.22 | +0.97 / +0.24 |
| facade | 76 | 87.42 / 95.67 | 92.10 / 97.36 | +4.68 / +1.70 |
| kicker | 31 | 91.38 / 97.12 | 92.98 / 97.63 | +1.60 / +0.51 |
| lecture_room | 23 | 86.45 / 95.36 | 86.93 / 95.50 | +0.48 / +0.14 |
| living_room | 65 | 85.57 / 94.82 | 88.39 / 95.92 | +2.82 / +1.09 |
| lounge | 10 | 80.47 / 93.49 | 78.22 / 92.52 | -2.25 / -0.97 |
| meadow | 15 | 80.69 / 93.30 | 81.01 / 93.46 | +0.32 / +0.16 |
| observatory | 27 | 72.83 / 90.15 | 70.36 / 89.25 | -2.47 / -0.90 |
| office | 26 | 61.41 / 82.36 | 60.50 / 82.24 | -0.91 / -0.12 |
| old_computer | 54 | 79.79 / 92.87 | 82.50 / 93.88 | +2.71 / +1.01 |
| pipes | 14 | 86.93 / 95.64 | 87.83 / 96.01 | +0.90 / +0.37 |
| playground | 38 | 89.25 / 96.28 | 89.62 / 96.44 | +0.37 / +0.16 |
| relief | 31 | 89.36 / 96.44 | 93.77 / 97.93 | +4.41 / +1.49 |
| relief_2 | 31 | 91.63 / 97.20 | 94.96 / 98.33 | +3.33 / +1.12 |
| statue | 11 | 96.45 / 98.82 | 97.29 / 99.10 | +0.84 / +0.28 |
| terrace | 23 | 92.07 / 97.36 | 94.43 / 98.14 | +2.36 / +0.79 |
| terrace_2 | 13 | 88.71 / 96.33 | 89.33 / 96.38 | +0.62 / +0.05 |
| terrains | 42 | 87.17 / 95.56 | 92.83 / 97.58 | +5.66 / +2.03 |
| **Overall** | 898 | 85.60 / 94.72 | 87.51 / 95.52 | +1.91 / +0.80 |

### RoMa v2 by BA round at eight references

| Scene | Round 1 | Round 2 | Round 3 | Δ round 3 − round 1 |
| --- | ---: | ---: | ---: | ---: |
| botanical_garden | 87.66 / 95.83 | 88.35 / 96.09 | 89.59 / 96.51 | +1.93 / +0.67 |
| boulders | 89.14 / 96.35 | 89.38 / 96.42 | 89.86 / 96.57 | +0.72 / +0.23 |
| bridge | 87.98 / 95.85 | 90.63 / 96.83 | 91.57 / 97.16 | +3.59 / +1.31 |
| courtyard | 91.03 / 97.01 | 91.22 / 97.08 | 91.55 / 97.18 | +0.52 / +0.17 |
| delivery_area | 90.89 / 96.96 | 91.32 / 97.11 | 92.30 / 97.43 | +1.41 / +0.47 |
| door | 85.38 / 94.61 | 86.22 / 95.37 | 86.81 / 95.93 | +1.43 / +1.32 |
| electro | 88.19 / 95.74 | 88.84 / 95.98 | 89.62 / 96.27 | +1.43 / +0.52 |
| exhibition_hall | 81.65 / 92.54 | 82.98 / 93.04 | 83.41 / 93.22 | +1.76 / +0.68 |
| facade | 91.51 / 97.17 | 91.84 / 97.27 | 92.10 / 97.36 | +0.60 / +0.20 |
| kicker | 90.56 / 96.80 | 91.71 / 97.20 | 92.98 / 97.63 | +2.42 / +0.83 |
| lecture_room | 85.39 / 94.93 | 86.34 / 95.27 | 86.93 / 95.50 | +1.54 / +0.57 |
| living_room | 84.39 / 94.50 | 86.82 / 95.35 | 88.39 / 95.92 | +4.01 / +1.42 |
| lounge | 69.58 / 89.15 | 74.39 / 91.13 | 78.22 / 92.52 | +8.64 / +3.37 |
| meadow | 80.15 / 93.16 | 80.96 / 93.44 | 81.01 / 93.46 | +0.86 / +0.30 |
| observatory | 70.36 / 89.29 | 70.31 / 89.25 | 70.36 / 89.25 | 0.00 / -0.04 |
| office | 48.44 / 73.29 | 57.18 / 80.29 | 60.50 / 82.24 | +12.06 / +8.95 |
| old_computer | 82.33 / 93.84 | 82.51 / 93.90 | 82.50 / 93.88 | +0.17 / +0.04 |
| pipes | 82.62 / 93.94 | 85.53 / 95.11 | 87.83 / 96.01 | +5.21 / +2.07 |
| playground | 87.46 / 95.70 | 88.50 / 96.05 | 89.62 / 96.44 | +2.16 / +0.74 |
| relief | 90.78 / 96.93 | 92.39 / 97.47 | 93.77 / 97.93 | +2.99 / +1.00 |
| relief_2 | 90.86 / 96.93 | 94.14 / 98.04 | 94.96 / 98.33 | +4.10 / +1.39 |
| statue | 96.98 / 98.99 | 97.17 / 99.06 | 97.29 / 99.10 | +0.31 / +0.10 |
| terrace | 94.43 / 98.14 | 94.44 / 98.15 | 94.43 / 98.14 | 0.00 / 0.00 |
| terrace_2 | 88.88 / 96.21 | 88.96 / 96.25 | 89.33 / 96.38 | +0.44 / +0.17 |
| terrains | 90.91 / 96.89 | 91.79 / 97.22 | 92.83 / 97.58 | +1.92 / +0.69 |
| **Overall** | 85.10 / 94.43 | 86.56 / 95.13 | 87.51 / 95.52 | +2.41 / +1.09 |

### FlexGlue by reference budget after three BA rounds

| Scene | Candidate pairs (8) | Candidate pairs (16) | 8 references | 16 references | Δ 16 − 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| botanical_garden | 150 | 185 | 85.58 / 95.15 | 86.55 / 95.49 | +0.97 / +0.34 |
| boulders | 140 | 190 | 84.82 / 94.85 | 86.36 / 95.34 | +1.55 / +0.49 |
| bridge | 793 | 1,459 | 89.38 / 96.40 | 90.67 / 96.88 | +1.29 / +0.48 |
| courtyard | 232 | 348 | 89.27 / 96.42 | 89.15 / 96.37 | -0.12 / -0.05 |
| delivery_area | 247 | 339 | 88.20 / 96.05 | 88.02 / 96.00 | -0.18 / -0.05 |
| door | 21 | 21 | 87.26 / 92.58 | 87.26 / 92.58 | 0.00 / 0.00 |
| electro | 267 | 409 | 85.40 / 94.68 | 85.39 / 94.77 | -0.01 / +0.10 |
| exhibition_hall | 451 | 720 | 82.44 / 92.98 | 82.30 / 92.85 | -0.14 / -0.12 |
| facade | 529 | 926 | 87.42 / 95.67 | 88.10 / 95.94 | +0.68 / +0.27 |
| kicker | 206 | 314 | 91.38 / 97.12 | 93.86 / 97.95 | +2.48 / +0.83 |
| lecture_room | 118 | 162 | 86.45 / 95.36 | 85.96 / 95.14 | -0.49 / -0.22 |
| living_room | 444 | 757 | 85.57 / 94.82 | 86.80 / 95.26 | +1.23 / +0.43 |
| lounge | 32 | 32 | 80.47 / 93.49 | 80.47 / 93.49 | 0.00 / 0.00 |
| meadow | 76 | 89 | 80.69 / 93.30 | 80.17 / 93.12 | -0.51 / -0.18 |
| observatory | 172 | 291 | 72.83 / 90.15 | 73.22 / 90.24 | +0.39 / +0.10 |
| office | 147 | 214 | 61.41 / 82.36 | 63.93 / 84.03 | +2.52 / +1.67 |
| old_computer | 342 | 548 | 79.79 / 92.87 | 79.61 / 92.76 | -0.19 / -0.11 |
| pipes | 66 | 72 | 86.93 / 95.64 | 86.64 / 95.55 | -0.29 / -0.10 |
| playground | 228 | 326 | 89.25 / 96.28 | 90.82 / 96.86 | +1.57 / +0.58 |
| relief | 186 | 294 | 89.36 / 96.44 | 89.87 / 96.61 | +0.50 / +0.17 |
| relief_2 | 179 | 273 | 91.63 / 97.20 | 91.97 / 97.32 | +0.34 / +0.12 |
| statue | 53 | 55 | 96.45 / 98.82 | 96.33 / 98.78 | -0.12 / -0.04 |
| terrace | 127 | 171 | 92.07 / 97.36 | 92.27 / 97.42 | +0.19 / +0.06 |
| terrace_2 | 69 | 78 | 88.71 / 96.33 | 88.02 / 96.09 | -0.69 / -0.24 |
| terrains | 268 | 397 | 87.17 / 95.56 | 89.31 / 96.31 | +2.14 / +0.75 |
| **Overall** | 5,543 | 8,670 | 85.60 / 94.72 | 86.12 / 94.93 | +0.53 / +0.21 |
