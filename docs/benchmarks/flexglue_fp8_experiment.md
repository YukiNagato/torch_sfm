# FlexGlue FP8 experiments on ETH3D and LaMAR

Both experiments used an NVIDIA RTX 4090 D (24 GB), PyTorch 2.11.0+cu128, and a covisibility threshold of 0.9. The FP8 matcher used `torch.compile(mode="reduce-overhead")` in both runs.

## ETH3D courtyard matching

On all 703 image pairs in ETH3D courtyard, the compiled FP8 backend matched **1.66× faster** than compiled original FlexGlue with all nine layers enabled. Early exit reduced FP8 matching time by a further **11.7%**.

The ETH3D host had an Intel Xeon Platinum 8474C. All configurations used the same 38 undistorted images at 1920×1080, cached RaCo keypoints (up to 4,096 per image), and FlexGlue descriptors extracted at 512×512. Both backends were compiled in the same mode. The FP8 backend included CuTe attention and the `cute-precise` matching head.

### Matching results

| Matcher | Time for 703 pairs | Time per pair | Rejected pairs | Retained matches |
| --- | ---: | ---: | ---: | ---: |
| Original FlexGlue | 16.931 s | 24.084 ms | 321 | 366,682 |
| FP8 FlexGlue | 10.218 s | 14.535 ms | 321 | 366,687 |
| FP8 FlexGlue with early exit (decay 0.5) | **9.027 s** | **12.840 ms** | 321 | 366,687 |

Times are medians of three full passes after warmup. Each pass started with an empty 32-image GPU feature cache. Timing covers feature transfers, matching, pair filtering, and returning results to the CPU. Feature extraction, disk I/O, model setup, and warmup are excluded. The early-exit measurement used the full-depth FP8 baseline from the preceding run.

The two full-depth backends rejected exactly the same pairs. Their retained match indices had **95.05% Jaccard agreement**. With early exit, FP8 rejected 296 pairs (42.1% of all pairs) before layer nine. These measurements cover matching only.

## LaMAR HGE reconstruction

On LaMAR HGE, compiled FP8 FlexGlue matched **649,518 image pairs in 2 h 24 min**. After two rounds of GlueMap bundle adjustment (BA), pose **AUC@3° was 57.507%**, compared with 57.555% for [the original FlexGlue run](lamar_hge_gluemap.md).

The dataset contains 7,553 images across 17 iOS mapping sequences. SALAD retrieval uses 100 global and 30 temporal neighbors. The frontend uses RaCo keypoints, 512×512 FlexGlue descriptor inputs, the FP8 v0.1 matcher, and early exit with `decay=0.5`. Tiled scheduling and Belady eviction manage a GPU cache of 1,024 images. Reconstruction uses Solv3R and fixed official intrinsics.

### Frontend runtime

| Stage | Wall time |
| --- | ---: |
| Feature extraction | 475.2 s |
| Model setup and warmup | 17.8 s |
| Matching | **8,671.2 s** |

Matching includes feature transfers and writing results. All pairs were computed, averaging **74.9 pairs/s**. Early exit stopped 53.3% of pairs and the matcher rejected 66.2% in total. The original run computed 449,501 pairs and read another 200,017 from cache in 15,894.2 s. Its effective rate was 28.3 new pairs/s, so the **2.65× throughput ratio** is a comparison between the two runs with different cache conditions.

### Reconstruction accuracy

| Metric | Original FlexGlue | Compiled FP8 |
| --- | ---: | ---: |
| Pose AUC@3° before BA | 9.120% | 9.103% |
| Pose AUC@3° after BA | 57.555% | 57.507% |
| Registered images after BA | 7,540 / 7,553 | 7,537 / 7,553 |
| Pose AUC@3° after BA on the same 7,537 images | 57.579% | 57.507% |

AUC follows the GlueMap evaluator and uses the larger of the relative rotation and translation direction errors for each directed image pair. The FP8 refinement excluded a disconnected component of two cameras and 84 observations before BA.
