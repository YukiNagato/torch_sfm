"""Run RaCo + FlexGlue with fixed intrinsics and the existing GLOMAP pipeline.

Install optional frontends (the tested revisions):
  pip install 'git+https://github.com/cvg/RaCo.git@ee06fa68dbb20303f334fbc257dad71d4149603b'
  pip install 'git+https://github.com/cvg/FlexGlue.git@c6cbc7a933bb70ab58608e8835d99b6cc4adca1a'

Workspace contains dataset.json with an ordered image list and camera definitions.
PINHOLE params are [fx, fy, cx, cy]. Images sharing a string cam_idx use the same camera.
Run: python examples/flexglue_glomap.py /path/to/workspace

Optional FP8 matcher (the descriptor still uses the original float weights):
  pip install -e '.[salad,flexglue,flexglue-fp8]'
  python examples/flexglue_glomap.py /path/to/workspace --backend fp8
The first FP8 match downloads the release checkpoint. Use --quantized-weights
/path/to/model_fp8.pt to override it with a local checkpoint.

The filesystem cache stores keypoints, original image size and dense float32
features once per image. Matching never decodes images. Set
--matching-schedule tiled_belady to group pairs for feature reuse and evict the
image needed furthest in the future. --gpu-cache-images sets the capacity.
No torch.compile is used. Scores and rejection flags persist in pair
cache metadata. Changing the covisibility threshold invalidates matches;
changing detector/backbone settings also invalidates features. Use a new
workspace when changing the image list or replacing model weights in place.
"""
import argparse
from pathlib import Path

from torch_sfm.correspondence import FlexGlueConfig
from torch_sfm.geometry import TwoViewConfig, BundleAdjustmentConfig
from torch_sfm.pipeline.glomap import GlomapConfig, GlomapPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('--manifest', type=Path, help='Path to the input JSON (default: workspace/dataset.json)')
    parser.add_argument('--min-covisibility', type=float, default=.9,
                        help='Set 0 to disable pair rejection')
    parser.add_argument('--max-keypoints', type=int, default=4096)
    parser.add_argument('--matching-schedule', choices=('original', 'tiled_belady'), default='original')
    parser.add_argument('--gpu-cache-images', type=int, default=32)
    parser.add_argument('--backend', choices=('float', 'fp8'),
                        help='Use fp8 to download the default FP8 matcher checkpoint on first use')
    parser.add_argument('--quantized-weights', type=str,
                        help='Use a local FP8 matcher checkpoint (implies --backend fp8 if omitted)')
    args = parser.parse_args()
    backend = args.backend or ('fp8' if args.quantized_weights is not None else 'float')
    config = GlomapConfig(
        correspondence_method='flexglue',
        flexglue=FlexGlueConfig(max_keypoints=args.max_keypoints,
                               min_covisibility=args.min_covisibility,
                               matching_schedule=args.matching_schedule,
                               gpu_cache_images=args.gpu_cache_images,
                               backend=backend,
                               quantized_weights=args.quantized_weights),
        retrieval_top_k=40,
        cache_backend='filesystem', cache_frame_images=False,
        geometry_device='cuda', calibration={'mode': 'trusted'},
        two_view=TwoViewConfig(backend='warp', pair_batch_size=512),
        bundle_adjustment=BundleAdjustmentConfig(refine_focal_length=False,
                                                 refine_principal_point=False),
    )
    with GlomapPipeline(config, args.workspace, manifest=args.manifest) as pipeline:
        result = pipeline.run()
        pipeline.visualize()
        print(pipeline.matching_stats)
        print(f'{int(result.registered.sum())}/{len(result.registered)} cameras registered')


if __name__ == '__main__':
    main()
