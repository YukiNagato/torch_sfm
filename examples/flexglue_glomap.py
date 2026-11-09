"""Run RaCo + FlexGlue with fixed intrinsics and the existing GLOMAP pipeline.

Install optional frontends (the tested revisions):
  pip install 'git+https://github.com/cvg/RaCo.git@ee06fa68dbb20303f334fbc257dad71d4149603b'
  pip install 'git+https://github.com/cvg/FlexGlue.git@c6cbc7a933bb70ab58608e8835d99b6cc4adca1a'

Workspace contains images.txt (one absolute image path per line) and
intrinsics.txt ([fx, fy, cx, cy], shared or one row per image).
Run: python examples/flexglue_glomap.py /path/to/workspace

The filesystem cache stores keypoints, original image size and dense float32
features once per image. Matching never decodes images. GPU LRU capacity is in
images. No torch.compile is used. Scores and rejection flags persist in pair
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
    parser.add_argument('--min-covisibility', type=float, default=.9,
                        help='Set 0 to disable pair rejection')
    parser.add_argument('--max-keypoints', type=int, default=4096)
    args = parser.parse_args()
    config = GlomapConfig(
        correspondence_method='flexglue',
        flexglue=FlexGlueConfig(max_keypoints=args.max_keypoints,
                               min_covisibility=args.min_covisibility),
        retrieval_top_k=40,
        cache_backend='filesystem', cache_frame_images=False,
        geometry_device='cuda', calibration={'mode': 'trusted'},
        two_view=TwoViewConfig(backend='warp', pair_batch_size=512),
        bundle_adjustment=BundleAdjustmentConfig(refine_focal_length=False,
                                                 refine_principal_point=False),
    )
    with GlomapPipeline(config, args.workspace) as pipeline:
        result = pipeline.run()
        pipeline.visualize()
        print(pipeline.matching_stats)
        print(f'{int(result.registered.sum())}/{len(result.registered)} cameras registered')


if __name__ == '__main__':
    main()
