"""Fuse Solv3R stars, then refine FlexGlue and virtual tracks with augmented BA.

Workspace needs images.txt, one absolute path per line; intrinsics.txt is not
required. Reuses GLOMAP's RaCo/FlexGlue filesystem caches when settings match.
Install with pip install -e '.[salad,flexglue,solv3r]'. Solv3R needs Python 3.11/3.12.

python examples/flexglue_gluemap.py --workspace /path/to/workspace
python examples/flexglue_gluemap.py --help
python examples/flexglue_gluemap.py --workspace /path/to/workspace --fuse-only
python examples/flexglue_gluemap.py --workspace /path/to/workspace --refine-only
python examples/flexglue_gluemap.py --workspace /path/to/workspace --config.intrinsics-source fixed --refine-only
python examples/flexglue_gluemap.py --workspace /path/to/workspace --config.retrieval-sequences /path/to/sequences.json --config.retrieval-top-k 100 --config.no-refine --config.no-save-virtual-candidates

Outputs: gluemap/summary.json and stars/<center>.pt, containing input IDs, local
c2w poses, predicted K, overlap/support scores, parallax statistics and sparse
virtual-point candidates. Dense depth maps and pixel masks are not persisted.
K/4px threshold refer to Solv3R's resized images, not original pixels.
Global results: global_poses.pt, global_summary.json, global_poses.html. Local
candidates retain their star gauges; star_to_world maps them into global space.
Depth sampling settings are baked into the cache; legacy dense caches still load.
Sequence retrieval reads JSON start/end ranges aligned with images.txt (end
exclusive, chronological order inside each range). It adds 30 temporal neighbors
per image by default, plus retrieval_top_k additional global neighbors excluding
that local window. Stars preserve temporal coverage and reserve non-temporal
slots; fusion multiplies temporal constraint scores by fusion.sequential_weight.
Pair and descriptor caches are separate from legacy retrieval.txt and validate
their image order, method and pairing configuration before reuse.
Refined results: reconstruction.pt/html, sparse/0, refinement_summary.json.
GLOMAP E/F/H verification filters individual matches before triangulation, using
fused K. Configure with --config.two-view; --config.no-verify-matches is an ablation.
Geometry is cached separately in output/cache; match_verification.json records counts/time.
Fixed calibration reads intrinsics.txt (fx fy cx cy at original resolution) and
disables focal/principal-point optimization. The model's local depth checks keep
their predicted calibration. The saved star transforms describe pre-BA poses only.
"""
from pathlib import Path

import tyro

from torch_sfm.pipeline.gluemap import GluemapConfig, GluemapPipeline


def main(workspace: Path, config: GluemapConfig, fuse_only: bool = False, refine_only: bool = False):
    if fuse_only and refine_only:
        raise ValueError('Choose only one resume stage')
    with GluemapPipeline(config, workspace) as pipeline:
        if refine_only:
            report = pipeline.refine_saved_poses()
            print(f"{report['registered']} cameras, {report['points']} points")
            print(report['timings_seconds'])
            print(pipeline.output / 'reconstruction.html')
            return
        if fuse_only:
            report = pipeline.fuse_saved_stars()
            print(f"{report['registered']} cameras registered")
            print(report['timings_seconds'])
            print(pipeline.output / 'global_poses.html')
            return
        result = pipeline.run()
        print(f"{result['num_stars']} stars; {result['num_accepted_edges']} accepted edges")
        print(result['timings_seconds'])
        if 'refinement' in result:
            print(pipeline.output / 'reconstruction.html')
        print(pipeline.output / 'summary.json')


if __name__ == '__main__':
    tyro.cli(main)
