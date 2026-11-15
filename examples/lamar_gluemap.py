"""Reconstruct LaMAR with RaCo/FlexGlue, Solv3R and two rounds of augmented BA.

Install: pip install -e '.[salad,flexglue,solv3r]'  # Python 3.11/3.12

Download the official benchmark archive, e.g.
https://cvg-data.inf.ethz.ch/lamar/benchmark/HGE.zip, and unzip it. Pass the map
session containing images.txt, sensors.txt and raw_data/ (not the query session):

python examples/lamar_gluemap.py --capture /datasets/HGE/sessions/map --workspace /work/HGE

This selects all iOS mapping images, preserving their original pixels and
per-frame PINHOLE calibration. No GT trajectories are used. The workspace gets:
  images.txt: absolute image paths sorted by sequence, timestamp, sensor ID.
  intrinsics.txt: one fx fy cx cy row per image, in exactly the same order.
  sequences.json: [{"name": "ios_...", "start": 0, "end": N, "count": N}, ...].
Sequence names come from sensor_id before '/'; ranges index images.txt and end
is exclusive. Images stay in raw_data/, so keep the extracted dataset in place.

Use --prepare-only to generate these files without running reconstruction.
Omit --capture to reuse an already prepared workspace.

Use --sequences to supply sequence ranges outside the workspace, --output to
choose the result directory, and --refine-only to resume saved global poses.
Model weights download automatically; --checkpoint and --flexglue.weights /
--flexglue.detector-weights select local files. Match the original weight paths
when reusing feature/match caches. Results: reconstruction.pt/html and sparse/0.
"""
import csv
from itertools import groupby
import json
import os
from pathlib import Path

os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
from PIL import Image
import torch
import tyro

from torch_sfm.correspondence import FlexGlueConfig
from torch_sfm.geometry.track_refinement import TrackRefinementConfig
from torch_sfm.pipeline.gluemap import GluemapConfig, GluemapPipeline


def prepare_workspace(capture: Path, workspace: Path):
    """Convert an extracted LaMAR map session to pipeline inputs; keep image bytes."""
    def rows(name):
        return [[value.strip() for value in row]
                for row in csv.reader((capture / name).read_text().splitlines())
                if row and row[0].strip() and not row[0].lstrip().startswith('#')]

    sensors = {row[0]: row for row in rows('sensors.txt')}
    records = []
    for timestamp, sensor_id, filename in rows('images.txt'):
        if not sensor_id.startswith('ios'):
            continue
        sensor = sensors[sensor_id]
        if len(sensor) != 10 or sensor[2:4] != ['camera', 'PINHOLE']:
            raise ValueError(f'Expected a PINHOLE iOS camera: {sensor_id}')
        width, height = map(int, sensor[4:6])
        intrinsics = np.asarray(sensor[6:10], dtype=np.float64)
        if not np.isfinite(intrinsics).all() or (intrinsics[:2] <= 0).any():
            raise ValueError(f'Invalid calibration: {sensor_id}')
        relative = Path(filename)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Invalid image path: {filename}')
        path = (capture / 'raw_data' / relative).resolve(strict=True)
        with Image.open(path) as image:
            if image.size != (width, height) or image.getexif().get(274, 1) != 1:
                raise ValueError(f'Image size/orientation does not match calibration: {path}')
        records.append((sensor_id.split('/')[0], int(timestamp), sensor_id, path, intrinsics))
    records.sort(key=lambda record: record[:3])
    if not records or len({record[3] for record in records}) != len(records):
        raise ValueError('Expected nonempty, unique iOS mapping images')

    sequences, start = [], 0
    for name, group in groupby(records, key=lambda record: record[0]):
        count = sum(1 for _ in group)
        sequences.append(dict(name=name, start=start, end=start + count, count=count))
        start += count
    inputs = {
        'images.txt': ''.join(f'{record[3]}\n' for record in records),
        'intrinsics.txt': ''.join(' '.join(f'{v:.12f}' for v in record[4]) + '\n' for record in records),
        'sequences.json': json.dumps(sequences, indent=2) + '\n',
    }
    # Re-running preparation is safe; a different dataset needs a new workspace.
    for name, content in inputs.items():
        path = workspace / name
        if path.exists() and path.read_text() != content:
            raise ValueError(f'{path} differs from these inputs; choose a new workspace')
    workspace.mkdir(parents=True, exist_ok=True)
    for name, content in inputs.items():
        (workspace / name).write_text(content)


def main(
    workspace: Path,
    capture: Path | None = None,
    prepare_only: bool = False,
    output: Path | None = None,
    sequences: Path | None = None,
    checkpoint: str | None = None,
    flexglue: FlexGlueConfig = FlexGlueConfig(),
    refine_only: bool = False,
):
    if prepare_only and capture is None:
        raise ValueError('--prepare-only requires --capture')
    if capture is not None:
        prepare_workspace(capture, workspace)
    if prepare_only:
        return
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)

    # Real FlexGlue tracks, fixed calibration, full-system Warp PCG.
    refinement = TrackRefinementConfig(
        track_source='flexglue', strategy='augmented', num_rounds=2,
        pcg_backend='warp',
    )
    refinement.virtual_tracks.enabled = False
    refinement.bundle_adjustment.refine_focal_length = False
    refinement.bundle_adjustment.refine_principal_point = False

    config = GluemapConfig(
        flexglue=flexglue,
        retrieval_top_k=100,
        retrieval_sequences=sequences or workspace / 'sequences.json',
        retrieval_sequential_neighbors=30,
        max_neighbors=25,
        checkpoint=checkpoint,
        calls=16,
        intrinsics_source='fixed',
        intrinsics_path=workspace / 'intrinsics.txt',
        save_virtual_candidates=False,
        verify_matches=True,
        refinement=refinement,
        output=output,
    )
    with GluemapPipeline(config, workspace) as pipeline:
        if refine_only:
            pipeline.refine_saved_poses()
        else:
            pipeline.run()
        print(pipeline.output / 'reconstruction.html')


if __name__ == '__main__':
    tyro.cli(main)
