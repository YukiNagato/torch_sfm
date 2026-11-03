import argparse
from pathlib import Path

from .reconstruction import visualize_reconstruction

parser = argparse.ArgumentParser(description='Export an offline point-cloud and camera HTML viewer.')
parser.add_argument('source', type=Path, help='Workspace directory or reconstruction.pt')
parser.add_argument('--output', type=Path)
parser.add_argument('--max-points', type=int, default=100000)
parser.add_argument('--camera-scale', type=float)
parser.add_argument('--view-up', choices=['auto', 'x', 'y', 'z', '-x', '-y', '-z'], default='auto')
args = parser.parse_args()
print(visualize_reconstruction(args.source, args.output,
    max_points=args.max_points, camera_scale=args.camera_scale, view_up=args.view_up))
