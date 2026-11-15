"""Optional COLMAP SIFT extraction/matching, returned as native torch-sfm inputs.

COLMAP only produces image measurements and descriptor matches. Camera geometry,
triangulation and BA remain in torch-sfm; no poses or GT calibration are read here.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time

import numpy as np
import torch
from torch_sfm.core import Frame
from .base import Matches


@dataclass
class SIFTConfig:
    colmap_binary: str = 'colmap'
    num_threads: int = 16
    use_gpu: bool = True
    gpu_index: str = '0'
    max_num_features: int = 8192
    max_image_size: int = -1
    max_ratio: float = .8
    max_distance: float = .7
    cross_check: bool = True
    # GlueMap keeps all-SIFT tracks, then adds tracked points for sparse pairs.
    track_pair_coverage: int = 512

    def __post_init__(self):
        if self.num_threads < 1 or self.max_num_features < 1 or self.track_pair_coverage < 0:
            raise ValueError('SIFT counts must be positive and coverage nonnegative')
        if not 0 < self.max_ratio <= 1 or self.max_distance <= 0:
            raise ValueError('Invalid SIFT matching thresholds')


def load_sift_database(database, frames, allowed_pairs):
    """Map COLMAP image IDs by filename, never by assumed insertion order."""
    with sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro', uri=True) as conn:
        ids = {name: int(i) for i, name in conn.execute('SELECT image_id,name FROM images')}
        output, inverse = [], {}
        for i, f in enumerate(frames):
            name = Path(f.frame_path).name
            if name not in ids:
                raise ValueError(f'SIFT database has no image {name}')
            image_id = ids[name]; inverse[image_id] = i
            row = conn.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?',(image_id,)).fetchone()
            if row is None:
                raise ValueError(f'SIFT database has no keypoints for {name}')
            n, c, blob = row
            if c < 2 and n:
                raise ValueError('Invalid SIFT keypoint shape')
            points = np.frombuffer(blob or b'', dtype=np.float32).reshape(n,c)[:,:2].copy()
            if not np.isfinite(points).all():
                raise ValueError('Nonfinite SIFT keypoints')
            output.append(Frame(frame_idx=i, frame_path=f.frame_path, image_size=f.image_size,
                                keypoints_2d=torch.from_numpy(points).double()))
        matches = {}
        for pair_id,n,c,blob in conn.execute('SELECT pair_id,rows,cols,data FROM matches'):
            if not n:
                continue
            a,b = divmod(int(pair_id),2147483647)
            if a not in inverse or b not in inverse:
                continue
            i,j=inverse[a],inverse[b]
            pair=tuple(sorted((i,j)))
            if pair not in allowed_pairs:
                continue
            if c != 2:
                raise ValueError('Invalid SIFT match shape')
            indices=np.frombuffer(blob,dtype=np.uint32).reshape(n,2).astype(np.int64)
            if i>j:
                indices=indices[:,::-1].copy()
            if ((indices[:,0]>=len(output[pair[0]].keypoints_2d)).any()
                    or (indices[:,1]>=len(output[pair[1]].keypoints_2d)).any()):
                raise ValueError('SIFT match index outside keypoints')
            matches[pair]=Matches(torch.from_numpy(indices),torch.ones(n),{'source':'sift'})
    return output,matches


def prepare_sift(frames, pairs, config, cache_directory):
    """Build a separately fingerprinted SIFT cache, or read a completed one."""
    directory=Path(cache_directory);directory.mkdir(parents=True,exist_ok=True)
    paths=[Path(f.frame_path).resolve() for f in frames]
    if len({p.parent for p in paths})!=1 or len({p.name for p in paths})!=len(paths):
        raise ValueError('COLMAP SIFT adapter requires unique image names in one directory')
    pairs=sorted({tuple(sorted(map(int,p))) for p in pairs})
    if any(i<0 or j>=len(frames) or i==j for i,j in pairs):
        raise ValueError('Invalid SIFT image pair')
    binary=shutil.which(config.colmap_binary)
    if binary is None:
        raise FileNotFoundError(config.colmap_binary)
    signature=dict(version=1,config=asdict(config),images=[dict(path=str(p),size=p.stat().st_size,
        mtime_ns=p.stat().st_mtime_ns) for p in paths],pairs=pairs)
    fingerprint=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    metadata=directory/'report.json'; database=directory/'database.db'
    start=time.perf_counter();reused=False
    previous=json.loads(metadata.read_text()) if metadata.exists() else None
    if previous is not None and previous.get('fingerprint')!=fingerprint:
        raise ValueError('SIFT cache inputs changed; choose a new cache directory')
    if previous is not None and previous.get('state')=='complete' and database.exists():
        report=previous;reused=True
    else:
        # A partial matching run is resumable; completed pairs are retained by COLMAP.
        if database.exists() and previous is None:
            raise ValueError('SIFT database lacks provenance; choose a new cache directory')
        report=previous or dict(state='running',fingerprint=fingerprint,inputs=signature,phases={},commands=[])
        def save(): metadata.write_text(json.dumps(report,indent=2)+'\n')
        save()
        image_list=directory/'images.txt';image_list.write_text(''.join(p.name+'\n' for p in paths))
        pair_list=directory/'pairs.txt';pair_list.write_text(''.join(f'{paths[i].name} {paths[j].name}\n' for i,j in pairs))
        def run(phase,args):
            cmd=[binary,*map(str,args)]
            report['commands'].append(cmd);report['active_phase']=phase;save()
            env=os.environ.copy();env.setdefault('QT_QPA_PLATFORM','offscreen')
            ts=time.perf_counter()
            print(f'SIFT_START {phase}',flush=True)
            with (directory/f'{phase}.log').open('a') as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
            report['phases'][phase]=report['phases'].get(phase,0)+time.perf_counter()-ts
            report['active_phase']=None;save()
            print(f'SIFT_END {phase} {report["phases"][phase]:.3f}s',flush=True)
        if 'extraction' not in report['phases']:
            run('extraction',['feature_extractor','--database_path',database,'--image_path',paths[0].parent,
                '--image_list_path',image_list,'--ImageReader.camera_model','PINHOLE','--ImageReader.single_camera','1',
                '--FeatureExtraction.type','SIFT','--FeatureExtraction.num_threads',config.num_threads,
                '--FeatureExtraction.use_gpu',int(config.use_gpu),'--FeatureExtraction.gpu_index',config.gpu_index,
                '--FeatureExtraction.max_image_size',config.max_image_size,'--SiftExtraction.max_num_features',config.max_num_features])
        run('matching',['matches_importer','--database_path',database,'--match_list_path',pair_list,'--match_type','pairs',
            '--FeatureMatching.type','SIFT_BRUTEFORCE','--FeatureMatching.num_threads',config.num_threads,
            '--FeatureMatching.use_gpu',int(config.use_gpu),'--FeatureMatching.gpu_index',config.gpu_index,
            '--FeatureMatching.skip_geometric_verification','1','--SiftMatching.max_ratio',config.max_ratio,
            '--SiftMatching.max_distance',config.max_distance,'--SiftMatching.cross_check',int(config.cross_check)])
        report.update(state='complete',active_phase=None);save()
    sift_frames,matches=load_sift_database(database,frames,set(pairs))
    result=dict(cache_reused=reused,config=asdict(config),fingerprint=fingerprint,
        extraction_seconds=0. if reused else report['phases']['extraction'],
        matching_seconds=0. if reused else report['phases']['matching'],
        original_phases=report['phases'],total_seconds=time.perf_counter()-start,
        keypoints=sum(len(f.keypoints_2d) for f in sift_frames),pairs=len(matches),
        matches=sum(len(m.indices) for m in matches.values()),
        geometry='Raw RootSIFT matches; merged correspondences use the pipeline Warp E/F/H verifier')
    return sift_frames,matches,result
