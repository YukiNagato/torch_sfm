"""Generate real multi-view correspondences with VGGSfM on saved star topology.

Tracks are image measurements, not depth projections. Saved Solv3R depths and
GT cameras are never read here. Normal geometric verification/triangulation
remain responsible for rejecting inconsistent observations.
"""
from collections import defaultdict, OrderedDict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
import torch

from torch_sfm.core import Frame
from torch_sfm.correspondence import Matches
from torch_sfm.correspondence.vggsfm import VGGSfMTracker, preprocess_image


def tracks_to_matches(frames, records, config, registered, allowed_pairs, *, snap_frames=None, base_matches=None):
    """Snap nearby tracks to detector identities, retain unsnapped measurements.

    Only center-to-neighbor edges are emitted, as in GlueMap. The original
    frames are not mutated and FlexGlue matches are not added implicitly.
    """
    if snap_frames is not None:
        return _tracks_with_base(frames, records, config, registered, allowed_pairs, snap_frames, base_matches or {})
    base = [f.keypoints_2d.detach().cpu().float().numpy() for f in frames]
    trees = [cKDTree(p) if len(p) else None for p in base]
    additions = [[] for _ in frames]
    counts = [len(p) for p in base]
    pairs, scores = defaultdict(list), defaultdict(list)
    total_valid = snapped = 0
    for record in records:
        ids = record['image_ids']
        center = ids[0]
        if not bool(registered[center]):
            continue
        xy = record['tracks'].numpy()
        vis = record['visibility'].numpy()
        qids = record['query_ids'].numpy()
        if xy.shape != (len(ids), len(qids), 2) or vis.shape != xy.shape[:2]:
            raise ValueError('Malformed VGGSfM track cache')
        if len(np.unique(qids)) != len(qids) or np.any(qids < 0) or np.any(qids >= len(base[center])):
            raise ValueError('Invalid tracker query identities')
        for slot, image_id in enumerate(ids[1:], 1):
            pair = tuple(sorted((center, image_id)))
            if not bool(registered[image_id]) or pair not in allowed_pairs:
                continue
            h, w = frames[image_id].image_size
            p = xy[slot]
            valid = (np.isfinite(p).all(-1) & np.isfinite(vis[slot]) & np.isfinite(vis[0])
                & (vis[slot] > config.visibility_threshold) & (vis[0] > config.visibility_threshold)
                & (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h))
            q = np.flatnonzero(valid)
            if not len(q):
                continue
            selected = p[q].copy()
            total_valid += len(q)
            if trees[image_id] is not None and config.snapping_threshold > 0:
                distances, nearest = trees[image_id].query(selected)
                close = distances < config.snapping_threshold * (h+w)/config.image_size
            else:
                nearest = np.zeros(len(q), dtype=np.int64)
                close = np.zeros(len(q), dtype=bool)
            snapped += int(close.sum())
            target = nearest.astype(np.int64)
            # Unsnapped predictions are real subpixel observations, not dropped.
            missing = ~close
            target[missing] = np.arange(counts[image_id], counts[image_id]+int(missing.sum()))
            additions[image_id].append(selected[missing])
            counts[image_id] += int(missing.sum())
            indices = np.stack((qids[q], target), axis=-1)
            if center > image_id:
                indices = indices[:, ::-1].copy()
            pairs[pair].append(indices)
            scores[pair].append(np.minimum(vis[slot, q], vis[0, q]))
    output_frames = []
    for i, frame in enumerate(frames):
        points = np.concatenate([base[i], *additions[i]], axis=0) if additions[i] else base[i].copy()
        output_frames.append(Frame(frame_idx=i, frame_path=frame.frame_path,
            image_size=frame.image_size, keypoints_2d=torch.from_numpy(points).double()))
    output_matches = {}
    for pair, chunks in pairs.items():
        indices = np.concatenate(chunks)
        values = np.concatenate(scores[pair])
        # Keep highest-confidence entry when two stars predict the same edge.
        order = np.argsort(-values, kind='stable')
        _, unique = np.unique(indices[order], axis=0, return_index=True)
        keep = order[unique]
        output_matches[pair] = Matches(torch.from_numpy(indices[keep]), torch.from_numpy(values[keep]),
                                       {'source': 'vggsfm'})
    return output_frames, output_matches, dict(stars=len(records), valid_neighbor_observations=total_valid,
        snapped_observations=snapped, keypoints=sum(counts), pairs=len(output_matches),
        matches=sum(len(m.indices) for m in output_matches.values()))


def track_stars(frames, manifest, initial, config, cache_directory, device='cuda', *, snap_frames=None, base_matches=None):
    """Cache one prediction per star; changing settings/queries/images invalidates it."""
    cache_directory = Path(cache_directory)
    cache_directory.mkdir(parents=True, exist_ok=True)
    if config.source is None or config.checkpoint is None:
        raise ValueError('VGGSfM requires source and checkpoint paths')
    def file_hash(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''):
                digest.update(chunk)
        return digest.hexdigest()
    backend = Path(config.source)/'thirdparty/vggsfm'
    signature = dict(version=1, config=asdict(config), checkpoint_sha256=file_hash(config.checkpoint),
        source_sha256={str(p.relative_to(backend)):file_hash(p) for p in sorted(backend.rglob('*.py'))},
        images=[dict(path=str(f.frame_path), size=Path(f.frame_path).stat().st_size,
                     mtime_ns=Path(f.frame_path).stat().st_mtime_ns) for f in frames])
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    (cache_directory/'inputs.json').write_text(json.dumps(signature, indent=2)+'\n')
    images = OrderedDict()
    image_capacity = max(1, int(config.image_cache_gib * 1024**3) // (3*config.image_size**2*4))
    def load_image(index):
        if index in images:
            images.move_to_end(index)
        if index not in images:
            with Image.open(frames[index].frame_path) as image:
                rgb = torch.from_numpy(np.asarray(ImageOps.exif_transpose(image).convert('RGB')).copy()).permute(2,0,1).float()/255.
            if tuple(rgb.shape[-2:]) != tuple(frames[index].image_size):
                raise ValueError('Tracker image dimensions differ from cached detector coordinates')
            images[index] = preprocess_image(rgb, config.image_size)
            while len(images) > image_capacity:
                images.popitem(last=False)
        return images[index]
    tracker = VGGSfMTracker(config, device)
    records, times = [], []
    start = time.perf_counter()
    inferred = 0
    for number, star in enumerate(manifest['stars']):
        ids = star['image_ids']
        center = ids[0]
        if len(ids)<2 or not bool(initial['registered'][center]):
            continue
        keypoints = frames[center].keypoints_2d.detach().cpu().float()
        if not len(keypoints):
            continue
        generator = torch.Generator().manual_seed(config.seed+center)
        query_ids = torch.randperm(len(keypoints), generator=generator)[:config.queries_per_star]
        query_points = keypoints[query_ids]
        key = dict(fingerprint=fingerprint, image_ids=ids,
                   query_sha256=hashlib.sha256(query_points.numpy().tobytes()+query_ids.numpy().tobytes()).hexdigest())
        path = cache_directory/f'{center:06d}.pt'
        record = torch.load(path, map_location='cpu', weights_only=True) if path.exists() else None
        if record is None or record.get('cache_key') != key:
            ts = time.perf_counter()
            data = [load_image(i) for i in ids]
            batch = torch.stack([v[0] for v in data])
            transforms = torch.stack([v[1] for v in data])
            record = tracker.predict(batch, query_points, transforms=transforms)
            record.update(cache_key=key, image_ids=ids, query_ids=query_ids,
                          seconds=time.perf_counter()-ts)
            temp = path.with_suffix('.tmp')
            torch.save(record, temp); temp.replace(path)
            inferred += 1
            del batch, transforms, data
        records.append(record); times.append(record['seconds'])
        if number % 5 == 0 or number+1 == len(manifest['stars']):
            print(f'VGGSFM_TRACKS {number+1}/{len(manifest["stars"])} inferred={inferred} elapsed={time.perf_counter()-start:.1f}s', flush=True)
    tracker.close(); images.clear()
    if torch.device(device).type=='cuda': torch.cuda.empty_cache()
    inference_seconds = time.perf_counter()-start
    allowed = {tuple(sorted(p)) for p in manifest['accepted_pairs']}
    rejected = {tuple(sorted(r['pair'])) for r in initial['fusion_report']['reverse_rejected']}
    allowed -= rejected
    ts = time.perf_counter()
    output_frames, matches, report = tracks_to_matches(frames, records, config, initial['registered'], allowed,
        snap_frames=snap_frames, base_matches=base_matches)
    report.update(source='vggsfm' if snap_frames is None else 'sift_vggsfm', config=asdict(config), virtual_tracks=False, inferred_stars=inferred,
        fingerprint=fingerprint, inference_seconds=inference_seconds,
        original_inference_seconds=sum(times), graph_seconds=time.perf_counter()-ts,
        total_seconds=time.perf_counter()-start,
        query_source='Seeded subset of cached RaCo keypoints; no FlexGlue match edges',
        admission='visibility > threshold, finite in-image measurements, accepted star pairs; followed by configured E/F/H verification and triangulation')
    (cache_directory/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return output_frames, matches, report


def _tracks_with_base(query_frames, records, config, registered, allowed_pairs, snap_frames, base_matches):
    """Keep SIFT matches and snap both endpoints of RaCo-seeded VGGSfM tracks.

    Query IDs belong to RaCo and must never be interpreted as SIFT IDs. SIFT
    observations stay at the start of each frame for GlueMap's track selector.
    """
    if len(query_frames)!=len(snap_frames):
        raise ValueError('Query and snapping frame counts differ')
    for a,b in zip(query_frames,snap_frames):
        if a.frame_idx!=b.frame_idx or a.frame_path!=b.frame_path or tuple(a.image_size)!=tuple(b.image_size):
            raise ValueError('Query and snapping image order/size differ')
    base=[f.keypoints_2d.detach().cpu().float().numpy() for f in snap_frames]
    trees=[cKDTree(p) if len(p) else None for p in base]
    additions=[[] for _ in base];counts=[len(p) for p in base]
    pairs,scores=defaultdict(list),defaultdict(list)
    base_edges=0
    for pair,m in base_matches.items():
        if pair in allowed_pairs and bool(registered[pair[0]] & registered[pair[1]]):
            indices=m.indices.cpu().numpy()
            if ((indices<0).any() or (indices[:,0]>=len(base[pair[0]])).any()
                    or (indices[:,1]>=len(base[pair[1]])).any()):
                raise ValueError('Base match IDs do not belong to the snapping features')
            pairs[pair].append(indices);scores[pair].append(m.scores.cpu().numpy());base_edges+=len(indices)
    snapped=valid_total=tracked_edges=0
    for record in records:
        ids=record['image_ids'];center=ids[0]
        if not bool(registered[center]): continue
        xy=record['tracks'].numpy().copy();vis=record['visibility'].numpy();qids=record['query_ids']
        if xy.shape!=(len(ids),len(qids),2) or vis.shape!=xy.shape[:2]:
            raise ValueError('Malformed VGGSfM track cache')
        queries=query_frames[center].keypoints_2d.detach().cpu()
        if len(qids)!=len(torch.unique(qids)) or (qids<0).any() or (qids>=len(queries)).any():
            raise ValueError('Invalid RaCo query identities')
        xy[0]=queries[qids].float().numpy()
        mapping=np.full(vis.shape,-1,dtype=np.int64)
        for slot,image_id in enumerate(ids):
            if not bool(registered[image_id]) or (slot and tuple(sorted((center,image_id))) not in allowed_pairs):
                continue
            h,w=snap_frames[image_id].image_size;p=xy[slot]
            valid=(np.isfinite(p).all(-1)&np.isfinite(vis[slot])&(vis[slot]>config.visibility_threshold)
                   &(p[:,0]>=0)&(p[:,0]<w)&(p[:,1]>=0)&(p[:,1]<h))
            q=np.flatnonzero(valid)
            if not len(q):continue
            selected=p[q];valid_total+=len(q)
            if trees[image_id] is not None and config.snapping_threshold>0:
                distance,nearest=trees[image_id].query(selected)
                close=distance<config.snapping_threshold*(h+w)/config.image_size
            else:
                nearest=np.zeros(len(q),dtype=np.int64);close=np.zeros(len(q),dtype=bool)
            snapped+=int(close.sum());target=nearest.astype(np.int64)
            missing=~close
            target[missing]=np.arange(counts[image_id],counts[image_id]+int(missing.sum()))
            counts[image_id]+=int(missing.sum());additions[image_id].append(selected[missing])
            mapping[slot,q]=target
        for slot,image_id in enumerate(ids[1:],1):
            good=(mapping[0]>=0)&(mapping[slot]>=0)
            if not good.any():continue
            pair=tuple(sorted((center,image_id)))
            indices=np.stack((mapping[0,good],mapping[slot,good]),axis=-1)
            if center>image_id:indices=indices[:,::-1].copy()
            pairs[pair].append(indices);scores[pair].append(np.minimum(vis[0,good],vis[slot,good]));tracked_edges+=len(indices)
    # Upstream identifies coincident tracked observations within 1e-3 px. Do
    # this only for appended measurements; original SIFT IDs remain unchanged.
    output_frames=[];remaps=[];duplicates=0
    for i,f in enumerate(snap_frames):
        extra=np.concatenate(additions[i],axis=0) if additions[i] else np.empty((0,2),dtype=np.float32)
        parent=np.arange(len(extra))
        def root(k):
            while parent[k]!=k:
                parent[k]=parent[parent[k]];k=parent[k]
            return k
        if len(extra):
            for a,b in cKDTree(extra).query_pairs(r=1e-3,output_type='ndarray'):
                ra,rb=root(a),root(b)
                if ra!=rb:parent[max(ra,rb)]=min(ra,rb)
            roots=np.array([root(k) for k in range(len(extra))],dtype=np.int64)
            unique,inverse=np.unique(roots,return_inverse=True)
        else:
            unique=inverse=np.empty(0,dtype=np.int64)
        duplicates+=len(extra)-len(unique)
        points=np.concatenate((base[i],extra[unique]),axis=0)
        remaps.append(np.r_[np.arange(len(base[i])),inverse+len(base[i])])
        output_frames.append(Frame(frame_idx=f.frame_idx,frame_path=f.frame_path,image_size=f.image_size,
                                   keypoints_2d=torch.from_numpy(points).double()))
    output={}
    for pair,chunks in pairs.items():
        indices=np.concatenate(chunks);values=np.concatenate(scores[pair])
        indices[:,0]=remaps[pair[0]][indices[:,0]];indices[:,1]=remaps[pair[1]][indices[:,1]]
        order=np.argsort(-values,kind='stable');_,unique=np.unique(indices[order],axis=0,return_index=True);keep=order[unique]
        output[pair]=Matches(torch.from_numpy(indices[keep]),torch.from_numpy(values[keep]),{'source':'sift_vggsfm'})
    return output_frames,output,dict(stars=len(records),valid_observations=valid_total,snapped_observations=snapped,
        base_keypoint_counts=[len(p) for p in base],base_matches=base_edges,tracked_matches=tracked_edges,
        deduplicated_tracked_observations=duplicates,keypoints=sum(len(f.keypoints_2d) for f in output_frames),
        pairs=len(output),matches=sum(len(m.indices) for m in output.values()),snapping_target='SIFT')
