"""Fuse local camera stars without rerunning image inference.

Conventions: R is world-to-camera, c2w local stars are anchored at their center.
The inverse star scale a satisfies a * (C_j-C_i) = -R_j.T @ t_ji(local).
Thus X_global = R_i.T @ X_local / a + C_i. Scales are shared by all edges in
one star, never by arbitrary pairs. The objective follows GlueMap's similarity
averaging; initialization uses a maximum spanning tree of overlapping stars,
including one-way stars when two common cameras determine a scale ratio.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
import math
import time

import torch
import torchls as tls

from .averaging import RotationConfig, rotation_averaging, _pcg_backend, _check_solution
from ..pipeline.stars import graph_components


@dataclass
class StarFusionConfig:
    rotation: RotationConfig = field(default_factory=lambda: RotationConfig(max_error=5.))
    reverse_rotation_degrees: float = 10.
    reverse_translation_degrees: float = 30.
    min_triangulation_degrees: float = 1.
    parallax_samples: int = 4096
    shared_intrinsics: bool = True
    camera_groups: tuple[int, ...] | None = None  # overrides shared_intrinsics
    max_iterations: int = 200
    huber_delta: float = .01  # in predicted local baseline units
    pcg_backend: str = 'auto'
    pcg_rtol: float = 1e-5
    pcg_iterations: int = 300
    sequential_weight: float = 2.

    def __post_init__(self):
        if not math.isfinite(self.sequential_weight) or self.sequential_weight <= 0:
            raise ValueError('sequential_weight must be finite and positive')
        if not all(math.isfinite(v) and 0 < v <= 180 for v in
                   (self.reverse_rotation_degrees, self.reverse_translation_degrees)):
            raise ValueError('Reverse consistency thresholds must be in (0,180]')
        if not 0 <= self.min_triangulation_degrees < 180:
            raise ValueError('Invalid parallax threshold')
        if min(self.parallax_samples, self.max_iterations, self.pcg_iterations) < 1:
            raise ValueError('Iteration/sample counts must be positive')
        if not math.isfinite(self.huber_delta) or self.huber_delta <= 0 or not 0 < self.pcg_rtol < 1:
            raise ValueError('Invalid Huber/PCG tolerance')
        if self.pcg_backend not in ('auto', 'torch', 'warp'):
            raise ValueError('Unknown PCG backend')


@dataclass
class LocalStar:
    image_ids: tuple[int, ...]
    camera_to_world: torch.Tensor
    intrinsics: torch.Tensor
    image_size: tuple[int, int]
    scores: torch.Tensor
    parallax_degrees: torch.Tensor


@dataclass
class RotationEdge:
    R: torch.Tensor
    score: float


def compact_star(data, cfg):
    """Discard dense arrays after estimating parallax in direct consistent pixels."""
    ids = tuple(int(i) for i in data['image_ids'])
    n = len(ids)
    poses = data['camera_to_world'].detach().cpu().double().clone()
    K = data['intrinsics'].detach().cpu().double().clone()
    scores = data['center_scores'].detach().cpu().double().clone()
    if n < 2 or len(set(ids)) != n or poses.shape != (n,4,4) or K.shape != (n,3,3):
        raise ValueError('Malformed local star')
    if scores.shape != (n,) or not torch.isfinite(scores).all() or ((scores<0)|(scores>1)).any():
        raise ValueError('Invalid local scores')
    if not torch.isfinite(poses).all() or not torch.isfinite(K).all() or (K[:,(0,1),(0,1)]<=0).any():
        raise ValueError('Invalid local cameras')
    if not torch.allclose(poses[0], torch.eye(4,dtype=poses.dtype), atol=1e-5):
        raise ValueError('Local star must be anchored at its first camera')
    scores[~data['valid_views'].cpu()] = 0
    angles = torch.zeros(n, dtype=torch.float64)
    if cfg.min_triangulation_degrees > 0 and 'parallax_degrees' in data:
        if data['parallax_samples'] != cfg.parallax_samples:
            raise ValueError('Parallax sampling differs from the compact star cache; regenerate stars')
        angles = data['parallax_degrees'].detach().cpu().double().clone()
        if angles.shape != (n,) or not torch.isfinite(angles).all():
            raise ValueError('Malformed cached parallax statistics')
    elif cfg.min_triangulation_degrees > 0:
        if data.get('depth') is None:
            raise ValueError('Parallax filtering requires saved depth; set min_triangulation_degrees=0 to disable')
        h,w = data['image_size']
        stride = max(1, math.ceil(math.sqrt(h*w/cfg.parallax_samples)))
        y,x = torch.meshgrid(torch.arange(0,h,stride),torch.arange(0,w,stride),indexing='ij')
        z = data['depth'][0,::stride,::stride].double()
        p = torch.stack((x+.5,y+.5,torch.ones_like(x)), -1).double() @ torch.linalg.inv(K[0]).T
        points = (p * z[...,None]).flatten(0,1)
        ray0 = torch.nn.functional.normalize(points,dim=-1)
        for k in range(1,n):
            valid = (data['center_masks'][k,::stride,::stride].flatten()
                     & torch.isfinite(z.flatten()) & (z.flatten()>0))
            if valid.any():
                rays = torch.nn.functional.normalize(points[valid]-poses[k,:3,3],dim=-1)
                angle = torch.acos((ray0[valid]*rays).sum(-1).clamp(-1,1)).rad2deg()
                # Near 180-degree rays are also degenerate for triangulation.
                angles[k] = torch.minimum(angle,180-angle).median()
    return LocalStar(ids,poses,K,tuple(data['image_size']),scores,angles)


def filter_reverse_constraints(stars, cfg):
    """Compare i->j with inverse(j->i) in one camera frame, ignoring scale."""
    by_center = {s.image_ids[0]: k for k,s in enumerate(stars)}
    if len(by_center) != len(stars):
        raise ValueError('Duplicate star centers')
    active = [s.scores > 0 for s in stars]
    for a in active:
        a[0] = False
    pairs = {(s.image_ids[0],j):(k,q) for k,s in enumerate(stars)
             for q,j in enumerate(s.image_ids[1:],1) if active[k][q]}
    rejected = []
    checked = 0
    for (i,j),(a,p) in pairs.items():
        if i >= j or (j,i) not in pairs:
            continue
        b,q = pairs[j,i]
        A, B = stars[a].camera_to_world[p], stars[b].camera_to_world[q]
        rot = A[:3,:3] @ B[:3,:3]
        rerr = math.degrees(math.acos(float(((rot.trace()-1)/2).clamp(-1,1))))
        # C_j expressed in i should oppose R_ij C_i expressed in j.
        u,v = A[:3,3], -(A[:3,:3] @ B[:3,3])
        norms = float(u.norm()*v.norm())
        terr = math.degrees(math.acos(float((u@v/norms).clamp(-1,1)))) if norms>1e-20 else 0.
        checked += 1
        if rerr>cfg.reverse_rotation_degrees or terr>cfg.reverse_translation_degrees:
            active[a][p] = active[b][q] = False
            rejected.append(dict(pair=(i,j),rotation_degrees=rerr,translation_degrees=terr))
    return active, dict(reverse_pairs_checked=checked, reverse_rejected=rejected)


def fuse_intrinsics(stars, active, original_sizes, cfg):
    """Median in original pixel coordinates, respecting explicit camera groups.

    A shared camera requires a shared original resolution. Inference uses a
    stretch resize and half-pixel centers, hence K_original = diag(W/w,H/h,1) K.
    """
    n = len(original_sizes)
    groups = list(cfg.camera_groups) if cfg.camera_groups is not None else ([0]*n if cfg.shared_intrinsics else list(range(n)))
    if len(groups)!=n or any(type(g) is not int or g<0 for g in groups):
        raise ValueError('camera_groups must contain one nonnegative integer per image')
    for g in set(groups):
        sizes = {tuple(original_sizes[i]) for i in range(n) if groups[i]==g}
        if len(sizes)!=1:
            raise ValueError('Images in a shared camera group must have the same original resolution')
    estimates = defaultdict(list)
    for s,mask in zip(stars,active):
        h,w = s.image_size
        for q,i in enumerate(s.image_ids):
            if q and not mask[q]:
                continue
            H,W = original_sizes[i]
            scale = s.intrinsics.new_tensor([W/w,H/h,1.])
            estimates[groups[i]].append(scale[:,None]*s.intrinsics[q])
    result = torch.full((n,3,3),torch.nan,dtype=torch.float64)
    for g,values in estimates.items():
        matrix = torch.stack(values).median(0).values
        matrix[2] = matrix.new_tensor([0,0,1])
        result[torch.tensor([v==g for v in groups])] = matrix
    return result, torch.tensor(groups)


def _maximum_tree(nodes, weighted_edges):
    parent = {i:i for i in nodes}
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    tree = {i:[] for i in nodes}
    for (a,b),weight in sorted(weighted_edges.items(),key=lambda v:(-v[1],v[0])):
        x,y = find(a),find(b)
        if x!=y:
            parent[x]=y
            tree[a].append(b)
            tree[b].append(a)
    return tree


def initialize_similarities(stars, active, rotations):
    """Maximum spanning tree on stars with two or more shared camera centers.

    Tree edge scale ratios use the longest common baseline. Single-camera
    overlap cannot determine relative scale: select the largest scale-connected
    component and report excluded stars instead of inventing unit scales.
    """
    offsets, members, weights = {}, defaultdict(list), {}
    for k,(s,mask) in enumerate(zip(stars,active)):
        if not mask.any():
            continue
        i = s.image_ids[0]
        offsets[k] = {i:torch.zeros(3,dtype=torch.float64)}
        for q in torch.where(mask)[0].tolist():
            j = s.image_ids[q]
            # Local w2c translation; rotate it with the averaged target rotation.
            t = -s.camera_to_world[q,:3,:3].T @ s.camera_to_world[q,:3,3]
            offsets[k][j] = -rotations[j].T @ t
        for j in offsets[k]:
            members[j].append(k)
    if not offsets:
        raise ValueError('No nondegenerate position constraints after filtering')
    overlap = defaultdict(list)
    for j,ss in members.items():
        for a,b in combinations(ss,2):
            overlap[a,b].append(j)
    ratios = {}
    for (a,b),common in overlap.items():
        if len(common)<2:
            continue
        va = torch.stack([offsets[a][j] for j in common])
        vb = torch.stack([offsets[b][j] for j in common])
        p,q = torch.triu_indices(len(common),len(common),offset=1)
        da,db = (va[p]-va[q]).norm(dim=-1),(vb[p]-vb[q]).norm(dim=-1)
        valid = (da>1e-10)&(db>1e-10)
        if not valid.any():
            continue
        quality = torch.where(valid,da/da.max()*db/db.max(),0.)
        best = int(quality.argmax())
        ratios[a,b] = float(db[best]/da[best])
        weights[a,b] = min(float(stars[a].scores[active[a]].mean()),float(stars[b].scores[active[b]].mean())) * len(common)
    components = [c for c in graph_components(len(stars),weights) if any(k in offsets for k in c)]
    components.sort(key=lambda c:(-len(set().union(*(set(offsets[k]) for k in c if k in offsets))),-len(c),min(c)))
    chosen = components[0]
    tree = _maximum_tree(chosen,{p:w for p,w in weights.items() if p[0] in chosen and p[1] in chosen})
    root = min(chosen)
    centers = {i:v.clone() for i,v in offsets[root].items()}
    scales = {root:1.}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in tree[a]:
            if b in scales:
                continue
            ratio = ratios[a,b] if (a,b) in ratios else 1/ratios[b,a]
            scales[b] = scales[a]*ratio
            local = offsets[b]
            center = stars[b].image_ids[0]
            common = sorted(set(local)&set(centers))
            translation = (centers[center] if center in centers else
                           torch.stack([centers[j]-local[j]/scales[b] for j in common]).mean(0))
            for j,v in local.items():
                if j not in centers:
                    centers[j] = translation+v/scales[b]
            stack.append(b)
    return centers,scales,offsets,dict(anchor_star=root,anchor_image=stars[root].image_ids[0],
        scale_components=components,selected_stars=sorted(scales),
        excluded_stars=sorted(set(range(len(stars)))-set(scales)))


class _WeightedHuber(tls.Loss):
    """Per-edge loss magnitudes, preserving the local-unit Huber threshold.

    Used on unchunked cost groups only; weights match the complete group rows.
    """
    def __init__(self, delta, weights):
        self.base,self.weights=tls.HuberLoss(delta),weights
    def evaluate(self,s):
        return tuple(self.weights*x for x in self.base.evaluate(s))
    def value(self,s):
        return self.weights*self.base.value(s)
    def _reduction(self,before,after,difference):
        return self.weights*self.base._reduction(before,after,difference)


def _similarity_residual(ci,cj,scale,baseline):
    return scale[0]*(cj-ci)-baseline


@torch.inference_mode(False)
def solve_similarities(stars, active, initial_centers, initial_scales, offsets, cfg, device):
    nodes,ss = sorted(initial_centers),sorted(initial_scales)
    anchor_star = ss[0]
    anchor_image = stars[anchor_star].image_ids[0]
    center_vars = [i for i in nodes if i!=anchor_image]
    scale_vars = [k for k in ss if k!=anchor_star]
    ci_map, s_map = {i:k for k,i in enumerate(center_vars)}, {s:k for k,s in enumerate(scale_vars)}
    params = {'centers':torch.stack([initial_centers[i] for i in center_vars]).to(device)}
    if scale_vars:
        params['scales']=torch.tensor([[initial_scales[s]] for s in scale_vars],device=device,dtype=torch.float64)
    problem = tls.Problem(params)
    grouped = defaultdict(list)
    for s in ss:
        i=stars[s].image_ids[0]
        for q in torch.where(active[s])[0].tolist():
            j=stars[s].image_ids[q]
            grouped[i==anchor_image,j==anchor_image,s==anchor_star].append((i,j,s,offsets[s][j],float(stars[s].scores[q])))
    for fixed,rows in grouped.items():
        args=[]
        for axis,(name,mapping) in enumerate([('centers',ci_map),('centers',ci_map),('scales',s_map)]):
            if fixed[axis]:
                value = initial_centers[anchor_image] if axis<2 else torch.tensor([initial_scales[anchor_star]],dtype=torch.float64)
                args.append(value.to(device).expand(len(rows),-1).clone())
            else:
                args.append((name,torch.tensor([mapping[r[axis]] for r in rows],device=device)))
        baseline=torch.stack([r[3] for r in rows]).to(device)
        weight=torch.tensor([r[4] for r in rows],device=device,dtype=torch.float64)
        problem.add_residual(_similarity_residual,*args,baseline,loss=_WeightedHuber(cfg.huber_delta,weight))
    backend=_pcg_backend(cfg.pcg_backend,params['centers'])
    solution,summary=problem.solve(linear_solver='conjugate_gradient',
        bounds={'scales':(1e-5,1e5)} if scale_vars else None,
        conjugate_gradient=tls.ConjugateGradientConfig(backend=backend,normal_matvec='warp' if backend=='warp' else 'torch',
            rtol=cfg.pcg_rtol,max_iterations=cfg.pcg_iterations),
        trust_region=tls.TrustRegionConfig(lambda_initial=1e-4,strategy='adaptive',damping_mode='diagonal'),
        termination=tls.TerminationConfig(max_iterations=cfg.max_iterations,cost_tolerance=1e-8,gradient_tolerance=1e-10),
        normal_blocks_backend='auto',jacobi_scaling=True,return_summary=True)
    _check_solution(summary,'Star similarity averaging')
    centers={anchor_image:initial_centers[anchor_image].clone()}
    centers.update({i:solution['centers'][k].detach().cpu() for k,i in enumerate(center_vars)})
    scales={anchor_star:initial_scales[anchor_star]}
    scales.update({s:float(solution['scales'][k]) for k,s in enumerate(scale_vars)})
    return centers,scales,summary


@torch.inference_mode(False)
def fuse_stars(stars, original_sizes, cfg=None, device='cuda', *, sequential_edges=()):
    cfg=StarFusionConfig() if cfg is None else cfg
    n=len(original_sizes)
    if not stars or any(i<0 or i>=n for s in stars for i in s.image_ids):
        raise ValueError('Stars must contain valid indices into the image list')
    timings={}
    def stage(name,fn,*args):
        if str(device).startswith('cuda'): torch.cuda.synchronize(device)
        start=time.perf_counter()
        out=fn(*args)
        if str(device).startswith('cuda'): torch.cuda.synchronize(device)
        timings[name]=time.perf_counter()-start
        return out
    active,report=stage('reverse_consistency',filter_reverse_constraints,stars,cfg)
    intrinsics,groups=stage('intrinsics',fuse_intrinsics,stars,active,original_sizes,cfg)
    sequential = {tuple(sorted(edge)) for edge in sequential_edges}
    if sequential:
        # Copy scores so resumes never compound the boost or mutate cached stars.
        weighted = []
        boosted = 0
        for s in stars:
            scores = s.scores.clone()
            for q, j in enumerate(s.image_ids[1:], 1):
                if (min(s.image_ids[0], j), max(s.image_ids[0], j)) in sequential:
                    scores[q] *= cfg.sequential_weight
                    boosted += 1
            weighted.append(LocalStar(s.image_ids, s.camera_to_world, s.intrinsics,
                                      s.image_size, scores, s.parallax_degrees))
        stars = weighted
        report['sequential_constraints_boosted'] = boosted
        report['sequential_weight'] = cfg.sequential_weight
    edges={(s.image_ids[0],s.image_ids[q]):RotationEdge(s.camera_to_world[q,:3,:3].T.to(device),float(s.scores[q]))
           for s,mask in zip(stars,active) for q in torch.where(mask)[0].tolist()}
    if not edges: raise ValueError('No rotation constraints after cross-star consistency checks')
    result=stage('rotation_averaging',rotation_averaging,n,edges,cfg.rotation)
    rotations=result.rotations.detach().cpu()
    report['rotation_registered']=int(result.registered.sum())
    report['rotation_edges_before']=len(edges)
    report['rotation_edges_after']=len(result.edges)
    kept=set(result.edges)
    low_parallax=0
    for s,mask in zip(stars,active):
        for q in torch.where(mask)[0].tolist():
            if (s.image_ids[0],s.image_ids[q]) not in kept:
                mask[q]=False
            elif s.parallax_degrees[q]<cfg.min_triangulation_degrees or s.camera_to_world[q,:3,3].norm()<1e-10:
                mask[q]=False
                low_parallax+=1
    initial,scales,offsets,init_report=stage('similarity_initialization',initialize_similarities,stars,active,rotations)
    centers,scales,solver=stage('similarity_averaging',solve_similarities,stars,active,initial,scales,offsets,cfg,device)
    C=torch.full((n,3),torch.nan,dtype=torch.float64)
    registered=torch.zeros(n,dtype=torch.bool)
    for i,c in centers.items():
        C[i]=c
        registered[i]=True
    inverse_scales=torch.full((len(stars),),torch.nan,dtype=torch.float64)
    transforms=torch.full((len(stars),4,4),torch.nan,dtype=torch.float64)
    for s,a in scales.items():
        i=stars[s].image_ids[0]
        inverse_scales[s]=a
        transforms[s]=torch.eye(4,dtype=torch.float64)
        transforms[s,:3,:3]=rotations[i].T/a
        transforms[s,:3,3]=C[i]
    report.update(init_report,low_parallax_edges=low_parallax,registered=int(registered.sum()),
        position_constraints=sum(int(active[s].sum()) for s in scales),
        solver=dict(initial_cost=solver.initial_cost,final_cost=solver.final_cost,
            accepted_steps=solver.accepted_steps,rejected_steps=solver.rejected_steps,termination=solver.termination_reason),
        timings_seconds=timings)
    return dict(rotations=rotations,centers=C,registered=registered,
        rotation_registered=result.registered.cpu(),intrinsics=intrinsics,camera_groups=groups,
        star_centers=torch.tensor([s.image_ids[0] for s in stars]),inverse_star_scales=inverse_scales,
        star_to_world=transforms,initial_centers=torch.stack([initial.get(i,torch.full((3,),torch.nan)) for i in range(n)]),
        points3d=torch.empty(0,3),observations=torch.empty(0,3,dtype=torch.long),fusion_report=report)
