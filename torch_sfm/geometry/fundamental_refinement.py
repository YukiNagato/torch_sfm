"""Batched rank-two F and shared-focal relative-pose Sampson refinement."""
import math
import torch
from torchls.geometry import skew, so3_exp

from .two_view import homogeneous


def _normalize_masked(points, mask):
    weights = mask.to(points)
    count = weights.sum(-1).clamp_min(1)
    center = (points * weights[..., None]).sum(-2) / count[:, None]
    centered = points - center[:, None]
    scale = math.sqrt(2) * count / (centered.norm(dim=-1) * weights).sum(-1).clamp_min(1e-12)
    transform = torch.eye(3, device=points.device, dtype=points.dtype).repeat(len(points), 1, 1)
    transform[:, 0, 0] = transform[:, 1, 1] = scale
    transform[:, :2, 2] = -center * scale[:, None]
    return centered * scale[:, None, None], transform


def sampson_matrix_jacobian(matrix, x, y, scale0, scale1, *, jacobian=True):
    """Pixel-space signed Sampson residual, with normalized-coordinate F.

    x/y are Hartley-normalized homogeneous pixels. The spatial derivatives in
    the denominator include their normalization scales, retaining pixel units.
    The returned Jacobian differentiates row-major matrix entries (9 columns).
    """
    line1 = x @ matrix.transpose(-1,-2)
    line0 = y @ matrix
    numerator = (y*line1).sum(-1)
    a = scale1[:,None].square()
    b = scale0[:,None].square()
    denominator = (a*line1[...,:2].square().sum(-1) + b*line0[...,:2].square().sum(-1)).clamp_min(1e-24)
    inv = denominator.rsqrt()
    residual = numerator*inv
    if not jacobian:
        return residual
    mask = x.new_ones(3)
    mask[2].zero_()
    derivative = (a[...,None,None]*(line1*mask)[..., :,None]*x[...,None,:]
                  + b[...,None,None]*y[..., :,None]*(line0*mask)[...,None,:])
    j = (y[..., :,None]*x[...,None,:]*inv[...,None,None]
         - derivative*(numerator*inv/denominator)[...,None,None])
    return residual, j.flatten(-2)


@torch.no_grad()
def refine_fundamental(matrix, points0, points1, valid, max_iterations=20):
    """Monotonic LM on fixed RANSAC inliers; rank two and unit norm by design.

    F = U diag(cos(theta), sin(theta), 0) V^T. Retraction rotates U/V and
    updates theta. This avoids an unconstrained nine-parameter fit followed
    by a rank projection that changes the optimized objective.
    """
    points0, points1, matrix = points0.double(), points1.double(), matrix.double()
    normalized0, t0 = _normalize_masked(points0, valid)
    normalized1, t1 = _normalize_masked(points1, valid)
    x, y = homogeneous(normalized0), homogeneous(normalized1)
    f = torch.linalg.solve(t1.transpose(-1,-2), matrix) @ torch.linalg.inv(t0)
    f = f/f.norm(dim=(-2,-1),keepdim=True).clamp_min(1e-30)
    weights = valid.to(points0)
    active = valid.sum(-1) >= 8
    damping = f.new_full((len(f),),1e-3)
    basis = skew(torch.eye(3,device=f.device,dtype=f.dtype))
    eye = torch.eye(7,device=f.device,dtype=f.dtype)
    for _ in range(max_iterations):
        u,s,vh = torch.linalg.svd(f)
        ratio_direction = u @ torch.diag_embed(torch.stack((-s[:,1],s[:,0],s[:,2]*0),-1)) @ vh
        tangent = torch.cat((basis[None]@f[:,None], f[:,None]@basis[None], ratio_direction[:,None]),1)
        tangent = tangent.flatten(-2).transpose(-1,-2)
        residual,j9 = sampson_matrix_jacobian(f,x,y,t0[:,0,0],t1[:,0,0])
        residual = residual*weights
        j = (j9@tangent)*weights[...,None]
        normal = j.transpose(-1,-2)@j
        gradient = (j.transpose(-1,-2)@residual[...,None]).squeeze(-1)
        diagonal = normal.diagonal(dim1=-2,dim2=-1).clamp_min(1e-10)
        delta,info = torch.linalg.solve_ex(normal+torch.diag_embed(damping[:,None]*diagonal)+eye*1e-12,-gradient)
        usable = active & (info==0) & torch.isfinite(delta).all(-1)
        delta = torch.where(usable[:,None],delta,0)
        theta = torch.atan2(s[:,1],s[:,0])+delta[:,6]
        values = torch.stack((theta.cos(),theta.sin(),theta*0),-1)
        trial = so3_exp(delta[:,:3]) @ u @ torch.diag_embed(values) @ vh @ so3_exp(delta[:,3:6])
        trial_residual = sampson_matrix_jacobian(trial,x,y,t0[:,0,0],t1[:,0,0],jacobian=False)*weights
        cost,proposed = residual.square().sum(-1),trial_residual.square().sum(-1)
        accept = usable & torch.isfinite(proposed) & (proposed<cost)
        f = torch.where(accept[:,None,None],trial,f)
        damping = (damping*torch.where(accept,1/3,10)).clamp(1e-12,1e12)
        converged = (gradient.abs().amax(-1)<1e-9) | (delta.norm(dim=-1)<1e-10)
        converged |= accept & ((cost-proposed)<1e-10*cost.clamp_min(1))
        active &= ~converged
    result = t1.transpose(-1,-2) @ f @ t0
    return result/result.norm(dim=(-2,-1),keepdim=True).clamp_min(1e-30)


def _shared_model(rotation, translation, focal, principal0, principal1):
    inverse0 = torch.eye(3,device=focal.device,dtype=focal.dtype).repeat(len(focal),1,1)
    inverse0[:,0,0] = inverse0[:,1,1] = focal.reciprocal()
    inverse1 = inverse0.clone()
    inverse0[:,:2,2] = -principal0/focal[:,None]
    inverse1[:,:2,2] = -principal1/focal[:,None]
    essential = skew(translation)@rotation
    return inverse1.transpose(-1,-2)@essential@inverse0, inverse0, inverse1, essential


def shared_focal_jacobian(rotation, translation, focal, principal0, principal1, x, y):
    """Analytic right-SO3/S2/log-focal Jacobian for a shared-camera pair."""
    from .relative_pose import tangent_basis
    matrix,inverse0,inverse1,essential = _shared_model(rotation,translation,focal,principal0,principal1)
    basis = skew(torch.eye(3,device=focal.device,dtype=focal.dtype))
    d_rotation = skew(translation)[:,None]@rotation[:,None]@basis[None]
    d_translation = skew(tangent_basis(translation).transpose(-1,-2))@rotation[:,None]
    diagonal = torch.eye(3,device=focal.device,dtype=focal.dtype)
    diagonal[2,2].zero_()
    d_focal = -(diagonal@essential+essential@diagonal)
    derivatives = torch.cat((d_rotation,d_translation,d_focal[:,None]),1)
    derivatives = inverse1.transpose(-1,-2)[:,None]@derivatives@inverse0[:,None]
    one = torch.ones_like(focal)
    residual,j9 = sampson_matrix_jacobian(matrix,x,y,one,one)
    return residual,j9@derivatives.flatten(-2).transpose(-1,-2)


@torch.no_grad()
def refine_shared_focal(matrix, points0, points1, valid, focal, principal0, principal1, max_iterations=20,
                        focal_ratio_bounds=(.1,10.), use_cuda_graph=False, backend='torch'):
    """Six-DOF local fit (R, t direction, shared focal) on RANSAC inliers.

    Used within six-point LO-RANSAC and for optional refinement of a seven-point
    F initialization. The camera-sharing constraint holds throughout the fit.
    """
    from .two_view_poses import _essential_pose_batch
    from .relative_pose import tangent_basis
    pose = _essential_pose_batch
    prepare = None
    if backend in ('auto','warp_fused') and focal.is_cuda and focal.dtype==torch.float64:
        try:
            from .warp.shared_focal_pose import choose_pose as pose
            from .warp.shared_focal_prepare import prepare
        except ImportError:
            if backend=='warp_fused':raise
    if prepare is not None:
        # Fused LM only reads these inputs; both initial focal views may alias.
        initial_focal = focal
        essential,rays0,rays1 = prepare(matrix,points0,points1,focal,principal0,principal1)
    else:
        focal = focal.clone()
        initial_focal = focal.clone()
        count = len(focal)
        rotation = torch.eye(3,device=focal.device,dtype=focal.dtype).repeat(count,1,1)
        translation = focal.new_tensor([1.,0.,0.]).expand(count,3).clone()
        _,inverse0,inverse1,_ = _shared_model(rotation,translation,focal,principal0,principal1)
        k0,k1 = torch.linalg.inv(inverse0),torch.linalg.inv(inverse1)
        x,y = homogeneous(points0),homogeneous(points1)
        essential = k1.transpose(-1,-2)@matrix@k0
        rays0,rays1 = x@inverse0.transpose(-1,-2),y@inverse1.transpose(-1,-2)
    rotation,translation = pose(essential,rays0,rays1,valid)[:2]
    args=(rotation,translation,focal,initial_focal,points0,points1,valid,principal0,principal1)
    if backend not in ('auto','torch','warp','warp_fused'):
        raise ValueError('Shared-focal refinement backend must be auto, torch, warp or warp_fused')
    implementation=_refine_shared_focal_lm
    resolved='torch'
    if backend!='torch':
        if focal.is_cuda and focal.dtype==torch.float64:
            try:
                if backend in ('auto','warp_fused'):
                    from .warp.shared_focal_lm import refine as implementation
                else:
                    from .warp.shared_focal_refinement import refine as implementation
            except ImportError:
                if backend in ('warp','warp_fused'):raise
            else:resolved='warp_fused' if backend in ('auto','warp_fused') else 'warp'
        elif backend in ('warp','warp_fused'):
            raise ValueError('Warp shared-focal refinement requires CUDA float64')
    if use_cuda_graph and resolved!='warp_fused' and focal.is_cuda and not torch.cuda.is_current_stream_capturing():
        from .shared_focal_graph import run
        return run(args,max_iterations,focal_ratio_bounds,resolved)
    return implementation(*args,max_iterations,focal_ratio_bounds)


@torch.no_grad()
def _refine_shared_focal_lm(rotation,translation,focal,initial_focal,points0,points1,valid,
                            principal0,principal1,max_iterations,focal_ratio_bounds):
    from .relative_pose import tangent_basis
    count=len(focal)
    x,y=homogeneous(points0),homogeneous(points1)
    weights = valid.to(focal)
    active = valid.sum(-1)>=8
    damping = focal.new_full((count,),1e-3)
    eye = torch.eye(6,device=focal.device,dtype=focal.dtype)
    for _ in range(max_iterations):
        residual,j = shared_focal_jacobian(rotation,translation,focal,principal0,principal1,x,y)
        residual,j = residual*weights,j*weights[...,None]
        normal = j.transpose(-1,-2)@j
        gradient = (j.transpose(-1,-2)@residual[...,None]).squeeze(-1)
        diag = normal.diagonal(dim1=-2,dim2=-1).clamp_min(1e-10)
        delta,info = torch.linalg.solve_ex(normal+torch.diag_embed(damping[:,None]*diag)+eye*1e-12,-gradient)
        usable = active & (info==0) & torch.isfinite(delta).all(-1)
        delta = torch.where(usable[:,None],delta,0)
        candidate_focal = focal*delta[:,5].exp()
        usable &= (torch.isfinite(candidate_focal) & (candidate_focal>focal_ratio_bounds[0]*initial_focal)
                   & (candidate_focal<focal_ratio_bounds[1]*initial_focal))
        candidate_focal = torch.where(usable,candidate_focal,focal)
        candidate_rotation = rotation@so3_exp(delta[:,:3])
        candidate_translation = torch.nn.functional.normalize(translation+
            (tangent_basis(translation)@delta[:,3:5,None]).squeeze(-1),dim=-1)
        trial,*_ = _shared_model(candidate_rotation,candidate_translation,candidate_focal,principal0,principal1)
        one = torch.ones_like(focal)
        trial_residual = sampson_matrix_jacobian(trial,x,y,one,one,jacobian=False)*weights
        cost,new_cost = residual.square().sum(-1),trial_residual.square().sum(-1)
        accept = usable & torch.isfinite(new_cost) & (new_cost<cost)
        rotation = torch.where(accept[:,None,None],candidate_rotation,rotation)
        translation = torch.where(accept[:,None],candidate_translation,translation)
        focal = torch.where(accept,candidate_focal,focal)
        damping = (damping*torch.where(accept,1/3,10)).clamp(1e-12,1e12)
        converged = (gradient.abs().amax(-1)<1e-9) | (delta.norm(dim=-1)<1e-10)
        converged |= accept & ((cost-new_cost)<1e-10*cost.clamp_min(1))
        active &= ~converged
    result,*_ = _shared_model(rotation,translation,focal,principal0,principal1)
    return result/result.norm(dim=(-2,-1),keepdim=True).clamp_min(1e-30), focal


@torch.no_grad()
def refine_view_graph_fundamentals(frames, geometries, *, device, batch_size=64, max_iterations=20,
                                   intrinsics=None, model='auto', focal_ratio_bounds=(.1,10.)):
    """Refine only the nonplanar F used for calibration; preserve cached inputs."""
    from dataclasses import replace
    result = dict(geometries)
    pairs = [p for p,g in geometries.items() if g.config in ('CALIBRATED','UNCALIBRATED')
             and g.F is not None and len(g.matches.indices)>=8]
    for start in range(0,len(pairs),batch_size):
        chunk = pairs[start:start+batch_size]
        size = max(len(geometries[p].matches.indices) for p in chunk)
        a = torch.zeros((len(chunk),size,2),device=device,dtype=torch.float64)
        b,mask = torch.zeros_like(a),torch.zeros(a.shape[:2],device=device,dtype=torch.bool)
        for row,(i,j) in enumerate(chunk):
            indices = geometries[i,j].matches.indices
            n = len(indices)
            a[row,:n] = frames[i].keypoints_2d.to(device)[indices[:,0].to(device)]
            b[row,:n] = frames[j].keypoints_2d.to(device)[indices[:,1].to(device)]
            mask[row,:n] = True
        f = torch.stack([geometries[p].F.to(device) for p in chunk])
        from .two_view import camera_matrix
        k = None if intrinsics is None else camera_matrix(intrinsics.to(device=device,dtype=torch.float64))
        if model == 'auto' and k is not None and k.ndim == 2:
            focal = ((k[0,0]+k[1,1])/2).expand(len(chunk))
            principal = k[:2,2].expand(len(chunk),2)
            fitted,_ = refine_shared_focal(f,a,b,mask,focal,principal,principal,max_iterations,focal_ratio_bounds)
        else:
            fitted = refine_fundamental(f,a,b,mask,max_iterations)
        for pair,value in zip(chunk,fitted):
            result[pair] = replace(geometries[pair],F=value)
    return result
