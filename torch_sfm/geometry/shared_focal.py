"""Batched shared-focal six-point solver using PoseLib's elimination template.

The generated polynomial file contains the upstream BSD notice. This is the pure PyTorch reference. The production Warp solver lives in
``geometry.warp`` and does not call this implementation.
"""
import torch
from . import shared_focal_polynomial as polynomial
from .two_view import homogeneous


@torch.no_grad()
def shared_focal_six_point(points0, points1, principal0, principal1, backend='torch'):
    """Return pixel F matrices, positive shared focals, and sample owners.

    Inputs have shape (B,6,2), principal points (B,2). Invalid/rank-deficient
    samples and non-real/non-physical polynomial roots produce no hypothesis.
    """
    if backend != 'torch':
        raise ValueError('Use the Warp verification backend for CUDA six-point estimation')
    a,b=points0-principal0[:,None],points1-principal1[:,None]
    scale=(a.norm(dim=-1).mean(-1)+b.norm(dim=-1).mean(-1))/2
    good=torch.isfinite(scale)&(scale>1e-12)
    scale=scale.clamp_min(1e-12)
    x=torch.nn.functional.normalize(homogeneous(a/scale[:,None,None]),dim=-1)
    y=torch.nn.functional.normalize(homogeneous(b/scale[:,None,None]),dim=-1)
    design=(x[:,:,:,None]*y[:,:,None,:]).flatten(-2)
    _,singular,vh=torch.linalg.svd(design,full_matrices=True)
    basis=vh[:,-3:]
    good &= singular[:,-1]>1e-10*singular[:,0]
    data=basis.flatten(-2)
    coefficients=polynomial.coefficients(data)
    c0=data.new_zeros((len(data),31,31));c1=data.new_zeros((len(data),31,15))
    i0=torch.tensor(polynomial.C0_IND,device=data.device);i1=torch.tensor(polynomial.C1_IND,device=data.device)
    c0[:,i0%31,i0//31]=coefficients[:,polynomial.COEFFS0_IND]
    c1[:,i1%31,i1//31]=coefficients[:,polynomial.COEFFS1_IND]
    solved,info=torch.linalg.solve_ex(c0,c1)
    good &= (info==0)&torch.isfinite(solved).all(dim=(-2,-1))
    ids=torch.arange(len(data),device=data.device)[good]
    if len(ids)==0:return points0.new_empty((0,3,3)),points0.new_empty(0),ids
    eye=torch.eye(15,device=data.device,dtype=data.dtype).expand(len(ids),-1,-1)
    reduction=torch.cat((-solved[good,-8:],eye),dim=1)
    action=reduction[:,polynomial.AM_IND]
    eigenvalues,vectors=torch.linalg.eig(action)
    coordinates=vectors[:,(1,12),:]/vectors[:,:1,:]
    z=eigenvalues.real;xx=coordinates[:,0].real;w=coordinates[:,1].real
    real=(eigenvalues.imag.abs()<1e-7*(1+z.abs()))&(coordinates.imag.abs().amax(1)<1e-7*(1+coordinates.real.abs().amax(1)))
    valid=real&(w>1e-8)&torch.isfinite(xx)&torch.isfinite(z)&torch.isfinite(w)
    owner=ids[:,None].expand_as(valid)[valid]
    xx,z,w=xx[valid],z[valid],w[valid]
    f_norm=w.rsqrt()
    fv=basis[owner,0]+xx[:,None]*basis[owner,1]+z[:,None]*basis[owner,2]
    fundamental=torch.nn.functional.normalize(fv,dim=-1).reshape(-1,3,3).transpose(-1,-2)
    factors=torch.stack((f_norm,f_norm,torch.ones_like(f_norm)),-1)
    essential=fundamental*factors[:,:,None]*factors[:,None,:]
    essential=torch.nn.functional.normalize(essential.flatten(-2),dim=-1).reshape(-1,3,3)
    eet=essential@essential.transpose(-1,-2)
    constraint=2*eet@essential-eet.diagonal(dim1=-2,dim2=-1).sum(-1)[:,None,None]*essential
    valid=torch.isfinite(fundamental).all(dim=(-2,-1))&(constraint.norm(dim=(-2,-1))<1e-5)
    fundamental,f_norm,owner,essential=fundamental[valid],f_norm[valid],owner[valid],essential[valid]
    if len(owner):
        ray_scale=torch.stack((1/f_norm,1/f_norm,torch.ones_like(f_norm)),-1)
        rays0,rays1=x[owner]*ray_scale[:,None],y[owner]*ray_scale[:,None]
        from .two_view_poses import _essential_pose_batch
        _,_,positive=_essential_pose_batch(essential,rays0,rays1,
            torch.ones((len(owner),6),device=data.device,dtype=torch.bool))
        cheiral=positive.all(-1)
        fundamental,f_norm,owner=fundamental[cheiral],f_norm[cheiral],owner[cheiral]
    transform0=torch.eye(3,device=data.device,dtype=data.dtype).repeat(len(owner),1,1)
    transform1=transform0.clone()
    transform0[:,:2,:2]/=scale[owner,None,None];transform1[:,:2,:2]/=scale[owner,None,None]
    transform0[:,:2,2]=-principal0[owner]/scale[owner,None]
    transform1[:,:2,2]=-principal1[owner]/scale[owner,None]
    fundamental=transform1.transpose(-1,-2)@fundamental@transform0
    return fundamental/fundamental.norm(dim=(-2,-1),keepdim=True),f_norm*scale[owner],owner
