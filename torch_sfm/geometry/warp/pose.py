"""Fused essential pre-refinement and final E/F pose, masks and angle metadata."""
import math
import torch
import warp as wp
from .scoring import vec, vote, error, BLOCK
from .six_point import _sync
from .essential_pose import _decompose_matrix
from .relative_pose import _matrix, _skew
from .relative_pose_lm import refine_pair
from ..essential_scoring import prepare_essential_observations
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False,'max_unroll':1})

@wp.func
def midpoint(r:wp.mat33d,t:wp.vec3d,a:wp.vec3d,b:wp.vec3d):
    a=wp.normalize(a);b=wp.transpose(r)*wp.normalize(b);center=-(wp.transpose(r)*t)
    cosine=wp.dot(a,b);den=wp.max(wp.float64(1.0)-cosine*cosine,wp.float64(1e-15))
    ac,bc=wp.dot(a,center),wp.dot(b,center)
    d0,d1=(ac-cosine*bc)/den,(cosine*ac-bc)/den
    xyz=(d0*a+center+d1*b)*wp.float64(.5)
    good=d0>wp.float64(1e-8) and d1>wp.float64(1e-8)
    for i in range(3):good=good and wp.isfinite(xyz[i])
    d=xyz/wp.max(wp.length(xyz),wp.float64(1e-12))-a
    return good,cosine,wp.dot(d,d)

@wp.func
def choose(model:wp.mat33d,ray0:wp.array3d(dtype=wp.float64),ray1:wp.array3d(dtype=wp.float64),
           masks:wp.array2d(dtype=wp.bool),pair:int,lane:int):
    r0,r1,t=wp.mat33d(),wp.mat33d(),wp.vec3d();good=False
    if lane==0:r0,r1,t,good=_decompose_matrix(model)
    good=wp.tile_extract(wp.tile(int(good)),0)!=0
    for i in range(3):
        t[i]=wp.tile_extract(wp.tile(t[i]),0)
        for j in range(3):
            r0[i,j]=wp.tile_extract(wp.tile(r0[i,j]),0);r1[i,j]=wp.tile_extract(wp.tile(r1[i,j]),0)
    # Warp needs explicit value copies here: the winner changes inside the loop.
    best=int(-1);cost=wp.float64(wp.inf);best_r,best_t=wp.mat33d(r0),wp.vec3d(t)
    for k in range(4):
        r=r0
        if k>=2:r=r1
        direction=t
        if k%2==1:direction=-t
        hits=int(0);partial=wp.float64(0.0)
        if good:
            for j in range(lane,masks.shape[1],BLOCK):
                if masks[pair,j]:
                    positive,cosine,e=midpoint(r,direction,vec(ray0,pair,j),vec(ray1,pair,j))
                    if positive:hits+=1;partial+=e
        count=wp.tile_extract(wp.tile_sum(wp.tile(hits)),0)
        total=wp.tile_extract(wp.tile_sum(wp.tile(partial)),0)
        if count>best or (count==best and total<cost):best=count;cost=total;best_r=r;best_t=direction
        _sync()
    return best_r,best_t,good

@wp.func
def initialize(model:wp.mat33d,ray0:wp.array3d(dtype=wp.float64),ray1:wp.array3d(dtype=wp.float64),
               masks:wp.array2d(dtype=wp.bool),pair:int,lane:int,minimum:int,
               rotations:wp.array3d(dtype=wp.float64),translations:wp.array2d(dtype=wp.float64),refine_mask:wp.array2d(dtype=wp.bool)):
    r,t,good=choose(model,ray0,ray1,masks,pair,lane)
    hits=int(0)
    for j in range(lane,masks.shape[1],BLOCK):
        positive,cosine,e=midpoint(r,t,vec(ray0,pair,j),vec(ray1,pair,j))
        inside=good and masks[pair,j] and positive
        refine_mask[pair,j]=inside;hits+=int(inside)
    count=wp.tile_extract(wp.tile_sum(wp.tile(hits)),0)
    if lane==0:
        for i in range(3):
            translations[pair,i]=t[i]
            for j in range(3):rotations[pair,i,j]=r[i,j]
    if count<minimum:
        for j in range(lane,masks.shape[1],BLOCK):refine_mask[pair,j]=False
    _sync()
    return good and count>=minimum

@wp.kernel
def pre_refine(models:wp.array3d(dtype=wp.float64),masks:wp.array2d(dtype=wp.bool),found:wp.array(dtype=wp.bool),
               p0:wp.array3d(dtype=wp.float64),p1:wp.array3d(dtype=wp.float64),valid:wp.array2d(dtype=wp.bool),
               ray0:wp.array3d(dtype=wp.float64),ray1:wp.array3d(dtype=wp.float64),jac0:wp.array4d(dtype=wp.float64),jac1:wp.array4d(dtype=wp.float64),
               x0:wp.array3d(dtype=wp.float64),x1:wp.array3d(dtype=wp.float64),inv0:wp.array3d(dtype=wp.float64),inv1:wp.array3d(dtype=wp.float64),
               threshold:wp.float64,minimum:int,rotations:wp.array3d(dtype=wp.float64),translations:wp.array2d(dtype=wp.float64),
               refine_mask:wp.array2d(dtype=wp.bool),proposal:wp.array2d(dtype=wp.bool)):
    pair,lane=wp.tid()
    if found[pair]:
        model=_matrix(models,pair)
        can_refine=initialize(model,ray0,ray1,masks,pair,lane,minimum,rotations,translations,refine_mask)
        if can_refine:
            refine_pair(pair,lane,rotations,translations,x0,x1,inv0,inv1,refine_mask,15)
            _sync()
            r=_matrix(rotations,pair);t=wp.vec3d(translations[pair,0],translations[pair,1],translations[pair,2])
            fitted=_skew(t)*r
            old_r,old_t,old_good=vote(model,ray0,ray1,valid,pair,lane)
            new_r,new_t,new_good=vote(fitted,ray0,ray1,valid,pair,lane)
            old_cost,new_cost=wp.float64(0.0),wp.float64(0.0)
            for j in range(lane,valid.shape[1],BLOCK):
                inside=False
                if valid[pair,j]:
                    old=error(model,0,old_r,old_t,p0,p1,ray0,ray1,jac0,jac1,pair,j)
                    new=error(fitted,0,new_r,new_t,p0,p1,ray0,ray1,jac0,jac1,pair,j)
                    old_cost+=wp.min(old,threshold);new_cost+=wp.min(new,threshold);inside=new<=threshold
                proposal[pair,j]=inside
            old_cost=wp.tile_extract(wp.tile_sum(wp.tile(old_cost)),0);new_cost=wp.tile_extract(wp.tile_sum(wp.tile(new_cost)),0)
            if new_good and new_cost<old_cost:
                for j in range(lane,valid.shape[1],BLOCK):masks[pair,j]=proposal[pair,j]
                if lane==0:
                    for i in range(3):
                        for j in range(3):models[pair,i,j]=fitted[i,j]

@wp.func_native('''
int left=0,right=a.shape[1]-1;
while(left<right){
    double pivot=wp::index(a,pair,(left+right)/2);int i=left,j=right;
    while(i<=j){
        while(wp::index(a,pair,i)<pivot)++i;
        while(wp::index(a,pair,j)>pivot)--j;
        if(i<=j){double v=wp::index(a,pair,i);wp::index(a,pair,i)=wp::index(a,pair,j);wp::index(a,pair,j)=v;++i;--j;}
    }
    if(k<=j)right=j;else if(k>=i)left=i;else break;
}
return wp::index(a,pair,k);
''')
def median(a:wp.array2d(dtype=wp.float64),pair:int,k:int)->wp.float64: ...

@wp.kernel
def final_pose(models:wp.array3d(dtype=wp.float64),masks:wp.array2d(dtype=wp.bool),calibrated:wp.array(dtype=wp.bool),minimum:wp.array(dtype=wp.int32),
               p0:wp.array3d(dtype=wp.float64),p1:wp.array3d(dtype=wp.float64),
               ray0:wp.array3d(dtype=wp.float64),ray1:wp.array3d(dtype=wp.float64),x0:wp.array3d(dtype=wp.float64),x1:wp.array3d(dtype=wp.float64),
               inv0:wp.array3d(dtype=wp.float64),inv1:wp.array3d(dtype=wp.float64),threshold:wp.float64,refine:bool,
               rotations:wp.array3d(dtype=wp.float64),translations:wp.array2d(dtype=wp.float64),refine_mask:wp.array2d(dtype=wp.bool),
               output:wp.array2d(dtype=wp.bool),angles:wp.array2d(dtype=wp.float64),metadata:wp.array2d(dtype=wp.float64)):
    pair,lane=wp.tid();model=_matrix(models,pair)
    can_refine=initialize(model,ray0,ray1,masks,pair,lane,minimum[pair],rotations,translations,refine_mask)
    if calibrated[pair] and refine and can_refine:
        refine_pair(pair,lane,rotations,translations,x0,x1,inv0,inv1,refine_mask,15)
        _sync()
    r=_matrix(rotations,pair);t=wp.vec3d(translations[pair,0],translations[pair,1],translations[pair,2])
    fitted=_skew(t)*r;f=wp.transpose(_matrix(inv1,pair))*fitted*_matrix(inv0,pair)
    count=int(0)
    for j in range(lane,masks.shape[1],BLOCK):
        positive,cosine,e=midpoint(r,t,vec(ray0,pair,j),vec(ray1,pair,j))
        inside=masks[pair,j] and positive
        if calibrated[pair]:
            a=wp.vec3d(p0[pair,j,0],p0[pair,j,1],wp.float64(1.0));b=wp.vec3d(p1[pair,j,0],p1[pair,j,1],wp.float64(1.0))
            l1,l0=f*a,wp.transpose(f)*b;n=wp.dot(b,l1)
            e=n*n/wp.max(l0[0]*l0[0]+l0[1]*l0[1]+l1[0]*l1[0]+l1[1]*l1[1],wp.float64(1e-30));inside=inside and e<=threshold
        output[pair,j]=inside;count+=int(inside)
        angles[pair,j]=wp.float64(wp.inf)
        if inside:angles[pair,j]=wp.acos(wp.clamp(cosine,wp.float64(-1.0),wp.float64(1.0)))*wp.float64(57.29577951308232)
    count=wp.tile_extract(wp.tile_sum(wp.tile(count)),0);_sync()
    if lane==0:
        angle=wp.float64(0.0)
        if count>0:angle=median(angles,pair,(count-1)//2)
        metadata[pair,0]=wp.float64(count);metadata[pair,1]=angle;metadata[pair,2]=wp.length(t)
        if calibrated[pair]:
            for i in range(3):
                for j in range(3):models[pair,i,j]=fitted[i,j]


def _workspace(points):
    n,m=points.shape[:2]
    return (points.new_empty((n,3,3)),points.new_empty((n,3)),torch.empty((n,m),device=points.device,dtype=torch.bool),
            torch.empty((n,m),device=points.device,dtype=torch.bool))

def _homogeneous(points):return torch.cat((points,torch.ones_like(points[...,:1])),-1)

def pre_refine_essential(models,masks,found,p0,p1,valid,inv0,inv1,cfg):
    args=[models,masks,found,p0,p1,valid,*prepare_essential_observations(p0,p1,inv0,inv1),
          _homogeneous(p0),_homogeneous(p1),inv0,inv1,wp.float64(cfg.max_error**2),cfg.min_num_inliers,*_workspace(p0)]
    wp.launch(pre_refine,dim=(len(p0),BLOCK),block_dim=BLOCK,inputs=[wp.from_torch(x) if isinstance(x,torch.Tensor) else x for x in args],
              stream=wp.stream_from_torch(torch.cuda.current_stream(p0.device)))


def finish_poses(results,members,essential,fundamental,p0,p1,rays0,rays1,masks,inv0,inv1,matrix0,matrix1,indices,scores,counts,cfg):
    from ...correspondence import Matches
    group=torch.tensor(members,device=p0.device);a,b=p0[group],p1[group]
    calibrated=torch.tensor([results[i].config=='CALIBRATED' for i in members],device=p0.device)
    inverse0,inverse1=inv0[group],inv1[group]
    # F -> E is a direct basis change; classification itself remains unchanged.
    from_f=matrix1[group].transpose(-1,-2)@fundamental[group]@matrix0[group]
    models=torch.where(calibrated[:,None,None],essential[group],from_f)
    minimums=[max(cfg.min_num_inliers,math.ceil(cfg.min_inlier_ratio*counts[i][3]))
              if results[i].config=='CALIBRATED' else cfg.min_num_inliers for i in members]
    minimum=torch.tensor(minimums,device=p0.device,dtype=torch.int32)
    rotation,translation,refine_mask,output=_workspace(a)
    angles=a.new_empty(output.shape);meta=a.new_empty((len(group),3))
    args=[models,masks[group],calibrated,minimum,a,b,rays0[group],rays1[group],_homogeneous(a),_homogeneous(b),inverse0,inverse1,
          wp.float64(cfg.max_error**2),cfg.refine_relative_pose,rotation,translation,refine_mask,output,angles,meta]
    wp.launch(final_pose,dim=(len(group),BLOCK),block_dim=BLOCK,inputs=[wp.from_torch(x) if isinstance(x,torch.Tensor) else x for x in args],
              stream=wp.stream_from_torch(torch.cuda.current_stream(p0.device)))
    metadata=meta.cpu().tolist();_,columns=torch.where(output);selected=columns.split([int(row[0]) for row in metadata])
    for local,i in enumerate(members):
        result=results[i];count,angle,norm=metadata[local]
        if result.config=='CALIBRATED':result.E=models[local]
        if norm<1e-8 or count<minimums[local]:
            result.config='DEGENERATE';result.matches=Matches(indices[i][:0],scores[i][:0]);continue
        result.R,result.t,result.tri_angle=rotation[local],translation[local],angle
        result.valid_translation=norm>1e-8 and angle>=cfg.min_tri_angle
        result.matches=Matches(indices[i][selected[local]],scores[i][selected[local]])
