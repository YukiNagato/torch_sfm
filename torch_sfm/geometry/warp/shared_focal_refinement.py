"""Fused six-DOF shared-focal LM, retaining Torch's small linear solve.

Residual/Jacobian terms are reduced directly into normal equations; no dense
per-observation Jacobian is materialized. Converged and padded models skip
residual work on the GPU. Damping, bounds and stopping rules match the reference.
"""
import torch
import warp as wp
from .relative_pose import _matrix, _coefficient, _skew, _tangent

wp.set_module_options({'enable_backward': False})
Vec6=wp.types.vector(length=6,dtype=wp.float64)
BLOCK=128


@wp.func
def _inverse(f: wp.float64,p: wp.array2d(dtype=wp.float64),i: int):
    z,o=wp.float64(0.0),wp.float64(1.0)
    return wp.mat33d(o/f,z,-p[i,0]/f,z,o/f,-p[i,1]/f,z,z,o)


@wp.kernel
def _prepare(rotations: wp.array3d(dtype=wp.float64),translations: wp.array2d(dtype=wp.float64),
             focal: wp.array(dtype=wp.float64),principal0: wp.array2d(dtype=wp.float64),
             principal1: wp.array2d(dtype=wp.float64),coefficients: wp.array4d(dtype=wp.float64)):
    i=wp.tid()
    r=_matrix(rotations,i)
    t=wp.vec3d(translations[i,0],translations[i,1],translations[i,2])
    e=_skew(t)*r
    inv0,inv1t=_inverse(focal[i],principal0,i),wp.transpose(_inverse(focal[i],principal1,i))
    first,second=_tangent(t)
    for k in range(7):
        derivative=e
        if k>0 and k<4:
            unit=wp.vec3d(wp.float64(0.0));unit[k-1]=wp.float64(1.0)
            derivative=e*_skew(unit)
        elif k==4:derivative=_skew(first)*r
        elif k==5:derivative=_skew(second)*r
        elif k==6:
            for row in range(3):
                for col in range(3):
                    factor=int(row<2)+int(col<2)
                    derivative[row,col]=-wp.float64(factor)*e[row,col]
        f=inv1t*derivative*inv0
        for row in range(3):
            for col in range(3):coefficients[i,k,row,col]=f[row,col]


@wp.func
def _terms(coefficients: wp.array4d(dtype=wp.float64),a: wp.vec3d,b: wp.vec3d,i: int,jacobian: int):
    f=_coefficient(coefficients,i,0)
    l1,l0=f*a,wp.transpose(f)*b
    num=wp.dot(b,l1)
    norm2=l0[0]*l0[0]+l0[1]*l0[1]+l1[0]*l1[0]+l1[1]*l1[1]
    inv=wp.float64(1.0)/wp.sqrt(wp.max(norm2,wp.float64(1e-24)))
    residual=num*inv;j=Vec6(wp.float64(0.0))
    if jacobian!=0:
        for k in range(6):
            df=_coefficient(coefficients,i,k+1)
            dl1,dl0=df*a,wp.transpose(df)*b
            # Matches the reference matrix Jacobian, including its finite guard.
            half=l0[0]*dl0[0]+l0[1]*dl0[1]+l1[0]*dl1[0]+l1[1]*dl1[1]
            j[k]=wp.dot(b,dl1)*inv-num*inv*inv*inv*half
    return residual,j


@wp.kernel
def _partials(coefficients: wp.array4d(dtype=wp.float64),points0: wp.array3d(dtype=wp.float64),
              points1: wp.array3d(dtype=wp.float64),valid: wp.array2d(dtype=wp.bool),
              active: wp.array(dtype=wp.bool),jacobian: int,partials: wp.array3d(dtype=wp.float64)):
    i,chunk,lane=wp.tid();p=chunk*BLOCK+lane
    residual=wp.float64(0.0);j=Vec6(wp.float64(0.0))
    if active[i] and p<points0.shape[1]:
        if valid[i,p]:
            a=wp.vec3d(points0[i,p,0],points0[i,p,1],wp.float64(1.0))
            b=wp.vec3d(points1[i,p,0],points1[i,p,1],wp.float64(1.0))
            residual,j=_terms(coefficients,a,b,i,jacobian)
    if jacobian!=0:
        index=int(0)
        for row in range(6):
            for col in range(row,6):
                wp.tile_store(partials[i,chunk],wp.tile_sum(wp.tile(j[row]*j[col])),offset=index)
                index+=1
        for row in range(6):
            wp.tile_store(partials[i,chunk],wp.tile_sum(wp.tile(j[row]*residual)),offset=21+row)
    wp.tile_store(partials[i,chunk],wp.tile_sum(wp.tile(residual*residual)),offset=27)


@wp.kernel
def _system(partials: wp.array3d(dtype=wp.float64),damping: wp.array(dtype=wp.float64),
            normal: wp.array3d(dtype=wp.float64),gradient: wp.array2d(dtype=wp.float64),
            rhs: wp.array2d(dtype=wp.float64),cost: wp.array(dtype=wp.float64)):
    i,index=wp.tid();value=wp.float64(0.0)
    for chunk in range(partials.shape[1]):value+=partials[i,chunk,index]
    if index<21:
        position=int(0)
        for row in range(6):
            for col in range(row,6):
                if position==index:
                    if row==col:value+=damping[i]*wp.max(value,wp.float64(1e-10))+wp.float64(1e-12)
                    normal[i,row,col]=value;normal[i,col,row]=value
                position+=1
    elif index<27:
        gradient[i,index-21]=value;rhs[i,index-21]=-value
    else:cost[i]=value


@wp.kernel
def _propose(rotations: wp.array3d(dtype=wp.float64),translations: wp.array2d(dtype=wp.float64),
             focal: wp.array(dtype=wp.float64),initial: wp.array(dtype=wp.float64),
             delta: wp.array2d(dtype=wp.float64),info: wp.array(dtype=wp.int32),
             gradient: wp.array2d(dtype=wp.float64),active: wp.array(dtype=wp.bool),
             principal0: wp.array2d(dtype=wp.float64),principal1: wp.array2d(dtype=wp.float64),
             lower: wp.float64,upper: wp.float64,proposed_r: wp.array3d(dtype=wp.float64),
             proposed_t: wp.array2d(dtype=wp.float64),proposed_f: wp.array(dtype=wp.float64),
             coefficients: wp.array4d(dtype=wp.float64),usable: wp.array(dtype=wp.bool),small: wp.array(dtype=wp.bool)):
    i=wp.tid();good=active[i] and info[i]==0
    for k in range(6):good=good and wp.isfinite(delta[i,k])
    d=Vec6(wp.float64(0.0));gradmax,norm2=wp.float64(0.0),wp.float64(0.0)
    for k in range(6):
        if good:d[k]=delta[i,k]
        gradmax=wp.max(gradmax,wp.abs(gradient[i,k]));norm2+=d[k]*d[k]
    f=focal[i]*wp.exp(d[5])
    good=good and wp.isfinite(f) and f>lower*initial[i] and f<upper*initial[i]
    if not good:f=focal[i]
    theta2=d[0]*d[0]+d[1]*d[1]+d[2]*d[2]
    a,b=wp.float64(1.0),wp.float64(0.5)
    if theta2<wp.float64(1e-4):
        a=wp.float64(1.0)-theta2/wp.float64(6.0)+theta2*theta2/wp.float64(120.0)
        b=wp.float64(0.5)-theta2/wp.float64(24.0)+theta2*theta2/wp.float64(720.0)
    else:
        theta=wp.sqrt(theta2);a=wp.sin(theta)/theta
        half=theta*wp.float64(.5);sinc=wp.sin(half)/half;b=wp.float64(.5)*sinc*sinc
    omega=_skew(wp.vec3d(d[0],d[1],d[2]))
    eye=wp.mat33d(wp.float64(1.0),wp.float64(0.0),wp.float64(0.0),wp.float64(0.0),wp.float64(1.0),wp.float64(0.0),wp.float64(0.0),wp.float64(0.0),wp.float64(1.0))
    r=_matrix(rotations,i)*(eye+a*omega+b*(omega*omega))
    t=wp.vec3d(translations[i,0],translations[i,1],translations[i,2]);first,second=_tangent(t)
    t=t+first*d[3]+second*d[4];t=t/wp.max(wp.length(t),wp.float64(1e-12))
    essential=_skew(t)*r
    matrix=wp.transpose(_inverse(f,principal1,i))*essential*_inverse(f,principal0,i)
    for row in range(3):
        proposed_t[i,row]=t[row]
        for col in range(3):
            proposed_r[i,row,col]=r[row,col];coefficients[i,0,row,col]=matrix[row,col]
    proposed_f[i]=f;usable[i]=good
    small[i]=gradmax<wp.float64(1e-9) or wp.sqrt(norm2)<wp.float64(1e-10)


@wp.kernel
def _accept(partials: wp.array3d(dtype=wp.float64),cost: wp.array(dtype=wp.float64),
            proposed_r: wp.array3d(dtype=wp.float64),proposed_t: wp.array2d(dtype=wp.float64),
            proposed_f: wp.array(dtype=wp.float64),usable: wp.array(dtype=wp.bool),small: wp.array(dtype=wp.bool),
            rotations: wp.array3d(dtype=wp.float64),translations: wp.array2d(dtype=wp.float64),
            focal: wp.array(dtype=wp.float64),damping: wp.array(dtype=wp.float64),active: wp.array(dtype=wp.bool),
            accepted_scale: wp.float64):
    i=wp.tid();proposed=wp.float64(0.0)
    for chunk in range(partials.shape[1]):proposed+=partials[i,chunk,27]
    accept=usable[i] and wp.isfinite(proposed) and proposed<cost[i]
    scale=wp.float64(10.0)
    if accept:
        scale=accepted_scale;focal[i]=proposed_f[i]
        for row in range(3):
            translations[i,row]=proposed_t[i,row]
            for col in range(3):rotations[i,row,col]=proposed_r[i,row,col]
    damping[i]=wp.clamp(damping[i]*scale,wp.float64(1e-12),wp.float64(1e12))
    converged=small[i] or (accept and cost[i]-proposed<wp.float64(1e-10)*wp.max(cost[i],wp.float64(1.0)))
    active[i]=active[i] and not converged


def refine(rotation,translation,focal,initial,points0,points1,valid,principal0,principal1,iterations,bounds):
    from ..fundamental_refinement import _shared_model
    rotation,translation,focal=rotation.clone(),translation.clone(),focal.clone()
    n=len(focal);chunks=(points0.shape[1]+BLOCK-1)//BLOCK
    coefficients=focal.new_empty((n,7,3,3));partials=focal.new_empty((n,chunks,28))
    system=focal.new_empty((n,6,6));gradient=focal.new_empty((n,6));rhs=torch.empty_like(gradient)
    cost=focal.new_empty(n);damping=focal.new_full((n,),1e-3);active=valid.sum(-1)>=8
    proposed_r=torch.empty_like(rotation);proposed_t=torch.empty_like(translation);proposed_f=torch.empty_like(focal)
    usable=torch.empty_like(active);small=torch.empty_like(active)
    stream=wp.stream_from_torch(torch.cuda.current_stream(focal.device))
    def arrays(*values):return [wp.from_torch(value) for value in values]
    prepare=arrays(rotation,translation,focal,principal0,principal1,coefficients)
    terms=arrays(coefficients,points0,points1,valid)
    system_args=arrays(partials,damping,system,gradient,rhs,cost)
    # torch.where(bool_mask, 1/3, 10) uses Torch's default floating dtype,
    # even when the state is float64. Preserve that rounding in the reference.
    accepted_scale=float(torch.tensor(1/3,device='cpu',dtype=torch.get_default_dtype()))
    accept_args=arrays(partials,cost,proposed_r,proposed_t,proposed_f,usable,small,rotation,translation,focal,damping,active)+[wp.float64(accepted_scale)]
    for _ in range(iterations):
        wp.launch(_prepare,dim=n,inputs=prepare,stream=stream)
        wp.launch(_partials,dim=(n,chunks,BLOCK),block_dim=BLOCK,
                  inputs=terms+arrays(active)+[1]+arrays(partials),stream=stream)
        wp.launch(_system,dim=(n,28),inputs=system_args,stream=stream)
        delta,info=torch.linalg.solve_ex(system,rhs)
        wp.launch(_propose,dim=n,inputs=arrays(rotation,translation,focal,initial,delta,info,gradient,active,principal0,principal1)
                  +[wp.float64(bounds[0]),wp.float64(bounds[1])]
                  +arrays(proposed_r,proposed_t,proposed_f,coefficients,usable,small),stream=stream)
        wp.launch(_partials,dim=(n,chunks,BLOCK),block_dim=BLOCK,
                  inputs=terms+arrays(usable)+[0]+arrays(partials),stream=stream)
        wp.launch(_accept,dim=n,inputs=accept_args,stream=stream)
    result,*_=_shared_model(rotation,translation,focal,principal0,principal1)
    return result/result.norm(dim=(-2,-1),keepdim=True).clamp_min(1e-30),focal
