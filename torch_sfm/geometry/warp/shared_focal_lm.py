"""One cooperative block per local shared-focal fit, including all LM steps."""
import torch
import warp as wp
from .relative_pose import _matrix, _skew, _tangent

wp.init()
wp.set_module_options({'enable_backward':False,'max_unroll':8,'fast_math':False})
Vec6=wp.types.vector(length=6,dtype=wp.float64)
Vec28=wp.types.vector(length=28,dtype=wp.float64)
Mat6=wp.types.matrix(shape=(6,6),dtype=wp.float64)
Coefficients=wp.types.matrix(shape=(7,9),dtype=wp.float64)
BLOCK=64


@wp.func
def _inverse(f: wp.float64,p: wp.vec2d):
    z,o=wp.float64(0.0),wp.float64(1.0)
    return wp.mat33d(o/f,z,-p[0]/f,z,o/f,-p[1]/f,z,z,o)


@wp.func
def _model(r: wp.mat33d,t: wp.vec3d,f: wp.float64,p: wp.vec2d,q: wp.vec2d):
    return wp.transpose(_inverse(f,q))*_skew(t)*r*_inverse(f,p)


@wp.func
def _coefficients(r: wp.mat33d,t: wp.vec3d,f: wp.float64,p: wp.vec2d,q: wp.vec2d):
    result=Coefficients();essential=_skew(t)*r
    inv0,inv1=_inverse(f,p),wp.transpose(_inverse(f,q));first,second=_tangent(t)
    for k in range(7):
        derivative=essential
        if k>0 and k<4:
            unit=wp.vec3d(wp.float64(0.0));unit[k-1]=wp.float64(1.0)
            derivative=essential*_skew(unit)
        elif k==4:derivative=_skew(first)*r
        elif k==5:derivative=_skew(second)*r
        elif k==6:
            for row in range(3):
                for col in range(3):derivative[row,col]=-wp.float64(int(row<2)+int(col<2))*essential[row,col]
        matrix=inv1*derivative*inv0
        for row in range(3):
            for col in range(3):result[k,3*row+col]=matrix[row,col]
    return result


@wp.func
def _coefficient(c: Coefficients,k: int):
    return wp.mat33d(c[k,0],c[k,1],c[k,2],c[k,3],c[k,4],c[k,5],c[k,6],c[k,7],c[k,8])


@wp.func
def _residual(f: wp.mat33d,a: wp.vec3d,b: wp.vec3d):
    l1,l0=f*a,wp.transpose(f)*b
    denominator=l0[0]*l0[0]+l0[1]*l0[1]+l1[0]*l1[0]+l1[1]*l1[1]
    return wp.dot(b,l1)/wp.sqrt(wp.max(denominator,wp.float64(1e-24)))


@wp.func
def _terms(c: Coefficients,a: wp.vec3d,b: wp.vec3d):
    f=_coefficient(c,0);l1,l0=f*a,wp.transpose(f)*b;num=wp.dot(b,l1)
    den=l0[0]*l0[0]+l0[1]*l0[1]+l1[0]*l1[0]+l1[1]*l1[1]
    inv=wp.float64(1.0)/wp.sqrt(wp.max(den,wp.float64(1e-24)))
    j=Vec6()
    for k in range(6):
        df=_coefficient(c,k+1);dl1,dl0=df*a,wp.transpose(df)*b
        half=l0[0]*dl0[0]+l0[1]*dl0[1]+l1[0]*dl1[0]+l1[1]*dl1[1]
        j[k]=wp.dot(b,dl1)*inv-num*inv*inv*inv*half
    return num*inv,j


@wp.func
def _solve(normal: Mat6,rhs: Vec6):
    a=normal;b=rhs;good=wp.bool(True)
    for col in range(6):
        pivot=col;largest=wp.abs(a[col,col])
        for row in range(col+1,6):
            if wp.abs(a[row,col])>largest:largest=wp.abs(a[row,col]);pivot=row
        if largest==wp.float64(0.0) or not wp.isfinite(largest):good=False;break
        for k in range(col,6):
            temporary=a[col,k];a[col,k]=a[pivot,k];a[pivot,k]=temporary
        temporary=b[col];b[col]=b[pivot];b[pivot]=temporary
        for row in range(col+1,6):
            factor=a[row,col]/a[col,col]
            for k in range(col+1,6):a[row,k]-=factor*a[col,k]
            b[row]-=factor*b[col]
    delta=Vec6()
    if good:
        for step in range(6):
            row=5-step;value=b[row]
            for col in range(row+1,6):value-=a[row,col]*delta[col]
            delta[row]=value/a[row,row];good=good and wp.isfinite(delta[row])
    if not good:delta=Vec6()
    return delta,good


@wp.func
def _rotation_update(r: wp.mat33d,delta: Vec6):
    omega=_skew(wp.vec3d(delta[0],delta[1],delta[2]))
    theta2=delta[0]*delta[0]+delta[1]*delta[1]+delta[2]*delta[2]
    a,b=wp.float64(1.0),wp.float64(.5)
    if theta2<wp.float64(1e-4):
        a=wp.float64(1.0)-theta2/wp.float64(6.0)+theta2*theta2/wp.float64(120.0)
        b=wp.float64(.5)-theta2/wp.float64(24.0)+theta2*theta2/wp.float64(720.0)
    else:
        theta=wp.sqrt(theta2);a=wp.sin(theta)/theta
        half=theta*wp.float64(.5);sinc=wp.sin(half)/half;b=wp.float64(.5)*sinc*sinc
    eye=wp.mat33d(wp.float64(1.0),wp.float64(0.0),wp.float64(0.0),wp.float64(0.0),wp.float64(1.0),wp.float64(0.0),wp.float64(0.0),wp.float64(0.0),wp.float64(1.0))
    return r*(eye+a*omega+b*(omega*omega))


@wp.kernel
def _refine(rotations: wp.array3d(dtype=wp.float64),translations: wp.array2d(dtype=wp.float64),
            focal: wp.array(dtype=wp.float64),initial: wp.array(dtype=wp.float64),
            points0: wp.array3d(dtype=wp.float64),points1: wp.array3d(dtype=wp.float64),
            valid: wp.array2d(dtype=wp.bool),principal0: wp.array2d(dtype=wp.float64),principal1: wp.array2d(dtype=wp.float64),
            iterations: int,lower: wp.float64,upper: wp.float64,accepted_scale: wp.float64,
            output: wp.array3d(dtype=wp.float64),output_focal: wp.array(dtype=wp.float64),steps: wp.array(dtype=wp.int32)):
    i,lane=wp.tid();r=_matrix(rotations,i)
    t=wp.vec3d(translations[i,0],translations[i,1],translations[i,2]);f=focal[i]
    p=wp.vec2d(principal0[i,0],principal0[i,1]);q=wp.vec2d(principal1[i,0],principal1[i,1])
    support=int(0)
    for point in range(lane,points0.shape[1],BLOCK):support+=int(valid[i,point])
    count=wp.tile_extract(wp.tile_sum(wp.tile(support)),0)
    damping=wp.float64(1e-3);done=int(0)
    for iteration in range(iterations):
        if count<8:break
        coefficients=_coefficients(r,t,f,p,q);partial=Vec28()
        for point in range(lane,points0.shape[1],BLOCK):
            if valid[i,point]:
                a=wp.vec3d(points0[i,point,0],points0[i,point,1],wp.float64(1.0))
                b=wp.vec3d(points1[i,point,0],points1[i,point,1],wp.float64(1.0))
                residual,j=_terms(coefficients,a,b);index=int(0)
                for row in range(6):
                    for col in range(row,6):partial[index]+=j[row]*j[col];index+=1
                for row in range(6):partial[21+row]+=j[row]*residual
                partial[27]+=residual*residual
        totals=Vec28()
        for k in range(28):totals[k]=wp.tile_extract(wp.tile_sum(wp.tile(partial[k])),0)
        normal=Mat6();gradient=Vec6();index=int(0)
        for row in range(6):
            gradient[row]=totals[21+row]
            for col in range(row,6):
                value=totals[index];index+=1
                if row==col:value+=damping*wp.max(value,wp.float64(1e-10))+wp.float64(1e-12)
                normal[row,col]=value;normal[col,row]=value
        delta,usable=_solve(normal,-gradient)
        proposed_f=f*wp.exp(delta[5])
        usable=usable and wp.isfinite(proposed_f) and proposed_f>lower*initial[i] and proposed_f<upper*initial[i]
        if not usable:proposed_f=f
        proposed_r=_rotation_update(r,delta);first,second=_tangent(t)
        proposed_t=t+first*delta[3]+second*delta[4]
        proposed_t=proposed_t/wp.max(wp.length(proposed_t),wp.float64(1e-12))
        model=_model(proposed_r,proposed_t,proposed_f,p,q);trial_sum=wp.float64(0.0)
        if usable:
            for point in range(lane,points0.shape[1],BLOCK):
                if valid[i,point]:
                    a=wp.vec3d(points0[i,point,0],points0[i,point,1],wp.float64(1.0))
                    b=wp.vec3d(points1[i,point,0],points1[i,point,1],wp.float64(1.0))
                    residual=_residual(model,a,b);trial_sum+=residual*residual
        proposed_cost=wp.tile_extract(wp.tile_sum(wp.tile(trial_sum)),0)
        accept=usable and wp.isfinite(proposed_cost) and proposed_cost<totals[27]
        multiplier=wp.float64(10.0)
        if accept:r=proposed_r;t=proposed_t;f=proposed_f;multiplier=accepted_scale
        damping=wp.clamp(damping*multiplier,wp.float64(1e-12),wp.float64(1e12))
        gradmax,norm2=wp.float64(0.0),wp.float64(0.0)
        for k in range(6):gradmax=wp.max(gradmax,wp.abs(gradient[k]));norm2+=delta[k]*delta[k]
        done=iteration+1
        if gradmax<wp.float64(1e-9) or wp.sqrt(norm2)<wp.float64(1e-10):break
        if accept and totals[27]-proposed_cost<wp.float64(1e-10)*wp.max(totals[27],wp.float64(1.0)):break
    if lane==0:
        model=_model(r,t,f,p,q);norm=wp.float64(0.0)
        for row in range(3):
            for col in range(3):norm+=model[row,col]*model[row,col]
        norm=wp.max(wp.sqrt(norm),wp.float64(1e-30))
        for row in range(3):
            for col in range(3):output[i,row,col]=model[row,col]/norm
        output_focal[i]=f;steps[i]=done


def refine(rotation,translation,focal,initial,points0,points1,valid,principal0,principal1,iterations,bounds,*,return_steps=False):
    output=focal.new_empty((len(focal),3,3));result=torch.empty_like(focal)
    steps=torch.empty(len(focal),device=focal.device,dtype=torch.int32)
    accepted_scale=float(torch.tensor(1/3,device='cpu',dtype=torch.get_default_dtype()))
    wp.launch(_refine,dim=(len(focal),BLOCK),block_dim=BLOCK,
        inputs=[wp.from_torch(v) for v in (rotation,translation,focal,initial,points0,points1,valid,principal0,principal1)]
               +[iterations,wp.float64(bounds[0]),wp.float64(bounds[1]),wp.float64(accepted_scale)]
               +[wp.from_torch(v) for v in (output,result,steps)],
        stream=wp.stream_from_torch(torch.cuda.current_stream(focal.device)))
    if return_steps:return output,result,steps
    return output,result
