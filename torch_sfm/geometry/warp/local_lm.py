"""Block-local LM loop reused by the new LO controller.

Arithmetic/stopping rules follow geometry.warp.shared_focal_lm._refine;
only observation indirection and the result handoff differ.
"""
import warp as wp
from .shared_focal_lm import Vec6, Vec28, Mat6, _coefficients, _terms, _solve, _rotation_update, _tangent, _model, _residual
BLOCK=64

@wp.func
def refine(r: wp.mat33d,t: wp.vec3d,f: wp.float64,initial: wp.float64,
           p: wp.vec2d,q: wp.vec2d,points0: wp.array3d(dtype=wp.float64),
           points1: wp.array3d(dtype=wp.float64),mask: wp.array2d(dtype=wp.bool),
           owner: int,worker: int,lane: int,iterations: int,
           lower: wp.float64,upper: wp.float64,accepted_scale: wp.float64):
    support=int(0)
    for point in range(lane,points0.shape[1],BLOCK):support+=int(mask[worker,point])
    count=wp.tile_extract(wp.tile_sum(wp.tile(support)),0)
    damping=wp.float64(1e-3);done=int(0)
    for iteration in range(iterations):
        if count<8:break
        coefficients=_coefficients(r,t,f,p,q);partial=Vec28()
        for point in range(lane,points0.shape[1],BLOCK):
            if mask[worker,point]:
                a=wp.vec3d(points0[owner,point,0],points0[owner,point,1],wp.float64(1.0))
                b=wp.vec3d(points1[owner,point,0],points1[owner,point,1],wp.float64(1.0))
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
        usable=usable and wp.isfinite(proposed_f) and proposed_f>lower*initial and proposed_f<upper*initial
        if not usable:proposed_f=f
        proposed_r=_rotation_update(r,delta);first,second=_tangent(t)
        proposed_t=t+first*delta[3]+second*delta[4]
        proposed_t=proposed_t/wp.max(wp.length(proposed_t),wp.float64(1e-12))
        model=_model(proposed_r,proposed_t,proposed_f,p,q);trial_sum=wp.float64(0.0)
        if usable:
            for point in range(lane,points0.shape[1],BLOCK):
                if mask[worker,point]:
                    a=wp.vec3d(points0[owner,point,0],points0[owner,point,1],wp.float64(1.0))
                    b=wp.vec3d(points1[owner,point,0],points1[owner,point,1],wp.float64(1.0))
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
    model=_model(r,t,f,p,q)
    norm=wp.float64(0.0)
    for row in range(3):
        for col in range(3):norm+=model[row,col]*model[row,col]
    model=model/wp.max(wp.sqrt(norm),wp.float64(1e-30))
    return model,f,done
