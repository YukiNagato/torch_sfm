"""Shared block scoring: fixed slots, no candidate-by-point error storage."""
import warp as wp
from ..geometry.warp_essential_pose import _decompose_matrix
from ..geometry.warp_relative_pose import _matrix
from .six_point import _sync
BLOCK=64
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False,'max_unroll':1})

@wp.func
def vec(a:wp.array3d(dtype=wp.float64),i:int,j:int):
    return wp.vec3d(a[i,j,0],a[i,j,1],a[i,j,2])

@wp.func
def positive(r:wp.mat33d,t:wp.vec3d,a:wp.vec3d,b:wp.vec3d):
    rotated=r*a;cosine=-wp.dot(rotated,b)
    x,y=-wp.dot(rotated,t),wp.dot(b,t)
    return x-cosine*y>wp.float64(0.0) and y-cosine*x>wp.float64(0.0)

@wp.func
def vote(model:wp.mat33d,ray0:wp.array3d(dtype=wp.float64),ray1:wp.array3d(dtype=wp.float64),
         valid:wp.array2d(dtype=wp.bool),owner:int,lane:int):
    r0,r1,t=wp.mat33d(),wp.mat33d(),wp.vec3d();good=False
    if lane==0:r0,r1,t,good=_decompose_matrix(model)
    good=wp.tile_extract(wp.tile(int(good)),0)!=0
    for i in range(3):
        t[i]=wp.tile_extract(wp.tile(t[i]),0)
        for j in range(3):
            r0[i,j]=wp.tile_extract(wp.tile(r0[i,j]),0);r1[i,j]=wp.tile_extract(wp.tile(r1[i,j]),0)
    # A loop-carried winner must not alias the immutable decomposition basis.
    best=int(-1);chosen_r,chosen_t=wp.mat33d(r0),wp.vec3d(t)
    for candidate in range(4):
        r=r0
        if candidate%2==1:r=r1
        direction=t
        if candidate>=2:direction=-t
        hits=int(0)
        if good:
            for j in range(lane,valid.shape[1],BLOCK):
                if valid[owner,j]:hits+=int(positive(r,direction,vec(ray0,owner,j),vec(ray1,owner,j)))
        count=wp.tile_extract(wp.tile_sum(wp.tile(hits)),0)
        if count>=best:best=count;chosen_r=r;chosen_t=direction
        _sync()
    return chosen_r,chosen_t,good

@wp.func
def error(model:wp.mat33d,kind:int,r:wp.mat33d,t:wp.vec3d,
          p0:wp.array3d(dtype=wp.float64),p1:wp.array3d(dtype=wp.float64),
          ray0:wp.array3d(dtype=wp.float64),ray1:wp.array3d(dtype=wp.float64),
          jac0:wp.array4d(dtype=wp.float64),jac1:wp.array4d(dtype=wp.float64),owner:int,j:int):
    result=wp.float64(wp.inf)
    if kind==0:
        a,b=vec(ray0,owner,j),vec(ray1,owner,j)
        if positive(r,t,a,b):
            l0,l1=wp.transpose(model)*b,model*a
            numerator=wp.dot(b,l1);den=wp.float64(0.0)
            for axis in range(2):
                x,y=wp.float64(0.0),wp.float64(0.0)
                for row in range(3):
                    x+=jac0[owner,j,row,axis]*l0[row];y+=jac1[owner,j,row,axis]*l1[row]
                den+=x*x+y*y
            if den>wp.float64(0.0):result=numerator*numerator/den
    else:
        a=wp.vec3d(p0[owner,j,0],p0[owner,j,1],wp.float64(1.0))
        b=wp.vec3d(p1[owner,j,0],p1[owner,j,1],wp.float64(1.0))
        l1=model*a
        if kind==2:
            if wp.abs(l1[2])>wp.float64(1e-12):
                d=wp.vec2d(l1[0]/l1[2]-b[0],l1[1]/l1[2]-b[1]);result=wp.dot(d,d)
        else:
            l0=wp.transpose(model)*b;n=wp.dot(b,l1)
            result=n*n/wp.max(l0[0]*l0[0]+l0[1]*l0[1]+l1[0]*l1[0]+l1[1]*l1[1],wp.float64(1e-30))
    if not wp.isfinite(result):result=wp.float64(wp.inf)
    return result

@wp.kernel
def score(models:wp.array3d(dtype=wp.float64),good:wp.array(dtype=wp.bool),active:wp.array(dtype=wp.int64),stride:int,kind:int,
          p0:wp.array3d(dtype=wp.float64),p1:wp.array3d(dtype=wp.float64),valid:wp.array2d(dtype=wp.bool),
          ray0:wp.array3d(dtype=wp.float64),ray1:wp.array3d(dtype=wp.float64),jac0:wp.array4d(dtype=wp.float64),jac1:wp.array4d(dtype=wp.float64),
          threshold:wp.float64,costs:wp.array(dtype=wp.float64),poses:wp.array3d(dtype=wp.float64)):
    candidate,lane=wp.tid();owner=int(active[candidate//stride]);cost=wp.float64(wp.inf)
    r,t=wp.mat33d(),wp.vec3d()
    if good[candidate]:
        model=_matrix(models,candidate);safe=True
        if kind==0:r,t,safe=vote(model,ray0,ray1,valid,owner,lane)
        cost=wp.float64(0.0)
        if safe:
            for j in range(lane,valid.shape[1],BLOCK):
                if valid[owner,j]:cost+=wp.min(error(model,kind,r,t,p0,p1,ray0,ray1,jac0,jac1,owner,j),threshold)
        else:cost=wp.float64(wp.inf)
    total=wp.tile_extract(wp.tile_sum(wp.tile(cost)),0)
    if lane==0:
        costs[candidate]=total
        for i in range(3):
            poses[candidate,3,i]=t[i]
            for j in range(3):poses[candidate,i,j]=r[i,j]
