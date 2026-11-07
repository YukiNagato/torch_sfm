"""Fused six-point cheirality checks, with Torch's float64 SVD retained."""
import torch
import warp as wp

wp.init()
wp.set_module_options({'enable_backward': False, 'fast_math': False})


@wp.func
def _matrix(array: wp.array3d(dtype=wp.float64), i: int):
    result=wp.mat33d()
    for r in range(3):
        for c in range(3):result[r,c]=array[i,r,c]
    return result


@wp.kernel
def _check(u_array: wp.array3d(dtype=wp.float64),vh_array: wp.array3d(dtype=wp.float64),
           rays0: wp.array3d(dtype=wp.float64),rays1: wp.array3d(dtype=wp.float64),
           accepted: wp.array(dtype=wp.bool)):
    i=wp.tid()
    u,vh=_matrix(u_array,i),_matrix(vh_array,i)
    du,dv=wp.determinant(u),wp.determinant(vh)
    for k in range(3):
        u[k,2]*=du;vh[2,k]*=dv
    w=wp.mat33d(wp.float64(0.0),wp.float64(-1.0),wp.float64(0.0),
               wp.float64(1.0),wp.float64(0.0),wp.float64(0.0),
               wp.float64(0.0),wp.float64(0.0),wp.float64(1.0))
    ok=wp.bool(False)
    for rotation in range(2):
        middle=w
        if rotation==1:middle=wp.transpose(w)
        r=u*middle*vh
        for sign in range(2):
            t=wp.vec3d(u[0,2],u[1,2],u[2,2])
            if sign==1:t=-t
            center=-(wp.transpose(r)*t)
            all_positive=wp.bool(True)
            for p in range(6):
                d0=wp.vec3d(rays0[i,p,0],rays0[i,p,1],rays0[i,p,2])
                d1=wp.vec3d(rays1[i,p,0],rays1[i,p,1],rays1[i,p,2])
                d0=d0/wp.max(wp.length(d0),wp.float64(1e-12))
                d1=wp.transpose(r)*(d1/wp.max(wp.length(d1),wp.float64(1e-12)))
                cosine=wp.dot(d0,d1)
                den=wp.max(wp.float64(1.0)-cosine*cosine,wp.float64(1e-15))
                a,b=wp.dot(d0,center),wp.dot(d1,center)
                depth0,depth1=(a-cosine*b)/den,(cosine*a-b)/den
                point=(depth0*d0+center+depth1*d1)*wp.float64(.5)
                all_positive=all_positive and depth0>wp.float64(1e-8) and depth1>wp.float64(1e-8)
                for c in range(3):all_positive=all_positive and wp.isfinite(point[c])
            ok=ok or all_positive
    accepted[i]=ok


def sample_cheirality(essential,rays0,rays1):
    """Whether one of the four poses has all six samples at positive depth."""
    u,_,vh=torch.linalg.svd(essential)
    accepted=torch.empty(len(essential),device=essential.device,dtype=torch.bool)
    wp.launch(_check,dim=len(essential),inputs=[wp.from_torch(x) for x in (u,vh,rays0,rays1,accepted)],
              stream=wp.stream_from_torch(torch.cuda.current_stream(essential.device)))
    return accepted
