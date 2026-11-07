"""Guarded float64 nonsymmetric roots for independent 15x15 matrices.

Power-of-two balancing, Householder Hessenberg reduction and real double-shift
QR. Coordinates are recovered from the original matrix with pivoted elimination.
Failures are explicit: callers must use a reference solver for flagged inputs.
"""
import torch
import warp as wp

wp.init()
wp.set_module_options({'enable_backward':False,'max_unroll':1,'fast_math':False,'fuse_fp':False})
Mat=wp.types.matrix(shape=(15,15),dtype=wp.float64)
Vec=wp.types.vector(length=15,dtype=wp.float64)


@wp.func
def recover_coordinate(sample: int,root: int,source: wp.array3d(dtype=wp.float64),real: wp.array2d(dtype=wp.float64),
                 imag: wp.array2d(dtype=wp.float64),status: wp.array(dtype=wp.int32),
                 coordinates: wp.array3d(dtype=wp.float64),valid: wp.array2d(dtype=wp.bool),
                 diagonal: wp.array2d(dtype=wp.float64)):
    valid[sample,root]=False
    if status[sample]!=0:return
    value=real[sample,root]
    imaginary=wp.abs(imag[sample,root])
    if imaginary>wp.float64(0.0):
        if imaginary<wp.float64(1e-7)*(wp.float64(1.0)+wp.abs(value)):wp.atomic_max(status,sample,2)
        return
    # Eigenvectors of clustered roots are sensitive even with a small residual.
    # Preserve all such hypotheses by delegating the whole matrix to LAPACK.
    for other in range(15):
        if other!=root:
            separation=wp.abs(real[sample,other]-value)+wp.abs(imag[sample,other])
            if separation<wp.float64(1e-6)*(wp.float64(1.0)+wp.abs(value)):
                wp.atomic_max(status,sample,2);return
    a=Mat();scale=wp.float64(0.0)
    for i in range(15):
        for j in range(14):
            entry=source[sample,i,j+1]
            if i==j+1:entry-=value
            a[i,j]=entry;scale=wp.max(scale,wp.abs(entry))
        a[i,14]=-source[sample,i,0]
        if i==0:a[i,14]+=value
    good=wp.bool(True)
    for j in range(14):
        pivot=j;largest=wp.abs(a[j,j])
        for i in range(j+1,15):
            if wp.abs(a[i,j])>largest:pivot=i;largest=wp.abs(a[i,j])
        if largest<=wp.float64(1e-8)*scale or not wp.isfinite(largest):good=False;break
        for k in range(j,15):
            tmp=a[j,k];a[j,k]=a[pivot,k];a[pivot,k]=tmp
        for i in range(j+1,15):
            factor=a[i,j]/a[j,j]
            for k in range(j+1,15):a[i,k]-=factor*a[j,k]
    v=Vec(wp.float64(0.0));v[0]=wp.float64(1.0)
    if good:
        for step in range(14):
            i=13-step;rhs=a[i,14]
            for j in range(i+1,14):rhs-=a[i,j]*v[j+1]
            v[i+1]=rhs/a[i,i]
        norm,error,magnitude=wp.float64(0.0),wp.float64(0.0),wp.float64(0.0)
        for i in range(15):norm=wp.max(norm,wp.abs(v[i]));good=good and wp.isfinite(v[i])
        for i in range(15):
            residual=-value*v[i];row=wp.float64(0.0)
            for j in range(15):residual+=source[sample,i,j]*v[j];row+=wp.abs(source[sample,i,j])
            error=wp.max(error,wp.abs(residual));magnitude=wp.max(magnitude,row)
        good=good and wp.isfinite(norm) and wp.isfinite(error) and error<=wp.float64(1e-11)*(magnitude+wp.abs(value))*norm
    if good:
        coordinates[sample,0,root]=v[1]*diagonal[sample,1]/diagonal[sample,0]
        coordinates[sample,1,root]=v[12]*diagonal[sample,12]/diagonal[sample,0]
        valid[sample,root]=True
    else:wp.atomic_max(status,sample,3)


def eigen_coordinates(matrix):
    n=len(matrix)
    real=matrix.new_zeros((n,15));imag=torch.zeros_like(real)
    status=torch.zeros(n,device=matrix.device,dtype=torch.int32);steps=torch.empty_like(status)
    coords=matrix.new_zeros((n,2,15));valid=torch.zeros_like(real,dtype=torch.bool)
    balanced=torch.empty_like(matrix);diagonal=matrix.new_empty((n,15))
    stream=wp.stream_from_torch(torch.cuda.current_stream(matrix.device))
    from .warp_eig15_block import roots
    if n:
        wp.launch(roots,dim=(n,32),inputs=[wp.from_torch(x) for x in
            (matrix,real,imag,status,steps,balanced,diagonal,coords,valid)],block_dim=32,stream=stream)
    return real,imag,coords,valid,status,steps


def coordinates_with_fallback(matrix,workers=0):
    """Return real root coordinates, repairing every flagged matrix via LAPACK."""
    from .cpu_linalg import batched_eig
    real,imag,coords,valid,status,_=eigen_coordinates(matrix)
    failed=torch.where(status!=0)[0]
    if len(failed):
        z,v=batched_eig(matrix[failed].cpu(),workers=workers,vector_rows=(0,1,12))
        c=v[:,1:]/v[:,:1]
        mask=(z.imag.abs()<1e-7*(1+z.real.abs()))&(c.imag.abs().amax(1)<1e-7*(1+c.real.abs().amax(1)))
        real[failed]=z.real.to(matrix.device);coords[failed]=c.real.to(matrix.device)
        valid[failed]=mask.to(matrix.device)
    return real,coords,valid,status
