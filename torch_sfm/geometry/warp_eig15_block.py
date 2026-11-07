"""One block per 15x15 action matrix: balance, Hessenberg, QR and coordinates."""
import warp as wp
from .warp_eig15 import recover_coordinate
wp.init()
wp.set_module_options({'enable_backward':False,'max_unroll':1,'fast_math':False,'fuse_fp':False})


@wp.kernel
def roots(source: wp.array3d(dtype=wp.float64),real: wp.array2d(dtype=wp.float64),
          imag: wp.array2d(dtype=wp.float64),status: wp.array(dtype=wp.int32),
          iterations: wp.array(dtype=wp.int32),balanced: wp.array3d(dtype=wp.float64),diagonal: wp.array2d(dtype=wp.float64),
          coordinates: wp.array3d(dtype=wp.float64),valid: wp.array2d(dtype=wp.bool)):
    sample,lane=wp.tid();largest=wp.float64(0.0)
    if lane<15:
        diagonal[sample,lane]=wp.float64(1.0)
        for j in range(15):largest=wp.max(largest,wp.abs(source[sample,lane,j]))
    scale=wp.tile_extract(wp.tile_max(wp.tile(largest)),0)
    if lane==0:status[sample]=1;iterations[sample]=0
    if not wp.isfinite(scale) or scale==wp.float64(0.0):return
    a=wp.tile_zeros(shape=(32,32),dtype=wp.float64,storage='shared')
    for j in range(15):
        value=wp.float64(0.0)
        if lane<15:value=source[sample,lane,j]/scale
        a[lane,j]=value
    for sweep in range(16):
        changed=wp.bool(False)
        for i in range(15):
            row_value,col_value=wp.float64(0.0),wp.float64(0.0)
            if lane<15 and lane!=i:row_value=wp.abs(a[i,lane]);col_value=wp.abs(a[lane,i])
            row=wp.tile_extract(wp.tile_sum(wp.tile(row_value)),0)
            col=wp.tile_extract(wp.tile_sum(wp.tile(col_value)),0)
            if row>wp.float64(0.0) and col>wp.float64(0.0):
                original=row+col;factor=wp.float64(1.0)
                for k in range(64):
                    if col>=row*wp.float64(.5):break
                    col*=wp.float64(4.0);factor*=wp.float64(2.0)
                for k in range(64):
                    if col<=row*wp.float64(2.0):break
                    col*=wp.float64(.25);factor*=wp.float64(.5)
                if (row+col)/factor<wp.float64(.95)*original:
                    changed=True
                    if lane==i:diagonal[sample,i]*=factor
                    a[i,lane]=a[i,lane]/factor
                    a[lane,i]=a[lane,i]*factor
        if not changed:break
    if lane<15:
        real[sample,lane]=wp.float64(0.0);imag[sample,lane]=wp.float64(0.0)
        for j in range(15):balanced[sample,lane,j]=a[lane,j]*scale
    v=wp.tile_zeros(shape=(32,),dtype=wp.float64,storage='shared')
    for k in range(13):
        entry=wp.float64(0.0)
        if lane>k and lane<15:entry=a[lane,k]
        norm=wp.tile_extract(wp.tile_max(wp.tile(wp.abs(entry))),0)
        if norm>wp.float64(0.0):
            value=entry/norm
            length=wp.sqrt(wp.tile_extract(wp.tile_sum(wp.tile(value*value)),0))
            v[lane]=value
            alpha=-length
            if v[k+1]<wp.float64(0.0):alpha=length
            if lane==k+1:value-=alpha
            v[lane]=value
            beta=wp.float64(2.0)/wp.tile_extract(wp.tile_sum(wp.tile(value*value)),0)
            dot=wp.float64(0.0)
            for i in range(k+1,15):dot+=v[i]*a[i,lane]
            for i in range(k+1,15):
                value=a[i,lane]
                if lane>=k and lane<15:value-=beta*v[i]*dot
                a[i,lane]=value
            dot=wp.float64(0.0)
            for j in range(k+1,15):dot+=a[lane,j]*v[j]
            for j in range(k+1,15):
                value=a[lane,j]
                if lane<15:value-=beta*dot*v[j]
                a[lane,j]=value
            value=a[lane,k]
            if lane==k+1:value=alpha*norm
            elif lane>k+1:value=wp.float64(0.0)
            a[lane,k]=value
    n=int(14);steps=int(0);since_deflation=int(0)
    while n>=0 and steps<600:
        low=n
        while low>0:
            diagonal_size=wp.abs(a[low-1,low-1])+wp.abs(a[low,low])
            if wp.abs(a[low,low-1])<=wp.float64(2e-15)*diagonal_size+wp.float64(1e-300):
                value=a[lane,low-1]
                if lane==low:value=wp.float64(0.0)
                a[lane,low-1]=value
                break
            low-=1
        if low==n:
            if lane==0:real[sample,n]=a[n,n]*scale
            n-=1;since_deflation=0
        elif low==n-1:
            x,y,b,c=a[n-1,n-1],a[n,n],a[n-1,n],a[n,n-1]
            half=(x-y)*wp.float64(.5);disc=half*half+b*c
            center=(x+y)*wp.float64(.5)
            if lane==0:
                if disc>=wp.float64(0.0):
                    root=wp.sqrt(disc);sign=wp.float64(1.0)
                    if half<wp.float64(0.0):sign=wp.float64(-1.0)
                    offset=half+sign*root
                    real[sample,n-1]=(y+offset)*scale;other=y
                    if offset!=wp.float64(0.0):other=y-b*c/offset
                    real[sample,n]=other*scale
                else:
                    real[sample,n-1]=center*scale;real[sample,n]=center*scale
                    imag[sample,n-1]=wp.sqrt(-disc)*scale;imag[sample,n]=-wp.sqrt(-disc)*scale
            n-=2;since_deflation=0
        else:
            trace=a[n-1,n-1]+a[n,n]
            determinant=a[n-1,n-1]*a[n,n]-a[n-1,n]*a[n,n-1]
            if since_deflation>0 and since_deflation%20==0:
                distance=wp.abs(a[n,n-1])+wp.abs(a[n-1,n-2])
                center=a[n,n]+wp.float64(.75)*distance
                trace=wp.float64(2.0)*center;determinant=center*center+wp.float64(.4375)*distance*distance
            v0=a[low,low]*a[low,low]+a[low,low+1]*a[low+1,low]-trace*a[low,low]+determinant
            v1=a[low+1,low]*(a[low,low]+a[low+1,low+1]-trace);v2=a[low+1,low]*a[low+2,low+1]
            for k in range(low,n):
                last=wp.min(k+2,n)
                if k>low:
                    v0=a[k,k-1];v1=a[k+1,k-1];v2=wp.float64(0.0)
                    if last==k+2:v2=a[k+2,k-1]
                magnitude=wp.max(wp.abs(v0),wp.max(wp.abs(v1),wp.abs(v2)))
                if magnitude>wp.float64(0.0):
                    reflector=wp.vec3d(v0/magnitude,v1/magnitude,v2/magnitude)
                    length=wp.length(reflector)
                    if reflector[0]<wp.float64(0.0):length=-length
                    reflector[0]+=length;beta=wp.float64(2.0)/wp.dot(reflector,reflector)
                    dot=wp.float64(0.0)
                    for i in range(k,last+1):dot+=reflector[i-k]*a[i,lane]
                    for i in range(k,last+1):
                        value=a[i,lane]
                        if lane>=wp.max(low,k-1) and lane<=n:value-=beta*reflector[i-k]*dot
                        a[i,lane]=value
                    dot=wp.float64(0.0)
                    for j in range(k,last+1):dot+=a[lane,j]*reflector[j-k]
                    for j in range(k,last+1):
                        value=a[lane,j]
                        if lane>=low and lane<=n:value-=beta*dot*reflector[j-k]
                        a[lane,j]=value
                    if k>low:
                        value=a[lane,k-1]
                        if lane==k+1 or (lane==k+2 and last==k+2):value=wp.float64(0.0)
                        a[lane,k-1]=value
            steps+=1;since_deflation+=1
    if lane==0:
        iterations[sample]=steps
        if n<0:status[sample]=0

    # Every lane reaches this shared write/barrier before reading status/roots.
    v[lane]=wp.float64(0.0)
    if lane<15:
        recover_coordinate(sample,lane,balanced,real,imag,status,coordinates,valid,diagonal)
