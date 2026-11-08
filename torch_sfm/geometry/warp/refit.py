"""Block-local normalized QR/Jacobi refit, callable inside persistent LO."""
import warp as wp
from .scoring import BLOCK
from .six_point import _sync
from .refit_finish import _project
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False,'max_unroll':1})

@wp.func_native('''
__shared__ double reduction[64], r[81], v[81], singular[9], scalars[3];
int lane=threadIdx.x;
for(int k=0;k<9;++k){
    double sum=0.;
    for(int i=k+lane;i<rows;i+=64){double x=wp::index(design,pair,i,k);sum+=x*x;}
    reduction[lane]=sum;__syncthreads();
    for(int step=32;step;step/=2){if(lane<step)reduction[lane]+=reduction[lane+step];__syncthreads();}
    if(lane==0){
        double pivot=wp::index(design,pair,k,k), norm=sqrt(reduction[0]);
        double alpha=pivot<0.?norm:-norm;
        scalars[0]=pivot-alpha;scalars[1]=norm>0.?(alpha-pivot)/alpha:0.;scalars[2]=alpha;
        r[k*9+k]=alpha;
    }
    __syncthreads();
    for(int i=k+lane;i<rows;i+=64)
        wp::index(design,pair,i,k)=scalars[1]==0.?0.:(i==k?1.:wp::index(design,pair,i,k)/scalars[0]);
    __syncthreads();
    for(int j=k+1;j<9;++j){
        double dot=0.;
        for(int i=k+lane;i<rows;i+=64)dot+=wp::index(design,pair,i,k)*wp::index(design,pair,i,j);
        reduction[lane]=dot;__syncthreads();
        for(int step=32;step;step/=2){if(lane<step)reduction[lane]+=reduction[lane+step];__syncthreads();}
        double value=reduction[0]*scalars[1];
        for(int i=k+lane;i<rows;i+=64)wp::index(design,pair,i,j)-=value*wp::index(design,pair,i,k);
        __syncthreads();
    }
}
if(lane==0){
    double scale=0.;
    for(int i=0;i<9;++i)for(int j=0;j<9;++j){
        if(i>j)r[i*9+j]=0.;else if(i<j)r[i*9+j]=wp::index(design,pair,i,j);
        scale=fmax(scale,fabs(r[i*9+j]));v[i*9+j]=double(i==j);
    }
    bool ok=isfinite(scale)&&scale>0.;
    if(ok)for(int i=0;i<81;++i)r[i]/=scale;
    if(ok)for(int sweep=0;sweep<64;++sweep){
        bool changed=false;
        for(int p=0;p<8;++p)for(int q=p+1;q<9;++q){
            double aa=0.,bb=0.,ab=0.;
            for(int i=0;i<9;++i){double a=r[i*9+p],b=r[i*9+q];aa+=a*a;bb+=b*b;ab+=a*b;}
            if(fabs(ab)>2e-15*sqrt(aa)*sqrt(bb)){
                double delta=bb-aa,twice=2.*ab,ps=fmax(fabs(delta),fabs(twice));
                double x=delta/ps,y=twice/ps,t=y/(x+copysign(sqrt(x*x+y*y),x));
                double c=1./sqrt(1.+t*t),s=c*t;
                for(int i=0;i<9;++i){
                    double a=r[i*9+p],b=r[i*9+q];r[i*9+p]=c*a-s*b;r[i*9+q]=s*a+c*b;
                    a=v[i*9+p];b=v[i*9+q];v[i*9+p]=c*a-s*b;v[i*9+q]=s*a+c*b;
                }changed=true;
            }
        }
        if(!changed)break;
    }
    double smallest=INFINITY,second=INFINITY,largest=0.;int slot=0;
    for(int j=0;j<9;++j){double n=0.;for(int i=0;i<9;++i)n+=r[i*9+j]*r[i*9+j];
        singular[j]=sqrt(n);largest=fmax(largest,singular[j]);
        if(singular[j]<smallest){second=smallest;smallest=singular[j];slot=j;}
        else second=fmin(second,singular[j]);
    }
    ok=ok&&isfinite(largest)&&second>1e-10*largest;
    for(int i=0;i<9;++i){double x=v[i*9+slot];wp::index(output,pair,i/3,i%3)=x;ok=ok&&isfinite(x);}
    wp::index(good,pair)=ok;
}
__syncthreads();
''')
def qr_svd(design:wp.array3d(dtype=wp.float64),pair:int,rows:int,
           output:wp.array3d(dtype=wp.float64),good:wp.array(dtype=wp.bool)): ...

@wp.func
def refit(p0:wp.array3d(dtype=wp.float64),p1:wp.array3d(dtype=wp.float64),mask:wp.array2d(dtype=wp.bool),
          pair:int,lane:int,kind:int,design:wp.array3d(dtype=wp.float64),output:wp.array3d(dtype=wp.float64),good:wp.array(dtype=wp.bool)):
    count=wp.float64(0.0);a,b=wp.vec2d(),wp.vec2d()
    for j in range(lane,mask.shape[1],BLOCK):
        if mask[pair,j]:
            count+=wp.float64(1.0);a+=wp.vec2d(p0[pair,j,0],p0[pair,j,1]);b+=wp.vec2d(p1[pair,j,0],p1[pair,j,1])
    count=wp.tile_extract(wp.tile_sum(wp.tile(count)),0)
    for i in range(2):
        a[i]=wp.tile_extract(wp.tile_sum(wp.tile(a[i])),0)/wp.max(count,wp.float64(1.0))
        b[i]=wp.tile_extract(wp.tile_sum(wp.tile(b[i])),0)/wp.max(count,wp.float64(1.0))
    d0,d1=wp.float64(0.0),wp.float64(0.0)
    for j in range(lane,mask.shape[1],BLOCK):
        if mask[pair,j]:
            d0+=wp.length(wp.vec2d(p0[pair,j,0],p0[pair,j,1])-a)
            d1+=wp.length(wp.vec2d(p1[pair,j,0],p1[pair,j,1])-b)
    d0=wp.tile_extract(wp.tile_sum(wp.tile(d0)),0);d1=wp.tile_extract(wp.tile_sum(wp.tile(d1)),0)
    s0=wp.sqrt(wp.float64(2.0))*count/wp.max(d0,wp.float64(1e-12));s1=wp.sqrt(wp.float64(2.0))*count/wp.max(d1,wp.float64(1e-12))
    rows=mask.shape[1]
    if kind==2:rows*=2
    rows=wp.max(rows,9)
    for row in range(lane,rows,BLOCK):
        j=row
        if kind==2:j=row//2
        x,y=wp.vec3d(),wp.vec3d()
        if j<mask.shape[1]:
            if mask[pair,j]:
                x=wp.vec3d((p0[pair,j,0]-a[0])*s0,(p0[pair,j,1]-a[1])*s0,wp.float64(1.0))
                y=wp.vec3d((p1[pair,j,0]-b[0])*s1,(p1[pair,j,1]-b[1])*s1,wp.float64(1.0))
        for col in range(9):
            value=y[col//3]*x[col%3]
            if kind==2:
                value=wp.float64(0.0)
                if col//3==row%2:value=-x[col%3]
                elif col//3==2:value=y[row%2]*x[col%3]
            design[pair,row,col]=value
    _sync()
    qr_svd(design,pair,rows,output,good)
    if lane==0:
        m=wp.mat33d()
        for i in range(3):
            for j in range(3):m[i,j]=output[pair,i,j]
        z,o=wp.float64(0.0),wp.float64(1.0)
        t0=wp.mat33d(s0,z,-a[0]*s0,z,s0,-a[1]*s0,z,z,o)
        t1=wp.mat33d(s1,z,-b[0]*s1,z,s1,-b[1]*s1,z,z,o)
        if kind==2:
            inv=wp.mat33d(o/s1,z,b[0],z,o/s1,b[1],z,z,o);m=inv*m*t0
            good[pair]=good[pair] and wp.abs(wp.determinant(m))>wp.float64(1e-12)
        else:
            m=wp.transpose(t1)*_project(m,False)*t0
            if kind==0:m=_project(m,True)
        good[pair]=good[pair] and count>=wp.float64(8.0)
        for i in range(3):
            for j in range(3):
                output[pair,i,j]=m[i,j];good[pair]=good[pair] and wp.isfinite(m[i,j])
    _sync()
