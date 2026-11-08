"""Shared scheduling and initialization for bounded adaptive RANSAC batches."""
import warp as wp
from .local_optimization import BLOCK
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':True,'max_unroll':1})

@wp.func_native('return log1p(x);')
def _log1p(x: wp.float64) -> wp.float64: ...


@wp.kernel
def _initialize(valid: wp.array2d(dtype=wp.bool),minimum: int,maximum: int,
                initial_limits: wp.array(dtype=wp.int64),limits: wp.array(dtype=wp.int64),
                sizes: wp.array(dtype=wp.int64),required: wp.array(dtype=wp.int64),
                models: wp.array3d(dtype=wp.float64),masks: wp.array2d(dtype=wp.bool),
                found: wp.array(dtype=wp.bool),focals: wp.array(dtype=wp.float64),
                costs: wp.array(dtype=wp.float64),trials: wp.array(dtype=wp.int64)):
    pair,lane=wp.tid();partial=int(0)
    for j in range(lane,valid.shape[1],BLOCK):
        partial+=int(valid[pair,j]);masks[pair,j]=False
    count=wp.tile_extract(wp.tile_sum(wp.tile(partial)),0)
    if lane==0:
        limit=wp.int64(maximum)
        if initial_limits.shape[0]>0:limit=initial_limits[pair]
        limits[pair]=limit
        sizes[pair]=wp.int64(count);required[pair]=wp.int64(0)
        if count>=minimum:required[pair]=limit
        costs[pair]=wp.float64(wp.inf);focals[pair]=wp.float64(1.0)
        found[pair]=False;trials[pair]=wp.int64(0)
        for i in range(3):
            for j in range(3):models[pair,i,j]=wp.float64(int(i==j))


@wp.kernel
def _schedule(required: wp.array(dtype=wp.int64),costs: wp.array(dtype=wp.float64),trial: int,
              active: wp.array(dtype=wp.int64),history: wp.array(dtype=wp.float64),count: wp.array(dtype=wp.int32)):
    # Pair batches are bounded (normally <=256). A serial stable compaction of
    # this small metadata array avoids an allocation or a second scan kernel.
    n=int(0)
    for pair in range(required.shape[0]):
        if wp.int64(trial)<required[pair]:
            active[n]=wp.int64(pair);history[n]=costs[pair];n+=1
    count[0]=n


@wp.func_native('return log1pf(x);')
def _log1p32(x: wp.float32) -> wp.float32: ...
