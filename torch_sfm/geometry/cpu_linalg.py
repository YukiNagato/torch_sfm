"""Parallel LAPACK batches for small independent matrices.

Torch's CPU geev loops over the batch serially. Independent calls release the
GIL, so bounded Python workers can keep the allocated CPU cores busy without
changing the eigensolver or the ordering of its results.
"""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import math
import os
from pathlib import Path
import torch


@lru_cache(maxsize=1)
def _allocated_cpus():
    count=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
    try:
        quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota!='max':count=min(count,max(1,math.ceil(int(quota)/int(period))))
    except (OSError,ValueError):
        try:
            quota=int(Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read_text())
            period=int(Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read_text())
            if quota>0:count=min(count,max(1,math.ceil(quota/period)))
        except (OSError,ValueError):pass
    return max(1,count)


@lru_cache(maxsize=4)
def _pool(pid,workers):
    # The PID prevents reusing a parent's thread pool after fork.
    return ThreadPoolExecutor(max_workers=workers,thread_name_prefix='sfm-lapack')


def batched_eig(matrix,workers=0,vector_rows=None):
    """CPU eig; optionally gather only needed eigenvector rows before packing.

    workers=0 uses up to eight allocated CPUs, 1 is serial. Gathering rows in
    each worker avoids concatenating/transferring unused eigenvector entries.
    Eigenvalues and eigenvector column ordering are unchanged.
    """
    if matrix.device.type!='cpu' or matrix.ndim!=3:
        raise ValueError('batched_eig expects a CPU batch of square matrices')
    if workers<0:raise ValueError('workers must be nonnegative')
    workers=min(workers or 8,_allocated_cpus(),max(1,len(matrix)//32))
    def solve(chunk):
        values,vectors=torch.linalg.eig(chunk)
        if vector_rows is not None:vectors=vectors[:,vector_rows,:]
        return values,vectors
    if workers==1:return solve(matrix)
    results=list(_pool(os.getpid(),workers).map(solve,matrix.chunk(workers)))
    return torch.cat([row[0] for row in results]),torch.cat([row[1] for row in results])
