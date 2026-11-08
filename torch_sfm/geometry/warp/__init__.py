"""Warp implementations of geometric verification, pose refinement, and tracks.

Kernel modules live here alongside the batched RANSAC controllers. Public
controller classes are imported lazily so using a triangulation kernel does
not initialize the unrelated two-view solvers.
"""
from importlib import import_module

_EXPORTS = {
    'SixPointBatch': ('six_point_cooperative', 'CooperativeSixPointBatch'),
    'CooperativeSixPointBatch': ('six_point_cooperative', 'CooperativeSixPointBatch'),
    'LocalOptimizer': ('local_optimization', 'LocalOptimizer'),
    'SharedFocalRansac': ('ransac', 'SharedFocalRansac'),
    'CalibratedRansac': ('calibrated_ransac', 'CalibratedRansac'),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module, attribute = _EXPORTS[name]
    value = getattr(import_module('.' + module, __name__), attribute)
    globals()[name] = value
    return value
