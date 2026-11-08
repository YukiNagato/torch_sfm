"""Warp shared-focal frontend used by the normal two-view verification pipeline.

SixPointBatch selects the four-kernel cooperative solver. Earlier prototypes
remain available from their own modules for comparison. The resulting
TwoViewGeometry outputs feed the existing global view-graph focal calibration.
"""
from .six_point_cooperative import CooperativeSixPointBatch
from .local_optimization import LocalOptimizer
from .ransac import SharedFocalRansac
from .calibrated_ransac import CalibratedRansac

SixPointBatch = CooperativeSixPointBatch

__all__ = ['SixPointBatch', 'CooperativeSixPointBatch', 'LocalOptimizer', 'SharedFocalRansac', 'CalibratedRansac']
