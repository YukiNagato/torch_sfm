"""Refinement of learned global poses with real and optional virtual tracks."""
from dataclasses import dataclass, field, replace, asdict
import time

import torch
import torchls as tls

from .averaging import _pcg_backend
from .bundle_adjustment import BundleAdjustmentConfig, bundle_adjustment, filter_tracks, reprojection_errors
from .triangulation import TriangulationConfig, IncrementalTriangulator
from .virtual_tracks import VirtualTrackConfig
from .augmented_bundle_adjustment import AugmentedBAConfig
from .track_initialization import TrackInitializationConfig, initialize_tracks
from torch_sfm.correspondence.vggsfm import VGGSfMTrackConfig
from torch_sfm.correspondence.sift import SIFTConfig


def _default_ba():
    return BundleAdjustmentConfig(fixed_rotation_iterations=0, min_track_length=2,
        retriangulate=False, linear_solver='conjugate_gradient',
        cost_tolerance=1e-6, gradient_tolerance=1e-10, parameter_tolerance=1e-8,
        conjugate_gradient=tls.ConjugateGradientConfig(rtol=1e-5,max_iterations=300))


@dataclass
class TrackRefinementConfig:
    track_source: str = 'flexglue'  # vggsfm or sift_vggsfm use RaCo queries
    tracker: VGGSfMTrackConfig = field(default_factory=VGGSfMTrackConfig)
    sift: SIFTConfig = field(default_factory=SIFTConfig)
    strategy: str = 'augmented'  # GlueMap; alternatives: retriangulate, glomap
    virtual_tracks: VirtualTrackConfig = field(default_factory=VirtualTrackConfig)
    augmented: AugmentedBAConfig = field(default_factory=AugmentedBAConfig)
    initialization: TrackInitializationConfig = field(default_factory=TrackInitializationConfig)
    num_rounds: int = 2
    pcg_backend: str = 'auto'
    triangulation: TriangulationConfig = field(default_factory=lambda: TriangulationConfig(
        implementation='tensor', ignore_two_view_tracks=False, create_max_angle_error=.5))
    bundle_adjustment: BundleAdjustmentConfig = field(default_factory=_default_ba)
    export_colmap: bool = True

    def __post_init__(self):
        if self.track_source not in ('flexglue', 'vggsfm', 'sift_vggsfm'):
            raise ValueError('Track source must be flexglue, vggsfm or sift_vggsfm')
        if self.num_rounds < 1 or self.pcg_backend not in ('auto','torch','warp'):
            raise ValueError('Invalid track refinement rounds or PCG backend')
        if self.strategy not in ('glomap', 'retriangulate', 'augmented'):
            raise ValueError('Unknown real-track refinement strategy')
        if self.strategy == 'augmented':
            if self.bundle_adjustment.optimizer_backend != 'torchls':
                raise ValueError('Augmented BA requires torchls for mixed robust losses')
            if self.bundle_adjustment.pose_parameterization != 'joint' or self.bundle_adjustment.fixed_camera_handling != 'masked':
                raise ValueError('Augmented BA uses joint poses and masked gauge anchors')
            if self.bundle_adjustment.principal_point_prior_sigma is not None:
                raise ValueError('Augmented BA does not use a principal-point prior')
        if self.strategy == 'retriangulate' and self.bundle_adjustment.retriangulate:
            raise ValueError('Track refinement owns retriangulation; set bundle_adjustment.retriangulate=False')
        if self.strategy == 'glomap' and not self.bundle_adjustment.retriangulate:
            raise ValueError('The glomap backend must own retriangulation')
        if self.bundle_adjustment.strategy != 'glomap' or not self.bundle_adjustment.enabled:
            raise ValueError('Track refinement requires enabled glomap-strategy BA')

    @classmethod
    def glomap(cls, **kwargs):
        """Three-view seed map followed by the standard GLOMAP refinement policy."""
        ba = kwargs.pop('bundle_adjustment', BundleAdjustmentConfig(
            linear_solver='conjugate_gradient', triangulation=TriangulationConfig(implementation='tensor')))
        return cls(strategy='glomap', bundle_adjustment=ba, **kwargs)


def refinement_intrinsics(initial, cfg):
    """Keep one shared intrinsic block shared throughout BA, not N equal copies."""
    K=initial['intrinsics'].detach().clone()
    n=len(initial['registered'])
    groups=initial.get('camera_groups')
    if groups is None:
        return K
    if groups.shape != (n,):
        raise ValueError('Expected one camera group per image')
    unique=torch.unique(groups)
    if len(unique)==1:
        if K.ndim==3:
            if not torch.allclose(K,K[:1].expand_as(K),equal_nan=True):
                raise ValueError('Shared-camera intrinsics disagree in saved poses')
            return K[0].clone()
        return K
    if len(unique)!=n and (cfg.refine_focal_length or cfg.refine_principal_point):
        raise ValueError('Grouped-camera BA currently supports one shared camera or per-image cameras; '
                         'for multiple partially shared groups, disable intrinsics refinement')
    return K


@torch.inference_mode(False)
def refine_tracks(frames, geometries, initial, cfg=None, device='cuda', *, star_manifest=None, star_directory=None):
    cfg=TrackRefinementConfig() if cfg is None else cfg
    if cfg.strategy == 'augmented':
        from .augmented_bundle_adjustment import refine_augmented
        if cfg.virtual_tracks.enabled and (star_manifest is None or star_directory is None):
            raise ValueError('Augmented BA with virtual tracks requires saved star manifest/directory')
        return refine_augmented(frames, geometries, initial, cfg, device, star_manifest, star_directory)
    R=initial['rotations'].detach().to(device=device,dtype=torch.float64).clone()
    C=initial['centers'].detach().to(R).clone()
    registered=initial['registered'].detach().to(device).clone()
    K=refinement_intrinsics(initial,cfg.bundle_adjustment).to(R)
    cg_backend=_pcg_backend(cfg.pcg_backend,R)
    ba=cfg.bundle_adjustment
    cg=replace(ba.conjugate_gradient,backend=cg_backend,
        normal_matvec='warp' if cg_backend=='warp' and ba.linear_solver=='conjugate_gradient' else 'torch')
    ba=replace(ba,conjugate_gradient=cg)
    if not geometries:
        raise ValueError('No accepted FlexGlue pairs available for refinement')
    # Inputs are feature identities in original pixel coordinates. No learned
    # depth or artificial observations enter triangulation or BA.
    implementation=cfg.triangulation.implementation
    if implementation=='tensor':
        from .triangulation_tensor import TensorTriangulator
        triangulator_type=TensorTriangulator
    elif implementation=='speculative':
        from .triangulation_speculative import SpeculativeTriangulator
        triangulator_type=SpeculativeTriangulator
    else:
        triangulator_type=IncrementalTriangulator
    stages=[]
    summaries=[]
    def sync():
        if R.is_cuda: torch.cuda.synchronize(R.device)
    sync()
    begin=time.perf_counter()
    initialization_report = None
    if cfg.strategy == 'glomap':
        points,observations,registered,initialization_report=initialize_tracks(
            frames,geometries,R,C,K,registered,ba,cfg.initialization)
        sync()
        tri_seconds=time.perf_counter()-begin
        before_points,before_observations=len(points),len(observations)
        start=time.perf_counter()
        fitted=bundle_adjustment(frames,R,C,points,observations,K,registered,ba,geometries)
        R,C,K=fitted.rotations,fitted.centers,fitted.intrinsics
        points,observations,registered=fitted.points3d,fitted.observations,fitted.registered
        sync()
        summaries.extend(asdict(s) for s in fitted.summaries)
        stages.append(dict(round=1,triangulation_seconds=tri_seconds,ba_seconds=time.perf_counter()-start,
            triangulated_points=before_points,triangulated_observations=before_observations,
            points=len(points),observations=len(observations),registered=int(registered.sum()),ba=fitted.stats))
    for iteration in range(cfg.num_rounds if cfg.strategy == 'retriangulate' else 0):
        sync()
        start=time.perf_counter()
        tri=triangulator_type(frames,geometries,registered,cfg.triangulation)
        tri.update_geometry(R,C,K)
        tri.triangulate()
        changes=tri.complete_and_merge()
        tri.retriangulate()
        points,observations=tri.export()
        del tri
        if not len(points):
            raise RuntimeError('No FlexGlue tracks triangulated from the current global poses')
        # Establish a supported connected component before introducing BA gauge
        # anchors. Learned pose registration alone does not imply track support.
        points,observations,supported=filter_tracks(frames,R,C,points,observations,K,ba,
            ba.initial_max_reprojection_error,ba.initial_max_angular_error)
        registered &= supported
        if not len(points) or int(registered.sum())<2:
            raise RuntimeError('Too few supported cameras/tracks for FlexGlue refinement')
        sync()
        tri_seconds=time.perf_counter()-start
        before_points,before_observations=len(points),len(observations)
        start=time.perf_counter()
        fitted=bundle_adjustment(frames,R,C,points,observations,K,registered,ba)
        R,C,K=fitted.rotations,fitted.centers,fitted.intrinsics
        points,observations,registered=filter_tracks(frames,R,C,fitted.points3d,
            fitted.observations,K,ba,ba.max_reprojection_error)
        if not len(points):
            raise RuntimeError('No tracks survive final refinement filtering')
        sync()
        ba_seconds=time.perf_counter()-start
        summaries.extend(asdict(s) for s in fitted.summaries)
        stages.append(dict(round=iteration+1,triangulation_seconds=tri_seconds,ba_seconds=ba_seconds,
            triangulated_points=before_points,triangulated_observations=before_observations,
            points=len(points),observations=len(observations),registered=int(registered.sum()),
            completion=changes,ba=fitted.stats))
    errors,positive=reprojection_errors(frames,R,C,points,observations,K)
    if not positive.all() or not torch.isfinite(errors).all():
        raise RuntimeError('Invalid observations survived refinement')
    R=R.clone(); C=C.clone()
    R[~registered]=torch.nan; C[~registered]=torch.nan
    sync()
    report=dict(track_source=cfg.track_source,virtual_tracks=0,pairs=len(geometries),
        initialization=initialization_report,
        registered=int(registered.sum()),points=len(points),observations=len(observations),
        reprojection_rms_px=float(errors.square().mean().sqrt()),
        reprojection_median_px=float(errors.median()),
        ba_solves=len(summaries),rounds=stages,
        timings_seconds=dict(triangulation=sum(s['triangulation_seconds'] for s in stages),
            ba=sum(s['ba_seconds'] for s in stages),total=time.perf_counter()-begin),config=asdict(cfg))
    return dict(rotations=R.cpu(),centers=C.cpu(),registered=registered.cpu(),
        translations=(-(R@C[...,None]).squeeze(-1)).cpu(),intrinsics=K.cpu(),
        points3d=points.cpu(),observations=observations.cpu(),reprojection_errors=errors.cpu(),
        camera_groups=initial.get('camera_groups'),image_paths=initial['image_paths'],
        initial_intrinsics=initial['intrinsics'].cpu(),solver_summaries=summaries,
        refinement_report=report,stage='flexglue_tracks_bundle_adjustment')
