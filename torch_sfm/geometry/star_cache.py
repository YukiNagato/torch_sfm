"""Compact persistent star results; dense predictions are inference scratch data."""
from dataclasses import replace

from .star_fusion import compact_star
from .virtual_tracks import prepare_virtual_candidates


def compact_star_cache(data, fusion_cfg, virtual_cfg, *, save_virtual_candidates=True):
    """Freeze depth-dependent summaries without retaining dense tensor storage.

    Cache all sparse candidates before global-dependent track selection. This
    preserves the existing selection RNG and permits changing global intrinsics,
    view validity and track budgets when resuming. Depth sampling and parallax
    sampling settings are recorded because they require inference to change.
    """
    star = compact_star(data, replace(fusion_cfg, min_triangulation_degrees=1.))
    result = {key: value for key, value in data.items()
              if key not in ('depth', 'center_masks', 'depth_conf')}
    result.update(version=2, depth=None, parallax_degrees=star.parallax_degrees,
                  parallax_samples=fusion_cfg.parallax_samples)
    if save_virtual_candidates:
        result['virtual_candidates'] = prepare_virtual_candidates(
            data, int(data['image_ids'][0]), virtual_cfg)
    return result
