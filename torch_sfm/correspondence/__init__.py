from .base import Extractor, Matcher, Matches
from .loma import LoMaExtractor, LoMaMatcher, create_loma
from .flexglue import FlexGlueConfig, create_flexglue

__all__ = ["Extractor", "Matcher", "Matches", "LoMaExtractor", "LoMaMatcher", "create_loma",
           "FlexGlueConfig", "create_flexglue", "get_extractor_and_matcher"]


def get_extractor_and_matcher(name: str, config=None) -> tuple[Extractor, Matcher]:
    if name == 'flexglue':
        return create_flexglue(config)
    if name == "loma":
        loma_model = create_loma()
        return LoMaExtractor(loma_model), LoMaMatcher(loma_model)
    else:
        raise ValueError(f"Unknown extractor and matcher: {name}")
