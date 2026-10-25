from .base import Extractor, Matcher, Matches
from .loma import LoMaExtractor, LoMaMatcher, create_loma

__all__ = ["Extractor", "Matcher", "Matches", "LoMaExtractor", "LoMaMatcher", "create_loma"]


def get_extractor_and_matcher(name: str) -> tuple[Extractor, Matcher]:
    if name == "loma":
        loma_model = create_loma()
        return LoMaExtractor(loma_model), LoMaMatcher(loma_model)
    else:
        raise ValueError(f"Unknown extractor and matcher: {name}")
