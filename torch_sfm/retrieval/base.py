from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Generic, TypeVar

import torch

Feature = TypeVar("Feature")


@dataclass
class RetrievalResult:
    indices: torch.Tensor  # (M,), int64; positions in the input feature list
    scores: torch.Tensor  # (M,), descending similarity, M <= k


class Retrieval(ABC, Generic[Feature]):
    @abstractmethod
    def extract(self, image: torch.Tensor) -> Feature:
        """Extract a feature from an RGB CHW float image in [0, 1]."""
        ...

    @abstractmethod
    def similarity(self, features: list[Feature]) -> torch.Tensor:
        """Return an (N, N) matrix; larger scores mean more similar images."""
        ...

    @torch.inference_mode()
    def retrieval_top_k(
        self,
        features: list[Feature],
        k: int,
        threshold: float | None = None,
        *,
        exclude_self: bool = True,
    ) -> list[RetrievalResult]:
        """One result per input feature; retain finite scores >= threshold."""
        if not isinstance(k, int) or k < 0:
            raise ValueError("k must be a nonnegative integer")
        if threshold is not None and not math.isfinite(threshold):
            raise ValueError("threshold must be finite or None")
        count = len(features)
        if count == 0:
            return []
        k = min(k, count - int(exclude_self))
        if k == 0:
            return [RetrievalResult(torch.empty(0, dtype=torch.long), torch.empty(0))
                    for _ in features]

        scores = self.similarity(features)
        if scores.shape != (count, count) or not scores.is_floating_point():
            raise ValueError("similarity must return a floating-point (N, N) tensor")
        scores = scores.clone()
        scores.masked_fill_(~torch.isfinite(scores), -torch.inf)
        if exclude_self:
            scores.fill_diagonal_(-torch.inf)
        values, indices = scores.topk(k, dim=1, sorted=True)
        valid = torch.isfinite(values)
        if threshold is not None:
            valid &= values >= threshold
        return [RetrievalResult(index[mask].cpu(), score[mask].cpu())
                for index, score, mask in zip(indices, values, valid)]
