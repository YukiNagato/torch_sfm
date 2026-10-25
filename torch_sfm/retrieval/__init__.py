from .base import Retrieval, RetrievalResult
from .sage import SAGERetrieval
from .salad import SALADRetrieval

__all__ = ["Retrieval", "RetrievalResult", "SAGERetrieval", "SALADRetrieval"]


def get_retrieval(name: str, device='cuda') -> Retrieval:
    if name == "sage":
        return SAGERetrieval(device=device)
    elif name == "salad":
        return SALADRetrieval(device=device)
    else:
        raise ValueError(f"Unknown retrieval method: {name}")

