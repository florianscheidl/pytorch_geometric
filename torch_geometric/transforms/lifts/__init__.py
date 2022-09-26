from .plain_to_hypergraph import LiftGraphToSimplicialComplex, LiftGraphToCellComplex

__all__ = [
    "LiftGraphToSimplicialComplex",
    "LiftGraphToCellComplex",
]

classes = __all__

from torch_geometric.deprecation import deprecated  # noqa
