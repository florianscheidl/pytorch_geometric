from .wiring_transform import WiringTransform
from .hypergraph_wiring import HypergraphWiring

__all__ = [
    "WiringTransform",
    "HypergraphWiring",
]

classes = __all__

from torch_geometric.deprecation import deprecated  # noqa
