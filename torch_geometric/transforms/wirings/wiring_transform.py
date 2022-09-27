# Define a HeteroData object given a hypergraph and a collection hyperedge types and adjacency types.

from typing import List

from torch_sparse import SparseTensor

from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform


class WiringTransform(BaseTransform):
    def __init__(self, adjacency_types: List[str], boundary_adjacency_tensors: List[SparseTensor]):
        self.adjacency_types = adjacency_types
        self.boundary_adjacency_tensors = boundary_adjacency_tensors
        super().__init__()

    def __call__(self, data) -> HeteroData:
        raise NotImplementedError