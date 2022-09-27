"""
Base class for transforming a plain graph to a higher-order graph data model.
"""

from builtins import str
from typing import List

from torch_sparse import SparseTensor

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class LiftTransform(BaseTransform):
    """Class for lifting transformation from plain graph to simplicial complex."""

    def __init__(self,
                 lift_method: str = None,
                 init_method: str = None,
                 boundary_adjacency_tensors: List[SparseTensor] = None):
        self.lift_method = lift_method
        self.boundary_adjacency_tensors = boundary_adjacency_tensors
        self.init_method = init_method
        super().__init__()

    def __call__(self, data: Data):
        raise NotImplementedError