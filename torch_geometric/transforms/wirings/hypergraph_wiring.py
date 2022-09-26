# Define a HeteroData object given a hypergraph and a collection hyperedge types and adjacency types.

from typing import List

import torch
from torch_sparse import SparseTensor

from torch_geometric.data import HeteroData
from torch_geometric.data.custom_complex import Complex
from torch_geometric.transforms.add_metapaths_hops import AddMetaPathsHops


class HypergraphWiring(object):
    def __init__(self, adjacency_types: List[str] = ["boundary","upper"]):
        self.adjacency_types = adjacency_types
        super().__init__()

    def __call__(self, data: Complex, boundary_adjacency_tensors: List[SparseTensor]) -> HeteroData:

        # Initialise the heteroData object
        het_data = HeteroData()

        # Map the Complex to a HeteroData object:
        het_data.y = data.y
        het_data.dim = data.dimension

        # Add higher-order vertices (HoVs) and their features:
        for chain in data.cochains:
            for key in data.cochains[chain].keys:
                    het_data[f"{data.cochains[chain].dim}_cell"][key] = data.cochains[chain][key]
                    het_data[f"{data.cochains[chain].dim}_cell"].num_nodes = data.cochains[chain].num_cells

        # Add higher-order edges (HoEs)
        for d in range(het_data.dim):
            if "boundary" not in self.adjacency_types:
                raise Exception("Require boundary adjacency.")
            else:
                boundaries, cells = boundary_adjacency_tensors[d].storage.row(), boundary_adjacency_tensors[d].storage.col()
                het_data[f"{d}_cell","boundary_of",f"{d+1}_cell"].edge_index = SparseTensor(row=boundaries, col=cells)
                het_data[f"{d+1}_cell","coboundary_of",f"{d}_cell"].edge_index = SparseTensor(row=cells, col=boundaries)

        metapaths = []

        if "upper" in self.adjacency_types:
            for d in range(het_data.dim):
                metapaths.append([(f"{d}_cell","boundary_of",f"{d+1}_cell"),(f"{d+1}_cell","coboundary_of",f"{d}_cell")])
        if "lower" in self.adjacency_types:
            for d in range(het_data.dim):
                metapaths.append([(f"{d+1}_cell", "coboundary_of", f"{d}_cell"), (f"{d}_cell", "boundary_of", f"{d + 1}_cell")])

        het_data = AddMetaPathsHops(metapaths,
                                    drop_orig_edges=False,
                                    keep_same_node_type=True,
                                    drop_unconnected_nodes=False,
                                    max_sample=10000,
                                    weighted=False)(het_data)

        return het_data