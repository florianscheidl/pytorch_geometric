# Define a HeteroData object given a hypergraph and a collection hyperedge types and adjacency types.

from typing import List

import torch
from torch_sparse import SparseTensor

from torch_geometric.data import HeteroData
from torch_geometric.data.custom_complex import Complex
from torch_geometric.transforms.add_metapaths_hops import AddMetaPathsHops
from torch_geometric.transforms.wirings import WiringTransform


class HypergraphWiring(WiringTransform):
    def __init__(self, adjacency_types: List[str] = ["boundary", "upper"], boundary_adjacency_tensors: List[SparseTensor] = None):
        self.adjacency_types = adjacency_types
        self.boundary_adjacency_tensors = boundary_adjacency_tensors
        super().__init__(self.adjacency_types, self.boundary_adjacency_tensors)

    def __call__(self, data: Complex) -> HeteroData:

        # Initialise the heteroData object
        het_data = HeteroData()

        # Map the Complex to a HeteroData object:
        for key in data.keys:
            setattr(het_data, key, data[key])


        # Add higher-order vertices (HoVs) and their features:
        for chain in data.cochains:
            for key in data.cochains[chain].keys:
                    het_data[f"{data.cochains[chain].dim}_cell"][key] = data.cochains[chain][key]
                    het_data[f"{data.cochains[chain].dim}_cell"].num_nodes = data.cochains[chain].num_cells

                    # Add higher-order edges (HoEs)
        for d in range(het_data.dimension):
            if "boundary" not in self.adjacency_types:
                raise Exception("Require boundary adjacency.")
            else:
                boundaries, cells = self.boundary_adjacency_tensors[d].storage.row(), self.boundary_adjacency_tensors[
                    d].storage.col()
                het_data[f"{d}_cell", "boundary_of", f"{d + 1}_cell"].edge_index = SparseTensor(row=boundaries,
                                                                                                col=cells)
                het_data[f"{d + 1}_cell", "coboundary_of", f"{d}_cell"].edge_index = SparseTensor(row=cells,
                                                                                                  col=boundaries)

        metapaths = []

        if "upper" in self.adjacency_types:
            for d in range(het_data.dimension):
                if [(f"{d}_cell","boundary_of",f"{d+1}_cell"),(f"{d+1}_cell","coboundary_of",f"{d}_cell")] not in metapaths:
                    metapaths.append([(f"{d}_cell","boundary_of",f"{d+1}_cell"),(f"{d+1}_cell","coboundary_of",f"{d}_cell")])
        if "lower" in self.adjacency_types:
            for d in range(het_data.dimension):
                if [(f"{d+1}_cell", "coboundary_of", f"{d}_cell"), (f"{d}_cell", "boundary_of", f"{d + 1}_cell")] not in metapaths:
                    metapaths.append([(f"{d+1}_cell", "coboundary_of", f"{d}_cell"), (f"{d}_cell", "boundary_of", f"{d + 1}_cell")])

        het_data = AddMetaPathsHops(metapaths,
                                    drop_orig_edges=False,
                                    keep_same_node_type=True,
                                    drop_unconnected_nodes=False,
                                    max_sample=10000,
                                    weighted=False)(het_data)

        # TODO: remove the to_dense potentially transform node_stores to SparseTensor (temporarily removed)
        # This is a temporary fix to avoid the error caused by collate
        for i in het_data.edge_types:
            row, col = het_data[i].edge_index.storage._row, het_data[i].edge_index.storage._col
            het_data[i].edge_index = torch.vstack([row, col])

        het_data.x_dict = {het_data.node_types[i]: het_data.node_stores[i]._Cochain__x for i in range(len(het_data.node_types)) if hasattr(het_data.node_stores[i], "_Cochain__x")}
        het_data.edge_index_dict = {het_data.edge_types[i]: het_data.edge_stores[i].edge_index for i in range(len(het_data.edge_types))}

        return het_data