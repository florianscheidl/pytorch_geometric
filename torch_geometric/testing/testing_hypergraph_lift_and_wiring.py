import torch

from torch_geometric.data import Data
from torch_geometric.transforms.lifts.plain_to_hypergraph import LiftGraphToSimplicialComplex, LiftGraphToCellComplex
from torch_geometric.transforms.wirings.hypergraph_wiring import HypergraphWiring

x = torch.tensor([0,1,2,3,4,5,6])
edge_index_rings = torch.tensor([[0,0,1,2,3,3,4,5],
                                 [1,2,2,3,4,6,5,6]])
edge_index_cliques = torch.tensor([[0,0,1,2,3,3,4,4,5,5],
                                   [1,2,2,3,4,6,5,6,3,6]])

base_graph_rings = Data(x=x, edge_index=edge_index_rings)
base_graph_cliques = Data(x=x, edge_index=edge_index_cliques)

boundary_adjs_cell, lift_cell = LiftGraphToCellComplex(return_HoEs=True,
                                                       lift_method="rings",
                                                       max_induced_cycle_length=4)(base_graph_rings)
boundary_adjs_simplicial, lift_simplex = LiftGraphToSimplicialComplex(return_HoEs=True,
                                                                      lift_method="clique",
                                                                      max_clique_dim=3)(base_graph_cliques)

wiring = HypergraphWiring(adjacency_types=["boundary","upper","lower"])

wired_cell = wiring(lift_cell, boundary_adjs_cell)
wired_simplicial = wiring(lift_simplex, boundary_adjs_simplicial)

print(wired_cell, wired_simplicial)