import torch

from torch_geometric.data import Data
from torch_geometric.transforms.lifts import LiftGraphToSimplicialComplex, LiftGraphToCellComplex
from torch_geometric.transforms.wirings import HypergraphWiring
from torch_geometric.transforms.liwich_transforms import LiftAndWire


x = torch.tensor([0,1,2,3,4,5,6])
edge_index_rings = torch.tensor([[0,0,1,2,3,3,4,5],
                                 [1,2,2,3,4,6,5,6]])
edge_index_cliques = torch.tensor([[0,0,1,2,3,3,4,4,5,5],
                                   [1,2,2,3,4,6,5,6,3,6]])

base_graph_rings = Data(x=x, edge_index=edge_index_rings)
base_graph_cliques = Data(x=x, edge_index=edge_index_cliques)

lift_cell = LiftGraphToCellComplex(lift_method="rings",
                                   max_induced_cycle_length=4)
lift_simplex = LiftGraphToSimplicialComplex(lift_method="clique",
                                            max_clique_dim=3)
wiring = HypergraphWiring(adjacency_types=["boundary","upper","lower"])
lift_and_wire_transform_cell = LiftAndWire(lift_cell, wiring)
lift_and_wire_transform_simplex = LiftAndWire(lift_simplex, wiring)

wired_cell = lift_and_wire_transform_cell(base_graph_rings)
wired_simplex = lift_and_wire_transform_simplex(base_graph_cliques)

print(wired_cell)