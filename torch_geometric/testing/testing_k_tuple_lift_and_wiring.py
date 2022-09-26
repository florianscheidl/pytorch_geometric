import torch

from torch_geometric.data import Data
# from torch_geometric.transforms.lifts.plain_to_k_tuple import LiftGraphToKTuple

x = torch.tensor([[0],[1],[2],[3]])
edge_index = torch.tensor([[0,0,1,2],
                           [1,2,2,3]])

base_graph = Data(x=x, edge_index=edge_index)

boundary_adjs, lift = LiftGraphToKTuple(return_HoEs=True)(base_graph)

wiring = kTupleWiring(adjacency_types=["boundary","upper","lower"])

wired_k_tuple = wiring(lift, boundary_adjs)

print(wired_k_tuple)