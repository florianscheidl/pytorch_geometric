import torch

from torch_geometric.datasets import TUDataset
from torch_geometric.transforms.lifts import LiftGraphToCellComplex
from torch_geometric.transforms.wirings import HypergraphWiring
from torch_geometric.transforms.liwich_transforms import LiftAndWire
from torch_geometric.loader import DataLoader

# configure lift and wiring
lifting_cell = LiftGraphToCellComplex(lift_method="rings",
                                      max_induced_cycle_length=10,
                                      init_edges=True,
                                      init_method = "mean")
wiring = HypergraphWiring(adjacency_types=["boundary","upper"])
lift_wire = LiftAndWire(lifting_cell, wiring)

#load example dataset with the given transform

normal = True
# normal = False

if normal:
    dataset = TUDataset(root='processed_dataset',
                        use_node_attr=True,
                        name='MUTAG')
else:
    dataset = TUDataset(pre_transform=lift_wire,
                        root='processed_dataset',
                        use_node_attr = True,
                        name='MUTAG')
train_dataset = dataset[len(dataset) // 10:]
train_loader = DataLoader(train_dataset, shuffle=True)
# [data for data in train_loader]
print(dataset[0])
