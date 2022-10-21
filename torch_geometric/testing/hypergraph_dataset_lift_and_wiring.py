import torch

from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.transforms.lifts import LiftGraphToCellComplex
from torch_geometric.transforms.wirings import HypergraphWiring
from torch_geometric.transforms.liwich_transforms import LiftAndWire
from torch_geometric.loader import DataLoader

# configure lift and wiring
lifting_cell = LiftGraphToCellComplex(lift_method="rings",
                                      max_induced_cycle_length=5,
                                      init_edges=True,
                                      init_rings=True,
                                      init_method="mean")
wiring = HypergraphWiring(adjacency_types=["boundary", "upper"])
lift_wire = LiftAndWire(lifting_cell, wiring)

#load example dataset with the given transform

normal = True
normal = False
pre_transform = True
# name='PROTEINS'
name='ENZYMES'
# name='NCI1'

if normal:
    dataset = TUDataset(root='processed_dataset',
                        use_node_attr=True,
                        name=name)
elif pre_transform:
    dataset = TUDataset(pre_transform=lift_wire,
                        root='processed_dataset',
                        use_node_attr = True,
                        name=name)
else:
    dataset = TUDataset(transform=lift_wire,
                        root='processed_dataset',
                        use_node_attr = True,
                        name=name)

# dataset_no_transform = Planetoid('processed_dataset', name, transform=lift_wire)
dataset_no_transform = TUDataset('processed_dataset', name)
# where are the masks?

# train_dataset = dataset[len(dataset) // 10:]
train_dataset = dataset_no_transform[0:19]
# train_dataset = dataset

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=4)
print(train_loader)
