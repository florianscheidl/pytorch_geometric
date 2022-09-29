from ctypes import Union
from typing import Dict, List

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.logging import log
from torch_geometric.transforms.lifts import LiftGraphToCellComplex
from torch_geometric.transforms.wirings import HypergraphWiring
from torch_geometric.transforms.liwich_transforms import LiftAndWire

from torch_geometric.nn.pool import global_add_pool
from torch_geometric.nn.conv import HANConv


# configure lift and wiring
lifting_cell = LiftGraphToCellComplex(lift_method="rings",
                                      max_induced_cycle_length=10,
                                      init_edges=True,
                                      init_rings=True,
                                      init_method="mean")
wiring = HypergraphWiring(adjacency_types=["boundary", "upper"])
lift_wire = LiftAndWire(lifting_cell, wiring)

# load example dataset with the given transform
dataset = TUDataset(pre_transform=lift_wire,
                    root='processed_dataset',
                    use_node_attr=True,
                    name='MUTAG')
data = dataset.data

train_dataset = dataset[len(dataset) // 10:]
train_loader = DataLoader(train_dataset, shuffle=True)

test_dataset = dataset[:len(dataset) // 10]
test_loader = DataLoader(test_dataset, shuffle=True)


class HAN(nn.Module):
    def __init__(self, in_channels: int|Dict[str, int],
                 out_channels: int, hidden_channels=128, heads=8, data=None):
        super().__init__()
        self.han_conv = HANConv(in_channels, hidden_channels, heads=heads,
                                dropout=0.6, metadata=data.metadata())
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        out = self.han_conv(x_dict, edge_index_dict) # a call corresponds to applying forward
        out = self.han_conv(x_dict, edge_index_dict)
        # out = self.lin(out)
        out = global_add_pool(out, batch=None)
        out = self.lin(out)
        out = torch.sigmoid(out)
        return out


# ML model
model = HAN(in_channels=-1,
            hidden_channels=128,
            heads=4,
            out_channels=1,
            data=data)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data, model = data.to(device), model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        data.edge_index_dict = {data.edge_types[i]: data.edge_stores[i]["edge_index"] for i in range(len(data.edge_types))}
        out = model(data.x_dict, data.edge_index_dict)
        loss = F.binary_cross_entropy(out, data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * float(data.num_graphs)
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        data.edge_index_dict = {data.edge_types[i]: data.edge_stores[i]["edge_index"] for i in
                                range(len(data.edge_types))}
        pred = model(data.x_dict, data.edge_index_dict).round()
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)


for epoch in range(1, 100 + 1):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    log(Epoch=epoch, Loss=loss, Train=train_acc, Test=test_acc)
