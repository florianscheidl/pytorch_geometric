from typing import Dict, List, Union

from torch import nn

from torch_geometric.nn.conv import HANConv
from torch_geometric.nn.pool import global_add_pool


class HAN(nn.Module):
    def __init__(self, in_channels: Union[int, Dict[str, int]],
                 out_channels: int, hidden_channels=128, heads=8, data=None):
        super().__init__()
        self.han_conv = HANConv(in_channels, hidden_channels, heads=heads,
                                dropout=0.6, metadata=data.metadata())
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        out = self.han_conv(x_dict, edge_index_dict) # a call corresponds to applying forward
        # out = self.lin(out['0_cell'])
        out = global_add_pool(out, batch=None)
        return out
