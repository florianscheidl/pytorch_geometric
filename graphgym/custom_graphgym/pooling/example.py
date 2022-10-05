import torch
import torch.nn as nn
from torch_scatter import scatter

from torch_geometric.graphgym.register import register_pooling


@register_pooling('example')
def global_example_pool(x, batch, size=None):
    size = batch.max().item() + 1 if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='add')

@register_pooling('broken_type_wise_hetero_pooling')
def type_wise_hetero_pooling(x_dict, size=None):
    x = torch.cat([x_dict[key] for key in x_dict.keys()], dim=0) # potentially change dimension
    indx = []
    for j, key in enumerate(list(x_dict.keys())):
        indx = indx + [j for _ in range(len(x_dict[key]))]
    indx = torch.tensor(indx)
    scattered = scatter(src=x, index= indx, dim=0, dim_size=size, reduce='add')
    out = torch.nn.Linear(scattered.shape[0],1)(scattered.T).flatten()
    return out

@register_pooling('type_wise_hetero_pooling')
def type_wise_hetero_pooling(batch, size=None):
    cell_dim_wise_scatter = {}
    # TODO: Ideally this layer should be shared by all cells:
    type_combination_layer = nn.Linear(len(batch.batch_dict.keys()),1)
    for key in batch.batch_dict.keys():
        cell_dim_wise_scatter[key] = scatter(batch.x_dict[key], batch.batch_dict[key], dim=0, reduce='add')
    out = []
    for i in range(len(batch)):
        single_batch_mixed_cells = torch.vstack([cell_dim_wise_scatter[key][i] for key in cell_dim_wise_scatter.keys()]).T
        out.append(torch.sigmoid(type_combination_layer(single_batch_mixed_cells)))
    out_tensor = torch.hstack(out).T
    return out_tensor

@register_pooling('hetero_add_pooling')
def type_wise_hetero_pooling(batch, size=None):
    # the batch_dict does not work for 1-cells (in some cases), also num_nodes is sometimes incorrectly inferred for 1-cells.
    cell_dim_wise_scatter = {key: scatter(batch.x_dict[key], batch.batch_dict[key], dim=0, reduce='add') for key in batch.x_dict.keys() if batch.x_dict[key] is not None}
    out = []
    for i in range(len(batch)):
        single_batch_mixed_cells = torch.vstack([cell_dim_wise_scatter[key][i] for key in cell_dim_wise_scatter.keys() if len(cell_dim_wise_scatter[key])>i]).sum(dim=0)
        out.append(single_batch_mixed_cells)
    out_tensor = torch.vstack(out)
    return out_tensor