import torch.nn as nn

from torch_geometric.graphgym.register import register_head
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import MLP, new_layer_config


@register_head('hetero_node_head')
class GNNHeteroNodeHead(nn.Module):
    """
    GNN prediction head for node prediction tasks.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.layer_post_mp = MLP(
            new_layer_config(dim_in, dim_out, cfg.gnn.layers_post_mp,
                             has_act=False, has_bias=True, cfg=cfg))

    def _apply_index(self, batch):
        mask = '{}_mask'.format(batch.split)
        try:
            batch_y = batch['0_cell']['y']
        except:
            Exception('Problem with finding y values of the batch.')
        return batch.x_dict['0_cell'][batch[mask]], batch_y[batch[mask]]

    def forward(self, batch):
        batch.x_dict['0_cell'] = self.layer_post_mp(batch.x_dict['0_cell'])
        pred, label = self._apply_index(batch)
        return pred, label