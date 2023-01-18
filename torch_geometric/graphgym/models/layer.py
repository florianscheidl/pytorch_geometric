import copy
from dataclasses import dataclass, replace
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import Tensor

import torch_geometric as pyg
import torch_geometric.graphgym.models.act
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.contrib.layer.generalconv import (
    GeneralConvLayer,
    GeneralEdgeConvLayer,
)
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn import Linear as Linear_pyg, inits
from torch_geometric.nn.dense import HeteroLinear
from torch_geometric.typing import Metadata
try:
    from pyg_lib.ops import segment_matmul  # noqa
    _WITH_PYG_LIB = True
except ImportError:
    _WITH_PYG_LIB = False

@dataclass
class LayerConfig:
    # batchnorm parameters.
    has_batchnorm: bool = False
    bn_eps: float = 1e-5
    bn_mom: float = 0.1

    # mem parameters.
    mem_inplace: bool = False

    # gnn parameters.
    dim_in: int = -1
    dim_out: int = -1
    edge_dim: int = -1
    dim_inner: int = None
    num_layers: int = 2
    has_bias: bool = True
    # regularizer parameters.
    has_l2norm: bool = True
    dropout: float = 0.0
    # activation parameters.
    has_act: bool = True
    final_act: bool = True
    act: str = 'relu'

    # other parameters.
    keep_edge: float = 0.5
    graph_type: str = 'homo'


def new_layer_config(dim_in, dim_out, num_layers, has_act, has_bias, cfg, graph_type: Optional[str] = 'homo'):
    return LayerConfig(
        has_batchnorm=cfg.gnn.batchnorm,
        bn_eps=cfg.bn.eps,
        bn_mom=cfg.bn.mom,
        mem_inplace=cfg.mem.inplace,
        dim_in=dim_in,
        dim_out=dim_out,
        edge_dim=cfg.dataset.edge_dim,
        has_l2norm=cfg.gnn.l2norm,
        dropout=cfg.gnn.dropout,
        has_act=has_act,
        final_act=True,
        act=cfg.gnn.act,
        has_bias=has_bias,
        keep_edge=cfg.gnn.keep_edge,
        dim_inner=cfg.gnn.dim_inner,
        num_layers=num_layers,
        graph_type=graph_type
    )


# General classes
class GeneralLayer(nn.Module):
    """
    General wrapper for layers

    Args:
        name (string): Name of the layer in registered :obj:`layer_dict`
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        has_act (bool): Whether has activation after the layer
        has_bn (bool):  Whether has BatchNorm in the layer
        has_l2norm (bool): Wheter has L2 normalization after the layer
        **kwargs (optional): Additional args
    """
    def __init__(self, name, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.has_l2norm = layer_config.has_l2norm
        self.has_bn = layer_config.has_batchnorm
        layer_config.has_bias = not self.has_bn

        # TODO: this was added to accomodate methods that require metadata and out_channels
        if name in ['hanconv','hgtconv']:
            self.layer = register.layer_dict[name](in_channels=layer_config.dim_in,
                                                   out_channels=layer_config.dim_out,
                                                   metadata=cfg.dataset.metadata, **kwargs)
        elif name=='heatconv':
            self.layer = register.layer_dict[name](in_channels=layer_config.dim_in,
                                                   out_channels=layer_config.dim_out,
                                                   num_node_types=len(cfg.dataset.metadata[0]),
                                                   num_edge_types=len(cfg.dataset.metadata[1]),
                                                   edge_type_emb_dim=cfg.gnn.heat_edge_type_emb_dim,
                                                   edge_dim=cfg.gnn.heat_edge_dim,
                                                   edge_attr_emb_dim=cfg.gnn.heat_edge_attr_emb_dim,
                                                   **kwargs)
        elif name=='heteroconv':
            heteroconv_dict = dict()
            for edge_type in cfg.dataset.metadata[1]:
                src_type = edge_type[0]
                dst_type = edge_type[2]
                name = f'_{src_type}_{dst_type}'
                heteroconv_dict[edge_type] = register.layer_dict[cfg.gnn.heteroconv[name]](layer_config,**kwargs)
            self.layer = register.layer_dict['heteroconv'](heteroconv_dict)
        else:
            self.layer = register.layer_dict[name](layer_config, **kwargs)
        layer_wrapper = []
        if self.has_bn and not cfg.gnn.graph_type.startswith('hetero'):
            layer_wrapper.append(
                nn.BatchNorm1d(layer_config.dim_out, eps=layer_config.bn_eps,
                               momentum=layer_config.bn_mom))
        if layer_config.dropout > 0:
            layer_wrapper.append(
                nn.Dropout(p=layer_config.dropout,
                           inplace=layer_config.mem_inplace))
        if layer_config.has_act:
            layer_wrapper.append(register.act_dict[layer_config.act]())
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, batch):

        if not isinstance(batch, Tensor) and cfg.gnn.layer_type in ['hanconv', 'hgtconv']:
            if not hasattr(batch, 'edge_index_dict') or len(batch.edge_index_dict)==0:
                batch.edge_index_dict = {batch.edge_types[i]: batch.edge_stores[i]["edge_index"] for i in range(len(batch.edge_types)) if "edge_index" in batch.edge_stores[i]} # TODO: this might be (super) inefficient -> more importantly, it is wrong.
            if not hasattr(batch, 'x_dict') or len(batch.x_dict)==0:
                batch.x_dict = {batch.node_types[i]: batch.node_stores[i]["_Cochain__x"] for i in range(len(batch.node_types)) if "_Cochain__x" in batch.node_stores[i]}
            batch = self.layer(batch)  # TODO: is this sufficient?
            if self.has_bn: # do batch normalisation "by hand" for hetero graphs
                batch.x_dict = {key: F.normalize(batch.x_dict[key], p=2, dim=-1) for key in batch.x_dict.keys()}
            batch.x_dict = {key: self.post_layer(batch.x_dict[key]) for key in batch.x_dict.keys() if batch.x_dict[key] is not None}# TODO: this applies various transformations to the single node types, not sure if this is desirable.

        if not isinstance(batch, Tensor) and cfg.gnn.layer_type=='heteroconv':
            if not hasattr(batch, 'edge_index_dict') or len(batch.edge_index_dict) == 0:
                batch.edge_index_dict = {batch.edge_types[i]: batch.edge_stores[i]["edge_index"] for i in
                                         range(len(batch.edge_types)) if "edge_index" in batch.edge_stores[
                                             i]}  # TODO: this might be (super) inefficient -> more importantly, it is wrong.
            if not hasattr(batch, 'x_dict') or len(batch.x_dict) == 0:
                batch.x_dict = {batch.node_types[i]: batch.node_stores[i]["_Cochain__x"] for i in
                                range(len(batch.node_types)) if "_Cochain__x" in batch.node_stores[i]}
            batch = self.layer(batch)  # TODO: is this sufficient?
            if self.has_bn:  # do batch normalisation "by hand" for hetero graphs
                batch.x_dict = {key: F.normalize(batch.x_dict[key], p=2, dim=-1) for key in batch.x_dict.keys()}
            batch.x_dict = {key: self.post_layer(batch.x_dict[key]) for key in batch.x_dict.keys() if batch.x_dict[
                key] is not None}  # TODO: this applies various transformations to the single node types, not sure if this is desirable.

        elif not isinstance(batch, Tensor) and cfg.gnn.layer_type == 'heatconv':
            raise NotImplementedError
            # batch = self.layer(batch.x, batch.edge_index, batch.node_type, batch.edge_type)
        elif cfg.gnn.graph_type == 'homo':
            batch = self.layer(batch)
            if isinstance(batch, torch.Tensor):
                batch = self.post_layer(batch)
                if self.has_l2norm:
                    batch = F.normalize(batch, p=2, dim=1)
            else:
                batch.x = self.post_layer(batch.x)
                if self.has_l2norm:
                    batch.x = F.normalize(batch.x, p=2, dim=1)
        return batch


class GeneralMultiLayer(nn.Module):
    """
    General wrapper for a stack of multiple layers

    Args:
        name (string): Name of the layer in registered :obj:`layer_dict`
        num_layers (int): Number of layers in the stack
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        dim_inner (int): The dimension for the inner layers
        final_act (bool): Whether has activation after the layer stack
        **kwargs (optional): Additional args
    """
    def __init__(self, name, layer_config: LayerConfig, **kwargs):
        super().__init__()
        dim_inner = layer_config.dim_out \
            if layer_config.dim_inner is None \
            else layer_config.dim_inner
        for i in range(layer_config.num_layers):
            d_in = layer_config.dim_in if i == 0 else dim_inner
            d_out = layer_config.dim_out \
                if i == layer_config.num_layers - 1 else dim_inner
            has_act = layer_config.final_act \
                if i == layer_config.num_layers - 1 else True
            inter_layer_config = copy.deepcopy(layer_config)
            inter_layer_config.dim_in = d_in
            inter_layer_config.dim_out = d_out
            inter_layer_config.has_act = has_act
            layer = GeneralLayer(name, inter_layer_config, **kwargs)
            self.add_module('Layer_{}'.format(i), layer)

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        return batch


# ---------- Core basic layers. Input: batch; Output: batch ----------------- #


@register_layer('linear')
class Linear(nn.Module):
    """
    Basic Linear layer.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        bias (bool): Whether has bias term
        **kwargs (optional): Additional args
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = Linear_pyg(layer_config.dim_in, layer_config.dim_out,
                                bias=layer_config.has_bias)
        self.layer_config = layer_config

    def forward(self, batch):
        if self.layer_config.graph_type.startswith('hetero'):
            batch.x_dict = {key: self.model(batch.x_dict[key]) for key in batch.x_dict.keys()}
        else:
            if isinstance(batch, torch.Tensor):
                batch = self.model(batch)
            else:
                 batch.x = self.model(batch.x)
        return batch

@register_layer('heterolinear')
class HeteroLinear(torch.nn.Module):
    r"""Applies separate linear tranformations to the incoming data according
    to types

    .. math::
        \mathbf{x}^{\prime}_{\kappa} = \mathbf{x}_{\kappa}
        \mathbf{W}^{\top}_{\kappa} + \mathbf{b}_{\kappa}

    for type :math:`\kappa`.
    It supports lazy initialization and customizable weight and bias
    initialization.

    Args:
        in_channels (int): Size of each input sample. Will be initialized
            lazily in case it is given as :obj:`-1`.
        out_channels (int): Size of each output sample.
        num_types (int): The number of types.
        is_sorted (bool, optional): If set to :obj:`True`, assumes that
            :obj:`type_vec` is sorted. This avoids internal re-sorting of the
            data and can improve runtime and memory efficiency.
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.Linear`.

    Shapes:
        - **input:**
          features :math:`(*, F_{in})`,
          type vector :math:`(*)`
        - **output:** features :math:`(*, F_{out})`
    """
    def __init__(self, in_channels: int, out_channels: int, num_types: int = 3,
                 is_sorted: bool = False, **kwargs): # todo: num_types should be inferred from the data
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_types = num_types
        self.is_sorted = is_sorted
        self.kwargs = kwargs

        self._WITH_PYG_LIB = torch.cuda.is_available() and _WITH_PYG_LIB

        if self._WITH_PYG_LIB:
            self.weight = torch.nn.Parameter(
                torch.Tensor(num_types, in_channels, out_channels))
            if kwargs.get('bias', True):
                self.bias = Parameter(torch.Tensor(num_types, out_channels))
            else:
                self.register_parameter('bias', None)
        else:
            self.lins = torch.nn.ModuleList([
                Linear(in_channels, out_channels, **kwargs)
                for _ in range(num_types)
            ])
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self._WITH_PYG_LIB:
            reset_weight_(self.weight, self.in_channels,
                          self.kwargs.get('weight_initializer', None))
            reset_weight_(self.bias, self.in_channels,
                          self.kwargs.get('bias_initializer', None))
        else:
            for lin in self.lins:
                lin.reset_parameters()

    def forward(self, x: Tensor, type_vec: Tensor) -> Tensor:
        r"""
        Args:
            x (Tensor): The input features.
            type_vec (LongTensor): A vector that maps each entry to a type.
        """
        if self._WITH_PYG_LIB:
            assert self.weight is not None

            if not self.is_sorted:
                if (type_vec[1:] < type_vec[:-1]).any():
                    type_vec, perm = type_vec.sort()
                    x = x[perm]

            type_vec_ptr = torch.ops.torch_sparse.ind2ptr(
                type_vec, self.num_types)
            out = segment_matmul(x, type_vec_ptr, self.weight)
            if self.bias is not None:
                out += self.bias[type_vec]
        else:
            out = x.new_empty(x.size(0), self.out_channels)
            for i, lin in enumerate(self.lins):
                mask = type_vec == i
                out[mask] = lin(x[mask])
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_types={self.num_types}, '
                f'bias={self.kwargs.get("bias", True)})')


class BatchNorm1dNode(nn.Module):
    """
    BatchNorm for node feature.

    Args:
        dim_in (int): Input dimension
    """
    def __init__(self, layer_config: LayerConfig):
        super().__init__()
        self.bn = nn.BatchNorm1d(layer_config.dim_in, eps=layer_config.bn_eps,
                                 momentum=layer_config.bn_mom)

    def forward(self, batch):
        batch.x = self.bn(batch.x)
        return batch


class BatchNorm1dEdge(nn.Module):
    """
    BatchNorm for edge feature.

    Args:
        dim_in (int): Input dimension
    """
    def __init__(self, layer_config: LayerConfig):
        super().__init__()
        self.bn = nn.BatchNorm1d(layer_config.dim_in, eps=layer_config.bn_eps,
                                 momentum=layer_config.bn_mom)

    def forward(self, batch):
        batch.edge_attr = self.bn(batch.edge_attr)
        return batch


@register_layer('mlp')
class MLP(nn.Module):
    """
    Basic MLP model.
    Here 1-layer MLP is equivalent to a LineAr layer.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        bias (bool): Whether has bias term
        dim_inner (int): The dimension for the inner layers
        num_layers (int): Number of layers in the stack
        **kwargs (optional): Additional args
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        dim_inner = layer_config.dim_in \
            if layer_config.dim_inner is None \
            else layer_config.dim_inner
        layer_config.has_bias = True
        layers = []
        if layer_config.num_layers > 1:
            sub_layer_config = LayerConfig(
                num_layers=layer_config.num_layers - 1,
                dim_in=layer_config.dim_in, dim_out=dim_inner,
                dim_inner=dim_inner, final_act=True)
            layers.append(GeneralMultiLayer('linear', sub_layer_config)) # toDO: potentially make adjustments in "Linear"
            layer_config = replace(layer_config, dim_in=dim_inner)
            layers.append(Linear(layer_config))
        else:
            layers.append(Linear(layer_config))
        self.model = nn.Sequential(*layers)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        elif type(batch).__name__ == "HeteroDataBatch":
            batch.x_dict['0_cell'] = self.model(batch.x_dict['0_cell'])
        else:
            batch.x = self.model(batch.x)
        return batch


@register_layer('gcnconv')
class GCNConv(nn.Module):
    """
    Graph Convolutional Network (GCN) layer
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.GCNConv(layer_config.dim_in, layer_config.dim_out,
                                    bias=layer_config.has_bias)

    def forward(self, batch):
        batch.x = self.model(x=batch.x, edge_index=batch.edge_index)
        return batch


@register_layer('sageconv')
class SAGEConv(nn.Module):
    """
    GraphSAGE Conv layer
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.SAGEConv(layer_config.dim_in, layer_config.dim_out,
                                     bias=layer_config.has_bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


@register_layer('gatconv')
class GATConv(nn.Module):
    """
    Graph Attention Network (GAT) layer
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.GATConv(layer_config.dim_in, layer_config.dim_out,
                                    bias=layer_config.has_bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


@register_layer('ginconv')
class GINConv(nn.Module):
    """
    Graph Isomorphism Network (GIN) layer
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        gin_nn = nn.Sequential(
            Linear_pyg(layer_config.dim_in, layer_config.dim_out), nn.ReLU(),
            Linear_pyg(layer_config.dim_out, layer_config.dim_out))
        self.model = pyg.nn.GINConv(gin_nn)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


@register_layer('splineconv')
class SplineConv(nn.Module):
    """
    SplineCNN layer
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.SplineConv(layer_config.dim_in,
                                       layer_config.dim_out, dim=1,
                                       kernel_size=2,
                                       bias=layer_config.has_bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        return batch


@register_layer('generalconv')
class GeneralConv(nn.Module):
    """A general GNN layer"""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = GeneralConvLayer(layer_config.dim_in,
                                      layer_config.dim_out,
                                      bias=layer_config.has_bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


@register_layer('generaledgeconv')
class GeneralEdgeConv(nn.Module):
    """A general GNN layer that supports edge features as well"""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = GeneralEdgeConvLayer(layer_config.dim_in,
                                          layer_config.dim_out,
                                          layer_config.edge_dim,
                                          bias=layer_config.has_bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index,
                             edge_feature=batch.edge_attr)
        return batch


@register_layer('generalsampleedgeconv')
class GeneralSampleEdgeConv(nn.Module):
    """A general GNN layer that supports edge features and edge sampling"""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = GeneralEdgeConvLayer(layer_config.dim_in,
                                          layer_config.dim_out,
                                          layer_config.edge_dim,
                                          bias=layer_config.has_bias)
        self.keep_edge = layer_config.keep_edge

    def forward(self, batch):
        edge_mask = torch.rand(batch.edge_index.shape[1]) < self.keep_edge
        edge_index = batch.edge_index[:, edge_mask]
        edge_feature = batch.edge_attr[edge_mask, :]
        batch.x = self.model(batch.x, edge_index, edge_feature=edge_feature)
        return batch

# additional functions
def reset_weight_(weight: Tensor, in_channels: int,
                  initializer: Optional[str] = None) -> Tensor:
    if in_channels <= 0:
        pass
    elif initializer == 'glorot':
        inits.glorot(weight)
    elif initializer == 'uniform':
        bound = 1.0 / math.sqrt(in_channels)
        torch.nn.init.uniform_(weight.data, -bound, bound)
    elif initializer == 'kaiming_uniform':
        inits.kaiming_uniform(weight, fan=in_channels, a=math.sqrt(5))
    elif initializer is None:
        inits.kaiming_uniform(weight, fan=in_channels, a=math.sqrt(5))
    else:
        raise RuntimeError(f"Weight initializer '{initializer}' not supported")

    return weight