import math
import warnings
from collections import defaultdict
from typing import List, Optional, Union, Dict

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter, Embedding, Module
import torch.nn.functional as F
from torch_sparse import SparseTensor

from torch_geometric.data import Batch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn.dense import Linear, HeteroLinear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.hgt_conv import group
from torch_geometric.nn.inits import glorot, zeros, reset, ones
from torch_geometric.nn import ModuleDict
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType, OptTensor
from torch_geometric.utils import softmax
from torch_geometric.utils.hetero import check_add_self_loops

# Note: A registered GNN layer should take 'batch' as input
# and 'batch' as output


# Example 1: Directly define a GraphGym format Conv
# take 'batch' as input and 'batch' as output
# @register_layer('exampleconv1')
class ExampleConv1(MessagePassing):
    r"""Example GNN layer

    """
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super().__init__(aggr=cfg.gnn.agg, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, batch):
        """"""
        x, edge_index = batch.x, batch.edge_index
        x = torch.matmul(x, self.weight)

        batch.x = self.propagate(edge_index, x=x)

        return batch

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

# ******************************************************************** #
# *************************** HeteroLayers *************************** #
# ******************************************************************** #

# *************************** HANConv *************************** #

@register_layer('hanconv')
class HANConv(MessagePassing):
    r"""
    The Heterogenous Graph Attention Operator from the
    `"Heterogenous Graph Attention Network"
    <https://arxiv.org/pdf/1903.07293.pdf>`_ paper.

    .. note::

        For an example of using HANConv, see `examples/hetero/han_imdb.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        hetero/han_imdb.py>`_.

    Args:
        in_channels (int or Dict[str, int]): Size of each input sample of every
            node type, or :obj:`-1` to derive the size from the first input(s)
            to the forward method.
        out_channels (int): Size of each output sample.
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
            of the heterogeneous graph, *i.e.* its node and edge types given
            by a list of strings and a list of string triplets, respectively.
            See :meth:`torch_geometric.data.HeteroData.metadata` for more
            information.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        negative_slope=0.2,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.metadata = metadata
        self.dropout = dropout
        self.k_lin = nn.Linear(out_channels, out_channels)
        self.q = nn.Parameter(torch.Tensor(1, out_channels))

        self.proj = nn.ModuleDict()
        for node_type, in_channels in self.in_channels.items():
            self.proj[node_type] = Linear(in_channels=in_channels, out_channels=out_channels)

        self.lin_src = nn.ParameterDict()
        self.lin_dst = nn.ParameterDict()
        dim = out_channels // heads
        for edge_type in metadata[1]:
            edge_type = '__'.join(edge_type)
            self.lin_src[edge_type] = nn.Parameter(torch.Tensor(1, heads, dim))
            self.lin_dst[edge_type] = nn.Parameter(torch.Tensor(1, heads, dim))

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.proj)
        glorot(self.lin_src)
        glorot(self.lin_dst)
        self.k_lin.reset_parameters()
        glorot(self.q)

    def forward(self, batch)-> Dict[NodeType, Optional[Tensor]]:
                #x_dict: Dict[NodeType, Tensor],
                #edge_index_dict: Dict[EdgeType,Adj]) \

        r"""
        Args:
            x_dict (Dict[str, Tensor]): A dictionary holding input node
                features  for each individual node type.
            edge_index_dict (Dict[str, Union[Tensor, SparseTensor]]): A
                dictionary holding graph connectivity information for each
                individual edge type, either as a :obj:`torch.LongTensor` of
                shape :obj:`[2, num_edges]` or a
                :obj:`torch_sparse.SparseTensor`.

        :rtype: :obj:`Dict[str, Optional[Tensor]]` - The output node embeddings
            for each node type.
            In case a node type does not receive any message, its output will
            be set to :obj:`None`.
        """
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict

        H, D = self.heads, self.out_channels // self.heads
        x_node_dict, out_dict = {}, {}

        # Iterate over node types:
        for node_type, x in x_dict.items():
            x_node_dict[node_type] = self.proj[node_type](x).view(-1, H, D)
            out_dict[node_type] = []

        # Iterate over edge types:
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)
            lin_src = self.lin_src[edge_type]
            lin_dst = self.lin_dst[edge_type]
            x_src = x_node_dict[src_type]
            x_dst = x_node_dict[dst_type]
            alpha_src = (x_src * lin_src).sum(dim=-1)
            alpha_dst = (x_dst * lin_dst).sum(dim=-1)
            # propagate_type: (x_dst: PairTensor, alpha: PairTensor)
            #print(edge_index)
            out = self.propagate(edge_index, x=(x_src, x_dst),
                                 alpha=(alpha_src, alpha_dst), size=None)
            # print(out)
            out = F.relu(out)
            out_dict[dst_type].append(out)

        # iterate over node types:
        for node_type, outs in out_dict.items():
            out = self.group(outs, self.q, self.k_lin)

            if out is None:
                out_dict[node_type] = None
                continue
            out_dict[node_type] = out

        batch.x_dict = out_dict
        return batch

    @staticmethod
    def group(xs: List[Tensor], q: nn.Parameter,
              k_lin: nn.Module) -> Optional[Tensor]:
        if len(xs) == 0:
            return None
        else:
            num_edge_types = len(xs)
            out = torch.stack(xs)
            if out.numel() == 0:
                return out.view(0, out.size(-1))
            attn_score = (q * torch.tanh(k_lin(out)).mean(1)).sum(-1)
            attn = F.softmax(attn_score, dim=0)
            out = torch.sum(attn.view(num_edge_types, 1, -1) * out, dim=0)
            return out

    def message(self, x_j: Tensor, alpha_i: Tensor, alpha_j: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:

        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = x_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.out_channels}, '
                f'heads={self.heads})')

@register_layer('hgtconv')
class HGTConv(MessagePassing):
    r"""The Heterogeneous Graph Transformer (HGT) operator from the
    `"Heterogeneous Graph Transformer" <https://arxiv.org/abs/2003.01332>`_
    paper.

    .. note::

        For an example of using HGT, see `examples/hetero/hgt_dblp.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        hetero/hgt_dblp.py>`_.

    Args:
        in_channels (int or Dict[str, int]): Size of each input sample of every
            node type, or :obj:`-1` to derive the size from the first input(s)
            to the forward method.
        out_channels (int): Size of each output sample.
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
            of the heterogeneous graph, *i.e.* its node and edge types given
            by a list of strings and a list of string triplets, respectively.
            See :meth:`torch_geometric.data.HeteroData.metadata` for more
            information.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        group (string, optional): The aggregation scheme to use for grouping
            node embeddings generated by different relations.
            (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`).
            (default: :obj:`"sum"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        group_agg: str = "sum",
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.group_agg = group_agg

        self.k_lin = torch.nn.ModuleDict()
        self.q_lin = torch.nn.ModuleDict()
        self.v_lin = torch.nn.ModuleDict()
        self.a_lin = torch.nn.ModuleDict()
        self.skip = torch.nn.ParameterDict()
        for node_type, in_channels in self.in_channels.items():
            self.k_lin[node_type] = Linear(in_channels, out_channels)
            self.q_lin[node_type] = Linear(in_channels, out_channels)
            self.v_lin[node_type] = Linear(in_channels, out_channels)
            self.a_lin[node_type] = Linear(out_channels, out_channels)
            self.skip[node_type] = Parameter(torch.Tensor(1))

        self.a_rel = torch.nn.ParameterDict()
        self.m_rel = torch.nn.ParameterDict()
        self.p_rel = torch.nn.ParameterDict()
        dim = out_channels // heads
        for edge_type in metadata[1]:
            edge_type = '__'.join(edge_type)
            self.a_rel[edge_type] = Parameter(torch.Tensor(heads, dim, dim))
            self.m_rel[edge_type] = Parameter(torch.Tensor(heads, dim, dim))
            self.p_rel[edge_type] = Parameter(torch.Tensor(heads))

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.k_lin)
        reset(self.q_lin)
        reset(self.v_lin)
        reset(self.a_lin)
        ones(self.skip)
        ones(self.p_rel)
        glorot(self.a_rel)
        glorot(self.m_rel)

    def forward(self,
                batch: Batch,)-> Dict[NodeType, Optional[Tensor]]:
        #x_dict: Dict[NodeType, Tensor],
        #edge_index_dict: Union[Dict[EdgeType, Tensor],Dict[EdgeType, SparseTensor]]  # Support both.

        r"""
        Args:
            x_dict (Dict[str, Tensor]): A dictionary holding input node
                features  for each individual node type.
            edge_index_dict (Dict[str, Union[Tensor, SparseTensor]]): A
                dictionary holding graph connectivity information for each
                individual edge type, either as a :obj:`torch.LongTensor` of
                shape :obj:`[2, num_edges]` or a
                :obj:`torch_sparse.SparseTensor`.

        :rtype: :obj:`Dict[str, Optional[Tensor]]` - The output node embeddings
            for each node type.
            In case a node type does not receive any message, its output will
            be set to :obj:`None`.
        """
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict

        H, D = self.heads, self.out_channels // self.heads

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        # Iterate over node-types:
        for node_type, x in x_dict.items():
            k_dict[node_type] = self.k_lin[node_type](x).view(-1, H, D)
            q_dict[node_type] = self.q_lin[node_type](x).view(-1, H, D)
            v_dict[node_type] = self.v_lin[node_type](x).view(-1, H, D)
            out_dict[node_type] = []

        # Iterate over edge-types:
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)

            a_rel = self.a_rel[edge_type]
            k = (k_dict[src_type].transpose(0, 1) @ a_rel).transpose(1, 0)

            m_rel = self.m_rel[edge_type]
            v = (v_dict[src_type].transpose(0, 1) @ m_rel).transpose(1, 0)

            # propagate_type: (k: Tensor, q: Tensor, v: Tensor, rel: Tensor)
            out = self.propagate(edge_index, k=k, q=q_dict[dst_type], v=v,
                                 rel=self.p_rel[edge_type], size=None)
            out_dict[dst_type].append(out)

        # Iterate over node-types:
        for node_type, outs in out_dict.items():
            out = self.group(outs, self.group_agg)

            if out is None:
                out_dict[node_type] = None
                continue

            out = self.a_lin[node_type](F.gelu(out))
            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = out

        batch.x_dict = out_dict
        return batch

    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, rel: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:

        alpha = (q_i * k_j).sum(dim=-1) * rel
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(-1, {self.out_channels}, '
                f'heads={self.heads})')

    def group(self, xs: List[Tensor], aggr: Optional[str]) -> Optional[Tensor]:
        if len(xs) == 0:
            return None
        elif aggr is None:
            return torch.stack(xs, dim=1)
        elif len(xs) == 1:
            return xs[0]
        else:
            out = torch.stack(xs, dim=0)
            out = getattr(torch, aggr)(out, dim=0)
            out = out[0] if isinstance(out, tuple) else out
            return out

@register_layer('heatconv')
class HEATConv(MessagePassing): # TODO: NOT INCORPORATED INTO THE MODEL YET
    r"""The heterogeneous edge-enhanced graph attentional operator from the
    `"Heterogeneous Edge-Enhanced Graph Attention Network For Multi-Agent
    Trajectory Prediction" <https://arxiv.org/abs/2106.07161>`_ paper, which
    enhances :class:`~torch_geometric.nn.conv.GATConv` by:

    1. type-specific transformations of nodes of different types
    2. edge type and edge feature incorporation, in which edges are assumed to
       have different types but contain the same kind of attributes

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        num_node_types (int): The number of node types.
        num_edge_types (int): The number of edge types.
        edge_type_emb_dim (int): The embedding size of edge types.
        edge_dim (int): Edge feature dimensionality.
        edge_attr_emb_dim (int): The embedding size of edge features.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          node types :math:`(|\mathcal{V}|)`,
          edge types :math:`(|\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_node_types: int,
                 num_edge_types: int,
                 edge_type_emb_dim: int,  # hyperparameter in config file
                 edge_dim: int,  # hyperparameter in config file
                 edge_attr_emb_dim: int,  #
                 heads: int = 1,
                 concat: bool = True,
                 negative_slope: float = 0.2,
                 dropout: float = 0.0,
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.root_weight = root_weight

        self.hetero_lin = HeteroLinear(in_channels, out_channels,
                                       num_node_types, bias=bias)

        self.edge_type_emb = Embedding(num_edge_types, edge_type_emb_dim)
        self.edge_attr_emb = Linear(edge_dim, edge_attr_emb_dim, bias=False)

        self.att = Linear(
            2 * out_channels + edge_type_emb_dim + edge_attr_emb_dim,
            self.heads, bias=False)

        self.lin = Linear(out_channels + edge_attr_emb_dim, out_channels,
                          bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.hetero_lin.reset_parameters()
        self.edge_type_emb.reset_parameters()
        self.edge_attr_emb.reset_parameters()
        self.att.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, node_type: Tensor,
                edge_type: Tensor, edge_attr: OptTensor = None) -> Tensor:
        """"""
        x = self.hetero_lin(x, node_type)

        edge_type_emb = F.leaky_relu(self.edge_type_emb(edge_type),
                                     self.negative_slope)

        # propagate_type: (x: Tensor, edge_type_emb: Tensor, edge_attr: OptTensor)  # noqa
        out = self.propagate(edge_index, x=x, edge_type_emb=edge_type_emb,
                             edge_attr=edge_attr, size=None)

        if self.concat:
            if self.root_weight:
                out = out + x.view(-1, 1, self.out_channels)
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
            if self.root_weight:
                out = out + x

        #return out
        raise NotImplementedError

    def message(self, x_i: Tensor, x_j: Tensor, edge_type_emb: Tensor,
                edge_attr: Tensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        edge_attr = F.leaky_relu(self.edge_attr_emb(edge_attr),
                                 self.negative_slope)

        alpha = torch.cat([x_i, x_j, edge_type_emb, edge_attr], dim=-1)
        alpha = F.leaky_relu(self.att(alpha), self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = self.lin(torch.cat([x_j, edge_attr], dim=-1)).unsqueeze(-2)
        return out * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})') # n # not implemented!

@register_layer('heteroconv')
class HeteroConv(Module):
    r"""A generic wrapper for computing graph convolution on heterogeneous
    graphs.
    This layer will pass messages from source nodes to target nodes based on
    the bipartite GNN layer given for a specific edge type.
    If multiple relations point to the same destination, their results will be
    aggregated according to :attr:`aggr`.
    In comparison to :meth:`torch_geometric.nn.to_hetero`, this layer is
    especially useful if you want to apply different message passing modules
    for different edge types.

    .. code-block:: python

        hetero_conv = HeteroConv({
            ('paper', 'cites', 'paper'): GCNConv(-1, 64),
            ('author', 'writes', 'paper'): SAGEConv((-1, -1), 64),
            ('paper', 'written_by', 'author'): GATConv((-1, -1), 64),
        }, aggr='sum')

        out_dict = hetero_conv(x_dict, edge_index_dict)

        print(list(out_dict.keys()))
        >>> ['paper', 'author']

    Args:
        convs (Dict[Tuple[str, str, str], Module]): A dictionary
            holding a bipartite
            :class:`~torch_geometric.nn.conv.MessagePassing` layer for each
            individual edge type.
        aggr (string, optional): The aggregation scheme to use for grouping
            node embeddings generated by different relations.
            (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`None`). (default: :obj:`"sum"`)
    """
    def __init__(self, convs: Dict[EdgeType, Module],
                 aggr: Optional[str] = "sum"):
        super().__init__()

        for edge_type, module in convs.items():
            check_add_self_loops(module, [edge_type])

        src_node_types = set([key[0] for key in convs.keys()])
        dst_node_types = set([key[-1] for key in convs.keys()])
        if len(src_node_types - dst_node_types) > 0:
            warnings.warn(
                f"There exist node types ({src_node_types - dst_node_types}) "
                f"whose representations do not get updated during message "
                f"passing as they do not occur as destination type in any "
                f"edge type. This may lead to unexpected behaviour.")

        self.convs = ModuleDict({'__'.join(k): v for k, v in convs.items()})
        self.aggr = aggr

    def reset_parameters(self):
        for conv in self.convs.values():
            conv.reset_parameters()

    def forward(self,
                batch: Batch,
                *args_dict,
                **kwargs_dict,
                ) -> Dict[NodeType, Tensor]:

        # x_dict: Dict[NodeType, Tensor],
        # edge_index_dict: Dict[EdgeType, Adj],

        r"""
        Args:
            x_dict (Dict[str, Tensor]): A dictionary holding node feature
                information for each individual node type.
            edge_index_dict (Dict[Tuple[str, str, str], Tensor]): A dictionary
                holding graph connectivity information for each individual
                edge type.
            *args_dict (optional): Additional forward arguments of invididual
                :class:`torch_geometric.nn.conv.MessagePassing` layers.
            **kwargs_dict (optional): Additional forward arguments of
                individual :class:`torch_geometric.nn.conv.MessagePassing`
                layers.
                For example, if a specific GNN layer at edge type
                :obj:`edge_type` expects edge attributes :obj:`edge_attr` as a
                forward argument, then you can pass them to
                :meth:`~torch_geometric.nn.conv.HeteroConv.forward` via
                :obj:`edge_attr_dict = { edge_type: edge_attr }`.
        """

        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict

        out_dict = defaultdict(list)
        for edge_type, edge_index in edge_index_dict.items():
            src, rel, dst = edge_type

            str_edge_type = '__'.join(edge_type)
            if str_edge_type not in self.convs:
                continue

            args = []
            for value_dict in args_dict:
                if edge_type in value_dict:
                    args.append(value_dict[edge_type])
                elif src == dst and src in value_dict:
                    args.append(value_dict[src])
                elif src in value_dict or dst in value_dict:
                    args.append(
                        (value_dict.get(src, None), value_dict.get(dst, None)))

            kwargs = {}
            for arg, value_dict in kwargs_dict.items():
                arg = arg[:-5]  # `{*}_dict`
                if edge_type in value_dict:
                    kwargs[arg] = value_dict[edge_type]
                elif src == dst and src in value_dict:
                    kwargs[arg] = value_dict[src]
                elif src in value_dict or dst in value_dict:
                    kwargs[arg] = (value_dict.get(src, None),
                                   value_dict.get(dst, None))

            conv = self.convs[str_edge_type]

            # BIG TODO: I might be misunderstanding, but why is the src==dst condition here? Why not just pass the src node features?
            # BIG TODO: Does this have to do with directed vs undirected graphs?

            local_batch = Batch()
            # local_batch.x = x_dict[src]
            # local_batch.edge_index = edge_index
            # out = conv(batch=local_batch, *args, **kwargs)

            if src == dst:
                local_batch.x = x_dict[src]
                local_batch.edge_index = edge_index
                out = conv(local_batch, *args, **kwargs)
                # out = conv(x=x_dict[src], edge_index=edge_index, *args,**kwargs)
            else:
                # local_batch.x = [x_dict[src], x_dict[dst]]
                # local_batch.x = torch.cat([x_dict[src], x_dict[dst]], dim=0)
                local_batch.x = (x_dict[src], x_dict[dst])
                local_batch.edge_index = edge_index
                out = conv(batch=local_batch, *args, **kwargs)
                # print('hey')
                # out = conv(x=(x_dict[src], x_dict[dst]), edge_index=edge_index, *args,**kwargs)
            out_dict[dst].append(out.x)

        for key, value in out_dict.items():
            out_dict[key] = group(value, self.aggr)

        batch.x_dict = out_dict
        return batch


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_relations={len(self.convs)})'
