from typing import List, Optional, Tuple, Any

import torch
import torch_sparse as ts
from torch_sparse import SparseTensor
import warnings

from torch_geometric.data import HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import EdgeType
from torch_geometric.utils import degree


@functional_transform('add_metapaths_hops')
class AddMetaPathsHops(BaseTransform):
    r""" Adds additional edge types to a
    :class:`~torch_geometric.data.HeteroData` object between the source node
    type and the destination node type of a given :obj:`metapath`, as described
    in the `"Heterogenous Graph Attention Networks"
    <https://arxiv.org/abs/1903.07293>`_ paper
    (functional name: :obj:`add_metapaths`).
    Meta-path based neighbors can exploit different aspects of structure
    information in heterogeneous graphs.
    Formally, a metapath is a path of the form

    .. math::

        \mathcal{V}_1 \xrightarrow{R_1} \mathcal{V}_2 \xrightarrow{R_2} \ldots
        \xrightarrow{R_{\ell-1}} \mathcal{V}_{\ell}

    in which :math:`\mathcal{V}_i` represents node types, and :math:`R_j`
    represents the edge type connecting two node types.
    The added edge type is given by the sequential multiplication  of
    adjacency matrices along the metapath, and is added to the
    :class:`~torch_geometric.data.HeteroData` object as edge type
    :obj:`(src_node_type, "metapath_*", dst_node_type)`, where
    :obj:`src_node_type` and :obj:`dst_node_type` denote :math:`\mathcal{V}_1`
    and :math:`\mathcal{V}_{\ell}`, repectively.

    In addition, a :obj:`metapath_dict` object is added to the
    :class:`~torch_geometric.data.HeteroData` object which maps the
    metapath-based edge type to its original metapath.

    .. code-block:: python

        from torch_geometric.datasets import DBLP
        from torch_geometric.data import HeteroData
        from torch_geometric.transforms import AddMetaPaths

        data = DBLP(root)[0]
        # 4 node types: "paper", "author", "conference", and "term"
        # 6 edge types: ("paper","author"), ("author", "paper"),
        #               ("paper, "term"), ("paper", "conference"),
        #               ("term, "paper"), ("conference", "paper")

        # Add two metapaths:
        # 1. From "paper" to "paper" through "conference"
        # 2. From "author" to "conference" through "paper"
        metapaths = [[("paper", "conference"), ("conference", "paper")],
                     [("author", "paper"), ("paper", "conference")]]
        data = AddMetaPaths(metapaths)(data)

        print(data.edge_types)
        >>> [("author", "to", "paper"), ("paper", "to", "author"),
             ("paper", "to", "term"), ("paper", "to", "conference"),
             ("term", "to", "paper"), ("conference", "to", "paper"),
             ("paper", "metapath_0", "paper"),
             ("author", "metapath_1", "conference")]

        print(data.metapath_dict)
        >>> {("paper", "metapath_0", "paper"): [("paper", "conference"),
                                                ("conference", "paper")],
             ("author", "metapath_1", "conference"): [("author", "paper"),
                                                      ("paper", "conference")]}

    Args:
        metapaths (List[List[Tuple[str, str, str]]]): The metapaths described
            by a list of lists of
            :obj:`(src_node_type, rel_type, dst_node_type)` tuples.
        drop_orig_edges (bool, optional): If set to :obj:`True`, existing edge
            types will be dropped. (default: :obj:`False`)
        keep_same_node_type (bool, optional): If set to :obj:`True`, existing
            edge types between the same node type are not dropped even in case
            :obj:`drop_orig_edges` is set to :obj:`True`.
            (default: :obj:`False`)
        drop_unconnected_nodes (bool, optional): If set to :obj:`True` drop
            node types not connected by any edge type. (default: :obj:`False`)
        max_sample (int, optional): If set, will sample at maximum
            :obj:`max_sample` neighbors within metapaths. Useful in order to
            tackle very dense metapath edges. (default: :obj:`None`)
        weighted (bool, optional): If set to :obj:`True` compute weights for
            each metapath edge and store them in :obj:`edge_weight`. The weight
            of each metapath edge is computed as the number of metapaths from
            the start to the end of the metapath edge.
            (default :obj:`False`)
    """

    def __init__(
            self,
            metapaths: List[List[EdgeType]],
            drop_orig_edges: bool = False,
            keep_same_node_type: bool = False,
            drop_unconnected_nodes: bool = False,
            max_sample: Optional[int] = None,
            weighted: bool = False,
    ):

        for path in metapaths:
            assert len(path) >= 2, f"Invalid metapath '{path}'"
            assert all([
                j[-1] == path[i + 1][0] for i, j in enumerate(path[:-1])
            ]), f"Invalid sequence of node types in '{path}'"

        self.metapaths = metapaths
        self.drop_orig_edges = drop_orig_edges
        self.keep_same_node_type = keep_same_node_type
        self.drop_unconnected_nodes = drop_unconnected_nodes
        self.max_sample = max_sample
        self.weighted = weighted

    def __call__(self, data: HeteroData) -> HeteroData:
        edge_types = data.edge_types  # save original edge types
        data.metapath_dict = {}  # do not seem to use this, maybe just for debugging?

        # new implementation: reverse through metapaths and track number of walks that pass through metapaths
        for j, metapath in enumerate(self.metapaths):

            # sanity check if all edge types in the metapaths are present in graph
            for edge_type in metapath:
                assert data._to_canonical(
                    edge_type) in edge_types, f"'{edge_type}' not present"

            metapath_length = len(metapath)
            node_type_abbrev = [metapath[0][0][0]]+[edge_type[2][0] for edge_type in metapath] # list of (first letters of) node-types along the metapath for easy labelling of metapath
            metapath_abbrev = ''.join(node_type_abbrev) # join abbreviations, e.g. author->paper->conference->paper becomes apcp.

            # assert that edge-types are present
            for edge_type in metapath:
                assert data._to_canonical(
                    edge_type) in edge_types, f"'{edge_type}' not present"

            # Go through edge types in the metapath backwards (from target to source), store the adjacency matrices in a list
            adjacencies_node_to_target : List[tuple[SparseTensor,int,str]] = [] # tracks adjacencies to target node following a walk along metapath, integer tracks the length of the walk

            # Start wth last edge type in metapath:
            edge_type = metapath[-1]
            edge_weight = self._get_edge_weight(data, edge_type)
            adj1 = SparseTensor.from_edge_index(
                edge_index=data[edge_type].edge_index,
                sparse_sizes=data[edge_type].size(), edge_attr=edge_weight)
            dist_adj_1 = 1
            from_node_type = edge_type[0]

            if self.max_sample is not None:
                adj1 = self.sample_adj(adj1)

            adjacencies_node_to_target.append((adj1, dist_adj_1, from_node_type))

            # Go through remaining edge types from target to source
            for i, edge_type in enumerate(reversed(metapath[:-1])):  # enumerate(metapath[1:])

                from_node_type = edge_type[0]
                edge_weight = self._get_edge_weight(data, edge_type)
                adj2 = SparseTensor.from_edge_index(
                    edge_index=data[edge_type].edge_index,
                    sparse_sizes=data[edge_type].size(), edge_attr=edge_weight)

                # update current walk matrix, corresponding to number of walks from current edge type to target following metapath.
                adj1 = adj2 @ adj1
                dist_adj_1 += 1
                # sample edges if max_sample is set
                if self.max_sample is not None:
                    adj1 = self.sample_adj(adj1)

                # add to list of intermediate walk adjacencies (after sampling)
                adjacencies_node_to_target.append((adj1,dist_adj_1, from_node_type))

            # add metapath edge from sources to targets
            row, col, edge_weight = adj1.coo()
            new_edge_type: tuple[str, str, str] = (
            metapath[0][0], f'metapath_{metapath_abbrev}_st',metapath[-1][-1]) # source_to_target
            data[new_edge_type].edge_index = torch.vstack([row, col])
            if self.weighted:
                data[new_edge_type].edge_weight = edge_weight
            data.metapath_dict[new_edge_type] = metapath

            # add metapath edges for intermediate nodes (weights correspond to the number of walks from source via intermediate node to target following a metapath walk)
            adjacencies_node_to_target.remove(adjacencies_node_to_target[-1])
            adjacency_next = None

            for (adj, m, from_node_type) in (reversed(adjacencies_node_to_target)):
                idx = metapath_length-m-1
                print(metapath[idx][0], from_node_type)

                adjacency_curr = SparseTensor.from_edge_index(edge_index=data[metapath[idx]].edge_index,
                                                         sparse_sizes=data[metapath[idx]].size(),
                                                         edge_attr=self._get_edge_weight(data, metapath[idx]))
                # row_curr, col_curr, edge_weight_curr = adjacency_curr.coo()
                if adjacency_next is None:
                    adjacency_next = SparseTensor.from_dense(torch.diag(adjacency_curr.sum(dim=0)))
                else:
                    adjacency_next = SparseTensor.from_dense(torch.diag(ts.matmul(adjacency_next, adjacency_curr).sum(dim=0)))

                intermediate_walk_adjacency = ts.matmul(adjacency_next, adj)

                row, col, edge_weight = intermediate_walk_adjacency.coo()
                new_edge_type: tuple[str, str, str] = (from_node_type, f'metapath_{metapath_abbrev}_via_{from_node_type[0]}_at_t_dist_{m}', metapath[-1][-1]) # t_dist = distance from target
                data[new_edge_type].edge_index = torch.vstack([row, col])
                if self.weighted:
                    data[new_edge_type].edge_weight = edge_weight
                data.metapath_dict[new_edge_type] = metapath

        if self.drop_orig_edges:
            for i in edge_types:
                if self.keep_same_node_type and i[0] == i[-1]:
                    continue
                else:
                    del data[i]

        # remove nodes not connected by any edge type.
        if self.drop_unconnected_nodes:
            new_edge_types = data.edge_types
            node_types = data.node_types
            connected_nodes = set()
            for i in new_edge_types:
                connected_nodes.add(i[0])
                connected_nodes.add(i[-1])
            for node in node_types:
                if node not in connected_nodes:
                    del data[node]

        return data

    def sample_adj(self, adj: SparseTensor) -> SparseTensor:
        row, col, _ = adj.coo()
        deg = degree(row, num_nodes=adj.size(0))
        prob = (self.max_sample * (1. / deg))[row]
        mask = torch.rand_like(prob) < prob
        return adj.masked_select_nnz(mask, layout='coo')

    def _get_edge_weight(self, data: HeteroData,
                         edge_type: EdgeType) -> torch.Tensor:
        if self.weighted:
            edge_weight = data[edge_type].get('edge_weight', None)
            if edge_weight is None:
                edge_weight = torch.ones(
                    data[edge_type].num_edges,
                    device=data[edge_type].edge_index.device)
            assert edge_weight.ndim == 1
        else:
            edge_weight = None
        return edge_weight
