"""
Transform a plain graph to a hypergraph, here focus lies on simplicial complices and cell complices.

This file was originally taken from the cwn github repository https://github.com/twitter-research/cwn.
For this project, its contents were modified and additional comments were added.
"""
from builtins import str, int

import graph_tool as gt
import graph_tool.topology as top
import numpy as np
import torch
import gudhi as gd
import itertools
import networkx as nx
from tqdm.auto import tqdm
from joblib import Parallel

from tqdm import tqdm
from torch_sparse import SparseTensor
from torch import nn

from torch_geometric.data import Data
from torch_geometric.data.custom_complex import Cochain, Complex
from torch_geometric.transforms.lifts.lift_transform import LiftTransform
from typing import List, Dict, Optional, Union, Tuple
from torch import Tensor
from torch_geometric.typing import Adj
from torch_scatter import scatter
from joblib import delayed


class LiftGraphToSimplicialComplex(LiftTransform):
    """Class for lifting transformation from plain graph to simplicial complex."""

    # TODO: depreciate this -> it is covered by LiftGraphToCellComplex and selecting clque_complex

    def __init__(self, lift_method: str = "inclusion", init_method: Optional[str] = "sum",
                 max_clique_dim: Optional[int] = 3, include_adj: dict = None,
                 skeleton_preserving: Optional[bool] = True,
                 ):
        self.lift_method = lift_method
        self.max_clique_dim = max_clique_dim
        self.include_adj = include_adj
        self.skeleton_preserving = skeleton_preserving
        self.init_method = init_method
        self.boundary_adjacency_tensors = None
        super().__init__(self.lift_method, self.init_method, self.boundary_adjacency_tensors)

        # Depreciate as soon as we only extract boundary (and coboundary) ajdacencies.
        if self.include_adj is None:
            self.include_adj = {"lower": False}

    def __call__(self, data: Data):
        if self.lift_method == "inclusion":
            self.boundary_adjacency_tensors, lifted_data = compute_clique_complex_with_gudhi(
                x=data.x,
                y=data.y,
                edge_attr=data.edge_attr,
                edge_index=data.edge_index,
                size=data.num_nodes,
                include_down_adj=self.include_adj["lower"],
                init_method="sum")
            return lifted_data

        elif self.lift_method == "clique":
            self.boundary_adjacency_tensors, lifted_data = compute_clique_complex_with_gudhi(x=data.x,
                                                                                             y=data.y,
                                                                                             edge_attr=data.edge_attr,
                                                                                             edge_index=data.edge_index,
                                                                                             expansion_dim=self.max_clique_dim,
                                                                                             size=data.num_nodes,
                                                                                             include_down_adj=
                                                                                             self.include_adj["lower"],
                                                                                             init_method="sum")
            return lifted_data

        else:
            raise Exception(f'Lift method not implemented. Please choose one of: "inclusion", "clique_complex".')

    def project_to_skeleton(self, dim: int):  # maybe move to the base class
        raise NotImplementedError


class LiftGraphToCellComplex(LiftTransform):
    """Class for lifting transformation from plain graph to cell complex of dimension 2."""

    # TODO: Potentially extend this to higher dimensions.

    def __init__(self, lift_method: str = "inclusion", max_clique_dim = 3 , max_simple_cycle_length: Optional[int] = 3,
                 max_induced_cycle_length: Optional[int] = 3, include_adj=None,
                 skeleton_preserving: Optional[bool] = True, init_method: Optional[str] = 'sum',
                 init_edges: bool = False, init_rings: bool = False):
        super().__init__()
        self.lift_method = lift_method
        self.max_clique_dim = max_clique_dim
        self.max_simple_cycle_length = max_simple_cycle_length
        self.max_induced_cycle_length = max_induced_cycle_length
        self.include_adj = include_adj
        self.skeleton_preserving = skeleton_preserving
        self.init_method = init_method
        self.init_edges = init_edges
        self.init_rings = init_rings
        self.boundary_adjacency_tensors = None

        # Depreciate as soon as we only extract boundary (and coboundary) ajdacencies.
        if self.include_adj is None:
            self.include_adj = {"lower": False}

    def __call__(self, data: Data):
        if self.lift_method == "inclusion":
            self.boundary_adjacency_tensors, lifted_data = compute_clique_complex_with_gudhi(data = data,
                                                                                             x=data.x,
                                                                                             y=data.y,
                                                                                             edge_attr=data.edge_attr,
                                                                                             edge_index=data.edge_index,
                                                                                             size=data.num_nodes,
                                                                                             include_down_adj=
                                                                                             self.include_adj["lower"],
                                                                                             init_method=self.init_method)
            return lifted_data

        elif self.lift_method == "clique":
            self.boundary_adjacency_tensors, lifted_data = compute_clique_complex_with_gudhi(data = data,
                                                                                             x=data.x,
                                                                                             y=data.y,
                                                                                             edge_attr=data.edge_attr,
                                                                                             edge_index=data.edge_index,
                                                                                             expansion_dim=self.max_clique_dim,
                                                                                             size=data.num_nodes,
                                                                                             include_down_adj=
                                                                                             self.include_adj["lower"],
                                                                                             init_method="sum")
            return lifted_data

        elif self.lift_method == "rings":
            self.boundary_adjacency_tensors, lifted_data = compute_ring_2complex(data=data,
                                                                                 x=data.x,
                                                                                 y=data.y,
                                                                                 edge_index=data.edge_index,
                                                                                 edge_attr=data.edge_attr,
                                                                                 size=data.num_nodes,
                                                                                 max_simple_cycle_length=self.max_simple_cycle_length,
                                                                                 max_induced_cycle_length=self.max_induced_cycle_length,
                                                                                 include_down_adj=self.include_adj["lower"],
                                                                                 init_method=self.init_method,
                                                                                 init_edges=self.init_edges,
                                                                                 init_rings=self.init_rings)
            return lifted_data

        else:
            return Exception(f'Lift method not implemented. Please choose one of: "inclusion", "clique", "rings".')


# **********************************************************************************************************************
# Functions used by both classes
# **********************************************************************************************************************


def pyg_to_simplex_tree(edge_index: Tensor, size: int):
    """Constructs a simplex tree from a PyG graph, see https://gudhi.inria.fr/python/latest/simplex_tree_ref.html.
    A simplex tree is a data structure for storing simplicial complices.

    This only changes the data structure to SimplexTree, without adding hyperedges.

    Args:
        edge_index: The edge_index of the graph (a tensor of shape [2, num_edges])
        size: The number of nodes in the graph.


    """
    st = gd.SimplexTree()
    # Add vertices to the simplex.
    for v in range(size):
        st.insert([v])

    # Add the edges to the simplex.
    edges = edge_index.numpy()
    for e in range(edges.shape[1]):
        edge = [edges[0][e], edges[1][e]]
        st.insert(edge)

    return st


def get_simplex_boundaries(simplex):
    boundaries = itertools.combinations(simplex, len(simplex) - 1)
    return [tuple(boundary) for boundary in boundaries]


def build_tables(simplex_tree, size):
    complex_dim = simplex_tree.dimension()
    # Each of these data structures has a separate entry per dimension.
    id_maps = [{} for _ in range(complex_dim + 1)]  # simplex -> id
    simplex_tables = [[] for _ in range(complex_dim + 1)]  # matrix of simplices
    boundaries_tables = [[] for _ in range(complex_dim + 1)]

    simplex_tables[0] = [[v] for v in range(size)]
    id_maps[0] = {tuple([v]): v for v in range(size)}

    # next_id = size

    for simplex, _ in simplex_tree.get_simplices():
        dim = len(simplex) - 1
        if dim == 0:
            continue

        # Assign this simplex the next unused ID: originally, IDs are repeated for each dimension, now we want to use unique IDs -> changed back to original
        # id_maps[dim][tuple(simplex)] = next_id
        # next_id += 1

        next_id = len(simplex_tables[dim])
        id_maps[dim][tuple(simplex)] = next_id

        simplex_tables[dim].append(simplex)

    return simplex_tables, id_maps


def extract_boundaries_and_coboundaries_from_simplex_tree(simplex_tree, id_maps, complex_dim: int,
                                                          return_HoEs: Optional[bool] = False):
    """Build two maps simplex -> its coboundaries and simplex -> its boundaries"""
    # The extra dimension is added just for convenience to avoid treating it as a special case.
    boundaries = [{} for _ in range(complex_dim + 2)]  # simplex -> boundaries
    coboundaries = [{} for _ in range(complex_dim + 2)]  # simplex -> coboundaries
    boundaries_tables = [[] for _ in range(complex_dim + 1)]

    boundary_adjacency_tensors = []  # list of tensors which map the ids of simplices of dimension k to their bondary adjacencies.

    simplex_ids = [[] for _ in range(complex_dim + 1)]
    boundary_ids = [[] for _ in range(complex_dim + 1)]

    for simplex, _ in simplex_tree.get_simplices():
        # Extract the relevant boundary and coboundary maps
        simplex_dim = len(simplex) - 1
        simplex_id = id_maps[simplex_dim][tuple(simplex)]

        level_coboundaries = coboundaries[simplex_dim]
        level_boundaries = boundaries[simplex_dim + 1]

        # Add the boundaries of the simplex to the boundaries table
        if simplex_dim > 0:
            boundaries_ids = [id_maps[simplex_dim - 1][boundary] for boundary in get_simplex_boundaries(simplex)]
            boundary_ids[simplex_dim] += boundaries_ids
            simplex_ids[simplex_dim] += [simplex_id for _ in range(len(boundaries_ids))]

            boundaries_tables[simplex_dim].append(boundaries_ids)

        # This operation should be roughly be O(dim_complex), so that is very efficient for us.
        # For details see pages 6-7 https://hal.inria.fr/hal-00707901v1/document
        simplex_coboundaries = simplex_tree.get_cofaces(simplex, codimension=1)
        for coboundary, _ in simplex_coboundaries:
            assert len(coboundary) == len(simplex) + 1

            if tuple(simplex) not in level_coboundaries:
                level_coboundaries[tuple(simplex)] = list()
            level_coboundaries[tuple(simplex)].append(tuple(coboundary))

            if tuple(coboundary) not in level_boundaries:
                level_boundaries[tuple(coboundary)] = list()
            level_boundaries[tuple(coboundary)].append(tuple(simplex))

    # collect boundaries into one tensor
    for dim in range(1, complex_dim + 1):
        boundary_adjacency_tensor = SparseTensor(row=torch.tensor(boundary_ids[dim], dtype=torch.long),
                                                 col=torch.tensor(simplex_ids[dim], dtype=torch.long))
        boundary_adjacency_tensors.append(boundary_adjacency_tensor)

    return boundary_adjacency_tensors, boundaries_tables, boundaries, coboundaries


def build_adj(boundaries: List[Dict], coboundaries: List[Dict], id_maps: List[Dict], complex_dim: int,
              include_down_adj: bool):
    """Builds the upper and lower adjacency data structures of the complex

    Args:
        boundaries: A list of dictionaries of the form
            boundaries[dim][simplex] -> List[simplex] (the boundaries)
        coboundaries: A list of dictionaries of the form
            coboundaries[dim][simplex] -> List[simplex] (the coboundaries)
        id_maps: A dictionary from simplex -> simplex_id
    """

    def initialise_structure():
        return [[] for _ in range(complex_dim + 1)]

    upper_indexes, lower_indexes = initialise_structure(), initialise_structure()
    all_shared_boundaries, all_shared_coboundaries = initialise_structure(), initialise_structure()

    # Go through all dimensions of the complex
    for dim in range(complex_dim + 1):
        # Go through all the simplices at that dimension
        for simplex, id in id_maps[dim].items():
            # Add the upper adjacent neighbours from the level below
            if dim > 0:
                for boundary1, boundary2 in itertools.combinations(boundaries[dim][simplex], 2):
                    id1, id2 = id_maps[dim - 1][boundary1], id_maps[dim - 1][boundary2]
                    upper_indexes[dim - 1].extend([[id1, id2], [id2, id1]])
                    all_shared_coboundaries[dim - 1].extend([id, id])

            # Add the lower adjacent neighbours from the level above
            if include_down_adj and dim < complex_dim and simplex in coboundaries[dim]:
                for coboundary1, coboundary2 in itertools.combinations(coboundaries[dim][simplex], 2):
                    id1, id2 = id_maps[dim + 1][coboundary1], id_maps[dim + 1][coboundary2]
                    lower_indexes[dim + 1].extend([[id1, id2], [id2, id1]])
                    all_shared_boundaries[dim + 1].extend([id, id])

    return all_shared_boundaries, all_shared_coboundaries, lower_indexes, upper_indexes


def construct_features(vx: Tensor, cell_tables, init_method: str, edge_attr: Optional[Tensor] = None, ) -> List:
    """Combines the features of the component vertices to initialise the cell features"""
    features = [vx]
    for dim in range(1, len(cell_tables)):
        aux_1 = []
        aux_0 = []
        for c, cell in enumerate(cell_tables[dim]):
            aux_1 += [c for _ in range(len(cell))]
            aux_0 += cell
        node_cell_index = torch.LongTensor([aux_0, aux_1])
        in_features = vx.index_select(0, node_cell_index[0])
        features.append(scatter(in_features, node_cell_index[1], dim=0,
                                dim_size=len(cell_tables[dim]), reduce=init_method))

    return features


def extract_labels(y, size):
    v_y, complex_y = None, None
    if y is None:
        return v_y, complex_y

    y_shape = list(y.size())

    if y_shape[0] == 1:
        # This is a label for the whole graph (for graph classification).
        # We will use it for the complex.
        complex_y = y
    else:
        # This is a label for the vertices of the complex.
        assert y_shape[0] == size
        v_y = y

    return v_y, complex_y


def generate_cochain(dim, x, all_upper_index, all_lower_index,
                     all_shared_boundaries, all_shared_coboundaries, cell_tables, boundaries_tables,
                     complex_dim, y=None):
    """Builds a Cochain given all the adjacency data extracted from the complex."""
    if dim == 0:
        assert len(all_lower_index[dim]) == 0
        assert len(all_shared_boundaries[dim]) == 0

    num_cells_down = len(cell_tables[dim - 1]) if dim > 0 else None
    num_cells_up = len(cell_tables[dim + 1]) if dim < complex_dim else 0

    up_index = (torch.tensor(all_upper_index[dim], dtype=torch.long).t()
                if len(all_upper_index[dim]) > 0 else None)
    down_index = (torch.tensor(all_lower_index[dim], dtype=torch.long).t()
                  if len(all_lower_index[dim]) > 0 else None)
    shared_coboundaries = (torch.tensor(all_shared_coboundaries[dim], dtype=torch.long)
                           if len(all_shared_coboundaries[dim]) > 0 else None)
    shared_boundaries = (torch.tensor(all_shared_boundaries[dim], dtype=torch.long)
                         if len(all_shared_boundaries[dim]) > 0 else None)

    boundary_index = None
    if len(boundaries_tables[dim]) > 0:
        boundary_index = [list(), list()]
        for s, cell in enumerate(boundaries_tables[dim]):
            for boundary in cell:
                boundary_index[1].append(s)
                boundary_index[0].append(boundary)
        boundary_index = torch.LongTensor(boundary_index)

    if num_cells_down is None:
        assert shared_boundaries is None
    if num_cells_up == 0:
        assert shared_coboundaries is None

    if up_index is not None:
        assert up_index.size(1) == shared_coboundaries.size(0)
    if down_index is not None:
        assert down_index.size(1) == shared_boundaries.size(0)

    return Cochain(dim=dim, x=x, upper_index=up_index,
                   lower_index=down_index, shared_coboundaries=shared_coboundaries,
                   shared_boundaries=shared_boundaries, y=y, num_cells_down=num_cells_down,
                   num_cells_up=num_cells_up, boundary_index=boundary_index)


def compute_clique_complex_with_gudhi(data: Data, x: Tensor, edge_index: Adj, size: int,
                                      expansion_dim: int = None, y: Tensor = None,
                                      edge_attr=None,
                                      include_down_adj=True,
                                      init_method: str = 'sum') -> Union[Complex, Tuple[List, Complex]]:
    """Generates a clique complex of a pyG graph via gudhi.

    Args:
        x: The feature matrix for the nodes of the graph
        edge_index: The edge_index of the graph (a tensor of shape [2, num_edges])
        size: The number of nodes in the graph
        expansion_dim: The dimension to expand the simplex to.
        y: Labels for the graph nodes or a label for the whole graph.
        include_down_adj: Whether to add down adj in the complex or not
        init_method: How to initialise features at higher levels.
    """
    assert x is not None
    assert isinstance(edge_index, Tensor)  # Support only tensor edge_index for now

    # Creates the gudhi-based simplicial complex
    simplex_tree = pyg_to_simplex_tree(edge_index, size)
    if expansion_dim is not None:
        simplex_tree.expansion(expansion_dim)  # Computes the clique complex up to the desired dim.
    complex_dim = simplex_tree.dimension()  # See what is the dimension of the complex now.

    # Builds tables of the simplicial complexes at each level and their IDs
    simplex_tables, id_maps = build_tables(simplex_tree, size)

    # Extracts the boundaries and coboundaries of each simplex in the complex
    boundary_adjacency_tensors, boundaries_tables, boundaries, co_boundaries = \
        extract_boundaries_and_coboundaries_from_simplex_tree(simplex_tree, id_maps, complex_dim, return_HoEs=True)

    # Computes the adjacencies between all the simplexes in the complex
    shared_boundaries, shared_coboundaries, lower_idx, upper_idx = build_adj(boundaries, co_boundaries, id_maps,
                                                                             complex_dim, include_down_adj)

    # Construct features for the higher dimensions
    # TODO: Make this handle edge features as well and add alternative options to compute this.
    xs = construct_features(x, simplex_tables, init_method, edge_attr=edge_attr)

    # Initialise the node / complex labels
    v_y, complex_y = extract_labels(y, size)

    cochains = []
    for i in range(complex_dim + 1):
        y = v_y if i == 0 else None
        cochain = generate_cochain(i, xs[i], upper_idx, lower_idx, shared_boundaries, shared_coboundaries,
                                   simplex_tables, boundaries_tables, complex_dim=complex_dim, y=y)
        cochains.append(cochain)

    complex = Complex(*cochains, y=complex_y, dimension=complex_dim)
    for key in data.keys:
        complex.keys.append(key)
        setattr(complex, key, data[key])

    return boundary_adjacency_tensors, complex


def convert_graph_dataset_with_gudhi(dataset, expansion_dim: int, include_down_adj=True,
                                     init_method: str = 'sum'):
    """Converts a pyG dataset to a clique complex dataset via gudhi."""
    # TODO(Cris): Add parallelism to this code like in the cell complex conversion code.
    dimension = -1
    complexes = []
    num_features = [None for _ in range(expansion_dim + 1)]

    for data in tqdm(dataset):
        complex = compute_clique_complex_with_gudhi(data.x, data.edge_index, data.num_nodes,
                                                    expansion_dim=expansion_dim, y=data.y,
                                                    include_down_adj=include_down_adj,
                                                    init_method=init_method)
        if complex.dimension > dimension:
            dimension = complex.dimension
        for dim in range(complex.dimension + 1):
            if num_features[dim] is None:
                num_features[dim] = complex.cochains[dim].num_features
            else:
                assert num_features[dim] == complex.cochains[dim].num_features
        complexes.append(complex)

    return complexes, dimension, num_features[:dimension + 1]


# ---- support for rings as cells

def get_simple_and_induced_cycles(edge_index,
                                  max_simple_cycle_length: int = 3,
                                  max_induced_cycle_length: Optional[int] = 3):
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()

    edge_list = edge_index.T
    graph_gt = gt.Graph(directed=False)
    graph_gt.add_edge_list(edge_list)
    gt.stats.remove_self_loops(graph_gt)
    gt.stats.remove_parallel_edges(graph_gt)
    # We represent rings with their original node ordering
    # so that we can easily read out the boundaries
    # The use of the `sorted_rings` set allows to discard
    # different isomorphisms which are however associated
    # to the same original ring – this happens due to the intrinsic
    # symmetries of cycles

    if max_simple_cycle_length >= max_induced_cycle_length:
        UserWarning(
            "Every induced cycle is a simple cycle, so setting max_induced_cycle_length <= max_simple_cycle_length is redundant.")

    simple_cycles = set()
    sorted_simple_cycles = set()
    induced_cycles = set()
    sorted_induced_cycles = set()

    # first we get all the simple cycles
    for k in range(3, min(max_induced_cycle_length,
                          max_simple_cycle_length) + 1):  # TODO: when adding cliques, only consider from size >3
        pattern = nx.cycle_graph(k)
        pattern_edge_list = list(pattern.edges)
        pattern_gt = gt.Graph(directed=False)
        pattern_gt.add_edge_list(pattern_edge_list)
        sub_isos = top.subgraph_isomorphism(pattern_gt, graph_gt, induced=False, subgraph=True,
                                            generator=True)
        sub_iso_sets = map(lambda isomorphism: tuple(isomorphism.a), sub_isos)
        for iso in sub_iso_sets:
            if tuple(sorted(iso)) not in sorted_simple_cycles:
                simple_cycles.add(iso)
                sorted_simple_cycles.add(tuple(sorted(iso)))
    for k in range(min(max_induced_cycle_length, max_simple_cycle_length) + 1, max_induced_cycle_length + 1):
        pattern = nx.cycle_graph(k)
        pattern_edge_list = list(pattern.edges)
        pattern_gt = gt.Graph(directed=False)
        pattern_gt.add_edge_list(pattern_edge_list)
        sub_isos = top.subgraph_isomorphism(pattern_gt, graph_gt, induced=True, subgraph=True,
                                            generator=True)
        sub_iso_sets = map(lambda isomorphism: tuple(isomorphism.a), sub_isos)
        for iso in sub_iso_sets:
            if tuple(sorted(iso)) not in sorted_induced_cycles:
                induced_cycles.add(iso)
                sorted_induced_cycles.add(tuple(sorted(iso)))
    rings = list(simple_cycles) + list(induced_cycles)
    return rings


def build_tables_with_rings(edge_index, simplex_tree, size, max_simple_k, max_induced_k):
    # Build simplex tables and id_maps up to edges by conveniently
    # invoking the code for simplicial complexes
    cell_tables, id_maps = build_tables(simplex_tree, size)

    # Find rings in the graph
    rings = get_simple_and_induced_cycles(edge_index,
                                          max_simple_cycle_length=max_simple_k,
                                          max_induced_cycle_length=max_induced_k)

    if len(rings) > 0:
        # Extend the tables with rings as 2-cells
        # next_id = sum([len(id_maps[i]) for i in range(len(id_maps))])
        id_maps += [{}]
        cell_tables += [[]]
        assert len(cell_tables) == 3, cell_tables
        for cell in rings:

            # original
            next_id = len(cell_tables[2])

            # unique ids --> causes problems in collate ...
            # next_id = len(cell_tables[2])+max([max(list(id_maps[i].values())) for i in range(len(id_maps)-1)])+1
            id_maps[2][cell] = next_id
            next_id += 1
            cell_tables[2].append(list(cell))

    return cell_tables, id_maps


def get_ring_boundaries(ring):
    boundaries = list()
    for n in range(len(ring)):
        a = n
        if n + 1 == len(ring):
            b = 0
        else:
            b = n + 1
        # We represent the boundaries in lexicographic order
        # so to be compatible with 0- and 1- dim cells
        # extracted as simplices with gudhi
        boundaries.append(tuple(sorted([ring[a], ring[b]])))
    return sorted(boundaries)


def extract_boundaries_and_coboundaries_with_rings(simplex_tree, id_maps):
    """Build two maps: cell -> its coboundaries and cell -> its boundaries"""

    # Find boundaries and coboundaries up to edges by conveniently
    # invoking the code for simplicial complexes
    assert simplex_tree.dimension() <= 1
    boundary_adjacency_tensors, boundaries_tables, boundaries, coboundaries = extract_boundaries_and_coboundaries_from_simplex_tree(
        simplex_tree, id_maps, simplex_tree.dimension())

    assert len(id_maps) <= 3
    if len(id_maps) == 3:
        # Extend tables with boundary and coboundary information of rings
        boundaries += [{}]
        coboundaries += [{}]
        boundaries_tables += [[]]

        cell_ids = []
        boundary_ids = []

        for cell_key in id_maps[2]:
            cell_boundaries = get_ring_boundaries(cell_key)

            cell_ids += [id_maps[2][cell_key] for _ in range(len(cell_boundaries))]
            boundary_ids += [id_maps[1][cell_boundary] for cell_boundary in cell_boundaries]

            boundaries[2][cell_key] = list()
            boundaries_tables[2].append([])
            for boundary in cell_boundaries:
                assert boundary in id_maps[1], boundary
                boundaries[2][cell_key].append(boundary)
                if boundary not in coboundaries[1]:
                    coboundaries[1][boundary] = list()
                coboundaries[1][boundary].append(cell_key)
                boundaries_tables[2][-1].append(id_maps[1][boundary])

        boundary_adjacency_tensor = SparseTensor(row=torch.tensor(boundary_ids, dtype=torch.long),
                                                 col=torch.tensor(cell_ids, dtype=torch.long))
        boundary_adjacency_tensors.append(boundary_adjacency_tensor)

    return boundary_adjacency_tensors, boundaries_tables, boundaries, coboundaries


def compute_ring_2complex(data: Data,
                          x: Union[Tensor, np.ndarray], edge_index: Union[Tensor, np.ndarray],
                          edge_attr: Optional[Union[Tensor, np.ndarray]],
                          size: int, y: Optional[Union[Tensor, np.ndarray]] = None,
                          max_simple_cycle_length: Optional[int] = 3,
                          max_induced_cycle_length: Optional[int] = None,
                          include_down_adj=True,
                          init_method: str = 'sum',
                          init_edges: bool = False,
                          init_rings: bool = False,
                          return_HoE: Optional[bool] = False) -> Union[tuple[List, Complex], Complex]:
    """Generates a ring 2-complex of a pyG graph via graph-tool.

    Args:
        x: The feature matrix for the nodes of the graph (shape [num_vertices, num_v_feats])
        edge_index: The edge_index of the graph (a tensor of shape [2, num_edges])
        edge_attr: The feature matrix for the edges of the graph (shape [num_edges, num_e_feats])
        size: The number of nodes in the graph
        y: Labels for the graph nodes or a label for the whole graph.
        include_down_adj: Whether to add down adj in the complex or not
        init_method: How to initialise features at higher levels.
    """
    # global boundary_adjacency_tensors
    assert x is not None
    assert isinstance(edge_index, np.ndarray) or isinstance(edge_index, Tensor)

    # For parallel processing with joblib we need to pass numpy arrays as inputs
    # Therefore, we convert here everything back to a tensor.
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    if isinstance(edge_index, np.ndarray):
        edge_index = torch.tensor(edge_index)
    if isinstance(edge_attr, np.ndarray):
        edge_attr = torch.tensor(edge_attr)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y)

    # Creates the gudhi-based simplicial complex up to edges
    simplex_tree = pyg_to_simplex_tree(edge_index, size)
    assert simplex_tree.dimension() <= 1
    if simplex_tree.dimension() == 0:
        assert edge_index.size(1) == 0

    # Builds tables of the cellular complexes at each level and their IDs
    cell_tables, id_maps = build_tables_with_rings(edge_index, simplex_tree, size, max_simple_cycle_length,
                                                   max_induced_cycle_length)
    assert len(id_maps) <= 3
    complex_dim = len(id_maps) - 1

    # Extracts the boundaries and coboundaries of each cell in the complex
    boundary_adjacency_tensors, boundaries_tables, boundaries, co_boundaries = extract_boundaries_and_coboundaries_with_rings(
        simplex_tree, id_maps)

    # Computes the adjacencies between all the cells in the complex;
    # here we force complex dimension to be 2
    shared_boundaries, shared_coboundaries, lower_idx, upper_idx = build_adj(boundaries, co_boundaries, id_maps,
                                                                             complex_dim, include_down_adj)

    # Construct features for the higher dimensions
    xs = [x, None, None]
    constructed_features = construct_features(x, cell_tables, init_method)  # TODO: construct ring features from edge features.
    if simplex_tree.dimension() == 0:
        assert len(constructed_features) == 1
    if init_rings and len(constructed_features) > 2:
        xs[2] = constructed_features[2]

    if init_edges and simplex_tree.dimension() >= 1:
        if edge_attr is None:
            xs[1] = constructed_features[1]
        # If we have edge-features we simply use them for 1-cells
        else:
            # If edge_attr is a list of scalar features, make it a matrix
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            # Retrieve feats and check edge features are undirected
            ex = dict()
            for e, edge in enumerate(edge_index.numpy().T):
                canon_edge = tuple(sorted(edge))
                edge_id = id_maps[1][canon_edge]
                edge_feats = edge_attr[e]
                if edge_id in ex:
                    assert torch.equal(ex[edge_id], edge_feats)
                else:
                    ex[edge_id] = edge_feats

            # Build edge feature matrix
            max_id = max(ex.keys())
            # min_id = min(ex.keys())
            edge_feats = []
            assert len(cell_tables[1]) == max_id + 1
            #assert len(cell_tables[1]) == max_id-min_id + 1

            for id in range(max_id + 1):
            #for id in range(min_id, max_id + 1):
                edge_feats.append(ex[id])
            xs[1] = torch.stack(edge_feats, dim=0)
            assert xs[1].dim() == 2
            assert xs[1].size(0) == len(id_maps[1])
            assert xs[1].size(1) == edge_attr.size(1)
            if edge_attr.size(1) != x.size(1):
                xs[1] = nn.Linear(edge_attr.size(1), x.size(1))(xs[1].float())

    # Initialise the node / complex labels
    v_y, complex_y = extract_labels(y, size)

    cochains = []
    for i in range(complex_dim + 1):
        y = v_y if i == 0 else None
        cochain = generate_cochain(i, xs[i], upper_idx, lower_idx, shared_boundaries, shared_coboundaries,
                                   cell_tables, boundaries_tables, complex_dim=complex_dim, y=y)
        cochains.append(cochain)
    complex = Complex(*cochains, y=complex_y, dimension=complex_dim)
    for key in data.keys:
        complex.keys.append(key)
        setattr(complex, key, data[key])

    return boundary_adjacency_tensors, complex


def convert_graph_dataset_with_rings(dataset, max_ring_size=7, include_down_adj=False,
                                     init_method: str = 'sum', init_edges=True, init_rings=False,
                                     n_jobs=1):
    dimension = -1
    num_features = [None, None, None]

    def maybe_convert_to_numpy(x):
        if isinstance(x, Tensor):
            return x.numpy()
        return x

    # Process the dataset in parallel
    parallel = ProgressParallel(n_jobs=n_jobs, use_tqdm=True, total=len(dataset))
    # It is important we supply a numpy array here. tensors seem to slow joblib down significantly.
    complexes = parallel(delayed(compute_ring_2complex)(data,
        maybe_convert_to_numpy(data.x), maybe_convert_to_numpy(data.edge_index),
        maybe_convert_to_numpy(data.edge_attr),
        data.num_nodes, y=maybe_convert_to_numpy(data.y), max_k=max_ring_size,
        include_down_adj=include_down_adj, init_method=init_method,
        init_edges=init_edges, init_rings=init_rings) for data in dataset)

    # NB: here we perform additional checks to verify the order of complexes
    # corresponds to that of input graphs after _parallel_ conversion
    for c, complex in enumerate(complexes):

        # Handle dimension and number of features
        if complex.dimension > dimension:
            dimension = complex.dimension
        for dim in range(complex.dimension + 1):
            if num_features[dim] is None:
                num_features[dim] = complex.cochains[dim].num_features
            else:
                assert num_features[dim] == complex.cochains[dim].num_features

        # Validate against graph
        graph = dataset[c]
        if complex.y is None:
            assert graph.y is None
        else:
            assert torch.equal(complex.y, graph.y)
        assert torch.equal(complex.cochains[0].x, graph.x)
        if complex.dimension >= 1:
            assert complex.cochains[1].x.size(0) == (graph.edge_index.size(1) // 2)

    return complexes, dimension, num_features[:dimension + 1]


class ProgressParallel(Parallel):
    """A helper class for adding tqdm progressbar to the joblib library."""

    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
