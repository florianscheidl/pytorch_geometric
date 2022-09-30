# import random
# from typing import Callable
#
# import torch
# from sklearn.model_selection import train_test_split
#
# import torch_geometric.graphgym.register as register
# from torch_geometric.graphgym.register import register_loader
# import torch_geometric.transforms as T
# from torch_geometric.transforms.liwich_transforms import LiftAndWire
# import torch_geometric.transforms.lifts as lifts
# import torch_geometric.transforms.wirings as wirings
# from torch_geometric.datasets import (
#     PPI,
#     Amazon,
#     Coauthor,
#     KarateClub,
#     MNISTSuperpixels,
#     Planetoid,
#     QM7b,
#     QM9,
#     TUDataset,
# )
# from torch_geometric.graphgym.config import cfg
# from torch_geometric.graphgym.models.transform import (
#     create_link_label,
#     neg_sampling_transform,
# )
# from torch_geometric.loader import (
#     ClusterLoader,
#     DataLoader,
#     GraphSAINTEdgeSampler,
#     GraphSAINTNodeSampler,
#     GraphSAINTRandomWalkSampler,
#     NeighborSampler,
#     RandomNodeSampler,
# )
# from torch_geometric.utils import (
#     index_to_mask,
#     negative_sampling,
#     to_undirected,
# )
#
# index2mask = index_to_mask  # TODO Backward compatibility
#
#
# def planetoid_dataset(name: str) -> Callable:
#     return lambda root: Planetoid(root, name)
#
#
# # register.register_dataset('Cora', planetoid_dataset('Cora'))
# # register.register_dataset('CiteSeer', planetoid_dataset('CiteSeer'))
# # register.register_dataset('PubMed', planetoid_dataset('PubMed'))
# # register.register_dataset('PPI', PPI)
#
#
# def load_pyg(name, dataset_dir, pre_transform=None):
#     """
#     Load PyG dataset objects. (More PyG datasets will be supported)
#
#     Args:
#         name (string): dataset name
#         dataset_dir (string): data directory
#
#     Returns: PyG dataset object
#
#     """
#     dataset_dir = '{}/{}'.format(dataset_dir, name)
#     if name in ['Cora', 'CiteSeer', 'PubMed']:
#         dataset = Planetoid(dataset_dir, name, pre_transform=pre_transform)
#     elif name[:3] == 'TU_':
#         # TU_IMDB doesn't have node features
#         if name[3:] == 'IMDB':
#             name = 'IMDB-MULTI'
#             dataset = TUDataset(dataset_dir, name, pre_transform=T.compose([T.Constant, pre_transform]))
#         else:
#             dataset = TUDataset(dataset_dir, name[3:], pre_transform=pre_transform) # TODO: could also put the transform here.
#     elif name == 'Karate':
#         dataset = KarateClub(pre_transform=pre_transform)
#     elif 'Coauthor' in name:
#         if 'CS' in name:
#             dataset = Coauthor(dataset_dir, name='CS',pre_transform=pre_transform)
#         else:
#             dataset = Coauthor(dataset_dir, name='Physics',pre_transform=pre_transform)
#     elif 'Amazon' in name:
#         if 'Computers' in name:
#             dataset = Amazon(dataset_dir, name='Computers',pre_transform=pre_transform)
#         else:
#             dataset = Amazon(dataset_dir, name='Photo', pre_transform=pre_transform)
#     elif name == 'MNIST':
#         dataset = MNISTSuperpixels(dataset_dir, pre_transform=pre_transform)
#     elif name == 'PPI':
#         dataset = PPI(dataset_dir, pre_transform=pre_transform)
#     elif name == 'QM7b':
#         dataset = QM7b(dataset_dir, pre_transform=pre_transform)
#     elif name == 'qm9':
#         dataset = QM9(dataset_dir, pre_transform=pre_transform)
#     else:
#         raise ValueError('{} not support'.format(name))
#
#     return dataset
#
#
# def set_dataset_attr(dataset, name, value, size):
#     dataset._data_list = None
#     dataset.data[name] = value
#     if dataset.slices is not None:
#         dataset.slices[name] = torch.tensor([0, size], dtype=torch.long)
#
#
# def load_ogb(name, dataset_dir):
#     r"""
#
#     Load OGB dataset objects.
#
#
#     Args:
#         name (string): dataset name
#         dataset_dir (string): data directory
#
#     Returns: PyG dataset object
#
#     """
#     from ogb.graphproppred import PygGraphPropPredDataset
#     from ogb.linkproppred import PygLinkPropPredDataset
#     from ogb.nodeproppred import PygNodePropPredDataset
#
#     if name[:4] == 'ogbn':
#         dataset = PygNodePropPredDataset(name=name, root=dataset_dir)
#         splits = dataset.get_idx_split()
#         split_names = ['train_mask', 'val_mask', 'test_mask']
#         for i, key in enumerate(splits.keys()):
#             mask = index_to_mask(splits[key], size=dataset.data.y.shape[0])
#             set_dataset_attr(dataset, split_names[i], mask, len(mask))
#         edge_index = to_undirected(dataset.data.edge_index)
#         set_dataset_attr(dataset, 'edge_index', edge_index,
#                          edge_index.shape[1])
#
#     elif name[:4] == 'ogbg':
#         dataset = PygGraphPropPredDataset(name=name, root=dataset_dir)
#         splits = dataset.get_idx_split()
#         split_names = [
#             'train_graph_index', 'val_graph_index', 'test_graph_index'
#         ]
#         for i, key in enumerate(splits.keys()):
#             id = splits[key]
#             set_dataset_attr(dataset, split_names[i], id, len(id))
#
#     elif name[:4] == "ogbl":
#         dataset = PygLinkPropPredDataset(name=name, root=dataset_dir)
#         splits = dataset.get_edge_split()
#         id = splits['train']['edge'].T
#         if cfg.dataset.resample_negative:
#             set_dataset_attr(dataset, 'train_pos_edge_index', id, id.shape[1])
#             dataset.transform = neg_sampling_transform
#         else:
#             id_neg = negative_sampling(edge_index=id,
#                                        num_nodes=dataset.data.num_nodes,
#                                        num_neg_samples=id.shape[1])
#             id_all = torch.cat([id, id_neg], dim=-1)
#             label = create_link_label(id, id_neg)
#             set_dataset_attr(dataset, 'train_edge_index', id_all,
#                              id_all.shape[1])
#             set_dataset_attr(dataset, 'train_edge_label', label, len(label))
#
#         id, id_neg = splits['valid']['edge'].T, splits['valid']['edge_neg'].T
#         id_all = torch.cat([id, id_neg], dim=-1)
#         label = create_link_label(id, id_neg)
#         set_dataset_attr(dataset, 'val_edge_index', id_all, id_all.shape[1])
#         set_dataset_attr(dataset, 'val_edge_label', label, len(label))
#
#         id, id_neg = splits['test']['edge'].T, splits['test']['edge_neg'].T
#         id_all = torch.cat([id, id_neg], dim=-1)
#         label = create_link_label(id, id_neg)
#         set_dataset_attr(dataset, 'test_edge_index', id_all, id_all.shape[1])
#         set_dataset_attr(dataset, 'test_edge_label', label, len(label))
#
#     else:
#         raise ValueError('OGB dataset: {} non-exist')
#     return dataset
#
# @register_loader('liwich_loader')
# def load_dataset():
#     r"""
#
#     Load dataset objects.
#
#     Returns: PyG dataset object
#
#     """
#     format = cfg.dataset.format
#     name = cfg.dataset.name
#     dataset_dir = cfg.dataset.dir
#     # Try to load customized data format
#     for func in register.loader_dict.values():
#         dataset = func(format, name, dataset_dir)
#         if dataset is not None:
#             return dataset
#     # Load from Pytorch Geometric dataset
#     if cfg.dataset.pre_transform == "lift_wire":
#         if cfg.lift.data_model == "simplicial_complex":
#             lift = lifts.LiftGraphToSimplicialComplex(cfg.lift.method,
#                                                       cfg.lift.init_method,
#                                                       cfg.lift.max_clique_dim
#                                                       )
#         elif cfg.lift.data_model == "cell_complex":
#             lift = lifts.LiftGraphToCellComplex(cfg.lift.method,
#                                                 cfg.lift.init_method,
#                                                 cfg.lift.max_simple_cycle_length,
#                                                 cfg.lift.max_induced_cycle_length,
#                                                 cfg.init_edges,
#                                                 cfg.init_rings
#                                                 )
#         else:
#             raise NotImplementedError
#         if cfg.lift.data_model in ["simplicial_complex", "cell_complex"]:
#             wiring = wirings.HypergraphWiring(cfg.wiring.adjacency_types)
#         else:
#             raise NotImplementedError
#         pre_transform = LiftAndWire(lift, wiring)
#     else:
#         pre_transform = None
#     if format == 'PyG':
#         dataset = load_pyg(name, dataset_dir, pre_transform=pre_transform)
#     # Load from OGB formatted data
#     elif format == 'OGB':
#         dataset = load_ogb(name.replace('_', '-'), dataset_dir)
#     else:
#         raise ValueError('Unknown data format: {}'.format(format))
#     return dataset
#
#
# def set_dataset_info(dataset):
#     r"""
#     Set global dataset information
#
#     Args:
#         dataset: PyG dataset object
#
#     """
#
#     # get dim_in and dim_out
#     try:
#         cfg.share.dim_in = dataset.data.x.shape[1]
#     except Exception:
#         cfg.share.dim_in = 1
#     try:
#         if cfg.dataset.task_type == 'classification':
#             cfg.share.dim_out = torch.unique(dataset.data.y).shape[0]
#         else:
#             cfg.share.dim_out = dataset.data.y.shape[1]
#     except Exception:
#         cfg.share.dim_out = 1
#
#     # count number of dataset splits
#     cfg.share.num_splits = 1
#     for key in dataset.data.keys:
#         if 'val' in key:
#             cfg.share.num_splits += 1
#             break
#     for key in dataset.data.keys:
#         if 'test' in key:
#             cfg.share.num_splits += 1
#             break
#
#
# def create_dataset():
#     r"""
#     Create dataset object
#
#     Returns: PyG dataset object
#
#     """
#     dataset = load_dataset()
#     set_dataset_info(dataset)
#
#     return dataset
#
#
# def get_loader(dataset, sampler, batch_size, shuffle=True):
#     if sampler == "full_batch" or len(dataset) > 1:
#         loader_train = DataLoader(dataset, batch_size=batch_size,
#                                   shuffle=shuffle, num_workers=cfg.num_workers,
#                                   pin_memory=True)
#     elif sampler == "neighbor":
#         loader_train = NeighborSampler(
#             dataset[0], sizes=cfg.train.neighbor_sizes[:cfg.gnn.layers_mp],
#             batch_size=batch_size, shuffle=shuffle,
#             num_workers=cfg.num_workers, pin_memory=True)
#     elif sampler == "random_node":
#         loader_train = RandomNodeSampler(dataset[0],
#                                          num_parts=cfg.train.train_parts,
#                                          shuffle=shuffle,
#                                          num_workers=cfg.num_workers,
#                                          pin_memory=True)
#     elif sampler == "saint_rw":
#         loader_train = \
#             GraphSAINTRandomWalkSampler(dataset[0],
#                                         batch_size=batch_size,
#                                         walk_length=cfg.train.walk_length,
#                                         num_steps=cfg.train.iter_per_epoch,
#                                         sample_coverage=0,
#                                         shuffle=shuffle,
#                                         num_workers=cfg.num_workers,
#                                         pin_memory=True)
#     elif sampler == "saint_node":
#         loader_train = \
#             GraphSAINTNodeSampler(dataset[0], batch_size=batch_size,
#                                   num_steps=cfg.train.iter_per_epoch,
#                                   sample_coverage=0, shuffle=shuffle,
#                                   num_workers=cfg.num_workers,
#                                   pin_memory=True)
#     elif sampler == "saint_edge":
#         loader_train = \
#             GraphSAINTEdgeSampler(dataset[0], batch_size=batch_size,
#                                   num_steps=cfg.train.iter_per_epoch,
#                                   sample_coverage=0, shuffle=shuffle,
#                                   num_workers=cfg.num_workers,
#                                   pin_memory=True)
#     elif sampler == "cluster":
#         loader_train = \
#             ClusterLoader(dataset[0],
#                           num_parts=cfg.train.train_parts,
#                           save_dir="{}/{}".format(cfg.dataset.dir,
#                                                   cfg.dataset.name.replace(
#                                                       "-", "_")),
#                           batch_size=batch_size, shuffle=shuffle,
#                           num_workers=cfg.num_workers,
#                           pin_memory=True)
#
#     else:
#         raise NotImplementedError("%s sampler is not implemented!" % sampler)
#     return loader_train
#
#
# def create_loader():
#     """
#     Create data loader object
#
#     Returns: List of PyTorch data loaders
#
#     """
#     dataset = create_dataset()
#     # train loader
#     if cfg.dataset.task == 'graph':
#         if hasattr(dataset.data, 'train_graph_index'):
#             id = dataset.data['train_graph_index']
#             loaders = [
#                 get_loader(dataset[id], cfg.train.sampler, cfg.train.batch_size,
#                            shuffle=True)
#             ]
#             delattr(dataset.data, 'train_graph_index')
#         else:
#             UserWarning('Custom loader: no train_graph_index in dataset, use full graph') #TODO
#             [train_ratio, val_ratio, test_ratio] = cfg.dataset.split
#             train_index, test_val_index = train_test_split(list(range(len(dataset))), test_size=val_ratio + test_ratio, shuffle=True, random_state=cfg.seed)
#             val_test_indices = train_test_split(test_val_index, test_size=test_ratio / (val_ratio + test_ratio), shuffle=True, random_state=cfg.seed)
#             loaders = [
#                 get_loader(dataset[train_index], cfg.train.sampler, cfg.train.batch_size,
#                            shuffle=True)
#             ]
#     else:
#         loaders = [
#             get_loader(dataset, cfg.train.sampler, cfg.train.batch_size,
#                        shuffle=True)
#         ]
#
#     # val and test loaders
#     for i in range(len(cfg.dataset.split)-1): # TODO: I changed this to the length of dataset.split
#         if cfg.dataset.task == 'graph':
#             if hasattr(dataset.data, 'val_graph_index') and hasattr(dataset.data, 'test_graph_index'):
#                 split_names = ['val_graph_index', 'test_graph_index']
#                 id = dataset.data[split_names[i]]
#                 loaders.append(
#                     get_loader(dataset[id], cfg.val.sampler, cfg.train.batch_size,
#                                shuffle=False))
#                 delattr(dataset.data, split_names[i])
#             else:
#                 UserWarning('Custom loader: no val_graph_index or test_graph_index in dataset, use customised test/val split.')
#                 loaders.append(
#                     get_loader(dataset[val_test_indices[i]], cfg.val.sampler, cfg.train.batch_size,
#                                shuffle=False))
#         else:
#             loaders.append(
#                 get_loader(dataset, cfg.val.sampler, cfg.train.batch_size,
#                            shuffle=False))
#
#     return loaders
