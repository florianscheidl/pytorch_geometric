import os.path as osp

import torch

import torch_geometric.transforms as T
import torch_geometric.sampler as S
from torch_geometric.datasets import AMiner

path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/AMiner')
dataset = AMiner(path)
data = dataset[0]

sampler = S.NeighborSampler(data, num_neighbors=[5], input_type='paper')
sampled_data = sampler.sample_from_nodes(torch.arange(10))

print(sampled_data)

metapaths = [[
    ('author', 'writes', 'paper'),
    ('paper', 'published_in', 'venue'),
    ('venue', 'publishes', 'paper')]
    ]

print("Adding metapaths...")
transform = T.AddMetaPathsHops(metapaths=metapaths, drop_orig_edges=False,
                           drop_unconnected_nodes=True)

print("Transforming...")
data = transform(data)
print(data)