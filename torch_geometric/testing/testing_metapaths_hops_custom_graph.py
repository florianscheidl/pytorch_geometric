import os

import torch

from torch_geometric.data.hetero_data import HeteroData
import torch_geometric.transforms as T

test_graph = HeteroData()
test_graph["food"].id = torch.tensor([[1],[2]]) # 1: Zucchine, 2: WhiteBread
test_graph["country"].population = torch.tensor([[9], [8], [25]]) # AUT, CH, NL
test_graph["person"].height = torch.tensor([[183],[174],[156],[152]]) # Flo, Bok, Lilly, Emily

test_graph["person", "likes", "food"].edge_index = torch.tensor([[0, 1, 2, 3, 3],
                                                                 [0, 1, 0, 0, 1]]) # Flo likes Zucchine, Bok likes WhiteBread, Lilly likes Zucchine, Emily likes Zucchine and WhiteBread

test_graph["country", "home_to", "person"].edge_index = torch.tensor([[1, 1, 1, 0, 0, 2],
                                                                      [0, 1, 2, 2, 3, 3]]) # CH home to Flo, AUT home to Mom, AUT&CH home to Lilly, AUT&NL home to Emily, AUT home to Daddy, CH home to Hanna

test_graph["food", "is_from", "country"].edge_index = torch.tensor([[0,0,1,1],
                                                                    [0,1,0,2]]) # Zucchine is from AUT&CH, WhiteBreak is from AUT&NL
test_graph["country", "produces", "food"].edge_index = torch.tensor([[0,1,0,2],
                                                                     [0,0,1,1]]) # CH produces Zucchine and Tofu, NL produces Gouda, AUT produces Tofu

metapaths = [[("person","likes","food"), ("food","is_from","country"), ("country","home_to","person")]]
             # [("food","is_from","country"), ("country","home_to","person"), ("person","likes","food")]]

transform = T.AddMetaPathsHops(metapaths=metapaths, drop_orig_edges=False, drop_unconnected_nodes=True, weighted=True)
data = transform(test_graph)
print(data)