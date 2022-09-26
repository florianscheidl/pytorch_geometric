# Test Liwich framework on TU Dataset

import os.path as osp

import torch

from torch_geometric.config.custom_liwich_config import (cfg,
                                                        dump_cfg,
                                                        set_cfg,
                                                        load_cfg,
                                                        set_out_dir,
                                                        set_run_dir,
                                                         )
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.transforms.lifts import LiftDataset
from torch_geometric.transforms.wirings import WireDataset

# Unpack config from argparser?
args = parse_args()
set_cfg(cfg)
load_cfg(cfg, args)
set_out_dir(cfg.out_dir, args.cfg_file)

# load dataset from config
# replace with corresponding dataset loader
from torch_geometric.datasets import TUDataset
path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/tu')
dataset = TUDataset

# Lift: transform dataset with one of our lifting methods
dataset_lift = LiftDataset(cfg.lift_params)
lifted_dataset = dataset_lift(dataset)

# Wire: transform higher-order graph data model to heterogeneous graph data model
dataset_wiring = WireDataset(cfg.wiring_params)
final_dataset = dataset_wiring(lifted_dataset)

# Chooose: design of ML model



# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model = MetaPath2Vec(data.edge_index_dict, embedding_dim=128,
#                      metapath=metapath, walk_length=50, context_size=7,
#                      walks_per_node=5, num_negative_samples=5,
#                      sparse=True).to(device)

# loader = model.loader(batch_size=128, shuffle=True, num_workers=6)
# optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)


def train(epoch, log_steps=100, eval_steps=2000):
    raise NotImplementedError
    # model.train()
    #
    # total_loss = 0
    # for i, (pos_rw, neg_rw) in enumerate(loader):
    #     optimizer.zero_grad()
    #     loss = model.loss(pos_rw.to(device), neg_rw.to(device))
    #     loss.backward()
    #     optimizer.step()
    #
    #     total_loss += loss.item()
    #     if (i + 1) % log_steps == 0:
    #         print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
    #                f'Loss: {total_loss / log_steps:.4f}'))
    #         total_loss = 0
    #
    #     if (i + 1) % eval_steps == 0:
    #         acc = test()
    #         print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
    #                f'Acc: {acc:.4f}'))


@torch.no_grad()
def test(train_ratio=0.1):
    raise NotImplementedError
    # model.eval()
    #
    # z = model('author', batch=data['author'].y_index.to(device))
    # y = data['author'].y
    #
    # perm = torch.randperm(z.size(0))
    # train_perm = perm[:int(z.size(0) * train_ratio)]
    # test_perm = perm[int(z.size(0) * train_ratio):]
    #
    # return model.test(z[train_perm], y[train_perm], z[test_perm], y[test_perm],
    #                   max_iter=150)


# for epoch in range(1, 6):
#     train(epoch)
#     acc = test()
#     print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')
