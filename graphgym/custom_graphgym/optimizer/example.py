from typing import Iterator

from torch.nn import Parameter
from torch.optim import Adagrad, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.graphgym.config import cfg

import torch_geometric.graphgym.register as register


@register.register_optimizer('adagrad')
def adagrad_optimizer(params: Iterator[Parameter], base_lr: float,
                      weight_decay: float) -> Adagrad:
    return Adagrad(params, lr=base_lr, weight_decay=weight_decay)


@register.register_scheduler('plateau')
def plateau_scheduler(optimizer: Optimizer, patience: int,
                      lr_decay: float) -> ReduceLROnPlateau:
    return ReduceLROnPlateau(optimizer,
                             patience=cfg.optim.patience,
                             factor=cfg.optim.lr_decay,
                             min_lr=cfg.optim.lr_scheduler_min_lr)
