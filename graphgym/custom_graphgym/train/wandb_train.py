import json

import custom_graphgym  # noqa, register custom modules
import pytorch_lightning as pl
import wandb

from torch_geometric.graphgym.config import (
    cfg
)
from torch_geometric.graphgym.checkpoint import get_ckpt_dir
from torch_geometric.graphgym.model_builder import create_model, GraphGymModule
from torch_geometric.graphgym.logger import LoggerCallback
from torch_geometric.graphgym.register import register_train


class LoggerCallbackWandb(LoggerCallback):
    @staticmethod
    def log_last(logger, split):
        with open('{}/stats.json'.format(logger.out_dir)) as f:
            last = json.loads(f.readlines()[-1])
            log = {}
            for k, v in last.items():
                if k in ['lr', 'params'] and split != 'train':
                    continue
                if k in ['epoch', 'lr', 'params']:
                    log[k] = v
                else:
                    log[f'{split}/{k}'] = v
            wandb.log(log)

    def on_train_epoch_end(
            self,
            trainer: 'pl.Trainer',
            pl_module: 'pl.LightningModule',
    ):
        self.train_logger.write_epoch(trainer.current_epoch)
        self.log_last(self.train_logger, 'train')

    def on_validation_epoch_end(
            self,
            trainer: 'pl.Trainer',
            pl_module: 'pl.LightningModule',
    ):
        self.val_logger.write_epoch(trainer.current_epoch)
        self.log_last(self.val_logger, 'val')

    def on_test_epoch_end(
            self,
            trainer: 'pl.Trainer',
            pl_module: 'pl.LightningModule',
    ):
        self.test_logger.write_epoch(trainer.current_epoch)
        self.log_last(self.test_logger, 'test')

@register_train('wandb_train')
def train(model: GraphGymModule, datamodule, logger: bool = True, use_wandb: bool = False):
    callbacks = []
    if logger:
        callbacks.append(LoggerCallbackWandb() if use_wandb else LoggerCallback())
    if cfg.train.enable_ckpt:
        ckpt_cbk = pl.callbacks.ModelCheckpoint(dirpath=get_ckpt_dir())
        callbacks.append(ckpt_cbk)

    trainer = pl.Trainer(
        enable_checkpointing=cfg.train.enable_ckpt,
        callbacks=callbacks,
        default_root_dir=cfg.out_dir,
        max_epochs=cfg.optim.max_epoch,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
