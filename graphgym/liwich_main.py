import logging
import json

import custom_graphgym  # noqa, register custom modules
import torch

from torch_geometric import seed_everything
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (
    cfg,
    dump_cfg,
    set_cfg,
    load_cfg,
    set_out_dir,
    set_run_dir,
)
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.train import GraphGymDataModule
from torch_geometric.graphgym.checkpoint import get_ckpt_dir
from torch_geometric.graphgym.model_builder import create_model, GraphGymModule
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.logger import LoggerCallback

import pytorch_lightning as pl

try:
    import wandb
    is_wandb_available = True
except ImportError:
    wandb = None
    is_wandb_available = False

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


def train(model: GraphGymModule, datamodule, logger: bool = True):
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


if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    opts = []
    for kv in args.opts:
        if '=' in kv:
            # 'key=value'
            opts.append(kv.split('=')[0])
            opts.append(kv.split('=')[1])
        else:
            # 'key' or 'value'
            opts.append(kv)
    args.opts = opts
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    use_wandb = cfg.use_wandb and is_wandb_available
    set_out_dir(cfg.out_dir, args.cfg_file)

    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    dump_cfg(cfg)

    set_run_dir(cfg.out_dir)
    set_printing()
    seed_everything(cfg.seed)
    auto_select_device()
    # Set machine learning pipeline
    datamodule = GraphGymDataModule()
    model = create_model()
    # Print model info
    if use_wandb:
        wandb.init(config=cfg)
    logging.info(model)
    logging.info(cfg)
    cfg.params = params_count(model)
    logging.info('Num parameters: %s', cfg.params)
    train(model, datamodule, logger=True)



# if __name__ == '__main__':
#     # Load cmd line args
#     args = parse_args()
#
#     # Load config file
#     load_cfg(cfg, args)
#     set_out_dir(cfg.out_dir, args.cfg_file)
#     # Set Pytorch environment
#     torch.set_num_threads(cfg.num_threads)
#     dump_cfg(cfg)
#     # Repeat for different random seeds
#     for i in range(args.repeat):
#         set_run_dir(cfg.out_dir)
#         set_printing()
#         # Set configurations for each run
#         cfg.seed = cfg.seed + 1
#         seed_everything(cfg.seed)
#         auto_select_device()
#         # Set machine learning pipeline
#         datamodule = GraphGymDataModule()
#         model = create_model()
#         # Print model info
#         logging.info(model)
#         logging.info(cfg)
#         cfg.params = params_count(model)
#         logging.info('Num parameters: %s', cfg.params)
#         train(model, datamodule, logger=True)
#
#     # Aggregate results from different seeds
#     agg_runs(cfg.out_dir, cfg.metric_best)
#     # When being launched in batch mode, mark a yaml as done
#     if args.mark_done:
#         os.rename(args.cfg_file, f'{args.cfg_file}_done')
