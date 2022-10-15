import logging
import sys
import os
sys.path.append(os.getcwd())

import custom_graphgym  # noqa, register custom modules
import torch
import wandb

from custom_graphgym.train import wandb_train
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
from torch_geometric.graphgym.loader import lift_wire_transform_formatter
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.train import GraphGymDataModule
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device


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
    set_cfg(cfg) # TMU, this is the default config
    load_cfg(cfg, args) # TMU, this merges the default config with the config file
    use_wandb = cfg.use_wandb
    set_out_dir(cfg.out_dir, args.cfg_file)

    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    dump_cfg(cfg) # TMU, this creates a config file in the output directory to record the config used

    set_run_dir(cfg.out_dir)
    set_printing()
    seed_everything(cfg.seed)
    auto_select_device()
    # Set machine learning pipeline
    datamodule = GraphGymDataModule() # how does this know which config to use?

    transformed_dataset = None
    # This is usually hidden in the GraphGymDataModule, but I need the dataset metadata for hanconv, so I'm loading it here too...
    if cfg.dataset.transform is not None:
        transformed_dataset = lift_wire_transform_formatter(name=cfg.dataset.name,
                                                            dataset_dir=cfg.dataset.dir+"metadata_information",
                                                            pre_transform=cfg.dataset.transform) # attention: here we pretransform with the transform to obtain the metadata.
    elif cfg.dataset.pre_transform is not None:
        transformed_dataset = lift_wire_transform_formatter(name=cfg.dataset.name,
                                                            dataset_dir=cfg.dataset.dir,
                                                            pre_transform=cfg.dataset.pre_transform)
    if transformed_dataset is not None:
        cfg.dataset.metadata = transformed_dataset.data.metadata()

    model = create_model() # how does this know which config to use?
    # Print model info
    if use_wandb:
        wandb.init(config=cfg)
    logging.info(model)
    logging.info(cfg)
    if transformed_dataset is not None:
        dummy_batch = transformed_dataset.data.to(cfg.accelerator)
        model(dummy_batch) # lazy initialisation, sometimes this seems to be necessary, not always though
    cfg.params = params_count(model)  # -> would need to initialize lazy modules.
    logging.info('Num parameters: %s', cfg.params)
    wandb_train.train(model, datamodule, logger=True, use_wandb=use_wandb)
