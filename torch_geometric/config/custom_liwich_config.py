import functools
import inspect
import logging
import os
import shutil
import warnings
from collections.abc import Iterable
from dataclasses import asdict
from typing import Any

from torch_geometric.data.makedirs import makedirs


try:  # Define global config object
    from yacs.config import CfgNode as CN
    cfg = CN()
except ImportError:
    cfg = None
    warnings.warn("Could not define global config object. Please install "
                  "'yacs' for using the GraphGym experiment manager via "
                  "'pip install yacs'.")

def set_cfg(cfg):
    r'''
    This function sets the default config value.

    :return: configuration use by the experiment.
    '''
    if cfg is None:
        return cfg

    # ----------------------------------------------------------------------- #
    # Lifting options
    # ----------------------------------------------------------------------- #

    cfg.lift = CN()

    cfg.lift.data_model = "simplicial_complex" # one of "simplicial_complex", "cell_complex
    cfg.lift.init_method = "sum"

    cfg.lift.method = "inclusion" # alternatively: clique_complex (for simplicial complex), or rings (for cell complex)
    cfg.lift.max_clique_dim = 3
    cfg.lift.max_simple_cycle_length = 2
    cfg.lift.max_induced_cycle_length = 2
    cfg.lift.init_edges = False
    cfg.lift.init_simplex = False
    cfg.lift.init_rings = False

    # ----------------------------------------------------------------------- #
    # Wiring options
    # ----------------------------------------------------------------------- #

    cfg.wiring = CN()

    cfg.wiring.adjacency_types = ["boundary","upper"]
    cfg.wiring.drop_orig_edges = False
    cfg.wiring.drop_unconnected_nodes = False
    cfg.wiring.max_sample = 10000
    cfg.wiring.weighted = False

    # ----------------------------------------------------------------------- #
    # GNN options
    # ----------------------------------------------------------------------- #

    # This should include: # layers, convs, aggr, norm, pool
    # Similar to graphgym set up. Ideally, we would merge these two config files and make adjustments where needed!

    cfg.gnn = CN()

    cfg.gnn.layers_pre_mp = 1
    cfg.gnn.layers_mp = 3
    cfg.gnn.layers_post_mp = 1

    cfg.gnn.dim_inner = [16 for _ in range(cfg.gnn.layers_mp+1)] # hidden dimensions
    cfg.gnn.convs = ["han_conv" for _ in range(cfg.gnn.layers_mp)]
    cfg.gnn.stage_type = 'stack'
    cfg.gnn.act = 'relu'
    cfg.gnn.dropout = 0.0
    cfg.gnn.agg = 'add'
    cfg.gnn.msg_direction = 'single'
    cfg.gnn.l2norm = True
    cfg.gnn.keep_edge = 0.5
    cfg.gnn.clear_feature = True

    return cfg


def assert_cfg(cfg):
    r"""Checks config values, do necessary post processing to the configs"""
    if cfg.dataset.task not in ['node', 'edge', 'graph', 'link_pred']:
        raise ValueError('Task {} not supported, must be one of node, '
                         'edge, graph, link_pred'.format(cfg.dataset.task))
    if 'classification' in cfg.dataset.task_type and cfg.model.loss_fun == \
            'mse':
        cfg.model.loss_fun = 'cross_entropy'
        logging.warning(
            'model.loss_fun changed to cross_entropy for classification.')
    if cfg.dataset.task_type == 'regression' and cfg.model.loss_fun == \
            'cross_entropy':
        cfg.model.loss_fun = 'mse'
        logging.warning('model.loss_fun changed to mse for regression.')
    if cfg.dataset.task == 'graph' and cfg.dataset.transductive:
        cfg.dataset.transductive = False
        logging.warning('dataset.transductive changed '
                        'to False for graph task.')
    if cfg.gnn.layers_post_mp < 1:
        cfg.gnn.layers_post_mp = 1
        logging.warning('Layers after message passing should be >=1')
    if cfg.gnn.head == 'default':
        cfg.gnn.head = cfg.dataset.task
    cfg.run_dir = cfg.out_dir


def dump_cfg(cfg):
    r"""
    Dumps the config to the output directory specified in
    :obj:`cfg.out_dir`

    Args:
        cfg (CfgNode): Configuration node

    """
    makedirs(cfg.out_dir)
    cfg_file = os.path.join(cfg.out_dir, cfg.cfg_dest)
    with open(cfg_file, 'w') as f:
        cfg.dump(stream=f)


def load_cfg(cfg, args):
    r"""
    Load configurations from file system and command line

    Args:
        cfg (CfgNode): Configuration node
        args (ArgumentParser): Command argument parser

    """
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    assert_cfg(cfg)


def makedirs_rm_exist(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)


def get_fname(fname):
    r"""
    Extract filename from file name path

    Args:
        fname (string): Filename for the yaml format configuration file
    """
    fname = fname.split('/')[-1]
    if fname.endswith('.yaml'):
        fname = fname[:-5]
    elif fname.endswith('.yml'):
        fname = fname[:-4]
    return fname


def set_out_dir(out_dir, fname):
    r"""
    Create the directory for full experiment run

    Args:
        out_dir (string): Directory for output, specified in :obj:`cfg.out_dir`
        fname (string): Filename for the yaml format configuration file

    """
    fname = get_fname(fname)
    cfg.out_dir = os.path.join(out_dir, fname)
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.out_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.out_dir)


def set_run_dir(out_dir):
    r"""
    Create the directory for each random seed experiment run

    Args:
        out_dir (string): Directory for output, specified in :obj:`cfg.out_dir`
        fname (string): Filename for the yaml format configuration file

    """
    cfg.run_dir = os.path.join(out_dir, str(cfg.seed))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)


set_cfg(cfg)


def from_config(func):
    if inspect.isclass(func):
        params = list(inspect.signature(func.__init__).parameters.values())[1:]
    else:
        params = list(inspect.signature(func).parameters.values())

    arg_names = [p.name for p in params]
    has_defaults = [p.default != inspect.Parameter.empty for p in params]

    @functools.wraps(func)
    def wrapper(*args, cfg: Any = None, **kwargs):
        if cfg is not None:
            cfg = dict(cfg) if isinstance(cfg, Iterable) else asdict(cfg)

            iterator = zip(arg_names[len(args):], has_defaults[len(args):])
            for arg_name, has_default in iterator:
                if arg_name in kwargs:
                    continue
                elif arg_name in cfg:
                    kwargs[arg_name] = cfg[arg_name]
                elif not has_default:
                    raise ValueError(f"'cfg.{arg_name}' undefined")
        return func(*args, **kwargs)

    return wrapper