from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


@register_config('lift_wire')
def set_cfg_lift_wire(cfg):
    r'''
    This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    '''

    # ----------------------------------------------------------------------- #
    # WandB options
    # ----------------------------------------------------------------------- #

    cfg.use_wandb = False

    # ----------------------------------------------------------------------- #
    # Lifting options
    # ----------------------------------------------------------------------- #

    cfg.lift = CN()

    cfg.lift.data_model = "simplicial_complex"  # one of "simplicial_complex", "cell_complex

    cfg.lift.method = "inclusion"  # alternatively: clique_complex (for simplicial complex), or rings (for cell complex)
    cfg.lift.init_method = "mean" # alternatively: "random" or "sum"

    # for simplicial_complex
    cfg.lift.max_clique_dim = 3
    cfg.lift.init_simplex = False

    # for cell complex
    cfg.lift.max_simple_cycle_length = 2
    cfg.lift.max_induced_cycle_length = 2
    cfg.lift.init_rings = True
    cfg.lift.init_edges = True

    # ----------------------------------------------------------------------- #
    # Wiring options
    # ----------------------------------------------------------------------- #

    cfg.wiring = CN()

    cfg.wiring.adjacency_types = ["boundary", "upper"]

    # Maybe add this later: would require a few changes in the Hypergraph Wiring class
    # cfg.wiring.drop_orig_edges = False
    # cfg.wiring.drop_unconnected_nodes = False
    # cfg.wiring.max_sample = 10000
    # cfg.wiring.weighted = False

    # ----------------------------------------------------------------------- #
    # Extending existing options
    # ----------------------------------------------------------------------- #

    cfg.dataset.pre_transform = "lift_wire"
    cfg.dataset.metadata = None

    cfg.gnn.graph_type = "hetero"

    # For pre_mp, we would want to implement heterolinear.
    cfg.gnn.layers_pre_mp = 0 # due to how we transform the data, pre_mp is not applicable.
    cfg.gnn.layers_mp = 3
    cfg.gnn.layers_post_mp = 1

    if cfg.dataset.task == "node" and cfg.gnn.graph_type=='hetero':
        cfg.gnn.head = 'hetero_node_head'
    # For post_mp, we first do graph_readout and then potentially apply post_mp.
    # Looks like this can be reactivated again:
    # cfg.gnn.batchnorm = False # run into heterograph problems otherwise
    # cfg.gnn.dropout = False
    # cfg.gnn.act =

    # for heatconv
    cfg.gnn.heat_edge_type_emb_dim = 16
    cfg.gnn.heat_edge_dim = 1
    cfg.gnn.heat_edge_attr_emb_dim = 1

    # for heteroconv -> this is quite general, so setting up the config could be lengthy
    cfg.gnn.heteroconv_dict = None # in here you need to specify the entire module dictionary - basta
    cfg.gnn.heteroconv_node_to_edge = None
    cfg.gnn.heteroconv_edge_to_node = None
    cfg.gnn.heteroconv_edge_to_cell = None
    cfg.gnn.heteroconv_cell_to_edge = None
