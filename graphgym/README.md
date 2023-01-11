# Instruction Manual

Welcome to the "Scalable Higher-Order Graph Representation Learning Through Heterogeneous Graph Neural Networks" repository!

## Quick start guide

### Installing dependencies
We ran our experiments using Python 3.10. We use Conda environments and assume that your system is compatible with PyTorch 1.12.1 and, if you have a GPU, with CUDA 11.3.
(Add more general description later.)

1. Open a terminal and navigate to the PyG/graphgym directory.
2. For systems with CUDA, you can install all dependencies by creating a Conda environment:

   ```
   $ conda env create --file pyg_GPU.yaml --name pyg_GPU
   ```

3. Alternatively, you can use the CPU-version of PyTorch. The corresponding Conda environment can be installed with

    ```
   $ conda env create --file pyg_CPU.yaml --name pyg_CPU
   ```    

### Running a single experiment

The experiments are designed to benchmark GNN methods. One experiment corresponds to a full training-testing cycle for a given configuration of method, dataset and hyperparameters. 

To run one experiment, we first create a configuration file. There are two configuration templates, one for CUDA: `GPU_confg.yaml`, 
and one for CPU: `CPU_confg.yaml`. An experiment can then be started by running

```
$ python liwich_main.py --cfg configs/GPU_confg.yaml
```

respectively,

```
$ python liwich_main.py --cfg configs/CPU_confg.yaml
```

The results are stored in the `PyG/graphgym/results` folder in a directory with the same name as the configuration and comprise statistics such as accuracy/mse, lr, etc.

### Running experiments with Weights & Biases
For more comprehensive experiments (testing hyperparameter configurations or different GNN methods), our code uses the [Weights and Biases](https://wandb.ai/site) (W&B) API. 
We briefly outline how we use W&B, please see the [docs](https://docs.wandb.ai/?_gl=1*4iegg6*_ga*MjA4NDc3NzIwNy4xNjYyNzI4MjMz*_ga_JH1SJHJQXJ*MTY3MzM1Nzg5OS4xMTYuMS4xNjczMzU3OTI5LjMwLjAuMA..) for a comprehensive introduction to W&B.

1. Create a **sweep** configuration (see `PyG/graphgym/sweep/example.yaml` for an example).
2. Login to W&B (only necessary once):

   ```
   $ wandb login
   ```

3. Activate the sweep:

   ```
   $ wandb sweep <path to sweep> --project <project name>
   ```
   
4. Follow the instructions in the terminal to run the sweep. This will be of the form
   ```
   $ wandb agent <username>/<project name>/<sweep ID>
   ```
   The results will be displayed in the Weights&Biases online application.


5. Reading out the results: this can be done with the W&B API, see e.g. `create_plots.py` and `generate_curves.py` in the `PyG/graphgym/wandb_plotting` directory.