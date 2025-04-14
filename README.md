# SIR-PINN

A Physics-Informed Neural Network approach for an inverse problem in a SIR
epidemiological model.

Written in Pytorch.

## Getting Started

The conda package manager is a prerequisite for this project. If you don't have
it installed, the [Miniforge](https://github.com/conda-forge/miniforge)
distribution is recommended.

Create a new conda environment with the required dependencies:

```bash
conda env create -f env.yaml
conda activate pinn
```

Run the jupyter notebook:

```bash
jupyter notebook
```

## Managing the Environment

If the environment already exists and the `env.yaml` file has been
updated, you can update the environment with:

```bash
conda env update -f env.yaml --prune
```

If you changed the environment and want to update the `env.yaml` file,
you can do so with:

```bash
conda env export --no-builds | grep -v "^prefix: " > env.yaml
```
