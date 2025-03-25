# EpiPINN

A Physics-Informed Neural Network approach for an inverse problem in
compartmental epidemiological models.

Written in Pytorch.

## Getting Started

The conda package manager is a prerequisite for this project. If you don't have
it installed, the [Miniforge](https://github.com/conda-forge/miniforge)
distribution is recommended.

To use conda environments:

```bash
conda env create -f environment.yaml
conda activate pinn
```

If the environment already exists and the `environment.yaml` file has been
updated, you can update the environment with:

```bash
conda env update -f environment.yaml --prune
```

If you changed the environment and want to update the `environment.yaml` file,
you can do so with:

```bash
conda env export --no-builds | grep -v "^prefix: " > environment.yaml
```
