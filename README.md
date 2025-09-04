# SIR-PINN: Physics-Informed Neural Networks for SIR Model Parameter Estimation

A Python implementation of Physics-Informed Neural Networks (PINNs) for estimating infection rate parameters in the SIR (Susceptible-Infected-Recovered) epidemiological model.

## 🚀 Quick Start

This project uses [**uv**](https://docs.astral.sh/uv) for dependency management.

### Setup

Install dependencies:

```bash
uv sync
```

Run the project:

```bash
uv run src/pinn/sir_pinn.py
```

## 📋 Overview

This project implements a Physics-Informed Neural Network approach to solve the inverse problem of the SIR epidemiological model. The primary objective is to estimate the infection rate parameter β from observed infection data while respecting the underlying physical laws described by the SIR differential equations.

### Key Features

- **Physics-Informed Neural Networks**: Combines deep learning with physical constraints
- **SIR Model Integration**: Respects the underlying differential equations
- **Comprehensive Ablation Studies**: Systematic evaluation of different configurations
- **Advanced Stopping Criteria**: SMMA and early stopping implementations
- **Visualization Tools**: Rich plotting and analysis capabilities

<!-- ## 🛠️ Usage -->
