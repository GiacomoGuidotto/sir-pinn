# %% [markdown]
# # SIR model parameter estimation: a Physics-Informed Neural Network approach
#
# This notebook presents an innovative approach to solving the inverse problem
# of the SIR (Susceptible-Infected-Recovered) epidemiological model using
# Physics-Informed Neural Networks (PINNs). The primary objective is to estimate
# the infection rate parameter $\beta$ from observed infection data, while
# respecting the underlying physical laws described by the SIR differential
# equations.
#
# ## Mathematical Model
#
# The SIR model is governed by the following system of ordinary differential
# equations (ODEs):
#
# $$
# \begin{cases}
# \frac{dS}{dt} &= -\frac{\beta}{N} I S, \\
# \frac{dI}{dt} &= \frac{\beta}{N} I S - \delta I, \\
# \frac{dR}{dt} &= \delta I,
# \end{cases}
# $$
#
# where:
# - $t \in [0, 90]$ days is the time domain
# - $S(t)$ is the number of susceptible individuals
# - $I(t)$ is the number of infected individuals
# - $R(t)$ is the number of recovered individuals
# - $N$ is the total population size
# - $\beta$ is the infection rate parameter
# - $\delta$ is the recovery rate
#
# Initial conditions are:
# - $S(0) = N - 1$
# - $I(0) = 1$
# - $R(0) = 0$
#
# ## Implementation Overview
#
# The implementation combines deep learning with physical constraints to create
# a hybrid model that:
# - Learns from observed infection data
# - Satisfies the SIR differential equations
# - Respects initial conditions
# - Provides uncertainty estimates
#
# The architecture uses a multi-layer perceptron (MLP) with custom activation
# functions and a novel loss function that balances data fitting with physical
# constraints. The loss function consists of three components:
# 1. PDE loss: Ensures the neural network satisfies the SIR differential equations
# 2. Initial condition loss: Enforces the correct initial values
# 3. Data loss: Fits the model to observed infection data
#
# ## Dependencies and Configuration
#
# The implementation leverages:
# - PyTorch for neural network operations
# - PyTorch Lightning for training orchestration
# - SciPy for ODE integration
# - Matplotlib and Seaborn for visualization
#
# Key features include:
# - Custom activation functions for better gradient flow
# - Adaptive learning rate scheduling
# - Early stopping to prevent overfitting
# - Comprehensive logging for monitoring training progress
# - TensorBoard integration for visualization

# %% [markdown]
# ## Environment setup
#
# Import the necessary libraries and set up the environment.

# %%
# std
import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

# third-party
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from lightning import Callback
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from scipy.integrate import odeint
from sklearn.metrics import mean_absolute_percentage_error
from torch.utils.data import Dataset, DataLoader

sns.set_theme(style="darkgrid")

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
tensorboard_dir = os.path.join(log_dir, "tensorboard")
os.makedirs(tensorboard_dir, exist_ok=True)
csv_dir = os.path.join(log_dir, "csv")
os.makedirs(csv_dir, exist_ok=True)
saved_models_dir = "./versions"
os.makedirs(saved_models_dir, exist_ok=True)
checkpoints_dir = "./checkpoints"

# %% [markdown]
# ## Module's Components
#
# The implementation consists of several key components:
#
# ### Data Structures
# - `SIRData`: A dataclass to store SIR compartment values (S, I, R)
# - `SIRConfig`: Configuration class for model and training parameters
#
# ### Neural Network Components
# - `Square`: Custom activation function for element-wise squaring
# - `create_mlp`: Utility function to create multi-layer perceptrons
# - `activation_map`: Dictionary of available activation functions
#
# ### Evaluation and Visualization
# - `evaluate_sir`: Function to compute various error metrics
# - `plot_sir_dynamics`: Function to visualize SIR trajectories
# - `print_metrics`: Utility to display metrics in tabular format
#
# ### Training Components
# - `SIRDataset`: Custom dataset class for training data
# - `SIRPINN`: Main PINN model class
# - Custom callbacks for training monitoring and early stopping
#
# Each component is designed to work together to solve the inverse problem
# of estimating the infection rate parameter while respecting the physical
# constraints of the SIR model.


# %%
@dataclass
class SIRData:
    """Data structure for SIR model compartments."""

    s: np.ndarray
    i: np.ndarray
    r: np.ndarray


class Square(nn.Module):
    """A module that squares its input element-wise."""

    @staticmethod
    def forward(x):
        return torch.square(x)


def create_mlp(layers_dims, activation, output_activation):
    """Create a multi-layer perceptron with specified architecture.

    Args:
        layers_dims: List of integers specifying the number of neurons in each layer
        activation: Activation function to use between layers
        output_activation: Activation function to use for the output layer

    Returns:
        A PyTorch Sequential model with the specified architecture
    """
    layers = []
    for i in range(len(layers_dims) - 1):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        if i < len(layers_dims) - 2:
            layers.append(activation)
    layers.append(output_activation)

    net = nn.Sequential(*layers)

    for layer in net:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    return net


# Dictionary mapping activation function names to their PyTorch implementations
activation_map = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "square": Square(),
    "softplus": nn.Softplus(),
    "identity": nn.Identity(),
}


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    return np.mean((pred - true) ** 2).item()


def re(pred: np.ndarray, true: np.ndarray) -> float:
    return np.linalg.norm(true - pred, 2).item() / np.linalg.norm(true, 2).item()


def mape(pred: np.ndarray, true: np.ndarray) -> float:
    return float(mean_absolute_percentage_error(true, pred))


def evaluate_sir(
    t: np.ndarray,
    sir_true: SIRData,
    beta_true: float,
    predictions: List[Tuple[str, SIRData, float]],
) -> List[Tuple[str, Dict[str, float]]]:
    """Evaluate SIR model predictions against ground truth data.

    Args:
        t: Array of time points
        sir_true: Ground truth SIR data
        beta_true: True value of the infection rate parameter
        predictions: List of tuples containing (model_name, predicted_sir_data, predicted_beta)

    Returns:
        List of tuples containing (model_name, metrics_dict) where metrics_dict includes:
        - MSE, MAPE, and relative error for each compartment (S, I, R)
        - Beta prediction error and percentage error
    """
    version_metrics = []

    for name, sir_pred, beta_pred in predictions:
        metrics = {}

        for comp, pred, true in zip(
            ["S", "I", "R"],
            [sir_pred.s, sir_pred.i, sir_pred.r],
            [sir_true.s, sir_true.i, sir_true.r],
        ):
            metrics[f"{comp}_mse"] = mse(pred, true)
            metrics[f"{comp}_mape"] = mape(pred, true)
            metrics[f"{comp}_re"] = re(pred, true)

        beta_error = abs(beta_pred - beta_true)
        beta_error_percent = beta_error / beta_true * 100
        metrics["beta_true"] = beta_true
        metrics["beta_pred"] = beta_pred
        metrics["beta_error"] = beta_error
        metrics["beta_error_percent"] = beta_error_percent

        version_metrics.append((name, metrics))

    return version_metrics


def plot_sir_dynamics(
    t: np.ndarray,
    sir_true: SIRData,
    predictions: List[Tuple[str, SIRData, float]],
) -> Figure:
    """Create visualization of SIR dynamics.

    Args:
        t: Time points
        sir_true: True SIR values
        predictions: List of tuples containing (name, predicted SIR values, predicted beta)

    Returns:
        Matplotlib figure with the visualization
    """
    fig = plt.figure(figsize=(12, 6))

    color_map = plt.colormaps.get_cmap("viridis")
    color_idx = np.random.rand()
    color = color_map(color_idx)
    sns.lineplot(x=t, y=sir_true.s, label="$S_{\\mathrm{true}}$", color=color)
    sns.lineplot(x=t, y=sir_true.i, label="$I_{\\mathrm{true}}$", color=color)
    sns.lineplot(x=t, y=sir_true.r, label="$R_{\\mathrm{true}}$", color=color)

    # Plot predictions
    for i, (name, sir_pred, _) in enumerate(predictions):
        subscript = f"_{{{name}}}" if len(predictions) > 1 else "_pred"
        new_color_idx = (color_idx + (i + 1) / (len(predictions) + 1)) % 1
        color = color_map(new_color_idx)

        sns.lineplot(
            x=t, y=sir_pred.s, label=f"$S{subscript}$", linestyle="--", color=color
        )
        sns.lineplot(
            x=t, y=sir_pred.i, label=f"$I{subscript}$", linestyle="--", color=color
        )
        sns.lineplot(
            x=t, y=sir_pred.r, label=f"$R{subscript}$", linestyle="--", color=color
        )

    plt.title("True vs Predicted SIR Dynamics")
    plt.xlabel("Time (days)")
    plt.ylabel("Fraction of Population")
    plt.legend()
    plt.tight_layout()

    return fig


def print_metrics(version_metrics: List[Tuple[str, Dict[str, float]]]):
    """Print evaluation metrics in a formatted table.

    Args:
        version_metrics: List of tuples containing (model_name, metrics_dict) where
            metrics_dict contains various error metrics for SIR compartments and beta
    """
    if not version_metrics:
        print("No metrics to display.")
        return

    metric_names = []
    for m in ["mse", "mape", "re"]:
        metric_names.extend([f"S_{m}", f"I_{m}", f"R_{m}"])
    metric_names.extend(["beta_pred", "beta_true", "beta_error", "beta_error_percent"])

    metric_name_width = max(len(name) for name in metric_names)
    metric_name_width = max(metric_name_width, 6)  # len("metric")

    values_width = max(len(name) for name, _ in version_metrics)
    values_width = max(values_width, 9)  # len("1.23e+05%")

    header = f"| {'metric':<{metric_name_width}} |"
    subheader = f"| {'-' * metric_name_width} |"
    for name, _ in version_metrics:
        header += f" {name:^{values_width}} |"
        subheader += f" {'-' * (values_width)} |"
    print(header)
    print(subheader)

    for metric in metric_names:
        row = f"| {metric:<{metric_name_width}} |"
        for _, metrics in version_metrics:
            value = metrics.get(metric)
            formatted_value = ""
            if value is None:
                formatted_value = " N/A"
            elif "_mape" in metric:
                formatted_value = f"{value:.1e}%"
            elif "_error_percent" in metric:
                formatted_value = f"{value:.5f}%"
            elif "_error" in metric or "_mse" in metric or "_re" in metric:
                formatted_value = f"{value:.2e}"
            else:
                formatted_value = f"{value:.6f}"

            row += f" {formatted_value:>{values_width}} |"
        print(row)


# %% [markdown]
# ## Module's configuration
#
# Define the configuration dictionary for the module.


# %%
@dataclass
class SIRConfig:
    """Configuration for SIR PINN model and training."""

    # Model parameters
    N: float = 56e6
    delta: float = 1 / 5
    r0: float = 3.0
    beta_true: float = delta * r0
    initial_beta: float = 0.5

    # Dataset parameters
    time_domain: Tuple[int, int] = (0, 90)
    collocation_points: int = 6000

    # Initial conditions (I0, R0)
    initial_conditions: List[float] = field(default_factory=lambda: [1.0, 0.0])

    # Network architecture
    hidden_layers: List[int] = field(default_factory=lambda: 4 * [50])
    activation: str = "tanh"
    output_activation: str = "square"

    # Loss weights
    pde_weight: float = 1.0
    ic_weight: float = 1.0
    data_weight: float = 1.0

    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 100
    max_epochs: int = 1000
    gradient_clip_val: float = 0.1

    # Scheduler parameters
    scheduler_factor: float = 0.5
    scheduler_patience: int = 65
    scheduler_threshold: float = 5e-3
    scheduler_min_lr: float = 1e-6

    # Early stopping
    early_stopping_enabled: bool = False
    early_stopping_patience: int = 100

    # SMMA stopping
    smma_stopping_enabled: bool = False
    smma_window: int = 50
    smma_threshold: float = 0.1
    smma_lookback: int = 50


# %% [markdown]
# ## Synthetic Data Generation
#
# Since real epidemiological data may be limited or noisy, we generate synthetic
# data using numerical integration of the SIR ODEs. This approach allows us to:
#
# 1. Control the ground truth parameters (e.g., $\beta$)
# 2. Generate noise-free data for validation
# 3. Add controlled noise to simulate real-world conditions
#
# The data generation process:
# 1. Solves the SIR ODEs using SciPy's `odeint`
# 2. Adds Poisson noise to the infected compartment to simulate real-world
#    counting processes
# 3. Returns time points, true SIR values, and noisy observations
#
# This synthetic data serves as both training data and ground truth for
# evaluating the model's performance.


def generate_sir_data(config: SIRConfig) -> Tuple[np.ndarray, SIRData, np.ndarray]:
    """Generate synthetic SIR data using ODE integration."""

    def sir(x, _, d, b):
        s, i, _ = x
        l = b * i / config.N
        ds_dt = -l * s
        di_dt = l * s - d * i
        dr_dt = d * i
        return np.array([ds_dt, di_dt, dr_dt])

    i0, r0 = config.initial_conditions
    t_start, t_end = config.time_domain
    t = np.linspace(t_start, t_end, t_end - t_start + 1)

    solution = odeint(
        sir, [config.N - i0 - r0, i0, r0], t, args=(config.delta, config.beta_true)
    )
    sir_true = SIRData(*solution.T)
    i_obs = np.random.poisson(sir_true.i)

    return t, sir_true, i_obs


# %% [markdown]
# ## Dataset Creation
#
# The `SIRDataset` class combines observed data with collocation points to
# create a comprehensive training dataset. Key features:
#
# ### Data Components
# - **Observation Points**: Time points where we have actual infection data
# - **Collocation Points**: Randomly sampled points where we enforce the
#   physical constraints (PDEs)
#
# ### Data Processing
# 1. Normalizes the infected population to [0, 1]
# 2. Generates collocation points using exponential sampling for better
#    coverage of the time domain
# 3. Combines observation and collocation points into a single dataset
#
# ### Batch Structure
# Each batch contains:
# - Time points (`t`)
# - Observation flags (`is_obs`)
# - Target values (`i_target`)
#
# This structure allows the model to:
# - Fit observed data where available
# - Enforce physical constraints everywhere
# - Handle missing data gracefully


# %%
class SIRDataset(Dataset):
    """Dataset for SIR PINN training."""

    def __init__(
        self,
        t_obs: np.ndarray,
        i_obs: np.ndarray,
        time_domain: Tuple[float, float],
        n_collocation: int,
        N: float,
    ):
        """
        Initialize dataset with observation points and random collocation points.
        The infected population is normalized to be in the range [0, 1].

        Args:
            t_obs: Observation time points
            i_obs: Observed infected population at each time point
            time_domain: (t_min, t_max) time range
            n_collocation: Number of random collocation points to generate
        """
        t_min, t_max = time_domain
        self.t_obs = torch.tensor(t_obs, dtype=torch.float32).reshape(-1, 1)

        i_norm = i_obs / N
        self.i_obs = torch.tensor(i_norm, dtype=torch.float32).reshape(-1, 1)

        t_rand = np.expm1(
            np.random.uniform(np.log1p(t_min), np.log1p(t_max), n_collocation)
        )
        self.t_collocation = torch.tensor(t_rand, dtype=torch.float32).reshape(-1, 1)

        self.t_combined = torch.cat([self.t_obs, self.t_collocation], dim=0)

        self.is_obs = torch.zeros(len(self.t_combined), dtype=torch.bool)
        self.is_obs[: len(self.t_obs)] = True

        self.i_targets = torch.zeros(len(self.t_combined), 1, dtype=torch.float32)
        self.i_targets[: len(self.t_obs)] = self.i_obs

    def __len__(self):
        return len(self.t_combined)

    def __getitem__(self, idx):
        return {
            "t": self.t_combined[idx],
            "is_obs": self.is_obs[idx],
            "i_target": self.i_targets[idx],
        }


# %% [markdown]
# ## Module Definition
#
# The `SIRPINN` class implements the core Physics-Informed Neural Network
# for the SIR model. Key aspects:
#
# ### Network Architecture
# - Two separate MLPs for S and I compartments
# - R compartment computed as R = N - S - I
# - Custom activation functions for better gradient flow
# - Learnable infection rate parameter $\beta$
#
# ### Loss Components
# 1. **PDE Loss**: Ensures the network satisfies the SIR differential equations
#    - Computes derivatives using automatic differentiation
#    - Evaluates residuals at collocation points
#
# 2. **Initial Condition Loss**: Enforces correct starting values
#    - Computes error at t = 0
#    - Ensures physical consistency
#
# 3. **Data Loss**: Fits the model to observed infection data
#    - Only evaluated at observation points
#    - Handles missing data gracefully
#
# ### Training Features
# - Adaptive learning rate scheduling
# - Gradient clipping for stability
# - Comprehensive logging
# - Early stopping based on SMMA of loss
#
# The implementation uses PyTorch Lightning for efficient training
# orchestration and monitoring.


# %%
class SIRPINN(LightningModule):
    """Physics-Informed Neural Network for SIR model parameter identification."""

    def __init__(self, config: SIRConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        layers_dims = [1] + config.hidden_layers + [1]
        activation = activation_map.get(config.activation)
        output_activation = activation_map.get(config.output_activation)

        self.net_S = create_mlp(layers_dims, activation, output_activation)
        self.net_I = create_mlp(layers_dims, activation, output_activation)

        self.beta = nn.Parameter(torch.tensor(config.initial_beta, dtype=torch.float32))

        self.N = 1.0
        self.delta = config.delta

        self.loss_fn = nn.MSELoss()

        self.t0_tensor = torch.zeros(1, 1, device=self.device, dtype=torch.float32)
        i0, r0 = map(lambda x: x / self.config.N, self.config.initial_conditions)
        ic = [self.N - i0 - r0, i0, r0]
        self.ic_true = torch.tensor(ic, dtype=torch.float32).reshape(1, 3)

        self.loss_buffer = []
        self.smma = None

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute S, I, R values at time t.

        Args:
            t: Time points tensor of shape [batch_size, 1]

        Returns:
            Tensor of shape [batch_size, 3] with [S, I, R] values
        """
        S = self.net_S(t)
        I = self.net_I(t)
        R = self.N - S - I

        return torch.cat([S, I, R], dim=1)

    @torch.inference_mode(False)
    def _compute_ode_residuals(
        self, t_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute residuals of the SIR ODEs using automatic differentiation.

        Args:
            t: Time points tensor of shape [batch_size, 1]

        Returns:
            Tuple of residual tensors (res_S, res_I)
        """
        t_tensor.requires_grad_(True)
        S = self.net_S(t_tensor)
        I = self.net_I(t_tensor)

        dS_dt = torch.autograd.grad(
            S, t_tensor, grad_outputs=torch.ones_like(S), create_graph=True
        )[0]
        dI_dt = torch.autograd.grad(
            I, t_tensor, grad_outputs=torch.ones_like(I), create_graph=True
        )[0]

        res_S = dS_dt + self.beta * S * I
        res_I = dI_dt - self.beta * S * I + self.delta * I

        return res_S, res_I

    def _compute_pde_loss(self, t: torch.Tensor) -> torch.Tensor:
        """Compute PDE residual loss."""
        res_S, res_I = self._compute_ode_residuals(t)
        loss_S = self.loss_fn(res_S, torch.zeros_like(res_S))
        loss_I = self.loss_fn(res_I, torch.zeros_like(res_I))

        return loss_S + loss_I

    def _compute_ic_loss(self) -> torch.Tensor:
        """Compute initial condition loss."""
        t0_tensor = self.t0_tensor.to(self.device)
        ic_true = self.ic_true.to(self.device)
        ic_pred = self(t0_tensor)

        return self.loss_fn(ic_pred, ic_true)

    def _compute_data_loss(
        self, t_obs: torch.Tensor, i_obs: torch.Tensor
    ) -> torch.Tensor:
        """Compute data fitting loss."""
        if t_obs.shape[0] == 0:  # No observations in batch
            return torch.tensor(0.0, device=self.device)

        i_pred = self(t_obs)[:, 1].reshape(-1, 1)
        return self.loss_fn(i_pred, i_obs)

    def training_step(self, batch):
        t = batch["t"]
        is_obs = batch["is_obs"]
        i_target = batch["i_target"]

        t_obs = t[is_obs] if is_obs.any() else torch.zeros((0, 1), device=self.device)
        i_obs = (
            i_target[is_obs]
            if is_obs.any()
            else torch.zeros((0, 1), device=self.device)
        )

        pde_loss_val = self._compute_pde_loss(t)
        ic_loss_val = self._compute_ic_loss()
        data_loss_val = self._compute_data_loss(t_obs, i_obs)

        total_loss = (
            self.config.pde_weight * pde_loss_val
            + self.config.ic_weight * ic_loss_val
            + self.config.data_weight * data_loss_val
        )

        self.log("train/pde_loss", pde_loss_val, on_epoch=True, on_step=False)
        self.log("train/ic_loss", ic_loss_val, on_epoch=True, on_step=False)
        self.log("train/data_loss", data_loss_val, on_epoch=True, on_step=False)
        self.log(
            "train/total_loss", total_loss, on_epoch=True, on_step=False, prog_bar=True
        )
        self.log(
            "train/beta", self.beta.item(), on_epoch=True, on_step=False, prog_bar=True
        )

        return total_loss

    def on_train_epoch_end(self):
        """At the end of each epoch: calculate and log SMMA of total loss."""
        loss = self.trainer.callback_metrics.get("train/total_loss")
        if loss is not None:
            loss = loss.item()
            n = self.config.smma_window

            if self.smma is None:
                self.loss_buffer.append(loss)
                if len(self.loss_buffer) == n:
                    self.smma = sum(self.loss_buffer) / n

            else:
                self.smma = ((n - 1) * self.smma + loss) / n
                self.log("train/total_loss_smma", self.smma)

    @torch.no_grad()
    def predict_sir(self, t):
        """Predict SIR values at specified time points."""
        t_tensor = torch.tensor(t, dtype=torch.float32).reshape(-1, 1).to(self.device)
        return self(t_tensor).cpu().numpy() * self.config.N

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
            threshold=self.config.scheduler_threshold,
            min_lr=self.config.scheduler_min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/total_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


# %% [markdown]
# ## Custom callbacks


# %%
class ProgressBar(TQDMProgressBar):
    """Custom progress bar for training that formats metrics for better readability.

    This class extends the TQDMProgressBar to provide custom formatting for
    training metrics, particularly for the total loss and beta values.
    """
    def get_metrics(self, *args, **kwargs):
        """Format metrics for display in the progress bar.

        Returns:
            Dictionary of formatted metrics with:
            - Total loss in scientific notation
            - Beta value with 4 decimal places
            - Other metrics as provided by the parent class
        """
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        if "train/total_loss" in items:
            items["train/total_loss"] = f"{items['train/total_loss']:.2e}"
        if "train/beta" in items:
            items["train/beta"] = f"{items['train/beta']:.4f}"
        return items


class SMMAStopping(Callback):
    """Early stopping callback based on the Smoothed Moving Average (SMMA) of the loss.

    This callback monitors the improvement in the SMMA of the total loss over a specified
    lookback period. Training is stopped if the improvement falls below a threshold.
    """
    def __init__(self, threshold: float, lookback: int):
        """Initialize the SMMA stopping callback.

        Args:
            threshold: Minimum required improvement in SMMA (as a fraction)
            lookback: Number of epochs to look back for computing improvement
        """
        super().__init__()
        self.threshold = threshold
        self.lookback = lookback
        self.smma_buffer = []

    def on_train_epoch_end(self, trainer: Trainer, module: SIRPINN):
        """Check if training should be stopped based on SMMA improvement.

        Args:
            trainer: The PyTorch Lightning trainer
            module: The SIRPINN model being trained
        """
        current_smma = trainer.callback_metrics.get("train/total_loss_smma")
        if current_smma is None:
            return

        self.smma_buffer.append(current_smma)
        if len(self.smma_buffer) <= self.lookback:
            return

        if len(self.smma_buffer) > self.lookback + 1:
            self.smma_buffer.pop(0)

        lookback_smma = self.smma_buffer[0]
        improvement = lookback_smma - current_smma
        improvement_percentage = improvement / lookback_smma

        if 0 < improvement_percentage < self.threshold:
            trainer.should_stop = True
            print(
                f"\nStopping training: SMMA improvement over {self.lookback} epochs ({improvement_percentage:.2%}) below threshold ({self.threshold:.2%})"
            )

        module.log("internal/smma_improvement", improvement_percentage)
        return


class SIREvaluation(Callback):
    """Callback for evaluating and visualizing SIR model predictions.

    This callback generates plots of the SIR dynamics and logs metrics to TensorBoard
    at the end of training. It compares the model's predictions against the ground truth
    data and computes various error metrics.
    """
    def __init__(self, t: np.ndarray, sir_true: SIRData):
        """Initialize the evaluation callback.

        Args:
            t: Array of time points
            sir_true: Ground truth SIR data
        """
        super().__init__()
        self.t = t
        self.sir_true = sir_true

    def on_train_end(self, trainer: Trainer, module: SIRPINN):
        """Generate evaluation plots and log metrics to TensorBoard.

        Args:
            trainer: The PyTorch Lightning trainer
            module: The SIRPINN model being trained
        """
        tb_logger = None
        for logger in trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_logger = logger.experiment
                break
        if tb_logger is None:
            raise ValueError("TensorBoard logger not found")

        sir_pred = SIRData(*module.predict_sir(self.t).T)
        beta_pred = module.beta.item()
        predictions = [("", sir_pred, beta_pred)]

        [(_, metrics)] = evaluate_sir(
            self.t,
            self.sir_true,
            module.config.beta_true,
            predictions,
        )
        for metric_name, metric_value in metrics.items():
            tb_logger.add_scalar(
                f"metrics/{metric_name}", metric_value, trainer.global_step
            )

        fig = plot_sir_dynamics(self.t, self.sir_true, predictions)
        tb_logger.add_figure("sir_dynamics", fig, global_step=trainer.global_step)
        plt.close(fig)


# %% [markdown]
# ## Execution
#
# The main execution block provides a flexible interface for training and
# evaluating the SIR PINN model. Key features:
#
# ### Command Line Interface
# - `--skip`: Skip training and load a saved model
# - `--version`: Specify model version(s) to load for evaluation
#
# ### Training Pipeline
# 1. Generate synthetic training data
# 2. Create and configure the dataset
# 3. Initialize the model and training components
# 4. Set up logging and callbacks
# 5. Train the model
# 6. Save the best model
#
# ### Evaluation Pipeline
# 1. Load specified model version(s)
# 2. Generate test data
# 3. Compute predictions
# 4. Calculate and display metrics
# 5. Generate visualization plots
#
# The implementation includes comprehensive logging to TensorBoard and CSV
# files for monitoring training progress and model performance.


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--skip", action="store_true", help="Skip training and load saved model"
    )
    parser.add_argument(
        "-v",
        "--version",
        type=int,
        nargs="+",
        help="Version number(s) of the model(s) to load for evaluation",
    )
    args = parser.parse_args()
    skip_training = args.skip
    load_versions = args.version

    if load_versions is not None or skip_training:
        latest_version = len(os.listdir(saved_models_dir)) - 1
        if latest_version < 0:
            raise FileNotFoundError("No saved models found")

        versions = load_versions if load_versions is not None else [latest_version]
        predictions = []

        for version in versions:
            model_path = saved_models_dir + f"/version_{version}.ckpt"

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model version {version} not found")

            model = SIRPINN.load_from_checkpoint(model_path)

            t, sir_true, i_obs = generate_sir_data(model.config)
            sir_pred = SIRData(*model.predict_sir(t).T)

            predictions.append((f"version_{version}", sir_pred, model.beta.item()))

        metrics = evaluate_sir(t, sir_true, model.config.beta_true, predictions)
        fig = plot_sir_dynamics(t, sir_true, predictions)

        print_metrics(metrics)
        plt.show()
        plt.close(fig)
        exit()

    subprocess.Popen(
        ["tensorboard", "--logdir", tensorboard_dir],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    config = SIRConfig(
        # Dataset parameters
        # collocation_points=8000,
        # Network architecture
        # hidden_layers=[64, 128, 128, 64],
        # output_activation="softplus",
        # Loss weights
        # pde_weight=10.0,
        # ic_weight=5.0,
        # data_weight=1.0,
        # Training parameters
        # batch_size=256,
        # Early stopping
        early_stopping_enabled=True,
        # SMMA stopping
        smma_stopping_enabled=True,
    )

    t, sir_true, i_obs = generate_sir_data(config)

    dataset = SIRDataset(
        t_obs=t,
        i_obs=i_obs,
        time_domain=config.time_domain,
        n_collocation=config.collocation_points,
        N=config.N,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=7,
        persistent_workers=True,
    )

    model = SIRPINN(config)

    if os.path.exists(checkpoints_dir):
        shutil.rmtree(checkpoints_dir)
    os.makedirs(checkpoints_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="{epoch:02d}",
        save_top_k=1,
        monitor="train/total_loss",
        mode="min",
        save_last=True,
    )

    callbacks: list[Callback] = [
        checkpoint_callback,
        LearningRateMonitor(
            logging_interval="epoch",
        ),
        ProgressBar(
            refresh_rate=10,
        ),
        SIREvaluation(
            t,
            sir_true,
        ),
    ]

    if config.early_stopping_enabled:
        callbacks.append(
            EarlyStopping(
                monitor="train/total_loss",
                patience=config.early_stopping_patience,
                check_on_train_epoch_end=True,
                mode="min",
            ),
        )

    if config.smma_stopping_enabled:
        callbacks.append(
            SMMAStopping(
                config.smma_threshold,
                config.smma_lookback,
            ),
        )

    loggers = [
        TensorBoardLogger(
            save_dir=tensorboard_dir,
            name="ablation_study",
            default_hp_metric=False,
        ),
        CSVLogger(
            save_dir=csv_dir,
            name="ablation_study",
        ),
    ]

    trainer = Trainer(
        max_epochs=config.max_epochs,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=1,  # ignored by the on_epoch=True
        gradient_clip_val=config.gradient_clip_val,
    )

    trainer.fit(model, data_loader)

    model = SIRPINN.load_from_checkpoint(checkpoint_callback.best_model_path)
    version = len(os.listdir(saved_models_dir))
    model_path = saved_models_dir + f"/version_{version}.ckpt"
    trainer.save_checkpoint(model_path)

    if os.path.exists(checkpoints_dir):
        shutil.rmtree(checkpoints_dir)
