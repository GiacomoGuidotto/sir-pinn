import argparse
import json
import os
import shutil
from dataclasses import dataclass, replace
from typing import Any, Dict, List

from sir_pinn import SIRConfig, train

STUDIES_DIR = "./data/studies"

@dataclass
class ConfigVariation:
    """Configuration variation for an ablation study."""

    name: str
    description: str
    config_updates: Dict[str, Any]


@dataclass
class AblationConfig:
    """Configuration for an ablation study experiment."""

    name: str
    description: str
    base_config: SIRConfig
    variations: List[ConfigVariation]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "base_config": self.base_config.__dict__,
            "variations": [
                {
                    "name": v.name,
                    "description": v.description,
                    "config_updates": v.config_updates,
                }
                for v in self.variations
            ],
        }


def create_ablation_configs() -> Dict[str, AblationConfig]:
    """Create configurations for the ablation study."""

    stopping_base_config = SIRConfig(
        study_name="stopping",
    )

    stopping_study = AblationConfig(
        name="stopping",
        description="Evaluation of different stopping criteria approaches...",
        base_config=stopping_base_config,
        variations=[
            ConfigVariation(
                name="no_stopping",
                description="Baseline without any stopping criteria",
                config_updates={},
            ),
            *[
                ConfigVariation(
                    name=f"smma_w{window}",
                    description=f"SMMA stopping with window={window}, lookback={window}",
                    config_updates={
                        "smma_stopping_enabled": True,
                        "smma_window": window,
                        "smma_lookback": window,
                        "smma_threshold": 0.1,
                    },
                )
                for window in [25, 50, 75, 100]
            ],
            *[
                ConfigVariation(
                    name=f"early_stopping_p{patience}",
                    description=f"Early stopping with patience={patience}",
                    config_updates={
                        "early_stopping_enabled": True,
                        "early_stopping_patience": patience,
                    },
                )
                for patience in [25, 50, 75]
            ],
        ],
    )

    activation_base_config = replace(
        stopping_base_config,
        study_name="activation",
        smma_stopping_enabled=True,
        smma_window=50,
        smma_lookback=50,
        smma_threshold=0.1,
    )

    activation_study = AblationConfig(
        name="activation",
        description="Evaluation of different output activation functions...",
        base_config=activation_base_config,
        variations=[
            ConfigVariation(
                name=activation,
                description=f"Using {activation} as output activation",
                config_updates={"output_activation": activation},
            )
            for activation in ["square", "relu", "sigmoid", "softplus"]
        ],
    )

    architecture_base_config = replace(
        activation_base_config,
        study_name="architecture",
        output_activation="softplus",
    )

    architecture_study = AblationConfig(
        name="architecture",
        description="Evaluation of different network architectures...",
        base_config=architecture_base_config,
        variations=[
            ConfigVariation(
                name=f"arch_l{num_layers}_n{neurons}",
                description=f"Architecture: {num_layers} layers, {neurons} neurons each",
                config_updates={
                    "hidden_layers": [neurons] * num_layers,
                },
            )
            for neurons in [16, 32, 50, 64]
            for num_layers in [2, 3, 4, 5]
        ],
    )

    chosen_architectures = [
        # (3, 50),
        # (3, 64),
        (4, 50),
        (5, 16),
        # (5, 32),
        (5, 64),
    ]

    batch_base_config = replace(
        architecture_base_config,
        study_name="batch_size",
    )

    batch_study = AblationConfig(
        name="batch_size",
        description="Evaluation of different batch sizes on top architectures...",
        base_config=batch_base_config,
        variations=[
            ConfigVariation(
                name=f"batch_{size}_l{num_layers}_n{neurons}",
                description=f"Architecture: {num_layers} layers, {neurons} neurons each. Training with batch size {size}",
                config_updates={
                    "batch_size": size,
                    "hidden_layers": [neurons] * num_layers,
                },
            )
            for size in [100, 256, 512]
            for num_layers, neurons in chosen_architectures
        ],
    )

    return {
        stopping_study.name: stopping_study,
        activation_study.name: activation_study,
        architecture_study.name: architecture_study,
        batch_study.name: batch_study,
    }


def run_ablation_study(study_name: str):
    """Execute the ablation study and save results."""

    study = create_ablation_configs().get(study_name)

    if study is None:
        raise ValueError(f"Study {study_name} not found")

    study_dir = os.path.join(STUDIES_DIR, study.name)
    os.makedirs(study_dir, exist_ok=True)

    with open(os.path.join(study_dir, "config.json"), "w") as f:
        json.dump(study.to_dict(), f, indent=2)

    print(study.description)

    repetitions = 3
    for variation in study.variations:
        for run in range(repetitions):
            print(f"[{run + 1}/{repetitions}] Variation: {variation.description}\n")

            config = replace(study.base_config, **variation.config_updates)
            config.run_name = variation.name

            trained_model_path, version = train(config)

            print(f"Saved version: {version}\n")

            study_model_path = f"{study_dir}/{version}.ckpt"
            shutil.copy(trained_model_path, study_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SIR-PINN ablation study")
    parser.add_argument(
        "-s",
        "--study",
        type=str,
        choices=[
            "stopping",
            "activation",
            "architecture",
            "batch_size",
        ],
        required=True,
        help="Which study to run",
    )
    args = parser.parse_args()

    run_ablation_study(args.study)
