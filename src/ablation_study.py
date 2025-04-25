import argparse
import json
import os
import shutil
from dataclasses import dataclass, replace
from typing import Any, Dict, List

from sir_pinn import SIRConfig, train


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

    # Base configuration that will be modified for each test
    stopping_base_config = SIRConfig(
        name="stopping",
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
                for window in [25, 50, 75]
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

    # TODO: replace with best stopping config
    activation_base_config = replace(
        stopping_base_config,
        name="activation",
        early_stopping_enabled=True,
        early_stopping_patience=50,
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

    # TODO: replace with best activation config
    architecture_base_config = replace(
        activation_base_config,
        name="architecture",
        output_activation="softplus",
    )

    variations = []
    for num_layers in [2, 3, 4, 5]:
        for neurons in [16, 32, 50, 64]:
            hidden_layers = [neurons] * num_layers
            variations.append(
                {
                    "name": f"arch_l{num_layers}_n{neurons}",
                    "description": f"Architecture with {num_layers} layers and {neurons} neurons each",
                    "config_updates": {"hidden_layers": hidden_layers},
                }
            )

    architecture_study = AblationConfig(
        name="architecture",
        description="Evaluation of different network architectures...",
        base_config=architecture_base_config,
        variations=variations,
    )

    # TODO: replace with best architecture config
    batch_base_config = replace(
        architecture_base_config,
        name="batch_size",
        early_stopping_enabled=True,
        early_stopping_patience=50,
        output_activation="softplus",
    )

    batch_study = AblationConfig(
        name="batch_size",
        description="Evaluation of different batch sizes on top architectures...",
        base_config=batch_base_config,
        variations=[
            ConfigVariation(
                name=f"batch_{size}",
                description=f"Training with batch size {size}",
                config_updates={"batch_size": size},
            )
            for size in [100, 256, 512]
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

    study_dir = f"studies/{study.name}"
    os.makedirs(study_dir, exist_ok=True)

    with open(f"{study_dir}/config.json", "w") as f:
        json.dump(study.to_dict(), f, indent=2)

    print(study.description)

    for variation in study.variations:
        print(f"Variation: {variation.description}\n")

        config = replace(study.base_config, **variation.config_updates)

        trained_model_path, version = train(config)

        study_model_path = f"{study_dir}/v{version}_{variation.name}.ckpt"
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
