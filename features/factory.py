# Trimmed vendor of srs-benchmark's features/factory.py.
# The RWKV data pipeline always builds FSRS-style (t_history, r_history) features
# (ALGO is "FSRS-6"/"FSRS-7"), so only FSRSFeatureEngineer is kept. The other
# engineers and the models/ package were removed to keep this repo RWKV-only.
from .base import BaseFeatureEngineer
from .fsrs_engineer import FSRSFeatureEngineer
from config import Config, ModelName
from typing import Type, get_args


# All FSRS-family / standard-tensor models map to the same FSRSFeatureEngineer.
FEATURE_ENGINEER_REGISTRY: dict[ModelName, Type[BaseFeatureEngineer]] = {
    "FSRSv1": FSRSFeatureEngineer,
    "FSRSv2": FSRSFeatureEngineer,
    "FSRSv3": FSRSFeatureEngineer,
    "FSRSv4": FSRSFeatureEngineer,
    "FSRS-4.5": FSRSFeatureEngineer,
    "FSRS-5": FSRSFeatureEngineer,
    "FSRS-6": FSRSFeatureEngineer,
    "FSRS-7": FSRSFeatureEngineer,
    "FSRS-rs": FSRSFeatureEngineer,
    "RNN": FSRSFeatureEngineer,
    "Transformer": FSRSFeatureEngineer,
    "SM2-trainable": FSRSFeatureEngineer,
    "Anki": FSRSFeatureEngineer,
    "90%": FSRSFeatureEngineer,
}


def create_feature_engineer(config: Config) -> BaseFeatureEngineer:
    """
    Factory function to create the appropriate feature engineer based on model name.

    Raises:
        ValueError: If config.model_name is not supported by this trimmed registry.
    """
    model_name = config.model_name
    if model_name not in FEATURE_ENGINEER_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' is not supported in this RWKV-only vendor. "
            f"Supported: {tuple(FEATURE_ENGINEER_REGISTRY)}"
        )
    feature_engineer_cls = FEATURE_ENGINEER_REGISTRY[model_name]
    return feature_engineer_cls(config)


def get_supported_models() -> tuple[str, ...]:
    return get_args(ModelName)
