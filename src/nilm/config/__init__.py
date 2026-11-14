"""
Configi projektu: dataclasses jako schemat + pliki YAML w configs/.

Importujemy z nilm.config, a nie z podmodułów; podział na pliki jest szczegółem implementacji.
"""

from nilm.config.experiment import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
    WandbConfig,
    WandbMode,
    load_config,
)
from nilm.config.preprocess import PreprocessConfig, load_preprocess_config

__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "ModelConfig",
    "PreprocessConfig",
    "TrainConfig",
    "WandbConfig",
    "WandbMode",
    "load_config",
    "load_preprocess_config",
]
