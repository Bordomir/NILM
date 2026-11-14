"""Config eksperymentu: obróbka danych, dane, model, trening i W&B; pliki YAML w configs/."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from nilm.config.loading import read_yaml, update_dataclass
from nilm.config.preprocess import PreprocessConfig

WandbMode = Literal["online", "offline", "disabled"]


@dataclass
class DataConfig:
    window_size: int = 100
    appliances: list[str] = field(default_factory=list)
    train_houses: list[str] = field(default_factory=list)
    val_houses: list[str] = field(default_factory=list)
    test_houses: list[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    name: str = "cnn1d"


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-3


@dataclass
class WandbConfig:
    # Wspólne miejsce logowania dla całego zespołu - nie zmieniać lokalnie
    entity: str = "DL_NILM"
    project: str = "nilm"
    # online | offline | disabled (zmienna środowiskowa WANDB_MODE ma pierwszeństwo)
    mode: WandbMode = "online"
    # debug=True loguje do osobnego projektu "<project>-dev", żeby nie zaśmiecać głównego
    debug: bool = False
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str | None = None


@dataclass
class ExperimentConfig:
    name: str | None = None
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    cfg = ExperimentConfig()
    update_dataclass(cfg, read_yaml(path))
    return cfg
