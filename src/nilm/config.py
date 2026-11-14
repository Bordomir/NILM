"""
Konfiguracja eksperymentów: dataclasses jako schemat + pliki YAML w configs/.

Wartości domyślne są tutaj, plik YAML nadpisuje tylko to, co podaje.
Nieznany klucz w YAML-u to błąd (literówka w nazwie parametru nie przejdzie po cichu).
"""

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

WandbMode = Literal["online", "offline", "disabled"]


@dataclass
class DataConfig:
    dataset_dir: str = "data"
    window_size: int = 100
    appliances: list[str] = field(default_factory=list)
    # Stały podział domów — musi być wspólny dla całego zespołu, inaczej wyniki są nieporównywalne
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
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _update_dataclass(obj: Any, values: dict[str, Any], prefix: str = "") -> None:
    known = {f.name for f in fields(obj)}
    for key, value in values.items():
        if key not in known:
            raise KeyError(f"Nieznany klucz konfiguracji: '{prefix}{key}'")
        current = getattr(obj, key)
        if is_dataclass(current):
            _update_dataclass(current, value or {}, prefix=f"{prefix}{key}.")
        else:
            setattr(obj, key, value)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path) as f:
        values = yaml.safe_load(f) or {}
    cfg = ExperimentConfig()
    _update_dataclass(cfg, values)
    return cfg
