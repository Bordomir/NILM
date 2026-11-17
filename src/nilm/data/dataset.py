"""
Z danych po obróbce (steps w pipeline) budujemy datasety dla treningu.

Jedyne wejście dla treningu to build_datasets. Okna są w windows.py, normalizacja w normalization.py.
"""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import get_args

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import ConcatDataset

from nilm.config import DataConfig, TargetPosition
from nilm.data.normalization import Normalizer
from nilm.data.preprocessing.io import SEGMENT_COL, read_parquet
from nilm.data.preprocessing.metadata import APPLIANCES, HouseMeta, load_houses
from nilm.data.windows import NILMWindowDataset


def appliance_target(df: pd.DataFrame, house: HouseMeta, appliance: str) -> pd.Series:
    """
    Pobór danego typu urządzenia w W.

    - kilka kanałów tego samego typu (np. 2 pralki w House_4) sumujemy,
    - w okresach, gdy na kanale jest inne urządzenie (np. House_10 po zmianie lodówki
      na blender), wartością jest NaN; takie próbki nie trafiają do okien jako cel.
    """
    channels = house.channels_for(appliance)
    if not channels:
        raise ValueError(f"{house.name}: brak urządzenia '{appliance}'")

    columns = []
    for ch in channels:
        values = df[ch.column].astype("float32")
        for start, end, a in ch.appliance_periods():
            if a == appliance:
                continue
            mask = np.ones(len(df), dtype=bool)
            if start is not None:
                mask &= df.index >= pd.Timestamp(start, tz="UTC")
            if end is not None:
                mask &= df.index < pd.Timestamp(end, tz="UTC")
            values = values.mask(mask)
        columns.append(values)

    # skipna=False: jeśli któryś z sumowanych kanałów jest niepewny, cała suma też
    return pd.concat(columns, axis=1).sum(axis=1, skipna=False).rename(appliance)


@dataclass
class HouseArrays:
    """Dane jednego domu jako tablice numpy, przed normalizacją."""

    name: str
    aggregate: np.ndarray  # (N,) pobór całego domu w W
    targets: np.ndarray  # (N, liczba_urządzeń) pobór urządzeń w W; NaN = brak wiarygodnego celu
    segments: np.ndarray  # (N,) numer ciągłego fragmentu danych


def validate_split(cfg: DataConfig, houses: dict[str, HouseMeta]) -> None:
    """
    Sprawdza config danych, zanim zaczniemy wczytywać gigabajty.
    Zbiera wszystkie problemy naraz, żeby nie poprawiać ich po jednym.
    """
    problems = []

    if not cfg.appliances:
        problems.append("lista appliances jest pusta")
    unknown = sorted(set(cfg.appliances) - APPLIANCES)
    if unknown:
        problems.append(f"nieznane urządzenia {unknown} (dozwolone: APPLIANCES w metadata.py)")
    known_appliances = [a for a in cfg.appliances if a in APPLIANCES]

    if not cfg.train_houses:
        problems.append("lista train_houses jest pusta")

    # YAML przyjmie dowolny tekst i liczbę, więc wartości sprawdzamy sami
    if cfg.target_position not in get_args(TargetPosition):
        problems.append(f"target_position '{cfg.target_position}', dozwolone: {get_args(TargetPosition)}")
    for name in ("window_size", "stride", "eval_stride"):
        value = getattr(cfg, name)
        if not isinstance(value, int) or value < 1:
            problems.append(f"{name} musi być liczbą całkowitą >= 1, podano {value!r}")

    splits = {"train": cfg.train_houses, "val": cfg.val_houses, "test": cfg.test_houses}

    # Ten sam dom w dwóch zbiorach = wyciek danych testowych do treningu
    for (name_a, houses_a), (name_b, houses_b) in combinations(splits.items(), 2):
        common = sorted(set(houses_a) & set(houses_b))
        if common:
            problems.append(f"domy {common} są jednocześnie w {name_a} i {name_b}")

    for split_name, split_houses in splits.items():
        for house_name in split_houses:
            if house_name not in houses:
                problems.append(f"{split_name}: dom {house_name} nie istnieje w houses.yaml")
                continue
            missing = [a for a in known_appliances if not houses[house_name].channels_for(a)]
            if missing:
                problems.append(f"{split_name}: {house_name} nie ma urządzeń {missing}")

    if problems:
        raise ValueError("Błędny config danych:\n- " + "\n- ".join(problems))


def load_house_arrays(data_dir: Path, house: HouseMeta, appliances: list[str]) -> HouseArrays:
    """Wczytuje dom z parquetu i zamienia kanały na urządzenia."""
    path = data_dir / f"{house.name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Brak {path} - uruchom scripts/preprocess.py")

    # Wczytujemy tylko potrzebne kanały: 9 domów w całości nie zmieściłoby się wygodnie w RAM
    columns = ["Aggregate"]
    for appliance in appliances:
        for channel in house.channels_for(appliance):
            if channel.column not in columns:
                columns.append(channel.column)
    has_segments = SEGMENT_COL in pq.read_schema(path).names
    if has_segments:
        columns.append(SEGMENT_COL)

    df = read_parquet(path, columns=columns)

    targets = np.column_stack([appliance_target(df, house, a).to_numpy() for a in appliances])
    if has_segments:
        segments = df[SEGMENT_COL].to_numpy()
    else:
        # Dopóki warstwa 1 nie dzieli danych na segmenty, cały dom to jeden segment
        segments = np.zeros(len(df), dtype=np.int64)

    return HouseArrays(house.name, df["Aggregate"].to_numpy(), targets, segments)


def make_dataset(houses: list[HouseArrays], normalizer: Normalizer, cfg: DataConfig, stride: int) -> ConcatDataset:
    """Łączy okna z kilku domów w jeden dataset; dane są normalizowane."""
    datasets = []
    for house in houses:
        dataset = NILMWindowDataset(
            aggregate=normalizer.normalize_aggregate(house.aggregate),
            targets=normalizer.normalize_targets(house.targets),
            segments=house.segments,
            window_size=cfg.window_size,
            target_position=cfg.target_position,
            stride=stride,
        )
        datasets.append(dataset)
    return ConcatDataset(datasets)


@dataclass
class NILMDatasets:
    """Wynik build_datasets. val/test są None, gdy w configu nie podano domów."""

    train: ConcatDataset
    val: ConcatDataset | None
    test: ConcatDataset | None
    normalizer: Normalizer


def build_datasets(cfg: DataConfig, data_dir: str | Path) -> NILMDatasets:
    """
    Jedyny punkt wejścia do danych dla treningu.

    data_dir to katalog data/processed/<hash>/ zwrócony przez run_pipeline.
    """
    houses = load_houses()
    validate_split(cfg, houses)
    data_dir = Path(data_dir)

    def load(house_names: list[str]) -> list[HouseArrays]:
        return [load_house_arrays(data_dir, houses[name], cfg.appliances) for name in house_names]

    # 1. Statystyki normalizacji tylko z domów treningowych
    train_houses = load(cfg.train_houses)
    normalizer = Normalizer(
        aggregates=[h.aggregate for h in train_houses],
        targets=[h.targets for h in train_houses],
        appliances=cfg.appliances,
    )

    # 2. Datasety; val i test normalizujemy statystykami z treningu
    train = make_dataset(train_houses, normalizer, cfg, cfg.stride)
    val = make_dataset(load(cfg.val_houses), normalizer, cfg, cfg.eval_stride) if cfg.val_houses else None
    test = make_dataset(load(cfg.test_houses), normalizer, cfg, cfg.eval_stride) if cfg.test_houses else None

    return NILMDatasets(train=train, val=val, test=test, normalizer=normalizer)
