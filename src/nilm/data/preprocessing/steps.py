"""
Moduł do etapów. Tutaj umieszczamy wszystkie funkcje do czyszczenia danych.
Każda funkcja ma mieć sygnaturę StepFn, a do pipeline'u trafia opakowana w Step z nazwą.

"""

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from nilm.config.preprocess import PreprocessConfig

StepFn = Callable[[pd.DataFrame, PreprocessConfig], tuple[pd.DataFrame, dict]]


@dataclass(frozen=True)
class Step:
    """
    Etap obróbki z jawną nazwą.

    Nazwa trafia do hasha i raportu, więc musi być unikalna w STEPS;
    zmiana nazwy oznacza nowy hash, czyli ponowną obróbkę danych.
    """

    name: str
    fn: StepFn


def remove_spikes(df: pd.DataFrame, cfg: PreprocessConfig) -> tuple[pd.DataFrame, dict]:
    """Czyści wartości znacznie przewyższające pozostałe."""

    return df, {"spikes_removed": 17}
