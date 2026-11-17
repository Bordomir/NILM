"""
Moduł do etapów. Tutaj umieszczamy wszystkie funkcje do czyszczenia danych.
Każda funkcja ma mieć sygnaturę StepFn, a do pipeline'u trafia opakowana w Step z nazwą.

"""

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from nilm.config.preprocess import PreprocessConfig
from nilm.data.preprocessing.io import SEGMENT_COL

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


def split_segments(df: pd.DataFrame, cfg: PreprocessConfig) -> tuple[pd.DataFrame, dict]:
    """
    Numeruje ciągłe fragmenty danych (kolumna "segment").

    W REFIT brakujące pomiary to brakujące wiersze, więc dwa sąsiednie wiersze mogą dzielić
    kilka sekund albo kilka dni. Tam, gdzie odstęp przekracza cfg.max_gap_s, zaczyna się
    nowy segment. Dataset nie tworzy okien przechodzących przez granicę segmentu.
    Żaden wiersz nie jest usuwany.
    """
    gaps = df.index.to_series().diff()
    is_new_segment = gaps > pd.Timedelta(seconds=cfg.max_gap_s)

    # cumsum: każda dziura zwiększa numer o 1, więc numery rosną 0, 0, 0, 1, 1, 2, ...
    df = df.copy()
    df[SEGMENT_COL] = is_new_segment.cumsum().to_numpy().astype("int32")

    gap_hours = gaps[is_new_segment].dt.total_seconds() / 3600
    report = {
        "segments": int(df[SEGMENT_COL].iloc[-1]) + 1 if len(df) else 0,
        "gaps_total_hours": round(float(gap_hours.sum()), 1),
        "longest_gap_hours": round(float(gap_hours.max()), 1) if len(gap_hours) else 0.0,
    }
    return df, report
