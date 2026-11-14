"""Config warstwy 1 obróbki danych (surowe CSV -> oczyszczone Parquet)."""

from dataclasses import dataclass
from pathlib import Path

from nilm.config.loading import read_yaml, update_dataclass


@dataclass
class PreprocessConfig:
    """
    Parametry warstwy 1, z których liczony jest hash wyniku.

    Zasady:
    - tylko pola, które zmieniają zawartość plików wyjściowych; ścieżki i opcje wykonania
      (np. liczba procesów) przekazujemy osobno, bo inaczej zmieniałyby hash bez powodu,
    - nowe pole dodajemy razem z etapem w steps.py, który go używa,
      a wartość domyślną uzasadniamy w notebooku z analizą danych.
    """


def load_preprocess_config(path: str | Path | None = None) -> PreprocessConfig:
    """Bez ścieżki zwraca config z wartościami domyślnymi."""
    cfg = PreprocessConfig()
    if path is not None:
        update_dataclass(cfg, read_yaml(path))
    return cfg
