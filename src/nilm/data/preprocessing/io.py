"""
Moduł do pobrania zbioru REFIT i transformacji danych csv na Parquet.

"""

from pathlib import Path

import kagglehub
import pandas as pd

KAGGLE_DATASET = "kyleahmurphy/uk-electrical-load"

APPLIANCE_COLS = [f"Appliance{i}" for i in range(1, 10)]
POWER_COLS = ["Aggregate", *APPLIANCE_COLS]


def download_refit(output_dir: str | Path = "data/raw") -> Path:
    """Pobiera surowe CSV zbioru REFIT z Kaggle do output_dir i zwraca ścieżkę do danych."""

    return Path(kagglehub.dataset_download(KAGGLE_DATASET, output_dir=str(output_dir)))


def read_house_csv(path: str | Path) -> pd.DataFrame:
    """Funkcja wczytująca dane do DataFrame dla pojedycznego domu."""

    df = pd.read_csv(
        path,
        usecols=["Unix", *POWER_COLS],
        dtype={"Unix": "int64", **dict.fromkeys(POWER_COLS, "int32")},
    )
    df["Unix"] = pd.to_datetime(df["Unix"], unit="s", utc=True)
    df = df.set_index("Unix")
    df.index.name = "time"


    return df


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Zapisuje DataFrame (razem z indeksem czasu) do pliku Parquet, tworząc brakujące katalogi."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Zapis do pliku tymczasowego i podmiana: przerwany zapis nie zostawi
    # niepełnego pliku, który cache pipeline'u uznałby za gotowy
    tmp_path = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp_path, index=True)
    tmp_path.replace(path)


def read_parquet(path: str | Path) -> pd.DataFrame:
    """Wczytuje DataFrame z pliku Parquet; indeks czasu i typy kolumn są odtwarzane z pliku."""

    return pd.read_parquet(path)