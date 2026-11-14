"""
Moduł spinający etapy z steps.py w całość.

"""

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from nilm.config import PreprocessConfig
from nilm.data.preprocessing.io import POWER_COLS, SEGMENT_COL, read_house_csv, write_json, write_parquet
from nilm.data.preprocessing.steps import Step

PIPELINE_VERSION = 1  # podbij przy KAŻDEJ zmianie logiki etapu
STEPS: list[Step] = []  # na razie pusto

HASH_LENGTH = 10

def compute_hash(cfg: PreprocessConfig) -> tuple[str, dict[str, Any]]:
    """
    Zwraca (hash, manifest). Manifest to wszystko, co wpływa na wynik obróbki,
    więc ten sam hash oznacza te same dane wyjściowe.
    """
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "steps": _step_names(),
        "config": asdict(cfg),
    }
    # sort_keys: kolejność kluczy w słownikach nie może zmieniać hasha
    payload = json.dumps(manifest, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:HASH_LENGTH], manifest


def process_house(csv_path: str | Path, cfg: PreprocessConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Wczytuje surowy CSV jednego domu i przepuszcza go przez STEPS; zwraca (dane, raport)."""
    _step_names()  # sprawdza unikalność nazw przed kosztownym przetwarzaniem
    df = read_house_csv(csv_path)
    _check_contract(df, "read_house_csv")

    report: dict[str, Any] = {"rows_raw": len(df), "steps": {}}
    for step in STEPS:
        rows_before, start = len(df), time.perf_counter()
        df, step_report = step.fn(df, cfg)
        _check_contract(df, step.name)
        report["steps"][step.name] = {
            **step_report,
            "rows_before": rows_before,
            "rows_after": len(df),
            "seconds": round(time.perf_counter() - start, 2),
        }
    report["rows_out"] = len(df)
    return df, report


def run_pipeline(
    cfg: PreprocessConfig,
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    houses: list[str] | None = None,
    force: bool = False,
) -> Path:
    """
    Przetwarza domy do processed_dir/<hash>/ i zwraca ten katalog.

    Dom z istniejącym parquetem i raportem jest pomijany (cache), chyba że force=True.
    Raport zapisujemy po parquecie, więc jego obecność oznacza, że dom przetworzono do końca.
    """
    digest, manifest = compute_hash(cfg)
    out_dir = Path(processed_dir) / digest
    write_json(manifest, out_dir / "manifest.json")

    for csv_path in _find_house_csvs(Path(raw_dir), houses):
        house = csv_path.stem
        parquet_path, report_path = out_dir / f"{house}.parquet", out_dir / f"{house}.report.json"
        if not force and parquet_path.exists() and report_path.exists():
            print(f"[preprocess] {house}: gotowe, pomijam")
            continue

        print(f"[preprocess] {house}: przetwarzam")
        df, report = process_house(csv_path, cfg)
        write_parquet(df, parquet_path)
        write_json(report, report_path)

    # Raport zbiorczy ze wszystkich domów w katalogu, także tych z wcześniejszych uruchomień
    reports = {p.name.removesuffix(".report.json"): json.loads(p.read_text()) for p in out_dir.glob("*.report.json")}
    write_json(dict(sorted(reports.items(), key=lambda item: _house_number(item[0]))), out_dir / "report.json")
    return out_dir


def _house_number(name: str) -> int:
    return int(name.removeprefix("House_"))


def _find_house_csvs(raw_dir: Path, houses: list[str] | None) -> list[Path]:
    available = {p.stem: p for p in raw_dir.glob("House_*.csv")}
    if not available:
        raise FileNotFoundError(f"Brak plików House_*.csv w {raw_dir}")
    if houses is None:
        houses = list(available)
    missing = sorted(set(houses) - set(available))
    if missing:
        raise FileNotFoundError(f"Brak plików CSV dla domów {missing} w {raw_dir}")
    return [available[h] for h in sorted(houses, key=_house_number)]


def _step_names() -> list[str]:
    names = [step.name for step in STEPS]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"Powtórzone nazwy etapów w STEPS: {duplicates}")
    return names


def _check_contract(df: pd.DataFrame, after: str) -> None:
    problems = []
    if not isinstance(df.index, pd.DatetimeIndex) or str(df.index.tz) != "UTC":
        problems.append("indeks nie jest czasem w UTC")
    elif not (df.index.is_monotonic_increasing and df.index.is_unique):
        problems.append("czas nie jest ściśle rosnący")

    columns, expected = set(df.columns) - {SEGMENT_COL}, set(POWER_COLS)
    if columns != expected:
        problems.append(f"brakujące kolumny {sorted(expected - columns)}, nadmiarowe {sorted(columns - expected)}")
    elif df["Aggregate"].isna().any():
        problems.append("NaN w Aggregate")

    if problems:
        raise ValueError(f"Kontrakt danych złamany po '{after}': " + "; ".join(problems))
