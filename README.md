# NILM

Non-Intrusive Load Monitoring (deep learning) na zbiorze REFIT (UK electrical load, Kaggle).

## Setup

Wymagany [uv](https://docs.astral.sh/uv/). Python 3.12 zostanie pobrany automatycznie.

```bash
uv sync                 # tworzy .venv i instaluje zależności (w tym dev)
```

## Dane

Zbiór REFIT z Kaggle (`kyleahmurphy/uk-electrical-load`, ok. 6,3 GB) trzymamy w `data/raw/`, poza gitem:

```bash
uv run python -c "import kagglehub; kagglehub.dataset_download('kyleahmurphy/uk-electrical-load', output_dir='data/raw')"
```

Opis zbioru i lista urządzeń w każdym domu: `data/raw/REFIT_Readme.txt`.

Obróbka surowych CSV do Parquet (wynik w `data/processed/<hash>/`; gotowe domy są pomijane):

```bash
uv run python scripts/preprocess.py            # --help pokazuje pozostałe opcje
```

## Typowe komendy

```bash
uv run ruff check . && uv run ruff format .
uv add <pakiet>                      # nowa zależność
uv add --dev <pakiet>                # zależność tylko deweloperska
```

Notebooki: w VS Code wybierz kernel z `.venv/bin/python`.

## Jak pracujemy

Krok po kroku, od nowej gałęzi do pull requesta: **[docs/workflow.md](docs/workflow.md)**.

## Weights & Biases

Wyniki treningów zapisujemy we wspólnym zespole **DL_NILM** na W&B.
Zanim puścisz pierwszy trening, przeczytaj **[docs/wandb.md](docs/wandb.md)**: jak zacząć i jakich zasad się trzymamy.

## Struktura

```
src/nilm/        kod pakietu (importowany jako `nilm`)
  config/        schematy configów (eksperyment, obróbka danych) i ich wczytywanie z YAML
  data/
    preprocessing/   obróbka surowych CSV: metadane domów, czyszczenie, zapis Parquet
    dataset.py       okna czasowe, Dataset
  models/        architektury sieci
  tracking.py    start runów W&B (init_run)
scripts/         skrypty uruchomieniowe (trening, ewaluacja)
configs/         konfiguracje eksperymentów
docs/            zasady pracy zespołu (workflow, W&B)
notebooks/       eksploracja i analizy
data/raw/        surowe CSV z Kaggle — poza gitem
data/processed/  wynik obróbki (Parquet) — poza gitem
outputs/         checkpointy, logi, wyniki — poza gitem
```
