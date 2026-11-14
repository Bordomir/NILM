# NILM

Non-Intrusive Load Monitoring (deep learning) na zbiorze REFIT (UK electrical load, Kaggle).

## Setup

Wymagany [uv](https://docs.astral.sh/uv/). Python 3.12 zostanie pobrany automatycznie.

```bash
uv sync                 # tworzy .venv i instaluje zależności (w tym dev)
```

## Typowe komendy

```bash
uv run python -m nilm.data.dataset   # pobranie danych + przygotowanie okien
uv run ruff check . && uv run ruff format .
uv add <pakiet>                      # nowa zależność
uv add --dev <pakiet>                # zależność tylko deweloperska
```

Notebooki: w VS Code wybierz kernel z `.venv/bin/python`.

## Weights & Biases

Wyniki treningów zapisujemy we wspólnym zespole **DL_NILM** na W&B.
Zanim puścisz pierwszy trening, przeczytaj **[docs/wandb.md](docs/wandb.md)**: jak zacząć i jakich zasad się trzymamy.

## Struktura

```
src/nilm/        kod pakietu (importowany jako `nilm`)
  data/          wczytywanie REFIT, mapowania urządzeń, okna czasowe, Dataset
  models/        architektury sieci
scripts/         skrypty uruchomieniowe (trening, ewaluacja)
configs/         konfiguracje eksperymentów
notebooks/       eksploracja i analizy
tests/           testy (pytest)
data/            dane surowe/przetworzone — poza gitem
outputs/         checkpointy, logi, wyniki — poza gitem
```
