"""Config warstwy 1 obróbki danych (surowe CSV -> oczyszczone Parquet)."""

from dataclasses import dataclass


@dataclass
class PreprocessConfig:
    """
    Parametry preprocessingu z których liczony jest hash wyniku.

    Zasady:
    - tylko pola, które zmieniają zawartość plików wyjściowych, bo inaczej zmieniałyby hash bez powodu,
    - nowe pole dodajemy razem z etapem w steps.py, który go używa
    """