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

    # split_segments: odstęp między kolejnymi pomiarami, powyżej którego uznajemy, że w danych jest dziura.
    # Typowy odstęp w REFIT to 7-15 s; 60 s to ok. 8 brakujących próbek.
    max_gap_s: float = 60