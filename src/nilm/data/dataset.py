"""
Warstwa 2: z danych po obróbce (warstwa 1) budujemy okna czasowe dla modelu.

Wejście to ramka jednego domu w formacie warstwy 1: indeks czasu, kolumny Aggregate
i Appliance1..9 (opcjonalnie kolumna "segment" z numerem ciągłego fragmentu).
Kanały zamieniamy na urządzenia dopiero tutaj, na podstawie houses.yaml.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from nilm.data.preprocessing.metadata import HouseMeta

SEGMENT_COL = "segment"


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


class NILMWindowDataset(Dataset):
    """
    Okna stałej długości z sygnału Aggregate i pobór urządzeń w ostatniej próbce okna.

    Okna nie są tworzone z góry (to byłyby gigabajty na dom), tylko wycinane w __getitem__.
    Okno nigdy nie przechodzi przez granicę segmentu (dziurę w danych),
    a próbki z celem NaN są pomijane.
    """

    def __init__(self, aggregate: np.ndarray, targets: np.ndarray, segments: np.ndarray, window_size: int):
        # aggregate: (N,), targets: (N, liczba_urządzeń), segments: (N,) - wszystko w kolejności czasu
        if np.isnan(aggregate).any():
            raise ValueError("Aggregate zawiera NaN - warstwa 1 powinna je usunąć")

        self.window_size = window_size
        self.aggregate = torch.from_numpy(aggregate.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))
        self.ends = torch.from_numpy(self._valid_window_ends(targets, segments, window_size))

    @staticmethod
    def _valid_window_ends(targets: np.ndarray, segments: np.ndarray, window_size: int) -> np.ndarray:
        ends = np.arange(window_size - 1, len(segments))
        # Segmenty są ciągłe i rosnące, więc ten sam numer na początku i końcu okna
        # oznacza, że całe okno leży w jednym segmencie
        same_segment = segments[ends - window_size + 1] == segments[ends]
        has_target = ~np.isnan(targets[ends]).any(axis=1)
        return ends[same_segment & has_target]

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = int(self.ends[idx])
        # Format wejściowy dla modeli w PyTorch: (Channels=1, Length)
        x = self.aggregate[end - self.window_size + 1 : end + 1].unsqueeze(0)
        return x, self.targets[end]


def build_house_dataset(
    df: pd.DataFrame, house: HouseMeta, appliances: list[str], window_size: int
) -> NILMWindowDataset:
    """Dataset jednego domu; dom musi mieć wszystkie podane urządzenia."""
    missing = [a for a in appliances if not house.channels_for(a)]
    if missing:
        raise ValueError(f"{house.name}: brak urządzeń {missing}")

    targets = np.column_stack([appliance_target(df, house, a).to_numpy() for a in appliances])
    # Dopóki warstwa 1 nie dzieli danych na segmenty, cały dom traktujemy jako jeden segment
    segments = df[SEGMENT_COL].to_numpy() if SEGMENT_COL in df else np.zeros(len(df), dtype=np.int64)
    return NILMWindowDataset(df["Aggregate"].to_numpy(), targets, segments, window_size)
