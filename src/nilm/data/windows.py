"""
Okna czasowe z jednego domu
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class NILMWindowDataset(Dataset):
    """
    Okna stałej długości z Aggregate i pobór urządzeń w ostatniej próbce okna.

    """

    def __init__(self, aggregate: np.ndarray, targets: np.ndarray, segments: np.ndarray, window_size: int):
        # aggregate: (N,), targets: (N, liczba_urządzeń), segments: (N,) - wszystko w kolejności czasu
        if np.isnan(aggregate).any():
            raise ValueError("Aggregate zawiera NaN - pipeline powinien je usunąć")

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
        x = self.aggregate[end - self.window_size + 1 : end + 1].unsqueeze(0)
        return x, self.targets[end]
