"""
Okna czasowe z jednego domu
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from nilm.config import TargetPosition


class NILMWindowDataset(Dataset):
    """
    Okna stałej długości z Aggregate i pobór urządzeń w jednej próbce okna (celu).

    """

    def __init__(
        self,
        aggregate: np.ndarray,
        targets: np.ndarray,
        segments: np.ndarray,
        window_size: int,
        target_position: TargetPosition = "last",
        stride: int = 1,
    ):
        # aggregate: (N,), targets: (N, liczba_urządzeń), segments: (N,) - wszystko w kolejności czasu
        if np.isnan(aggregate).any():
            raise ValueError("Aggregate zawiera NaN - pipeline powinien je usunąć")

        self.window_size = window_size
        self.target_offset = window_size - 1 if target_position == "last" else window_size // 2
        self.aggregate = torch.from_numpy(aggregate.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))

        starts = self._valid_window_starts(targets, segments)
        # Przerzedzamy dopiero poprawne okna, więc liczba okien to dokładnie len(poprawne) / stride
        self.starts = torch.from_numpy(starts[::stride])

    def _valid_window_starts(self, targets: np.ndarray, segments: np.ndarray) -> np.ndarray:
        starts = np.arange(len(segments) - self.window_size + 1)
        ends = starts + self.window_size - 1

        # Segmenty są ciągłe i rosnące, więc ten sam numer na początku i końcu okna
        # oznacza, że całe okno leży w jednym segmencie
        same_segment = segments[starts] == segments[ends]
        has_target = ~np.isnan(targets[starts + self.target_offset]).any(axis=1)
        return starts[same_segment & has_target]

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[idx])
        x = self.aggregate[start : start + self.window_size].unsqueeze(0)  # (1, window_size)
        y = self.targets[start + self.target_offset]  # (liczba_urządzeń,)
        return x, y
