"""
Podstawowa sieć konwolucyjna 1D w stylu seq2point (Zhang i in., 2018), w mniejszej wersji.

Okno Aggregate -> kilka warstw konwolucji (wykrywają kształty: skoki, plateau, cykle)
-> warstwy w pełni połączone -> pobór każdego urządzenia w jednej próbce okna.
"""

from collections.abc import Sequence

import torch
from torch import nn

from nilm.models.base import NILMModel


class CNN1D(NILMModel):
    def __init__(
        self,
        window_size: int,
        n_appliances: int,
        channels: Sequence[int] = (32, 32, 64),
        kernel_size: int = 5,
        hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__(window_size, n_appliances)

        conv_layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            conv_layers += [nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"), nn.ReLU()]
            in_channels = out_channels
        self.features = nn.Sequential(*conv_layers)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * window_size, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_appliances),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
