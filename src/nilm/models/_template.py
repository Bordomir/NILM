"""
Szablon nowego modelu.

"""

import torch
from torch import nn

from nilm.models.base import NILMModel


class TemplateModel(NILMModel):
    # Hiperparametry jako argumenty konstruktora z wartościami domyślnymi.
    # Każdy z nich można nadpisać w configu (model.params) i trafi do W&B razem z configiem.
    def __init__(self, window_size: int, n_appliances: int, hidden: int = 32):
        super().__init__(window_size, n_appliances)

        self.net = nn.Sequential(
            nn.Flatten(),  # (batch, 1, window_size) -> (batch, window_size)
            nn.Linear(window_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_appliances),  # bez aktywacji na końcu (patrz NILMModel)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
