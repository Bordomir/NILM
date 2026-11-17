"""
Wspólny kontrakt wszystkich modeli NILM.
"""

import torch
from torch import nn


class NILMModel(nn.Module):
    """
    Klasa bazowa modeli.

    """

    def __init__(self, window_size: int, n_appliances: int):
        super().__init__()
        self.window_size = window_size
        self.n_appliances = n_appliances

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Model musi zaimplementować forward")
