"""
Normalizacja danych dla sieci i powrót z wyjścia sieci do watów.
"""

import numpy as np
import torch


class Normalizer:
    """
    Standaryzacja danych: (x - średnia) / odchylenie standardowe.

    Statystyki liczymy TYLKO na domach treningowych. Są częścią modelu: bez nich
    nie da się zamienić wyjścia sieci na waty, więc zapisujemy je w W&B (to_dict).
    """

    def __init__(self, aggregates: list[np.ndarray], targets: list[np.ndarray], appliances: list[str]):
        # Po jednej tablicy na dom treningowy: aggregates (N,), targets (N, liczba_urządzeń)
        aggregate = np.concatenate(aggregates).astype(np.float64)
        targets_all = np.concatenate(targets).astype(np.float64)

        self.appliances = list(appliances)

        self.aggregate_mean = float(aggregate.mean())
        self.aggregate_std = _safe_std(aggregate.std())

        # nanmean/nanstd pomijają NaN, czyli okresy, w których na kanale było inne urządzenie
        self.target_mean = np.nanmean(targets_all, axis=0)
        self.target_std = np.array([_safe_std(s) for s in np.nanstd(targets_all, axis=0)])

    def normalize_aggregate(self, aggregate: np.ndarray) -> np.ndarray:
        normalized = (aggregate - self.aggregate_mean) / self.aggregate_std
        return normalized.astype(np.float32)

    def normalize_targets(self, targets: np.ndarray) -> np.ndarray:
        normalized = (targets - self.target_mean) / self.target_std
        return normalized.astype(np.float32)

    def denormalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Wyjście sieci -> waty. Tensor o kształcie (batch, liczba_urządzeń)."""
        mean = torch.tensor(self.target_mean, dtype=targets.dtype, device=targets.device)
        std = torch.tensor(self.target_std, dtype=targets.dtype, device=targets.device)
        return targets * std + mean

    def to_dict(self) -> dict:
        """Statystyki w formie do zapisania w configu W&B."""
        stats = {"aggregate": {"mean": self.aggregate_mean, "std": self.aggregate_std}}
        for i, appliance in enumerate(self.appliances):
            stats[appliance] = {"mean": float(self.target_mean[i]), "std": float(self.target_std[i])}
        return stats


def _safe_std(std: float) -> float:
    # Stały sygnał (np. urządzenie, które nigdy się nie włączyło) dałby dzielenie przez zero
    if std < 1e-6:
        return 1.0
    return float(std)
