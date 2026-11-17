"""
Pętla treningowa i ewaluacja; scripts/train.py tylko je spina z configiem i W&B.
"""

import random
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from nilm.data.normalization import Normalizer
from nilm.data.windows import NILMWindowDataset
from nilm.metrics import compute_metrics


def set_seed(seed: int) -> None:
    """Ten sam seed = ta sama inicjalizacja wag i kolejność batchy."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    step: int,
    log_every_steps: int,
    log: Callable[..., object],
) -> tuple[float, int]:
    """
    Jedna epoka. Strata MSE na znormalizowanych danych.

    """
    
    model.train()
    loss_fn = nn.MSELoss()
    total_loss, batches = 0.0, 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        total_loss += loss_value
        batches += 1
        step += 1
        if step % log_every_steps == 0:
            log({"train/loss": loss_value}, step=step)

    return total_loss / max(batches, 1), step


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, normalizer: Normalizer, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predykcje i cele dla całego zbioru, w watach; oba o kształcie (N, liczba_urządzeń), na CPU."""
    model.eval()
    predictions, targets = [], []

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        predictions.append(normalizer.denormalize_targets(model(x)).cpu())
        targets.append(normalizer.denormalize_targets(y).cpu())

    return torch.cat(predictions), torch.cat(targets)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    normalizer: Normalizer,
    device: torch.device,
    split: str,
    prefix: str = "",
) -> dict[str, float]:
    """Wszystkie metryki z metrics.py dla jednego zbioru, np. {"val/mae/kettle": ...}."""
    pred_w, target_w = predict(model, loader, normalizer, device)
    return compute_metrics(pred_w, target_w, normalizer.appliances, split, prefix)


def window_targets(dataset: ConcatDataset) -> torch.Tensor:
    """
    Znormalizowane cele wszystkich okien naraz, (N, liczba_urządzeń).

    """
    windows: list[NILMWindowDataset] = dataset.datasets  # type: ignore[assignment]
    return torch.cat([ds.targets[ds.starts + ds.target_offset] for ds in windows])


def baseline_metrics(dataset: ConcatDataset, normalizer: Normalizer, split: str) -> dict[str, float]:
    """
    Punkty odniesienia bez uczenia; sieć, która ich nie bije, niczego się nie nauczyła.

    - baseline_mean: zawsze średni pobór z domów treningowych,
    - baseline_zero: zawsze 0 W; dla rzadko włączanych urządzeń (czajnik, mikrofalówka)
      zwykle trudniejszy do pobicia niż średnia.
    Nazwy w W&B np. val/baseline_mean_mae/kettle, val/baseline_zero_mae/kettle.
    """
    target_w = normalizer.denormalize_targets(window_targets(dataset))
    mean_w = torch.as_tensor(normalizer.target_mean, dtype=target_w.dtype).expand_as(target_w)
    zero_w = torch.zeros_like(target_w)

    appliances = normalizer.appliances
    return {
        **compute_metrics(mean_w, target_w, appliances, split, prefix="baseline_mean_"),
        **compute_metrics(zero_w, target_w, appliances, split, prefix="baseline_zero_"),
    }
