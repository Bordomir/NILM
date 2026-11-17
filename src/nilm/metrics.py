"""
Metryki NILM liczone w watach, osobno dla każdego urządzenia.

Każda metryka to zwykła funkcja: dostaje predykcje i cele o kształcie (N, liczba_urządzeń)
i zwraca jedną wartość na urządzenie, czyli tensor (liczba_urządzeń,).

Nowa metryka = nowa funkcja + jedna linijka w compute_metrics.
Nazwy w W&B: zbiór/metryka/urządzenie (docs/wandb.md, zasada 5), np. val/mae/kettle.
"""

import torch


def mae(pred_w: torch.Tensor, target_w: torch.Tensor) -> torch.Tensor:
    """Średni błąd bezwzględny w W: o ile średnio mylimy się w każdej chwili."""
    return (pred_w - target_w).abs().mean(dim=0)


def energy_error(pred_w: torch.Tensor, target_w: torch.Tensor) -> torch.Tensor:
    """
    Względny błąd łącznego zużycia (SAE): |przewidziana energia - prawdziwa| / prawdziwa.

    0.1 oznacza, że łączne zużycie urządzenia oszacowaliśmy z błędem 10%.
    Pomyłki w różnych chwilach mogą się tu znosić, dlatego to uzupełnienie MAE, a nie zamiennik.
    """
    predicted_energy = pred_w.sum(dim=0)
    true_energy = target_w.sum(dim=0)
    return (predicted_energy - true_energy).abs() / true_energy.clamp(min=1e-6)


def compute_metrics(
    pred_w: torch.Tensor,
    target_w: torch.Tensor,
    appliances: list[str],
    split: str,
    prefix: str = "",
) -> dict[str, float]:
    """
    Wszystkie metryki z nazwami do W&B, np. {"val/mae/kettle": 12.3, "val/mae/mean": 20.1, ...}.

    prefix dokleja się do nazwy metryki, np. prefix="baseline_mean_" daje "val/baseline_mean_mae/kettle".
    """
    per_appliance = {
        "mae": mae(pred_w, target_w),
        "energy_error": energy_error(pred_w, target_w),
    }

    results = {}
    for name, values in per_appliance.items():
        for i, appliance in enumerate(appliances):
            results[f"{split}/{prefix}{name}/{appliance}"] = float(values[i])
        # Średnia po urządzeniach: jedna liczba do porównywania runów
        results[f"{split}/{prefix}{name}/mean"] = float(values.double().mean())
    return results
