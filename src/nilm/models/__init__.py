"""
Modele NILM. train.py tworzy model wyłącznie przez build_model, na podstawie sekcji model w configu.

Nowy model: skopiuj _template.py i dopisz go do MODELS (instrukcja w _template.py).
"""

import inspect

import torch

from nilm.config import ModelConfig
from nilm.models.base import NILMModel
from nilm.models.cnn1d import CNN1D

# Nazwa w configu (model.name) -> klasa modelu
MODELS: dict[str, type[NILMModel]] = {
    "cnn1d": CNN1D,
}


def build_model(cfg: ModelConfig, window_size: int, n_appliances: int) -> NILMModel:
    """Tworzy model z configu i od razu sprawdza, czy spełnia kontrakt NILMModel."""
    if cfg.name not in MODELS:
        raise ValueError(
            f"Nieznany model '{cfg.name}', dostępne: {sorted(MODELS)}. "
            "Nowy model dopisz do MODELS w src/nilm/models/__init__.py"
        )
    model_class = MODELS[cfg.name]
    _check_params(model_class, cfg.params)

    model = model_class(window_size=window_size, n_appliances=n_appliances, **cfg.params)
    _check_output_shape(model, window_size, n_appliances)
    return model


def _check_params(model_class: type[NILMModel], params: dict) -> None:
    # Literówka w model.params dałaby niejasny TypeError; tu mówimy wprost, czego model nie zna
    allowed = set(inspect.signature(model_class.__init__).parameters) - {"self", "window_size", "n_appliances"}
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(f"{model_class.__name__} nie ma parametrów {unknown}, dostępne: {sorted(allowed)}")


def _check_output_shape(model: NILMModel, window_size: int, n_appliances: int) -> None:
    # Próbny przebieg na zerach: zły kształt wyjścia wychodzi przed treningiem
    was_training = model.training
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 1, window_size))
    model.train(was_training)

    expected = (2, n_appliances)
    if tuple(output.shape) != expected:
        raise ValueError(
            f"{type(model).__name__}: dla wejścia (2, 1, {window_size}) "
            f"oczekiwano wyjścia {expected}, jest {tuple(output.shape)}"
        )


__all__ = ["MODELS", "NILMModel", "build_model"]
