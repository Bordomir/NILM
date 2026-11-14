"""
Wspólne wczytywanie configów z YAML-a do dataclasses.

Wartości domyślne są w dataclassach, plik YAML nadpisuje tylko to, co podaje.
Nieznany klucz w YAML-u to błąd.
"""

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def update_dataclass(obj: Any, values: dict[str, Any], prefix: str = "") -> None:
    known = {f.name for f in fields(obj)}
    for key, value in values.items():
        if key not in known:
            raise KeyError(f"Nieznany klucz konfiguracji: '{prefix}{key}'")
        current = getattr(obj, key)
        if is_dataclass(current):
            update_dataclass(current, value or {}, prefix=f"{prefix}{key}.")
        else:
            setattr(obj, key, value)
