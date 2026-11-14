"""
Metadane domów REFIT z houses.yaml: co jest podłączone do którego kanału i co zmieniało się w czasie.

Plik jest walidowany przy wczytaniu: nieznany klucz, typ urządzenia spoza APPLIANCES
albo brakujący kanał kończą się błędem.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, cast, get_args

import yaml

HOUSES_YAML = Path(__file__).with_name("houses.yaml")

CHANNELS = tuple(f"Appliance{i}" for i in range(1, 10))

# Słownik typów kanonicznych; nowy typ w houses.yaml trzeba najpierw dopisać tutaj
APPLIANCES = frozenset(
    {
        "fridge",
        "freezer",
        "fridge_freezer",
        "washing_machine",
        "washer_dryer",
        "tumble_dryer",
        "dishwasher",
        "microwave",
        "kettle",
        "toaster",
        "television",
        "computer",
        "electric_heater",
        "hifi",
        "router",
        "lamp",
        "extractor_fan",
        "blender",
        "food_mixer",
        "bread_maker",
        "vivarium",
        "pond_pump",
    }
)

EventKind = Literal["appliance_changed", "device_replaced", "device_added", "signature_change"]

# Okres [start, end) z danym typem urządzenia; None = od początku / do końca nagrań
Period = tuple[date | None, date | None, str | None]


@dataclass(frozen=True)
class ChannelEvent:
    kind: EventKind
    date: date | None  # None = data nieznana
    approx: bool = False
    appliance: str | None = None  # nowy typ, tylko dla appliance_changed
    note: str | None = None


@dataclass(frozen=True)
class Channel:
    column: str  # nazwa kolumny w CSV, np. "Appliance1"
    label: str  # oryginalna nazwa z README
    appliance: str | None  # typ na początku nagrań; None = nieużywany / nieznany
    mixed: bool = False
    note: str | None = None
    events: tuple[ChannelEvent, ...] = ()

    def appliance_periods(self) -> list[Period]:
        """Podział czasu nagrań na okresy z jednym typem urządzenia (zmienia go tylko appliance_changed)."""
        periods: list[Period] = []
        start, current = None, self.appliance
        changes = sorted((e for e in self.events if e.kind == "appliance_changed"), key=lambda e: cast(date, e.date))
        for e in changes:
            periods.append((start, e.date, current))
            start, current = e.date, e.appliance
        periods.append((start, None, current))
        return periods

    def has_appliance(self, appliance: str) -> bool:
        """Czy kanał w którymkolwiek momencie mierzy dany typ urządzenia."""
        return any(a == appliance for _, _, a in self.appliance_periods())


@dataclass(frozen=True)
class HouseMeta:
    name: str
    channels: dict[str, Channel]
    solar: bool = False
    notes: tuple[str, ...] = ()

    def channels_for(self, appliance: str) -> list[Channel]:
        """Kanały, na których w którymkolwiek momencie jest dany typ urządzenia (może być kilka, np. 2 zamrażarki)."""
        if appliance not in APPLIANCES:
            raise ValueError(f"Nieznany typ urządzenia '{appliance}'")
        return [ch for ch in self.channels.values() if ch.has_appliance(appliance)]


def _check_keys(values: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise KeyError(f"{where}: nieznane klucze {sorted(unknown)}")


def _check_appliance(appliance: Any, where: str) -> str | None:
    if appliance is not None and appliance not in APPLIANCES:
        raise ValueError(f"{where}: nieznany typ urządzenia '{appliance}' (dopisz go do APPLIANCES)")
    return appliance


def _parse_event(values: dict[str, Any], where: str) -> ChannelEvent:
    _check_keys(values, {"kind", "date", "approx", "appliance", "note"}, where)
    kind = values.get("kind")
    if kind not in get_args(EventKind):
        raise ValueError(f"{where}: nieznany rodzaj zdarzenia '{kind}', dozwolone: {get_args(EventKind)}")
    event_date = values.get("date")
    if event_date is not None and not isinstance(event_date, date):
        raise ValueError(f"{where}: data musi mieć format YYYY-MM-DD, jest '{event_date}'")
    appliance = _check_appliance(values.get("appliance"), where)
    if kind == "appliance_changed" and (event_date is None or appliance is None):
        raise ValueError(f"{where}: appliance_changed wymaga pól date i appliance")
    if kind != "appliance_changed" and appliance is not None:
        raise ValueError(f"{where}: pole appliance dozwolone tylko dla appliance_changed")
    return ChannelEvent(
        kind=cast(EventKind, kind),
        date=event_date,
        approx=values.get("approx", False),
        appliance=appliance,
        note=values.get("note"),
    )


def _parse_channel(column: str, values: dict[str, Any], where: str) -> Channel:
    _check_keys(values, {"label", "appliance", "mixed", "note", "events"}, where)
    if "label" not in values or "appliance" not in values:
        raise KeyError(f"{where}: wymagane pola label i appliance")
    events = tuple(_parse_event(e, f"{where}.events[{i}]") for i, e in enumerate(values.get("events", [])))
    return Channel(
        column=column,
        label=values["label"],
        appliance=_check_appliance(values["appliance"], where),
        mixed=values.get("mixed", False),
        note=values.get("note"),
        events=events,
    )


def _parse_house(name: str, values: dict[str, Any]) -> HouseMeta:
    _check_keys(values, {"channels", "solar", "notes"}, name)
    channels = values.get("channels", {})
    if set(channels) != set(CHANNELS):
        raise KeyError(f"{name}: kanały muszą być dokładnie {CHANNELS}, są {sorted(channels)}")
    return HouseMeta(
        name=name,
        channels={col: _parse_channel(col, channels[col], f"{name}.{col}") for col in CHANNELS},
        solar=values.get("solar", False),
        notes=tuple(values.get("notes", [])),
    )


def load_houses(path: str | Path = HOUSES_YAML) -> dict[str, HouseMeta]:
    with open(path) as f:
        values = yaml.safe_load(f) or {}
    return {name: _parse_house(name, house) for name, house in values.items()}
