"""
Jedyny punkt startu runów W&B w projekcie. Nie wywołujemy wandb.init bezpośrednio,
żeby każdy run miał ten sam entity/project, pełny config i informację o stanie gita.
"""

import os
import subprocess
from dataclasses import asdict
from typing import cast, get_args

import wandb
from nilm.config import ExperimentConfig, WandbMode

JOB_TYPES = ("train", "eval", "data", "sweep")


def _resolve_mode(default: WandbMode) -> WandbMode:
    # WANDB_MODE ze środowiska ma pierwszeństwo; YAML i env to zwykłe stringi, więc sprawdzamy wartość
    mode = os.environ.get("WANDB_MODE", default)
    if mode not in get_args(WandbMode):
        raise ValueError(f"Nieprawidłowy tryb W&B '{mode}', dozwolone: {get_args(WandbMode)}")
    return cast(WandbMode, mode)


def _git_dirty() -> bool | None:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
        return bool(out.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def init_run(cfg: ExperimentConfig, job_type: str):
    if job_type not in JOB_TYPES:
        raise ValueError(f"job_type musi być jednym z {JOB_TYPES}, podano '{job_type}'")

    w = cfg.wandb
    project = f"{w.project}-dev" if w.debug else w.project
    mode = _resolve_mode(w.mode)

    git_dirty = _git_dirty()
    if git_dirty:
        print("[wandb] UWAGA: niezacommitowane zmiany — run może być trudny do odtworzenia.")

    return wandb.init(
        entity=w.entity,
        project=project,
        name=cfg.name,
        group=w.group,
        job_type=job_type,
        tags=w.tags,
        notes=w.notes,
        config={**asdict(cfg), "git_dirty": git_dirty},
        mode=mode,
    )
