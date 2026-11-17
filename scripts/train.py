"""
Trening modelu z configu: dane -> model -> W&B -> checkpoint.
"""

import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from nilm.config import ExperimentConfig, load_config
from nilm.data.dataset import NILMDatasets, build_datasets
from nilm.data.normalization import Normalizer
from nilm.data.preprocessing.pipeline import run_pipeline
from nilm.models import build_model
from nilm.tracking import init_run
from nilm.training import baseline_metrics, evaluate, get_device, set_seed, train_one_epoch


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    device = get_device()

    # Dane i model przygotowujemy przed startem runa: błąd w configu nie zostawi pustego runa w W&B
    data_dir = run_pipeline(cfg.preprocess, args.raw_dir, args.processed_dir)
    data = build_datasets(cfg.data, data_dir)
    model = build_model(cfg.model, window_size=cfg.data.window_size, n_appliances=len(cfg.data.appliances))
    model = model.to(device)

    # with: run zostanie zamknięty także po błędzie lub Ctrl+C
    with init_run(cfg, job_type="train", data_hash=data_dir.name) as run:
        run.config.update(describe_run(data, model, device))
        checkpoint_path = Path(args.output_dir) / run.id / "best.pt"
        checkpoint_info = {"normalizer": data.normalizer.to_dict(), "config": asdict(cfg), "data_hash": data_dir.name}
        print(f"[train] urządzenie: {device}, okna treningowe: {len(data.train):,}, checkpoint: {checkpoint_path}")

        train_loader = make_loader(data.train, cfg, train=True)
        val_loader = make_loader(data.val, cfg, train=False) if data.val else None
        test_loader = make_loader(data.test, cfg, train=False) if data.test else None

        # Baseline: wynik, który model musi pobić
        for split, dataset in (("val", data.val), ("test", data.test)):
            if dataset is not None:
                log_baseline(run, dataset, data.normalizer, split)

        fit(run, model, train_loader, val_loader, data.normalizer, cfg, device, checkpoint_path, checkpoint_info)

        # Zbiór testowy oceniamy raz, na końcu, najlepszym modelem z walidacji
        if test_loader is not None:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model"])
            test_metrics = evaluate(model, test_loader, data.normalizer, device, "test")
            run.summary.update(test_metrics)
            print(f"[train] model test:\n{format_metrics(test_metrics)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", default="configs/base.yaml")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def fit(
    run: Any,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    normalizer: Normalizer,
    cfg: ExperimentConfig,
    device: torch.device,
    checkpoint_path: Path,
    checkpoint_info: dict[str, Any],
) -> None:
    """Pętla epok. Po każdej epoce walidacja; najlepszy model trafia do checkpoint_path."""
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    step = 0
    best_val_mae = float("inf")

    for epoch in range(1, cfg.train.epochs + 1):
        start = time.perf_counter()
        epoch_loss, step = train_one_epoch(
            model, train_loader, optimizer, device, step, cfg.train.log_every_steps, run.log
        )
        metrics = {"epoch": epoch, "train/epoch_loss": epoch_loss, "train/epoch_seconds": time.perf_counter() - start}
        summary = f"[train] epoka {epoch}/{cfg.train.epochs}: loss {epoch_loss:.4f}"

        if val_loader is None:
            # Bez walidacji nie da się wybrać najlepszej epoki, więc zapisujemy ostatnią
            is_best = True
        else:
            metrics.update(evaluate(model, val_loader, normalizer, device, "val"))
            val_mae = metrics["val/mae/mean"]
            summary += f", val/mae/mean {val_mae:.1f} W"
            is_best = val_mae < best_val_mae
            if is_best:
                best_val_mae = val_mae
                run.summary["val/mae/best_mean"] = val_mae

        run.log(metrics, step=step)
        print(summary)

        if is_best:
            save_checkpoint(checkpoint_path, model, epoch, checkpoint_info)
            run.summary["best_epoch"] = epoch


def make_loader(dataset: Dataset, cfg: ExperimentConfig, train: bool) -> DataLoader:
    return DataLoader(
        dataset,
        # Ewaluacja nie liczy gradientów, więc większy batch mieści się w pamięci
        batch_size=cfg.train.batch_size if train else cfg.train.batch_size * 4,
        shuffle=train,
        # Generator z seedem: powtarzalna kolejność batchy
        generator=torch.Generator().manual_seed(cfg.train.seed) if train else None,
        num_workers=cfg.train.num_workers,
        persistent_workers=cfg.train.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def describe_run(data: NILMDatasets, model: nn.Module, device: torch.device) -> dict[str, Any]:
    """Informacje dopisywane do configu runa w W&B, poza samym configiem eksperymentu."""

    return {
        # Bez statystyk normalizacji nie da się zamienić wyjścia modelu na waty
        "normalizer": data.normalizer.to_dict(),
        "n_windows": {
            "train": len(data.train),
            "val": len(data.val) if data.val else 0,
            "test": len(data.test) if data.test else 0,
        },
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device),
    }


def log_baseline(run: Any, dataset: ConcatDataset, normalizer: Normalizer, split: str) -> None:
    """Metryki stałych predykcji (średnia z treningu i 0 W) do summary runa."""

    metrics = baseline_metrics(dataset, normalizer, split)
    run.summary.update(metrics)
    print(f"[train] baseline {split}:\n{format_metrics(metrics)}")


def save_checkpoint(path: Path, model: nn.Module, epoch: int, info: dict[str, Any]) -> None:
    """Wagi modelu + wszystko, czego trzeba, żeby go później użyć (normalizacja, config, wersja danych)."""
    
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": epoch, **info}, path)


def format_metrics(metrics: dict[str, float]) -> str:
    groups: dict[str, list[str]] = {}
    for key, value in metrics.items():
        _, metric, appliance = key.split("/")
        groups.setdefault(metric, []).append(f"{appliance} {value:.2f}")
    return "\n".join(f"    {metric}: {', '.join(values)}" for metric, values in groups.items())


if __name__ == "__main__":
    main()
