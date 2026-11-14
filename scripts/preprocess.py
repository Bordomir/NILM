"""
Obróbka surowych danych REFIT: data/raw/House_*.csv -> data/processed/<hash>/House_*.parquet.

Przykłady:
    uv run python scripts/preprocess.py
    uv run python scripts/preprocess.py --houses House_2 House_3
    uv run python scripts/preprocess.py --config configs/base.yaml --force
"""

import argparse
import json

from nilm.config import ExperimentConfig, load_config
from nilm.data.preprocessing.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="config eksperymentu; używana jest jego sekcja preprocess")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--houses", nargs="+", help="np. House_2 House_3 (domyślnie wszystkie)")
    parser.add_argument("--force", action="store_true", help="przelicz także domy, które są już w cache")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else ExperimentConfig()
    out_dir = run_pipeline(cfg.preprocess, args.raw_dir, args.processed_dir, args.houses, args.force)

    reports = json.loads((out_dir / "report.json").read_text())
    print(f"\nWynik: {out_dir} ({len(reports)} domów)")
    for house, report in reports.items():
        print(f"  {house:9} {report['rows_raw']:>9} -> {report['rows_out']:>9} wierszy")


if __name__ == "__main__":
    main()
