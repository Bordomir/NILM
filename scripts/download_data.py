"""
Pobranie surowego zbioru REFIT z Kaggle (ok. 6,3 GB) do data/raw/.

Jeśli wszystkie domy z houses.yaml są już na dysku, nic nie jest pobierane.

Przykłady:
    uv run python scripts/download_data.py
    uv run python scripts/download_data.py --force
"""

import argparse
from pathlib import Path

from nilm.data.preprocessing.io import KAGGLE_DATASET, download_refit
from nilm.data.preprocessing.metadata import load_houses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--force", action="store_true", help="pobierz ponownie, nawet jeśli dane są już na dysku")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    missing = sorted(h for h in load_houses() if not (output_dir / f"{h}.csv").exists())
    if not missing and not args.force:
        print(f"Dane są już w {output_dir}, nic nie pobieram (--force wymusza ponowne pobranie).")
        return

    print(f"Pobieram {KAGGLE_DATASET} do {output_dir} (ok. 6,3 GB)...")
    path = download_refit(output_dir, force=args.force)
    print(f"Gotowe: {path}\nNastępny krok: uv run python scripts/preprocess.py")


if __name__ == "__main__":
    main()
