"""Write the dataset to CSV so it can be opened and inspected directly.

The generator is the source of truth; these files are a snapshot of it.
A test asserts the snapshot matches the generator, so the CSVs cannot
drift from what the app computes on.

    python -m data.export
"""
from pathlib import Path

import pandas as pd

from data.generate import generate_market_data
from src.corruption import apply_default_faults
from src import risk

HERE = Path(__file__).resolve().parent
FILES = {
    "clean": HERE / "market_data_clean.csv",
    "corrupted": HERE / "market_data_corrupted.csv",
    "faults": HERE / "fault_log.csv",
    "portfolio": HERE / "portfolio.csv",
}


def export() -> dict:
    clean = generate_market_data()
    corrupted, fault_log = apply_default_faults(clean)
    clean.rename_axis("date").to_csv(FILES["clean"], float_format="%.6f")
    corrupted.rename_axis("date").to_csv(FILES["corrupted"], float_format="%.6f")
    fault_log.to_csv(FILES["faults"], index=False)
    risk.sensitivities_table().to_csv(FILES["portfolio"], index=False)
    return FILES


def load_clean() -> pd.DataFrame:
    return pd.read_csv(FILES["clean"], index_col="date", parse_dates=True)


if __name__ == "__main__":
    for name, path in export().items():
        print(f"{name:<10} {path.name:<28} {path.stat().st_size/1024:6.1f} KB")
