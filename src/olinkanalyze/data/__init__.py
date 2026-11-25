"""Bundled example NPX datasets."""

from __future__ import annotations

from pathlib import Path

from olinkanalyze.data.synthetic import make_npx_data

_DATA_DIR = Path(__file__).resolve().parent


def data_path(filename: str) -> Path:
    """Return the path of a file shipped with the package."""
    path = _DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Bundled data file not found: {path}")
    return path


def load_example_csv() -> Path:
    return data_path("npx_data_long.csv")


npx_data1 = make_npx_data(project="data1", seed=1)
npx_data2 = make_npx_data(project="data2", seed=2)

__all__ = ["data_path", "load_example_csv", "npx_data1", "npx_data2"]
