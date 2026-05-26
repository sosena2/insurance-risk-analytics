from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    InsuranceDataError,
    InsuranceDataValidationError,
    load_insurance_data,
    read_insurance_file,
)


def test_read_insurance_file_rejects_empty_path():
    with pytest.raises(ValueError, match="non-empty"):
        read_insurance_file("")


def test_read_insurance_file_rejects_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="not found"):
        read_insurance_file(tmp_path / "missing.txt")


def test_read_insurance_file_rejects_directory(tmp_path: Path):
    with pytest.raises(IsADirectoryError, match="directory"):
        read_insurance_file(tmp_path)


def test_load_insurance_data_raises_on_missing_required_columns(tmp_path: Path):
    sample = pd.DataFrame(
        {
            "PolicyID": [1, 2],
            "Province": ["A", "B"],
        }
    )
    input_file = tmp_path / "sample.csv"
    sample.to_csv(input_file, index=False)

    with pytest.raises(InsuranceDataValidationError, match="Missing required columns"):
        load_insurance_data(input_file)


def test_load_insurance_data_builds_derived_columns(tmp_path: Path):
    sample = pd.DataFrame(
        {
            "TransactionMonth": ["2023-01-01", "2023-02-01"],
            "TotalPremium": [1000, 2000],
            "TotalClaims": [100, 300],
        }
    )
    input_file = tmp_path / "sample.csv"
    sample.to_csv(input_file, index=False)

    df = load_insurance_data(input_file)

    for required in ["LossRatio", "Margin", "HasClaim", "Month", "Year", "MonthName"]:
        assert required in df.columns


def test_load_insurance_data_wraps_parsing_errors(monkeypatch):
    def broken_reader(*_args, **_kwargs):
        raise RuntimeError("simulated parser crash")

    monkeypatch.setattr("src.data_loader.read_insurance_file", broken_reader)

    with pytest.raises(InsuranceDataError, match="Unexpected error while loading insurance data"):
        load_insurance_data("dummy-path.csv")
