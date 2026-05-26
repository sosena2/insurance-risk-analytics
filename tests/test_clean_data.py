from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clean_data import InsuranceCleaningError, clean_insurance_data


def test_clean_insurance_data_rejects_missing_input_path():
    with pytest.raises(ValueError, match="input_path"):
        clean_insurance_data("", "output.csv")


def test_clean_insurance_data_rejects_missing_required_columns(tmp_path: Path):
    sample = pd.DataFrame({"PolicyID": [1, 2], "Province": ["A", "B"]})
    input_file = tmp_path / "input.txt"
    sample.to_csv(input_file, sep="|", index=False)

    with pytest.raises(InsuranceCleaningError, match="Missing required columns"):
        clean_insurance_data(input_file, tmp_path / "output.csv")


def test_clean_insurance_data_rejects_output_directory(tmp_path: Path):
    sample = pd.DataFrame(
        {
            "TotalPremium": [100.0],
            "TotalClaims": [10.0],
            "TransactionMonth": ["2024-01-01"],
        }
    )
    input_file = tmp_path / "input.txt"
    sample.to_csv(input_file, sep="|", index=False)

    with pytest.raises(IsADirectoryError, match="output file path"):
        clean_insurance_data(input_file, tmp_path)


def test_clean_insurance_data_writes_cleaned_output(tmp_path: Path):
    sample = pd.DataFrame(
        {
            "TotalPremium": [100.0, 200.0, 100.0],
            "TotalClaims": [10.0, 20.0, 10.0],
            "TransactionMonth": ["2024-01-01", "2024-02-01", "2024-01-01"],
            "Province": ["Gauteng", None, "Gauteng"],
        }
    )
    input_file = tmp_path / "input.txt"
    output_file = tmp_path / "cleaned.csv"
    sample.to_csv(input_file, sep="|", index=False)

    clean_insurance_data(input_file, output_file)

    assert output_file.exists()
    cleaned = pd.read_csv(output_file)
    assert "TransactionMonth" in cleaned.columns
    assert "TotalPremium" in cleaned.columns
    assert len(cleaned) >= 1


def test_clean_insurance_data_wraps_read_errors(monkeypatch, tmp_path: Path):
    sample = pd.DataFrame({"TotalPremium": [100.0], "TotalClaims": [10.0]})
    input_file = tmp_path / "input.txt"
    sample.to_csv(input_file, sep="|", index=False)

    def broken_read_csv(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("src.clean_data.pd.read_csv", broken_read_csv)

    with pytest.raises(InsuranceCleaningError, match="Failed to read insurance data file"):
        clean_insurance_data(input_file, tmp_path / "output.csv")