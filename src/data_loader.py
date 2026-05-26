"""
Data loading and preprocessing utilities for insurance analytics
"""

from pathlib import Path

import pandas as pd
import numpy as np


class InsuranceDataError(Exception):
    """Raised when the insurance dataset cannot be loaded or validated."""


class InsuranceDataValidationError(InsuranceDataError):
    """Raised when loaded insurance data does not meet expected schema or quality checks."""


def _validate_required_columns(df, required_columns):
    if required_columns is None:
        required_columns = {'TotalPremium', 'TotalClaims'}

    if not isinstance(required_columns, (set, list, tuple)):
        raise TypeError("required_columns must be a set, list, tuple, or None")

    normalized = {str(col).strip() for col in required_columns if str(col).strip()}
    if not normalized:
        raise ValueError("required_columns must include at least one non-empty column name")

    missing_columns = normalized.difference(df.columns)
    if missing_columns:
        raise InsuranceDataValidationError(
            f"Missing required columns in insurance dataset: {', '.join(sorted(missing_columns))}"
        )

    return normalized


def _validate_filepath(filepath):
    if filepath is None or str(filepath).strip() == "":
        raise ValueError("filepath must be a non-empty string or Path-like value")

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Insurance data file not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Expected a data file but found a directory: {path}")

    return path


def _detect_separator(path):
    with path.open('r', encoding='utf-8-sig') as file_handle:
        header_line = ""
        for line in file_handle:
            stripped_line = line.strip()
            if stripped_line:
                header_line = stripped_line
                break

    if not header_line:
        raise InsuranceDataError(f"The insurance data file is empty: {path}")

    for separator in ('|', '\t', ';'):
        if separator in header_line:
            return separator

    return ','


def read_insurance_file(filepath):
    """Read the insurance dataset with automatic delimiter detection."""
    try:
        path = _validate_filepath(filepath)
        separator = _detect_separator(path)
    except (ValueError, FileNotFoundError, IsADirectoryError, InsuranceDataError):
        raise
    except Exception as exc:
        raise InsuranceDataError(f"Unable to prepare insurance file '{filepath}' for reading: {exc}") from exc

    try:
        return pd.read_csv(path, sep=separator, low_memory=False)
    except pd.errors.ParserError as exc:
        raise InsuranceDataError(
            f"Failed to parse insurance data file '{path}' using separator '{separator}': {exc}"
        ) from exc
    except Exception as exc:
        raise InsuranceDataError(f"Failed to read insurance data file '{path}': {exc}") from exc


def load_insurance_data(filepath):
    """Load insurance dataset with proper data types and derived metrics"""
    print(f"Loading data from {filepath}...")

    try:
        df = read_insurance_file(filepath)
    except InsuranceDataError:
        raise
    except Exception as exc:
        raise InsuranceDataError(f"Unexpected error while loading insurance data from '{filepath}': {exc}") from exc

    if not isinstance(df, pd.DataFrame):
        raise InsuranceDataValidationError("Loaded insurance data is not a pandas DataFrame")
    if df.empty:
        raise InsuranceDataValidationError("Loaded insurance dataset is empty")

    _validate_required_columns(df, {'TotalPremium', 'TotalClaims'})

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    try:
        # Convert date columns
        if 'TransactionMonth' in df.columns:
            df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
            df['Month'] = df['TransactionMonth'].dt.month
            df['Year'] = df['TransactionMonth'].dt.year
            df['MonthName'] = df['TransactionMonth'].dt.month_name()

        # Ensure numeric columns are correct type
        numeric_cols = [
            'TotalPremium',
            'TotalClaims',
            'CalculatedPremiumPerTerm',
            'SumInsured',
            'CapitalOutstanding',
            'CustomValueEstimate',
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate derived metrics
        df['LossRatio'] = np.where(df['TotalPremium'] > 0, df['TotalClaims'] / df['TotalPremium'], 0)
        df['Margin'] = df['TotalPremium'] - df['TotalClaims']
        df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)
    except Exception as exc:
        raise InsuranceDataValidationError(f"Failed while transforming insurance dataset: {exc}") from exc

    print("\nDerived metrics:")
    print(f"  - Loss Ratio mean: {df['LossRatio'].mean():.3f}")
    print(f"  - Margin mean: R{df['Margin'].mean():.2f}")
    print(f"  - Claim Frequency: {df['HasClaim'].mean()*100:.1f}%")

    return df


def check_data_quality(df):
    """Perform data quality checks"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if df.empty:
        raise ValueError("df must not be empty")

    quality_report = []
    
    # Missing values
    missing_pct = (df.isnull().sum() / len(df)) * 100
    for col, pct in missing_pct.items():
        if pct > 0:
            status = '⚠️ Warning' if pct > 20 else '✅ OK'
            quality_report.append({'check': 'missing_values', 'column': col, 
                                  'value': f'{pct:.2f}%', 'status': status})
    
    # Duplicate rows
    duplicates = df.duplicated().sum()
    status = '✅ OK' if duplicates == 0 else '⚠️ Warning'
    quality_report.append({'check': 'duplicate_rows', 'column': 'all', 
                          'value': duplicates, 'status': status})
    
    # Negative values check
    for col in ['TotalPremium', 'TotalClaims']:
        if col in df.columns:
            negative = (df[col] < 0).sum()
            if negative > 0:
                quality_report.append({'check': 'negative_values', 'column': col, 
                                      'value': negative, 'status': '❌ Critical'})
    
    return pd.DataFrame(quality_report)