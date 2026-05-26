"""
Data cleaning script for insurance dataset
Creates second version of data for DVC tracking
"""

from pathlib import Path

import pandas as pd
import numpy as np
import sys


class InsuranceCleaningError(Exception):
    """Raised when the cleaning pipeline cannot complete successfully."""


def _validate_input_path(input_path):
    if input_path is None or str(input_path).strip() == "":
        raise ValueError("input_path must be a non-empty string or Path-like value")

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input data file not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Expected a data file but found a directory: {path}")

    return path


def _validate_output_path(output_path):
    if output_path is None or str(output_path).strip() == "":
        raise ValueError("output_path must be a non-empty string or Path-like value")

    path = Path(output_path)
    if path.exists() and path.is_dir():
        raise IsADirectoryError(f"Expected an output file path but found a directory: {path}")

    return path


def _read_cleaning_input(input_path):
    path = _validate_input_path(input_path)

    try:
        return pd.read_csv(path, sep='|')
    except pd.errors.ParserError as exc:
        raise InsuranceCleaningError(
            f"Failed to parse insurance data file '{path}' with pipe delimiter: {exc}"
        ) from exc
    except Exception as exc:
        raise InsuranceCleaningError(f"Failed to read insurance data file '{path}': {exc}") from exc


def clean_insurance_data(input_path, output_path):
    """
    Clean insurance data and save cleaned version
    
    Cleaning steps:
    1. Remove duplicates
    2. Handle missing values
    3. Remove extreme outliers
    4. Fix data types
    """
    
    input_file = _validate_input_path(input_path)
    output_file = _validate_output_path(output_path)

    print(f"Loading data from {input_file}")
    df = _read_cleaning_input(input_file)

    if not isinstance(df, pd.DataFrame):
        raise InsuranceCleaningError("Loaded input is not a pandas DataFrame")
    if df.empty:
        raise InsuranceCleaningError("Input dataset is empty")

    original_shape = df.shape

    required_columns = {'TotalPremium', 'TotalClaims'}
    missing_required = required_columns.difference(df.columns)
    if missing_required:
        raise InsuranceCleaningError(
            f"Missing required columns for cleaning: {', '.join(sorted(missing_required))}"
        )

    if not all(isinstance(col, str) or pd.api.types.is_hashable(col) for col in df.columns):
        raise InsuranceCleaningError("Input dataset contains invalid column names")
    
    # 1. Remove duplicates
    df = df.drop_duplicates()
    print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")
    
    # 2. Handle missing values
    missing_before = df.isnull().sum().sum()
    
    # Numeric columns: fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Categorical columns: fill with mode
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
            df[col] = df[col].fillna(mode_val)
    
    missing_after = df.isnull().sum().sum()
    print(f"Filled {missing_before - missing_after} missing values")
    
    # 3. Remove outliers (beyond 3 standard deviations for TotalClaims)
    if 'TotalClaims' in df.columns:
        claims_std = df['TotalClaims'].std()
        if pd.notna(claims_std) and claims_std > 0:
            z_scores = np.abs((df['TotalClaims'] - df['TotalClaims'].mean()) / claims_std)
            outliers_before = len(df)
            df = df[z_scores < 3]
            print(f"Removed {outliers_before - len(df)} outliers in TotalClaims")
        else:
            print("Skipped outlier filtering for TotalClaims because standard deviation is zero or undefined")
    
    # 4. Ensure date column is proper datetime
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
    
    # 5. Remove any rows with negative premiums or claims
    if 'TotalPremium' in df.columns:
        df = df[df['TotalPremium'] > 0]
    if 'TotalClaims' in df.columns:
        df = df[df['TotalClaims'] >= 0]
    
    # Save cleaned data
    try:
        df.to_csv(output_file, index=False)
    except Exception as exc:
        raise InsuranceCleaningError(f"Failed to write cleaned data to '{output_file}': {exc}") from exc

    print(f"\nCleaned data saved to {output_file}")
    print(f"Original shape: {original_shape}")
    print(f"Final shape: {df.shape}")
    print(f"Rows removed: {original_shape[0] - df.shape[0]} ({((original_shape[0] - df.shape[0])/original_shape[0])*100:.2f}%)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_data.py <input_path> <output_path>")
        sys.exit(1)
    
    clean_insurance_data(sys.argv[1], sys.argv[2])