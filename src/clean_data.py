"""
Data cleaning script for insurance dataset
Creates second version of data for DVC tracking
"""

import pandas as pd
import numpy as np
import sys


def clean_insurance_data(input_path, output_path):
    """
    Clean insurance data and save cleaned version
    
    Cleaning steps:
    1. Remove duplicates
    2. Handle missing values
    3. Remove extreme outliers
    4. Fix data types
    """
    
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path, sep='|')
    original_shape = df.shape
    
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
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
            df[col] = df[col].fillna(mode_val)
    
    missing_after = df.isnull().sum().sum()
    print(f"Filled {missing_before - missing_after} missing values")
    
    # 3. Remove outliers (beyond 3 standard deviations for TotalClaims)
    if 'TotalClaims' in df.columns:
        z_scores = np.abs((df['TotalClaims'] - df['TotalClaims'].mean()) / df['TotalClaims'].std())
        outliers_before = len(df)
        df = df[z_scores < 3]
        print(f"Removed {outliers_before - len(df)} outliers in TotalClaims")
    
    # 4. Ensure date column is proper datetime
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
    
    # 5. Remove any rows with negative premiums or claims
    if 'TotalPremium' in df.columns:
        df = df[df['TotalPremium'] > 0]
    if 'TotalClaims' in df.columns:
        df = df[df['TotalClaims'] >= 0]
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to {output_path}")
    print(f"Original shape: {original_shape}")
    print(f"Final shape: {df.shape}")
    print(f"Rows removed: {original_shape[0] - df.shape[0]} ({((original_shape[0] - df.shape[0])/original_shape[0])*100:.2f}%)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_data.py <input_path> <output_path>")
        sys.exit(1)
    
    clean_insurance_data(sys.argv[1], sys.argv[2])