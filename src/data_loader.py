"""
Data loading and preprocessing utilities for insurance analytics
"""

import pandas as pd
import numpy as np


def load_insurance_data(filepath):
    """Load insurance dataset with proper data types and derived metrics"""
    print(f"Loading data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8-sig') as file_handle:
        header_line = file_handle.readline()

    if '|' in header_line:
        separator = '|'
    elif '\t' in header_line:
        separator = '\t'
    else:
        separator = ','

    df = pd.read_csv(filepath, sep=separator)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Convert date columns
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
        df['Month'] = df['TransactionMonth'].dt.month
        df['Year'] = df['TransactionMonth'].dt.year
        df['MonthName'] = df['TransactionMonth'].dt.month_name()
    
    # Ensure numeric columns are correct type
    numeric_cols = ['TotalPremium', 'TotalClaims', 'CalculatedPremiumPerTerm', 
                    'SumInsured', 'CapitalOutstanding', 'CustomValueEstimate']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate derived metrics
    if 'TotalPremium' in df and 'TotalClaims' in df:
        df['LossRatio'] = np.where(df['TotalPremium'] > 0, df['TotalClaims'] / df['TotalPremium'], 0)
        df['Margin'] = df['TotalPremium'] - df['TotalClaims']
        df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)
        
        print(f"\nDerived metrics:")
        print(f"  - Loss Ratio mean: {df['LossRatio'].mean():.3f}")
        print(f"  - Margin mean: R{df['Margin'].mean():.2f}")
        print(f"  - Claim Frequency: {df['HasClaim'].mean()*100:.1f}%")
    
    return df


def check_data_quality(df):
    """Perform data quality checks"""
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