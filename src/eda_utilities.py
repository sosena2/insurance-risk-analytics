"""
EDA visualization utilities for insurance data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def setup_style():
    """Configure plotting style"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12


def _safe_loss_ratio(frame):
    total_premium = frame['TotalPremium'].sum()
    total_claims = frame['TotalClaims'].sum()
    if total_premium <= 0:
        return np.nan
    return total_claims / total_premium


def plot_loss_ratio_by_category(df, category_col, title=None):
    """Plot loss ratio across categories"""
    loss_ratio = df.groupby(category_col).apply(_safe_loss_ratio).dropna().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    loss_ratio.plot(kind='barh', ax=ax, color='coral')
    ax.set_xlabel('Loss Ratio')
    ax.set_title(title or f'Loss Ratio by {category_col}')
    ax.axvline(x=1, color='red', linestyle='--', label='Break-even')
    ax.axvline(x=loss_ratio.mean(), color='blue', linestyle=':', label=f'Avg: {loss_ratio.mean():.2f}')
    ax.legend()
    
    for i, v in enumerate(loss_ratio.values):
        ax.text(v + 0.02, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    return fig


def plot_claims_distribution(df):
    """Plot distribution of TotalClaims and TotalPremium"""
    fig1, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    claims_positive = df[df['TotalClaims'] > 0]['TotalClaims']
    axes[0].hist(claims_positive, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Total Claims Amount (Rands)')
    axes[0].set_ylabel('Frequency (log scale)')
    axes[0].set_title('Distribution of Claims')
    axes[0].set_yscale('log')
    
    axes[1].hist(df['TotalPremium'], bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[1].set_xlabel('Total Premium Amount (Rands)')
    axes[1].set_ylabel('Frequency (log scale)')
    axes[1].set_title('Distribution of Premiums')
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    df[['TotalClaims', 'TotalPremium']].boxplot(ax=axes2[0])
    axes2[0].set_title('Boxplot - Original Scale')
    
    df_log = df[['TotalClaims', 'TotalPremium']].apply(lambda x: np.log1p(x))
    df_log.boxplot(ax=axes2[1])
    axes2[1].set_title('Boxplot - Log Scale')
    
    plt.tight_layout()
    return fig1, fig2


def plot_temporal_trends(df):
    """Plot claim frequency and severity over time"""
    if 'TransactionMonth' not in df.columns:
        print("TransactionMonth column not found")
        return None
    
    monthly = df.groupby(df['TransactionMonth'].dt.to_period('M')).agg({
        'TotalClaims': ['sum', 'mean'],
        'TotalPremium': 'sum',
        'HasClaim': 'mean'
    }).round(3)
    
    monthly.columns = ['TotalClaims', 'AvgClaimAmount', 'TotalPremium', 'ClaimFrequency']
    monthly['LossRatio'] = np.where(monthly['TotalPremium'] > 0, monthly['TotalClaims'] / monthly['TotalPremium'], np.nan)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    monthly['ClaimFrequency'].plot(ax=axes[0, 0], marker='o', color='blue')
    axes[0, 0].set_title('Claim Frequency Over Time')
    axes[0, 0].set_ylabel('Claim Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    monthly['AvgClaimAmount'].plot(ax=axes[0, 1], marker='s', color='red')
    axes[0, 1].set_title('Average Claim Severity Over Time')
    axes[0, 1].set_ylabel('Avg Claim Amount (Rands)')
    axes[0, 1].grid(True, alpha=0.3)
    
    monthly['LossRatio'].plot(ax=axes[1, 0], marker='^', color='green')
    axes[1, 0].set_title('Loss Ratio Over Time')
    axes[1, 0].set_ylabel('Loss Ratio')
    axes[1, 0].axhline(y=1, color='red', linestyle='--', label='Break-even')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    monthly[['TotalPremium', 'TotalClaims']].plot(ax=axes[1, 1], marker='o')
    axes[1, 1].set_title('Total Premiums vs Claims')
    axes[1, 1].set_ylabel('Amount (Rands)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def detect_outliers_iqr(df, column, threshold=1.5):
    """Detect outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    print(f"\n{'='*50}")
    print(f"Outlier Analysis for {column}")
    print(f"{'='*50}")
    print(f"Lower bound: {lower_bound:,.2f}")
    print(f"Upper bound: {upper_bound:,.2f}")
    print(f"Number of outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    
    return outliers