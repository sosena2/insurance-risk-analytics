"""
Hypothesis testing utilities for insurance risk analysis
A/B testing for provinces, zip codes, and gender differences
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
import warnings
warnings.filterwarnings('ignore')


def test_province_risk(df, province_a, province_b, metric='LossRatio'):
    """
    Test risk difference between two provinces
    
    H₀: No risk difference between provinces
    H₁: Significant risk difference exists
    
    Parameters:
    -----------
    df : DataFrame
        Insurance dataset
    province_a, province_b : str
        Province names to compare
    metric : str
        Metric to compare ('LossRatio', 'Margin', 'HasClaim')
    
    Returns:
    --------
    dict with test results
    """
    # Filter data for the two provinces
    data_a = df[df['Province'] == province_a][metric].dropna()
    data_b = df[df['Province'] == province_b][metric].dropna()
    
    # Perform t-test
    t_stat, p_value = ttest_ind(data_a, data_b)
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(data_a) + np.var(data_b)) / 2)
    cohens_d = abs(data_a.mean() - data_b.mean()) / pooled_std if pooled_std > 0 else 0
    
    result = {
        'hypothesis': f'Risk difference: {province_a} vs {province_b}',
        'metric': metric,
        'test': 'Independent t-test',
        'group_a': province_a,
        'group_b': province_b,
        'mean_a': data_a.mean(),
        'mean_b': data_b.mean(),
        'difference': data_a.mean() - data_b.mean(),
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'effect_size': cohens_d,
        'interpretation': ''
    }
    
    # Add interpretation
    if result['significant']:
        if result['mean_a'] > result['mean_b']:
            result['interpretation'] = f"Reject H₀: {province_a} has significantly HIGHER {metric} than {province_b} (p={p_value:.4f})"
        else:
            result['interpretation'] = f"Reject H₀: {province_a} has significantly LOWER {metric} than {province_b} (p={p_value:.4f})"
    else:
        result['interpretation'] = f"Fail to reject H₀: No significant difference in {metric} between {province_a} and {province_b} (p={p_value:.4f})"
    
    return result


def test_zipcode_margin_difference(df, zipcode_a, zipcode_b):
    """
    Test margin difference between two zip codes
    
    H₀: No significant margin difference between zip codes
    H₁: Significant margin difference exists
    
    Parameters:
    -----------
    df : DataFrame
        Insurance dataset
    zipcode_a, zipcode_b : str
        Postal codes to compare
    
    Returns:
    --------
    dict with test results
    """
    # Filter data for the two zip codes
    data_a = df[df['PostalCode'] == zipcode_a]['Margin'].dropna()
    data_b = df[df['PostalCode'] == zipcode_b]['Margin'].dropna()
    
    # Check sample sizes
    if len(data_a) < 30 or len(data_b) < 30:
        # Use Mann-Whitney U test for small samples
        stat, p_value = stats.mannwhitneyu(data_a, data_b, alternative='two-sided')
        test_name = 'Mann-Whitney U test'
    else:
        # Use t-test for large samples
        t_stat, p_value = ttest_ind(data_a, data_b)
        test_name = 'Independent t-test'
    
    result = {
        'hypothesis': f'Margin difference: Zip {zipcode_a} vs Zip {zipcode_b}',
        'metric': 'Margin',
        'test': test_name,
        'group_a': str(zipcode_a),
        'group_b': str(zipcode_b),
        'mean_a': data_a.mean(),
        'mean_b': data_b.mean(),
        'difference': data_a.mean() - data_b.mean(),
        'sample_size_a': len(data_a),
        'sample_size_b': len(data_b),
        'p_value': p_value,
        'significant': p_value < 0.05,
        'interpretation': ''
    }
    
    if result['significant']:
        result['interpretation'] = f"Reject H₀: Significant margin difference between zip codes (p={p_value:.4f}). Zip {zipcode_a} margin: R{data_a.mean():.0f}, Zip {zipcode_b}: R{data_b.mean():.0f}"
    else:
        result['interpretation'] = f"Fail to reject H₀: No significant margin difference between zip codes (p={p_value:.4f})"
    
    return result


def test_gender_risk_difference(df):
    """
    Test risk difference between Women and Men using claim frequency
    
    H₀: No significant risk difference between genders
    H₁: Significant risk difference exists
    
    Parameters:
    -----------
    df : DataFrame
        Insurance dataset
    
    Returns:
    --------
    dict with test results
    """
    # Create contingency table for claim frequency by gender
    contingency = pd.crosstab(df['Gender'], df['HasClaim'])
    
    # Perform chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    # Calculate claim frequency by gender
    male_freq = df[df['Gender'] == 'Male']['HasClaim'].mean()
    female_freq = df[df['Gender'] == 'Female']['HasClaim'].mean()
    
    result = {
        'hypothesis': 'Risk difference between Genders',
        'metric': 'Claim Frequency',
        'test': 'Chi-square test of independence',
        'male_claim_frequency': male_freq,
        'female_claim_frequency': female_freq,
        'difference': male_freq - female_freq,
        'chi2_statistic': chi2,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'significant': p_value < 0.05,
        'interpretation': ''
    }
    
    if result['significant']:
        if male_freq > female_freq:
            result['interpretation'] = f"Reject H₀: Men have significantly HIGHER claim frequency than Women (p={p_value:.4f})"
        else:
            result['interpretation'] = f"Reject H₀: Women have significantly HIGHER claim frequency than Men (p={p_value:.4f})"
    else:
        result['interpretation'] = f"Fail to reject H₀: No significant risk difference between genders (p={p_value:.4f})"
    
    return result


def test_claim_severity_by_province(df, province_a, province_b):
    """
    Test claim severity difference between two provinces
    (Only for policies with claims)
    
    Parameters:
    -----------
    df : DataFrame
        Insurance dataset
    province_a, province_b : str
        Province names to compare
    
    Returns:
    --------
    dict with test results
    """
    # Filter only policies with claims
    claims_df = df[df['TotalClaims'] > 0]
    
    # Get claim amounts for each province
    claims_a = claims_df[claims_df['Province'] == province_a]['TotalClaims']
    claims_b = claims_df[claims_df['Province'] == province_b]['TotalClaims']
    
    if len(claims_a) < 5 or len(claims_b) < 5:
        return {'error': 'Insufficient sample size for claim severity test'}
    
    # Use t-test for severity comparison
    t_stat, p_value = ttest_ind(claims_a, claims_b)
    
    result = {
        'hypothesis': f'Claim severity difference: {province_a} vs {province_b}',
        'metric': 'Claim Severity (Avg Claim Amount)',
        'test': 'Independent t-test',
        'group_a': province_a,
        'group_b': province_b,
        'mean_severity_a': claims_a.mean(),
        'mean_severity_b': claims_b.mean(),
        'difference': claims_a.mean() - claims_b.mean(),
        'p_value': p_value,
        'significant': p_value < 0.05,
        'interpretation': ''
    }
    
    if result['significant']:
        result['interpretation'] = f"Reject H₀: Claim severity differs significantly between {province_a} (R{claims_a.mean():.0f}) and {province_b} (R{claims_b.mean():.0f}) (p={p_value:.4f})"
    else:
        result['interpretation'] = f"Fail to reject H₀: No significant difference in claim severity between provinces (p={p_value:.4f})"
    
    return result


def create_results_table(results_list):
    """
    Create a formatted results table from hypothesis test results
    
    Parameters:
    -----------
    results_list : list
        List of result dictionaries from test functions
    
    Returns:
    --------
    DataFrame with formatted results
    """
    table_data = []
    for result in results_list:
        if 'error' not in result:
            table_data.append({
                'Hypothesis': result['hypothesis'],
                'Metric': result['metric'],
                'Test Used': result['test'],
                'Group A (Mean)': f"{result.get('mean_a', 'N/A'):.3f}" if 'mean_a' in result else result.get('group_a', 'N/A'),
                'Group B (Mean)': f"{result.get('mean_b', 'N/A'):.3f}" if 'mean_b' in result else result.get('group_b', 'N/A'),
                'Difference': f"{result.get('difference', 0):.3f}",
                'P-Value': f"{result['p_value']:.4f}",
                'Significant (α=0.05)': '✅ YES' if result['significant'] else '❌ NO',
                'Business Interpretation': result['interpretation'][:100] + '...' if len(result['interpretation']) > 100 else result['interpretation']
            })
    
    return pd.DataFrame(table_data)