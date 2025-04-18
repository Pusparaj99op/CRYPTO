"""
Descriptive Statistics Module

This module provides functions for calculating various descriptive statistics
and measures for cryptocurrency data analysis.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional
from scipy import stats

def calculate_summary_stats(data: Union[np.ndarray, pd.Series, pd.DataFrame]) -> Dict[str, float]:
    """
    Calculate basic summary statistics for the data.
    
    Args:
        data: Input data array, series, or dataframe
        
    Returns:
        Dictionary containing summary statistics
    """
    if isinstance(data, pd.DataFrame):
        return {col: calculate_summary_stats(data[col]) for col in data.columns}
    
    if isinstance(data, pd.Series):
        data = data.values
    
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'range': np.max(data) - np.min(data),
        'iqr': stats.iqr(data),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data)
    }

def calculate_distribution_stats(data: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """
    Calculate distribution-related statistics.
    
    Args:
        data: Input data array or series
        
    Returns:
        Dictionary containing distribution statistics
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    # Fit normal distribution
    mu, std = stats.norm.fit(data)
    
    # Calculate Kolmogorov-Smirnov test
    ks_stat, ks_pval = stats.kstest(data, 'norm', args=(mu, std))
    
    # Calculate Anderson-Darling test
    ad_stat, ad_crit, ad_sig = stats.anderson(data, dist='norm')
    
    return {
        'normal_mu': mu,
        'normal_std': std,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pval,
        'ad_statistic': ad_stat,
        'ad_critical_values': ad_crit,
        'ad_significance_levels': ad_sig
    }

def calculate_moments(data: Union[np.ndarray, pd.Series], 
                     order: int = 4) -> Dict[str, float]:
    """
    Calculate statistical moments up to specified order.
    
    Args:
        data: Input data array or series
        order: Maximum order of moments to calculate (default: 4)
        
    Returns:
        Dictionary containing moments
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    moments = {}
    for i in range(1, order + 1):
        moments[f'moment_{i}'] = stats.moment(data, moment=i)
        if i > 1:
            moments[f'central_moment_{i}'] = stats.moment(data, moment=i, center=True)
    
    return moments

def calculate_quantiles(data: Union[np.ndarray, pd.Series],
                       quantiles: List[float] = None) -> Dict[str, float]:
    """
    Calculate specified quantiles of the data.
    
    Args:
        data: Input data array or series
        quantiles: List of quantiles to calculate (default: [0.25, 0.5, 0.75])
        
    Returns:
        Dictionary containing quantile values
    """
    if quantiles is None:
        quantiles = [0.25, 0.5, 0.75]
    
    if isinstance(data, pd.Series):
        data = data.values
    
    return {f'q_{q}': np.quantile(data, q) for q in quantiles}

def calculate_rolling_stats(data: Union[np.ndarray, pd.Series],
                          window: int = 20) -> Dict[str, np.ndarray]:
    """
    Calculate rolling window statistics.
    
    Args:
        data: Input data array or series
        window: Rolling window size (default: 20)
        
    Returns:
        Dictionary containing rolling statistics
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    return {
        'rolling_mean': data.rolling(window).mean().values,
        'rolling_std': data.rolling(window).std().values,
        'rolling_skew': data.rolling(window).skew().values,
        'rolling_kurt': data.rolling(window).kurt().values,
        'rolling_min': data.rolling(window).min().values,
        'rolling_max': data.rolling(window).max().values
    } 