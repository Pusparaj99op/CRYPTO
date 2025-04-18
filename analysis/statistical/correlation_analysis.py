"""
Correlation Analysis Module

This module provides functions for analyzing correlations between different
cryptocurrency assets and market factors.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional
from scipy import stats
from statsmodels.stats.correlation_tools import cov_nearest
from statsmodels.tsa.stattools import ccf

def calculate_correlation_matrix(data: pd.DataFrame,
                               method: str = 'pearson') -> pd.DataFrame:
    """
    Calculate correlation matrix for multiple assets.
    
    Args:
        data: DataFrame with asset prices/returns
        method: Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        Correlation matrix
    """
    return data.corr(method=method)

def calculate_partial_correlation(data: pd.DataFrame,
                                target: str,
                                control_vars: List[str]) -> Dict[str, float]:
    """
    Calculate partial correlation controlling for other variables.
    
    Args:
        data: DataFrame with all variables
        target: Target variable name
        control_vars: List of control variable names
        
    Returns:
        Dictionary containing partial correlation results
    """
    # Create design matrix
    X = data[control_vars]
    y = data[target]
    
    # Calculate residuals
    X_resid = pd.DataFrame(index=data.index)
    y_resid = pd.Series(index=data.index)
    
    for col in X.columns:
        X_resid[col] = stats.linregress(X.drop(col, axis=1), X[col])[2]
    
    y_resid = stats.linregress(X, y)[2]
    
    # Calculate partial correlation
    partial_corr = {}
    for col in X.columns:
        partial_corr[col] = stats.pearsonr(X_resid[col], y_resid)[0]
    
    return partial_corr

def calculate_rank_correlation(data: pd.DataFrame,
                             method: str = 'spearman') -> pd.DataFrame:
    """
    Calculate rank-based correlation matrix.
    
    Args:
        data: DataFrame with asset prices/returns
        method: Rank correlation method ('spearman' or 'kendall')
        
    Returns:
        Rank correlation matrix
    """
    return data.rank().corr(method=method)

def calculate_rolling_correlation(data: pd.DataFrame,
                                window: int = 20,
                                method: str = 'pearson') -> pd.DataFrame:
    """
    Calculate rolling correlation between assets.
    
    Args:
        data: DataFrame with asset prices/returns
        window: Rolling window size
        method: Correlation method
        
    Returns:
        DataFrame with rolling correlations
    """
    return data.rolling(window).corr()

def detect_correlation_breakdown(data: pd.DataFrame,
                               window: int = 20,
                               threshold: float = 0.5) -> Dict[str, List[pd.Timestamp]]:
    """
    Detect significant changes in correlation structure.
    
    Args:
        data: DataFrame with asset prices/returns
        window: Rolling window size
        threshold: Correlation change threshold
        
    Returns:
        Dictionary of detected breakdown points
    """
    # Calculate rolling correlations
    rolling_corr = calculate_rolling_correlation(data, window)
    
    # Calculate correlation changes
    corr_changes = rolling_corr.diff()
    
    # Find significant changes
    breakdown_points = {}
    for col in data.columns:
        significant_changes = corr_changes[corr_changes[col].abs() > threshold].index
        if not significant_changes.empty:
            breakdown_points[col] = significant_changes.tolist()
    
    return breakdown_points

def calculate_cross_correlation(data1: pd.Series,
                              data2: pd.Series,
                              max_lag: int = 20) -> Dict[str, np.ndarray]:
    """
    Calculate cross-correlation between two time series.
    
    Args:
        data1: First time series
        data2: Second time series
        max_lag: Maximum lag to consider
        
    Returns:
        Dictionary containing cross-correlation results
    """
    # Calculate cross-correlation
    ccf_values = ccf(data1, data2, adjusted=True)
    
    # Get lags
    lags = np.arange(-max_lag, max_lag + 1)
    
    return {
        'cross_correlation': ccf_values,
        'lags': lags
    } 