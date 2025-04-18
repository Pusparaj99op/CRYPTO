"""
Causal Inference Module

This module provides functions for performing causal inference analysis
on cryptocurrency data.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.tools import cfa_simulation
from scipy import stats

def perform_granger_causality(data: pd.DataFrame,
                            max_lag: int = 5,
                            test: str = 'ssr_chi2test') -> Dict[str, Dict[str, float]]:
    """
    Perform Granger causality tests between time series.
    
    Args:
        data: DataFrame with time series
        max_lag: Maximum lag to test
        test: Test method ('ssr_chi2test', 'ssr_ftest', 'lrtest', 'params_ftest')
        
    Returns:
        Dictionary containing Granger causality test results
    """
    results = {}
    for col1 in data.columns:
        for col2 in data.columns:
            if col1 != col2:
                test_result = grangercausalitytests(
                    data[[col1, col2]],
                    maxlag=max_lag,
                    verbose=False
                )
                
                # Extract p-values for each lag
                p_values = {
                    lag: test_result[lag][0][test][1]
                    for lag in range(1, max_lag + 1)
                }
                
                results[f'{col1}_causes_{col2}'] = p_values
    
    return results

def calculate_impulse_response(data: pd.DataFrame,
                             periods: int = 10,
                             orthogonalized: bool = True) -> Dict[str, np.ndarray]:
    """
    Calculate impulse response functions.
    
    Args:
        data: DataFrame with time series
        periods: Number of periods for response
        orthogonalized: Whether to use orthogonalized IRF
        
    Returns:
        Dictionary containing impulse response results
    """
    # Fit VAR model
    model = VAR(data)
    results = model.fit()
    
    # Calculate impulse responses
    irf = results.irf(periods=periods)
    
    if orthogonalized:
        response = irf.orth_irfs
    else:
        response = irf.irfs
    
    return {
        'impulse_responses': response,
        'cumulative_effects': np.cumsum(response, axis=0),
        'variance_decomposition': irf.orth_irfs
    }

def perform_structural_break_test(data: Union[np.ndarray, pd.Series],
                                test: str = 'chow') -> Dict[str, Union[float, np.ndarray]]:
    """
    Test for structural breaks in time series.
    
    Args:
        data: Time series data
        test: Test method ('chow', 'cusum', 'cusumsq')
        
    Returns:
        Dictionary containing structural break test results
    """
    if test == 'chow':
        # Implement Chow test
        n = len(data)
        break_point = n // 2
        
        # Split data
        data1 = data[:break_point]
        data2 = data[break_point:]
        
        # Calculate statistics
        ssr_pooled = np.sum((data - np.mean(data))**2)
        ssr1 = np.sum((data1 - np.mean(data1))**2)
        ssr2 = np.sum((data2 - np.mean(data2))**2)
        
        # Calculate F-statistic
        f_stat = ((ssr_pooled - (ssr1 + ssr2)) / 2) / ((ssr1 + ssr2) / (n - 4))
        p_value = 1 - stats.f.cdf(f_stat, 2, n - 4)
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'break_point': break_point
        }
    
    elif test == 'cusum':
        # Implement CUSUM test
        residuals = data - np.mean(data)
        cumulative_sum = np.cumsum(residuals)
        sigma = np.std(residuals)
        
        # Calculate test statistic
        test_stat = np.max(np.abs(cumulative_sum)) / (sigma * np.sqrt(len(data)))
        
        return {
            'test_statistic': test_stat,
            'critical_value': 1.36,  # 5% significance level
            'cumulative_sum': cumulative_sum
        }

def calculate_causal_impact(data: pd.DataFrame,
                          intervention_point: int,
                          control_series: List[str],
                          target_series: str) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate causal impact of an intervention.
    
    Args:
        data: DataFrame with time series
        intervention_point: Point of intervention
        control_series: List of control series names
        target_series: Target series name
        
    Returns:
        Dictionary containing causal impact results
    """
    # Split data into pre and post intervention
    pre_data = data[:intervention_point]
    post_data = data[intervention_point:]
    
    # Fit model on pre-intervention data
    X_pre = pre_data[control_series]
    y_pre = pre_data[target_series]
    
    model = VAR(pre_data)
    results = model.fit()
    
    # Predict post-intervention
    forecast = results.forecast(post_data[control_series].values, steps=len(post_data))
    
    # Calculate impact
    actual = post_data[target_series].values
    predicted = forecast[:, 0]  # First column is target series
    
    impact = actual - predicted
    
    return {
        'actual': actual,
        'predicted': predicted,
        'impact': impact,
        'cumulative_impact': np.cumsum(impact),
        'relative_effect': impact / predicted * 100
    }

def perform_counterfactual_analysis(data: pd.DataFrame,
                                  intervention_point: int,
                                  scenario: Dict[str, float]) -> Dict[str, Union[float, np.ndarray]]:
    """
    Perform counterfactual analysis under different scenarios.
    
    Args:
        data: DataFrame with time series
        intervention_point: Point of intervention
        scenario: Dictionary of scenario parameters
        
    Returns:
        Dictionary containing counterfactual analysis results
    """
    # Split data
    pre_data = data[:intervention_point]
    post_data = data[intervention_point:]
    
    # Fit VAR model
    model = VAR(pre_data)
    results = model.fit()
    
    # Generate counterfactual scenarios
    counterfactuals = {}
    for name, params in scenario.items():
        # Modify parameters
        modified_data = post_data.copy()
        for param, value in params.items():
            modified_data[param] = value
        
        # Forecast
        forecast = results.forecast(modified_data.values, steps=len(post_data))
        counterfactuals[name] = forecast
    
    return {
        'actual': post_data.values,
        'counterfactuals': counterfactuals,
        'baseline_forecast': results.forecast(post_data.values, steps=len(post_data))
    } 