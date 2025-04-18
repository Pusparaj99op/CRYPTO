"""
Tail Risk Analysis

This module implements tail risk analysis tools for assessing extreme market
movements and their impact on portfolio performance.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Tuple, List, Optional
from .extreme_value import fit_gpd, calculate_var_extreme

def calculate_tail_risk(returns: Union[np.ndarray, pd.Series],
                       confidence_level: float = 0.99,
                       threshold: Optional[float] = None) -> Dict[str, float]:
    """
    Calculate comprehensive tail risk metrics.
    
    Args:
        returns: Array or Series of returns
        confidence_level: Confidence level for tail risk calculation
        threshold: Threshold for tail risk analysis
        
    Returns:
        Dict[str, float]: Dictionary of tail risk metrics
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    # Fit GPD to tail
    xi, sigma, threshold = fit_gpd(returns, threshold)
    
    # Calculate tail risk metrics
    var = calculate_var_extreme(returns, confidence_level, threshold)
    es = calculate_es_extreme(returns, confidence_level, threshold)
    
    # Calculate tail index (Hill estimator)
    tail_index = 1 / xi if xi > 0 else np.inf
    
    # Calculate expected shortfall ratio
    es_ratio = es / abs(var)
    
    # Calculate tail dependence
    left_tail_dep = calculate_tail_dependence(returns, side='left')
    right_tail_dep = calculate_tail_dependence(returns, side='right')
    
    return {
        'VaR': var,
        'Expected_Shortfall': es,
        'Tail_Index': tail_index,
        'ES_Ratio': es_ratio,
        'Left_Tail_Dependence': left_tail_dep,
        'Right_Tail_Dependence': right_tail_dep,
        'Threshold': threshold,
        'Shape_Parameter': xi,
        'Scale_Parameter': sigma
    }

def calculate_tail_dependence(returns: Union[np.ndarray, pd.Series],
                            side: str = 'left',
                            threshold: float = 0.05) -> float:
    """
    Calculate tail dependence coefficient.
    
    Args:
        returns: Array or Series of returns
        side: Which tail to analyze ('left' or 'right')
        threshold: Threshold for tail analysis
        
    Returns:
        float: Tail dependence coefficient
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    if side.lower() == 'left':
        tail_returns = returns[returns < np.percentile(returns, threshold * 100)]
    else:
        tail_returns = returns[returns > np.percentile(returns, (1 - threshold) * 100)]
    
    # Calculate tail dependence as the ratio of joint exceedances
    n = len(returns)
    n_tail = len(tail_returns)
    tail_dependence = n_tail / (n * threshold)
    
    return tail_dependence

def analyze_tail_dependence(returns1: Union[np.ndarray, pd.Series],
                          returns2: Union[np.ndarray, pd.Series],
                          threshold: float = 0.05) -> Dict[str, float]:
    """
    Analyze tail dependence between two return series.
    
    Args:
        returns1: First return series
        returns2: Second return series
        threshold: Threshold for tail analysis
        
    Returns:
        Dict[str, float]: Tail dependence analysis results
    """
    if isinstance(returns1, pd.Series):
        returns1 = returns1.values
    if isinstance(returns2, pd.Series):
        returns2 = returns2.values
    
    # Calculate quantiles
    q1_left = np.percentile(returns1, threshold * 100)
    q1_right = np.percentile(returns1, (1 - threshold) * 100)
    q2_left = np.percentile(returns2, threshold * 100)
    q2_right = np.percentile(returns2, (1 - threshold) * 100)
    
    # Calculate joint exceedances
    left_tail_joint = np.mean((returns1 < q1_left) & (returns2 < q2_left))
    right_tail_joint = np.mean((returns1 > q1_right) & (returns2 > q2_right))
    
    # Calculate tail dependence coefficients
    left_dependence = left_tail_joint / threshold
    right_dependence = right_tail_joint / threshold
    
    return {
        'Left_Tail_Dependence': left_dependence,
        'Right_Tail_Dependence': right_dependence,
        'Left_Tail_Joint_Probability': left_tail_joint,
        'Right_Tail_Joint_Probability': right_tail_joint
    }

def calculate_es_extreme(returns: Union[np.ndarray, pd.Series],
                       confidence_level: float = 0.99,
                       threshold: Optional[float] = None) -> float:
    """
    Calculate Expected Shortfall using Extreme Value Theory.
    
    Args:
        returns: Array or Series of returns
        confidence_level: Confidence level for ES calculation
        threshold: Threshold for GPD fitting
        
    Returns:
        float: Extreme Expected Shortfall value
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    # Fit GPD to exceedances
    xi, sigma, threshold = fit_gpd(returns, threshold)
    
    # Calculate VaR
    var = calculate_var_extreme(returns, confidence_level, threshold)
    
    # Calculate ES using GPD
    p_exceed = len(returns[returns > threshold]) / len(returns)
    es = (var + sigma - xi*threshold) / (1 - xi)
    
    return es

def plot_tail_analysis(returns: Union[np.ndarray, pd.Series],
                     threshold: Optional[float] = None,
                     ax=None) -> None:
    """
    Plot tail analysis results.
    
    Args:
        returns: Array or Series of returns
        threshold: Threshold for tail analysis
        ax: Matplotlib axis object (if None, will create new figure)
    """
    import matplotlib.pyplot as plt
    
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    if threshold is None:
        threshold = np.percentile(returns, 95)
    
    # Fit GPD
    xi, sigma, threshold = fit_gpd(returns, threshold)
    
    # Get exceedances
    exceedances = returns[returns > threshold] - threshold
    
    # Create QQ plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Theoretical quantiles
    theoretical_quantiles = stats.genpareto.ppf(
        np.linspace(0.01, 0.99, len(exceedances)),
        xi,
        scale=sigma
    )
    
    # Empirical quantiles
    empirical_quantiles = np.sort(exceedances)
    
    # Plot
    ax.scatter(theoretical_quantiles, empirical_quantiles)
    ax.plot([min(theoretical_quantiles), max(theoretical_quantiles)],
            [min(theoretical_quantiles), max(theoretical_quantiles)],
            'r--')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Empirical Quantiles')
    ax.set_title('QQ Plot of Tail Data')
    ax.grid(True)
    
    if ax is None:
        plt.show() 