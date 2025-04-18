"""
Value at Risk (VaR) Models

This module implements various Value at Risk calculation methodologies including:
- Historical VaR
- Parametric (Normal) VaR
- Monte Carlo VaR
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, List, Tuple

def calculate_historical_var(returns: Union[np.ndarray, pd.Series], 
                           confidence_level: float = 0.95) -> float:
    """
    Calculate Historical Value at Risk.
    
    Args:
        returns: Array or Series of returns
        confidence_level: Confidence level for VaR calculation (default: 0.95)
        
    Returns:
        float: Historical VaR value
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    return np.percentile(returns, (1 - confidence_level) * 100)

def calculate_parametric_var(returns: Union[np.ndarray, pd.Series],
                           confidence_level: float = 0.95) -> float:
    """
    Calculate Parametric (Normal) Value at Risk.
    
    Args:
        returns: Array or Series of returns
        confidence_level: Confidence level for VaR calculation (default: 0.95)
        
    Returns:
        float: Parametric VaR value
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    mean = np.mean(returns)
    std = np.std(returns)
    z_score = stats.norm.ppf(1 - confidence_level)
    
    return mean + z_score * std

def calculate_var(returns: Union[np.ndarray, pd.Series],
                 method: str = 'historical',
                 confidence_level: float = 0.95,
                 **kwargs) -> float:
    """
    Calculate Value at Risk using specified method.
    
    Args:
        returns: Array or Series of returns
        method: VaR calculation method ('historical' or 'parametric')
        confidence_level: Confidence level for VaR calculation
        **kwargs: Additional parameters for specific methods
        
    Returns:
        float: VaR value
    """
    if method.lower() == 'historical':
        return calculate_historical_var(returns, confidence_level)
    elif method.lower() == 'parametric':
        return calculate_parametric_var(returns, confidence_level)
    else:
        raise ValueError(f"Unsupported VaR calculation method: {method}")

def backtest_var(returns: Union[np.ndarray, pd.Series],
                var_estimates: Union[np.ndarray, pd.Series],
                confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Backtest VaR estimates.
    
    Args:
        returns: Array or Series of actual returns
        var_estimates: Array or Series of VaR estimates
        confidence_level: Confidence level used for VaR calculation
        
    Returns:
        Tuple[float, float]: Number of violations and violation rate
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    if isinstance(var_estimates, pd.Series):
        var_estimates = var_estimates.values
    
    violations = returns < var_estimates
    num_violations = np.sum(violations)
    violation_rate = num_violations / len(returns)
    
    return num_violations, violation_rate 