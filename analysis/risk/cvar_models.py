"""
Conditional Value at Risk (CVaR) Models

This module implements Conditional Value at Risk (Expected Shortfall) calculation
methodologies for risk assessment.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Tuple

def calculate_cvar(returns: Union[np.ndarray, pd.Series],
                  confidence_level: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (Expected Shortfall).
    
    Args:
        returns: Array or Series of returns
        confidence_level: Confidence level for CVaR calculation (default: 0.95)
        
    Returns:
        float: CVaR value
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    var = np.percentile(returns, (1 - confidence_level) * 100)
    cvar = returns[returns <= var].mean()
    
    return cvar

def calculate_expected_shortfall(returns: Union[np.ndarray, pd.Series],
                               confidence_level: float = 0.95) -> float:
    """
    Calculate Expected Shortfall (alternative name for CVaR).
    
    Args:
        returns: Array or Series of returns
        confidence_level: Confidence level for calculation (default: 0.95)
        
    Returns:
        float: Expected Shortfall value
    """
    return calculate_cvar(returns, confidence_level)

def calculate_parametric_cvar(returns: Union[np.ndarray, pd.Series],
                            confidence_level: float = 0.95) -> float:
    """
    Calculate Parametric (Normal) Conditional Value at Risk.
    
    Args:
        returns: Array or Series of returns
        confidence_level: Confidence level for CVaR calculation (default: 0.95)
        
    Returns:
        float: Parametric CVaR value
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    mean = np.mean(returns)
    std = np.std(returns)
    z_score = stats.norm.ppf(1 - confidence_level)
    
    # Calculate the normal density at the VaR level
    density = stats.norm.pdf(z_score)
    
    # Calculate CVaR using the normal distribution formula
    cvar = mean - std * (density / (1 - confidence_level))
    
    return cvar

def backtest_cvar(returns: Union[np.ndarray, pd.Series],
                 cvar_estimates: Union[np.ndarray, pd.Series],
                 confidence_level: float = 0.95) -> Tuple[float, float, float]:
    """
    Backtest CVaR estimates.
    
    Args:
        returns: Array or Series of actual returns
        cvar_estimates: Array or Series of CVaR estimates
        confidence_level: Confidence level used for CVaR calculation
        
    Returns:
        Tuple[float, float, float]: Number of violations, violation rate, and average violation magnitude
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    if isinstance(cvar_estimates, pd.Series):
        cvar_estimates = cvar_estimates.values
    
    violations = returns < cvar_estimates
    num_violations = np.sum(violations)
    violation_rate = num_violations / len(returns)
    
    if num_violations > 0:
        avg_violation_magnitude = np.mean(returns[violations] - cvar_estimates[violations])
    else:
        avg_violation_magnitude = 0.0
    
    return num_violations, violation_rate, avg_violation_magnitude 