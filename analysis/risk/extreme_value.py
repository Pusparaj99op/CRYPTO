"""
Extreme Value Theory Implementation

This module implements Extreme Value Theory (EVT) for modeling tail risk and
calculating extreme quantiles using the Generalized Pareto Distribution (GPD).
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from typing import Union, Tuple, Optional

def fit_gpd(returns: Union[np.ndarray, pd.Series],
           threshold: Optional[float] = None,
           method: str = 'mle') -> Tuple[float, float, float]:
    """
    Fit Generalized Pareto Distribution to exceedances over threshold.
    
    Args:
        returns: Array or Series of returns
        threshold: Threshold for exceedances (if None, will be estimated)
        method: Estimation method ('mle' for maximum likelihood, 'pwm' for probability weighted moments)
        
    Returns:
        Tuple[float, float, float]: Shape parameter (xi), scale parameter (sigma), and threshold
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    # If threshold not provided, use 95th percentile
    if threshold is None:
        threshold = np.percentile(returns, 95)
    
    # Get exceedances
    exceedances = returns[returns > threshold] - threshold
    
    if method.lower() == 'mle':
        # Maximum likelihood estimation
        def neg_log_likelihood(params):
            xi, sigma = params
            if sigma <= 0:
                return np.inf
            if xi < 0 and max(exceedances) > -sigma/xi:
                return np.inf
            return -np.sum(stats.genpareto.logpdf(exceedances, xi, scale=sigma))
        
        # Initial guess
        xi_0, sigma_0 = 0.1, np.std(exceedances)
        result = optimize.minimize(neg_log_likelihood, [xi_0, sigma_0])
        xi, sigma = result.x
        
    elif method.lower() == 'pwm':
        # Probability weighted moments estimation
        n = len(exceedances)
        sorted_exceedances = np.sort(exceedances)
        
        # Calculate probability weighted moments
        b0 = np.mean(sorted_exceedances)
        b1 = np.mean(sorted_exceedances * np.arange(n) / (n - 1))
        
        # Estimate parameters
        xi = 2 - b0 / (b0 - 2 * b1)
        sigma = 2 * b0 * b1 / (b0 - 2 * b1)
        
    else:
        raise ValueError(f"Unsupported estimation method: {method}")
    
    return xi, sigma, threshold

def calculate_var_extreme(returns: Union[np.ndarray, pd.Series],
                        confidence_level: float = 0.99,
                        threshold: Optional[float] = None) -> float:
    """
    Calculate Value at Risk using Extreme Value Theory.
    
    Args:
        returns: Array or Series of returns
        confidence_level: Confidence level for VaR calculation
        threshold: Threshold for GPD fitting
        
    Returns:
        float: Extreme VaR value
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    # Fit GPD to exceedances
    xi, sigma, threshold = fit_gpd(returns, threshold)
    
    # Calculate exceedance probability
    p_exceed = len(returns[returns > threshold]) / len(returns)
    
    # Calculate VaR using GPD
    var = threshold + (sigma/xi) * (((1 - confidence_level)/p_exceed)**(-xi) - 1)
    
    return var

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

def plot_tail_fit(returns: Union[np.ndarray, pd.Series],
                 threshold: Optional[float] = None,
                 ax=None) -> None:
    """
    Plot the fitted GPD against empirical data for visual inspection.
    
    Args:
        returns: Array or Series of returns
        threshold: Threshold for GPD fitting
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
    
    # Create empirical CDF
    sorted_exceedances = np.sort(exceedances)
    empirical_cdf = np.arange(1, len(sorted_exceedances) + 1) / len(sorted_exceedances)
    
    # Create theoretical CDF
    theoretical_cdf = stats.genpareto.cdf(sorted_exceedances, xi, scale=sigma)
    
    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(sorted_exceedances, empirical_cdf, 'b-', label='Empirical')
    ax.plot(sorted_exceedances, theoretical_cdf, 'r--', label='GPD Fit')
    ax.set_xlabel('Exceedance')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('GPD Fit to Tail Data')
    ax.legend()
    ax.grid(True)
    
    if ax is None:
        plt.show() 