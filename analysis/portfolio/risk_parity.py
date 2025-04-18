"""
Risk Parity Portfolio Construction for cryptocurrency portfolios.
This module provides functions to construct portfolios where each asset
contributes equally to the overall portfolio risk, which can be beneficial
for crypto portfolios due to their highly varying volatilities.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

def calc_portfolio_var(weights, cov):
    """
    Calculate portfolio variance given weights and covariance matrix.
    
    Parameters:
    -----------
    weights : np.array
        Portfolio weights
    cov : np.array
        Covariance matrix of asset returns
        
    Returns:
    --------
    float
        Portfolio variance
    """
    return np.dot(weights.T, np.dot(cov, weights))

def calc_risk_contribution(weights, cov):
    """
    Calculate the risk contribution of each asset in the portfolio.
    
    Parameters:
    -----------
    weights : np.array
        Portfolio weights
    cov : np.array
        Covariance matrix of asset returns
        
    Returns:
    --------
    np.array
        Risk contribution of each asset
    """
    port_variance = calc_portfolio_var(weights, cov)
    marginal_contrib = np.dot(cov, weights)
    risk_contrib = np.multiply(marginal_contrib, weights) / port_variance
    return risk_contrib

def risk_budget_objective(weights, args):
    """
    Calculate the sum of squared error between target risk budget
    and actual risk contribution.
    
    Parameters:
    -----------
    weights : np.array
        Portfolio weights
    args : tuple
        Tuple containing covariance matrix and target risk budget
        
    Returns:
    --------
    float
        Sum of squared errors between target and actual risk contributions
    """
    cov, target_risk = args
    actual_risk_contrib = calc_risk_contribution(weights, cov)
    
    # Sum of squared errors
    sse = sum((actual_risk_contrib - target_risk)**2)
    return sse

def risk_parity_portfolio(returns, risk_budget=None, max_iterations=500):
    """
    Construct a risk parity portfolio where each asset contributes
    equally to the total portfolio risk.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns for assets
    risk_budget : np.array, optional
        Target risk budget, by default None (equal risk contribution)
    max_iterations : int, optional
        Maximum number of optimization iterations, by default 500
        
    Returns:
    --------
    dict
        Dictionary with optimal weights and portfolio metrics
    """
    n_assets = len(returns.columns)
    cov = returns.cov().values
    
    # Default to equal risk contribution if no risk budget specified
    if risk_budget is None:
        risk_budget = np.ones(n_assets) / n_assets
    else:
        # Normalize to ensure risk budget sums to 1
        risk_budget = np.array(risk_budget) / sum(risk_budget)
    
    # Initial guess (equal weights)
    init_weights = np.ones(n_assets) / n_assets
    
    # Weight constraints
    bounds = tuple((0.001, 1) for _ in range(n_assets))
    
    # The weights must sum to 1
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # Minimize the risk parity objective
    options = {'maxiter': max_iterations, 'ftol': 1e-9, 'disp': False}
    result = minimize(
        risk_budget_objective,
        init_weights,
        args=(cov, risk_budget),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options=options
    )
    
    # Check if optimization was successful
    if not result['success']:
        print(f"Warning: Optimization did not converge. Status: {result['message']}")
    
    # Calculate portfolio metrics
    weights = result['x']
    portfolio_volatility = np.sqrt(calc_portfolio_var(weights, cov))
    expected_returns = returns.mean().values
    portfolio_return = np.sum(weights * expected_returns)
    risk_contrib = calc_risk_contribution(weights, cov)
    
    return {
        'weights': weights,
        'expected_return': portfolio_return * 252,  # Annualized
        'expected_volatility': portfolio_volatility * np.sqrt(252),  # Annualized
        'risk_contribution': risk_contrib
    }

def risk_parity_with_target_volatility(returns, target_volatility, risk_budget=None):
    """
    Construct a risk parity portfolio with a target volatility.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns for assets
    target_volatility : float
        Target portfolio volatility (annualized)
    risk_budget : np.array, optional
        Target risk budget, by default None (equal risk contribution)
        
    Returns:
    --------
    dict
        Dictionary with optimal weights, portfolio metrics, and leverage
    """
    # Get the risk parity weights
    risk_parity_result = risk_parity_portfolio(returns, risk_budget)
    weights = risk_parity_result['weights']
    current_vol = risk_parity_result['expected_volatility']
    
    # Calculate the leverage needed to reach the target volatility
    leverage = target_volatility / current_vol
    
    # Adjust the weights
    leveraged_weights = weights * leverage
    
    # Calculate the expected return with leverage
    expected_return = risk_parity_result['expected_return'] * leverage
    
    return {
        'weights': weights,
        'leveraged_weights': leveraged_weights,
        'expected_return': expected_return,
        'expected_volatility': target_volatility,
        'leverage': leverage,
        'risk_contribution': risk_parity_result['risk_contribution']
    }

def dynamic_risk_parity(returns, lookback_window=60, rolling_window=20):
    """
    Implement a dynamic risk parity strategy that adjusts weights
    based on rolling covariance estimates.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns for assets
    lookback_window : int, optional
        Lookback window for calculating rolling covariance, by default 60
    rolling_window : int, optional
        Window for rolling risk parity calculation, by default 20
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with time-varying weights for each asset
    """
    assets = returns.columns
    n_assets = len(assets)
    
    # Initialize weights DataFrame
    weights_df = pd.DataFrame(index=returns.index[lookback_window:], columns=assets)
    
    # Calculate weights for each period
    for i in range(lookback_window, len(returns)):
        if i % rolling_window == 0:  # Recalculate every rolling_window periods
            # Use returns from the lookback window
            hist_returns = returns.iloc[i-lookback_window:i]
            
            # Calculate risk parity weights
            rp_result = risk_parity_portfolio(hist_returns)
            
            # Store the weights
            weights_df.loc[returns.index[i]] = rp_result['weights']
        else:
            # Use the last calculated weights
            weights_df.loc[returns.index[i]] = weights_df.iloc[i-lookback_window-1]
    
    # Forward fill any missing values
    weights_df = weights_df.ffill()
    
    return weights_df

def relative_risk_parity(returns, risk_scaling_factors):
    """
    Implement a relative risk parity strategy where assets contribute
    to risk proportionally to their risk scaling factors.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns for assets
    risk_scaling_factors : dict or pd.Series
        Risk scaling factors for each asset
        
    Returns:
    --------
    dict
        Dictionary with optimal weights and portfolio metrics
    """
    assets = returns.columns
    
    # Convert risk_scaling_factors to numpy array in the same order as assets
    if isinstance(risk_scaling_factors, dict):
        risk_budget = np.array([risk_scaling_factors.get(asset, 1.0) for asset in assets])
    else:  # pd.Series
        risk_budget = np.array([risk_scaling_factors.get(asset, 1.0) for asset in assets])
    
    # Normalize risk budget
    risk_budget = risk_budget / sum(risk_budget)
    
    # Calculate risk parity with the specified risk budget
    return risk_parity_portfolio(returns, risk_budget) 