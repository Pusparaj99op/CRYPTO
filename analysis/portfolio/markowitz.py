"""
Modern Portfolio Theory (MPT) implementation for cryptocurrency portfolios.
This module provides functions to calculate optimal portfolio weights based on
historical returns, covariance, and efficient frontier calculations.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

def portfolio_expected_return(weights, returns):
    """
    Calculate the expected return of a portfolio.
    
    Parameters:
    -----------
    weights : np.array
        Portfolio weights
    returns : pd.DataFrame
        Historical returns for assets
        
    Returns:
    --------
    float
        Expected portfolio return
    """
    return np.sum(returns.mean() * weights) * 252  # Annualized return

def portfolio_volatility(weights, cov_matrix):
    """
    Calculate the volatility/risk of a portfolio.
    
    Parameters:
    -----------
    weights : np.array
        Portfolio weights
    cov_matrix : pd.DataFrame
        Covariance matrix of asset returns
        
    Returns:
    --------
    float
        Portfolio volatility (annualized)
    """
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

def portfolio_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate=0.01):
    """
    Calculate the Sharpe ratio of a portfolio.
    
    Parameters:
    -----------
    weights : np.array
        Portfolio weights
    returns : pd.DataFrame
        Historical returns for assets
    cov_matrix : pd.DataFrame
        Covariance matrix of asset returns
    risk_free_rate : float, optional
        Risk-free rate, by default 0.01
        
    Returns:
    --------
    float
        Portfolio Sharpe ratio
    """
    p_ret = portfolio_expected_return(weights, returns)
    p_vol = portfolio_volatility(weights, cov_matrix)
    return (p_ret - risk_free_rate) / p_vol

def neg_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate=0.01):
    """
    Calculate the negative Sharpe ratio of a portfolio (for minimization).
    
    Parameters:
    -----------
    weights : np.array
        Portfolio weights
    returns : pd.DataFrame
        Historical returns for assets
    cov_matrix : pd.DataFrame
        Covariance matrix of asset returns
    risk_free_rate : float, optional
        Risk-free rate, by default 0.01
        
    Returns:
    --------
    float
        Negative of the portfolio Sharpe ratio
    """
    return -portfolio_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate)

def optimize_portfolio(returns, risk_free_rate=0.01, target_return=None, target_risk=None):
    """
    Optimize a portfolio using MPT.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns for assets
    risk_free_rate : float, optional
        Risk-free rate, by default 0.01
    target_return : float, optional
        Target portfolio return, by default None
    target_risk : float, optional
        Target portfolio risk/volatility, by default None
        
    Returns:
    --------
    dict
        Dictionary with optimal weights and portfolio metrics
    """
    n_assets = len(returns.columns)
    cov_matrix = returns.cov() * 252  # Annualized covariance
    
    # Initial guess (equal weights)
    init_weights = np.array([1/n_assets] * n_assets)
    
    # Constraints
    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # Additional constraints based on target return or risk
    if target_return is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda x: portfolio_expected_return(x, returns) - target_return
        })
    if target_risk is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda x: portfolio_volatility(x, cov_matrix) - target_risk
        })
    
    # Optimization
    opt_result = minimize(
        neg_sharpe_ratio,
        init_weights,
        args=(returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    # Prepare results
    optimal_weights = opt_result['x']
    expected_return = portfolio_expected_return(optimal_weights, returns)
    expected_volatility = portfolio_volatility(optimal_weights, cov_matrix)
    sharpe_ratio = portfolio_sharpe_ratio(optimal_weights, returns, cov_matrix, risk_free_rate)
    
    return {
        'weights': optimal_weights,
        'expected_return': expected_return,
        'expected_volatility': expected_volatility,
        'sharpe_ratio': sharpe_ratio
    }

def min_variance_portfolio(returns):
    """
    Calculate the minimum variance portfolio.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns for assets
        
    Returns:
    --------
    dict
        Dictionary with optimal weights and portfolio metrics
    """
    n_assets = len(returns.columns)
    cov_matrix = returns.cov() * 252  # Annualized covariance
    
    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    # Initial guess (equal weights)
    init_weights = np.array([1/n_assets] * n_assets)
    
    # Constraints
    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # Optimization
    opt_result = minimize(
        portfolio_variance,
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    # Prepare results
    optimal_weights = opt_result['x']
    expected_return = portfolio_expected_return(optimal_weights, returns)
    expected_volatility = portfolio_volatility(optimal_weights, cov_matrix)
    
    return {
        'weights': optimal_weights,
        'expected_return': expected_return,
        'expected_volatility': expected_volatility
    } 