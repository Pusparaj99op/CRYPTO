"""
Black-Litterman model implementation for cryptocurrency portfolios.
This module provides functions to incorporate investor views into portfolio optimization
using the Black-Litterman model, which is particularly useful for cryptocurrency markets
where subjective views often differ from historical data.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import inv

def market_implied_risk_aversion(market_return, risk_free_rate, market_volatility):
    """
    Calculate the implied risk aversion parameter from market data.
    
    Parameters:
    -----------
    market_return : float
        Expected market return (annualized)
    risk_free_rate : float
        Risk-free rate (annualized)
    market_volatility : float
        Market volatility (annualized)
        
    Returns:
    --------
    float
        Implied risk aversion parameter
    """
    return (market_return - risk_free_rate) / (market_volatility ** 2)

def market_implied_returns(weights, risk_aversion, cov_matrix):
    """
    Calculate the implied equilibrium returns using reverse optimization.
    
    Parameters:
    -----------
    weights : np.array
        Market capitalization weights
    risk_aversion : float
        Risk aversion parameter
    cov_matrix : np.array
        Covariance matrix of asset returns
        
    Returns:
    --------
    np.array
        Implied equilibrium returns
    """
    return risk_aversion * np.dot(cov_matrix, weights)

def black_litterman_returns(market_weights, cov_matrix, views, view_confidences, 
                           risk_aversion, tau=0.05):
    """
    Calculate the posterior expected returns using the Black-Litterman model.
    
    Parameters:
    -----------
    market_weights : np.array
        Market capitalization weights
    cov_matrix : np.array
        Covariance matrix of asset returns
    views : dict
        Dictionary with views where keys are tuples of (asset_idx, weight)
        and values are expected returns. For relative views, weights should sum to 0.
    view_confidences : np.array
        Confidence in each view (inversely related to variance)
    risk_aversion : float
        Risk aversion parameter
    tau : float, optional
        Scalar indicating the uncertainty of the CAPM prior, by default 0.05
        
    Returns:
    --------
    np.array
        Posterior expected returns
    """
    n_assets = len(market_weights)
    
    # Calculate prior returns (Pi) using reverse optimization
    pi = market_implied_returns(market_weights, risk_aversion, cov_matrix)
    
    # Setup the views matrix P and the view returns Q
    n_views = len(views)
    P = np.zeros((n_views, n_assets))
    Q = np.zeros(n_views)
    
    for i, (view_assets, view_return) in enumerate(views.items()):
        for asset_idx, weight in view_assets:
            P[i, asset_idx] = weight
        Q[i] = view_return
    
    # Compute the variance of the views (Omega)
    omega = np.zeros((n_views, n_views))
    np.fill_diagonal(omega, 1 / np.array(view_confidences))
    
    # Calculate the posterior expected returns
    tau_cov = tau * cov_matrix
    A = inv(inv(tau_cov) + np.dot(P.T, np.dot(inv(omega), P)))
    B = np.dot(inv(tau_cov), pi) + np.dot(P.T, np.dot(inv(omega), Q))
    posterior_returns = np.dot(A, B)
    
    return posterior_returns

def optimize_bl_portfolio(returns, market_weights, views, view_confidences,
                         risk_aversion=2.5, risk_free_rate=0.01, tau=0.05):
    """
    Optimize a portfolio using the Black-Litterman model.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns for assets
    market_weights : np.array
        Market capitalization weights
    views : dict
        Dictionary with views where keys are tuples of (asset_idx, weight)
        and values are expected returns
    view_confidences : np.array
        Confidence in each view (inversely related to variance)
    risk_aversion : float, optional
        Risk aversion parameter, by default 2.5
    risk_free_rate : float, optional
        Risk-free rate, by default 0.01
    tau : float, optional
        Scalar indicating the uncertainty of the CAPM prior, by default 0.05
        
    Returns:
    --------
    dict
        Dictionary with optimal weights and portfolio metrics
    """
    n_assets = len(returns.columns)
    cov_matrix = returns.cov() * 252  # Annualized covariance
    
    # Calculate Black-Litterman expected returns
    bl_returns = black_litterman_returns(
        market_weights, cov_matrix, views, view_confidences, risk_aversion, tau
    )
    
    # Optimize the portfolio using mean-variance optimization
    def portfolio_return(weights):
        return np.sum(weights * bl_returns)
    
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    def neg_sharpe_ratio(weights):
        return -(portfolio_return(weights) - risk_free_rate) / portfolio_volatility(weights)
    
    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial weights (use market weights as starting point)
    initial_weights = market_weights
    
    # Optimization
    result = minimize(
        neg_sharpe_ratio,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    # Calculate portfolio metrics
    optimal_weights = result['x']
    expected_return = portfolio_return(optimal_weights)
    expected_volatility = portfolio_volatility(optimal_weights)
    sharpe_ratio = (expected_return - risk_free_rate) / expected_volatility
    
    return {
        'weights': optimal_weights,
        'expected_return': expected_return,
        'expected_volatility': expected_volatility,
        'sharpe_ratio': sharpe_ratio,
        'bl_returns': bl_returns
    }

def prepare_views(asset_names, absolute_views=None, relative_views=None):
    """
    Prepare views for the Black-Litterman model.
    
    Parameters:
    -----------
    asset_names : list
        List of asset names in the portfolio
    absolute_views : dict, optional
        Dictionary mapping asset names to expected returns, by default None
    relative_views : dict, optional
        Dictionary mapping tuples of (asset1, asset2) to the expected
        return difference (asset1 - asset2), by default None
        
    Returns:
    --------
    tuple
        (views, view_confidences) formatted for the Black-Litterman model
    """
    asset_dict = {name: i for i, name in enumerate(asset_names)}
    views = {}
    view_confidences = []
    
    # Process absolute views
    if absolute_views:
        for asset, expected_return in absolute_views.items():
            idx = asset_dict[asset]
            views[((idx, 1.0),)] = expected_return
            # Higher confidence for absolute views (arbitrary scale, adjust as needed)
            view_confidences.append(1.0)
    
    # Process relative views
    if relative_views:
        for (asset1, asset2), return_diff in relative_views.items():
            idx1 = asset_dict[asset1]
            idx2 = asset_dict[asset2]
            views[((idx1, 1.0), (idx2, -1.0))] = return_diff
            # Lower confidence for relative views (arbitrary scale, adjust as needed)
            view_confidences.append(0.8)
    
    return views, np.array(view_confidences)

def calculate_market_cap_weights(market_caps):
    """
    Calculate market capitalization weights.
    
    Parameters:
    -----------
    market_caps : dict or pd.Series
        Market capitalization values keyed by asset
        
    Returns:
    --------
    np.array
        Market capitalization weights
    """
    total_cap = sum(market_caps.values())
    weights = {asset: cap / total_cap for asset, cap in market_caps.items()}
    
    if isinstance(market_caps, pd.Series):
        return pd.Series(weights, index=market_caps.index).values
    return np.array(list(weights.values())) 