"""
Efficient Frontier calculations for cryptocurrency portfolio optimization.
This module provides functions to calculate and visualize the efficient frontier
for crypto asset portfolios.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def portfolio_return(weights, returns):
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
        Expected portfolio return (annualized)
    """
    return np.sum(returns.mean() * weights) * 252

def portfolio_volatility(weights, cov_matrix):
    """
    Calculate the volatility of a portfolio.
    
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

def minimize_volatility(target_return, returns, cov_matrix):
    """
    Find the portfolio with minimum volatility for a given target return.
    
    Parameters:
    -----------
    target_return : float
        Target portfolio return
    returns : pd.DataFrame
        Historical returns for assets
    cov_matrix : pd.DataFrame
        Covariance matrix of asset returns
        
    Returns:
    --------
    np.array
        Optimal portfolio weights
    """
    n_assets = len(returns.columns)
    args = (cov_matrix,)
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: portfolio_return(x, returns) - target_return}
    )
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    initial_weights = np.array([1/n_assets] * n_assets)
    
    result = minimize(
        lambda weights: portfolio_volatility(weights, cov_matrix),
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result['x']

def calculate_efficient_frontier(returns, n_points=100):
    """
    Calculate the efficient frontier points.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns for assets
    n_points : int, optional
        Number of points to calculate for the efficient frontier, by default 100
        
    Returns:
    --------
    tuple
        (returns, volatilities) for the efficient frontier points
    """
    cov_matrix = returns.cov() * 252  # Annualized covariance
    
    # Calculate min and max returns
    n_assets = len(returns.columns)
    init_weights = np.array([1/n_assets] * n_assets)
    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # Find minimum variance portfolio
    min_func = lambda x: portfolio_volatility(x, cov_matrix)
    min_result = minimize(min_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    min_vol_ret = portfolio_return(min_result['x'], returns)
    
    # Find maximum return portfolio - usually a single asset portfolio
    max_func = lambda x: -portfolio_return(x, returns)
    max_result = minimize(max_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    max_ret = portfolio_return(max_result['x'], returns)
    
    # Generate target returns
    target_returns = np.linspace(min_vol_ret, max_ret, n_points)
    efficient_returns = []
    efficient_volatilities = []
    
    for target in target_returns:
        weights = minimize_volatility(target, returns, cov_matrix)
        efficient_returns.append(portfolio_return(weights, returns))
        efficient_volatilities.append(portfolio_volatility(weights, cov_matrix))
    
    return np.array(efficient_returns), np.array(efficient_volatilities)

def find_optimal_portfolio(returns, risk_free_rate=0.01):
    """
    Find the optimal portfolio based on the Sharpe ratio.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns for assets
    risk_free_rate : float, optional
        Risk-free rate, by default 0.01
        
    Returns:
    --------
    dict
        Dictionary with optimal portfolio information
    """
    n_assets = len(returns.columns)
    cov_matrix = returns.cov() * 252  # Annualized covariance
    
    def neg_sharpe_ratio(weights):
        p_ret = portfolio_return(weights, returns)
        p_vol = portfolio_volatility(weights, cov_matrix)
        return -(p_ret - risk_free_rate) / p_vol
    
    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial weights
    initial_weights = np.array([1/n_assets] * n_assets)
    
    # Optimization
    result = minimize(
        neg_sharpe_ratio,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    optimal_weights = result['x']
    optimal_return = portfolio_return(optimal_weights, returns)
    optimal_volatility = portfolio_volatility(optimal_weights, cov_matrix)
    optimal_sharpe = (optimal_return - risk_free_rate) / optimal_volatility
    
    return {
        'weights': optimal_weights,
        'return': optimal_return,
        'volatility': optimal_volatility,
        'sharpe_ratio': optimal_sharpe
    }

def plot_efficient_frontier(returns, risk_free_rate=0.01, show_assets=True, show_optimal=True):
    """
    Plot the efficient frontier.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns for assets
    risk_free_rate : float, optional
        Risk-free rate, by default 0.01
    show_assets : bool, optional
        Whether to show individual assets, by default True
    show_optimal : bool, optional
        Whether to show the optimal portfolio, by default True
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    efficient_returns, efficient_volatilities = calculate_efficient_frontier(returns)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(efficient_volatilities, efficient_returns, 'b-', linewidth=2, label='Efficient Frontier')
    
    # Plot individual assets if requested
    if show_assets:
        n_assets = len(returns.columns)
        asset_returns = returns.mean() * 252
        asset_volatilities = returns.std() * np.sqrt(252)
        ax.scatter(asset_volatilities, asset_returns, marker='o', s=100, 
                   c='red', alpha=0.7, label='Individual Assets')
        
        # Add asset labels
        for i, txt in enumerate(returns.columns):
            ax.annotate(txt, (asset_volatilities[i], asset_returns[i]), 
                        xytext=(5, 5), textcoords='offset points')
    
    # Plot optimal portfolio if requested
    if show_optimal:
        optimal_portfolio = find_optimal_portfolio(returns, risk_free_rate)
        ax.scatter(optimal_portfolio['volatility'], optimal_portfolio['return'], 
                  marker='*', s=200, c='green', label='Optimal Portfolio')
        
        # Plot the Capital Market Line
        x_vals = np.linspace(0, max(efficient_volatilities) * 1.2, 50)
        y_vals = risk_free_rate + (optimal_portfolio['return'] - risk_free_rate) / \
                optimal_portfolio['volatility'] * x_vals
        ax.plot(x_vals, y_vals, 'r-', label='Capital Market Line')
    
    # Set labels and title
    ax.set_title('Efficient Frontier', fontsize=14)
    ax.set_xlabel('Volatility (Standard Deviation)', fontsize=12)
    ax.set_ylabel('Expected Return', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig

def get_portfolio_at_volatility(target_volatility, returns):
    """
    Find a portfolio with a target volatility.
    
    Parameters:
    -----------
    target_volatility : float
        Target portfolio volatility
    returns : pd.DataFrame
        Historical returns for assets
        
    Returns:
    --------
    dict
        Portfolio weights and metrics
    """
    n_assets = len(returns.columns)
    cov_matrix = returns.cov() * 252  # Annualized covariance
    
    def target_func(weights):
        return abs(portfolio_volatility(weights, cov_matrix) - target_volatility)
    
    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial weights
    initial_weights = np.array([1/n_assets] * n_assets)
    
    # Optimization
    result = minimize(
        target_func,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    weights = result['x']
    return {
        'weights': weights,
        'return': portfolio_return(weights, returns),
        'volatility': portfolio_volatility(weights, cov_matrix)
    } 