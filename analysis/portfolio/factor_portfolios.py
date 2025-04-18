"""
Factor-based portfolio construction for cryptocurrency portfolios.
This module provides functions to create and optimize portfolios based on
factor exposures, which is particularly useful for cryptocurrency markets
where traditional factors may be augmented with crypto-specific factors.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import statsmodels.api as sm

def extract_factors(returns, factors, rolling_window=None):
    """
    Extract factor loadings (betas) for each asset.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns for assets
    factors : pd.DataFrame
        Factor returns with the same index as asset returns
    rolling_window : int, optional
        If provided, calculate rolling betas using this window, by default None
        
    Returns:
    --------
    pd.DataFrame or dict of pd.DataFrame
        Factor loadings for each asset (static or rolling)
    """
    assets = returns.columns
    factor_names = factors.columns
    
    if rolling_window is None:
        # Static factor loadings
        factor_loadings = pd.DataFrame(index=assets, columns=factor_names)
        
        for asset in assets:
            # Add constant to factors for regression
            X = sm.add_constant(factors)
            # Fit linear regression
            model = sm.OLS(returns[asset], X).fit()
            # Store factor loadings (skip the constant term)
            factor_loadings.loc[asset] = model.params[1:]
        
        return factor_loadings
    else:
        # Rolling factor loadings
        rolling_loadings = {}
        for asset in assets:
            # Initialize DataFrame for this asset's rolling loadings
            asset_loadings = pd.DataFrame(index=returns.index[rolling_window-1:], columns=factor_names)
            
            # Calculate rolling betas
            for i in range(rolling_window, len(returns) + 1):
                # Select window of returns and factors
                asset_window = returns[asset].iloc[i-rolling_window:i]
                factors_window = factors.iloc[i-rolling_window:i]
                
                # Add constant to factors for regression
                X = sm.add_constant(factors_window)
                # Fit linear regression
                model = sm.OLS(asset_window, X).fit()
                # Store factor loadings (skip the constant term)
                asset_loadings.loc[returns.index[i-1]] = model.params[1:]
            
            rolling_loadings[asset] = asset_loadings
        
        return rolling_loadings

def factor_mimicking_portfolio(returns, factors, target_exposures):
    """
    Create a portfolio with specific factor exposures.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns for assets
    factors : pd.DataFrame
        Factor returns with the same index as asset returns
    target_exposures : dict
        Dictionary mapping factor names to target exposures
        
    Returns:
    --------
    dict
        Dictionary with optimal weights and portfolio metrics
    """
    # Extract factor loadings
    factor_loadings = extract_factors(returns, factors)
    
    # Number of assets
    n_assets = len(returns.columns)
    
    # Convert target exposures to array (same order as in factor_loadings)
    target_array = np.array([target_exposures.get(factor, 0) for factor in factor_loadings.columns])
    
    # Objective function: minimize tracking error to target factor exposures
    def objective(weights):
        # Calculate portfolio factor exposures
        portfolio_exposures = factor_loadings.values.T @ weights
        # Return sum of squared differences (tracking error)
        return np.sum((portfolio_exposures - target_array) ** 2)
    
    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
    bounds = tuple((0, 1) for _ in range(n_assets))  # Long-only constraints
    
    # Initial guess (equal weights)
    init_weights = np.ones(n_assets) / n_assets
    
    # Optimization
    result = minimize(
        objective,
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    # Extract optimal weights
    weights = result['x']
    
    # Calculate portfolio metrics
    cov_matrix = returns.cov() * 252  # Annualized covariance
    expected_return = np.sum(returns.mean() * weights) * 252  # Annualized return
    expected_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # Annualized volatility
    
    # Calculate actual factor exposures
    actual_exposures = factor_loadings.values.T @ weights
    factor_exposure_dict = {factor: exposure for factor, exposure in zip(factor_loadings.columns, actual_exposures)}
    
    return {
        'weights': pd.Series(weights, index=returns.columns),
        'expected_return': expected_return,
        'expected_volatility': expected_volatility,
        'factor_exposures': factor_exposure_dict
    }

def factor_tilted_portfolio(returns, factors, factor_views, risk_tolerance=2):
    """
    Create a portfolio with tilts toward factors with positive expected returns.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns for assets
    factors : pd.DataFrame
        Factor returns with the same index as asset returns
    factor_views : dict
        Dictionary mapping factor names to expected returns
    risk_tolerance : float, optional
        Risk tolerance parameter (higher = more aggressive), by default 2
        
    Returns:
    --------
    dict
        Dictionary with optimal weights and portfolio metrics
    """
    # Extract factor loadings
    factor_loadings = extract_factors(returns, factors)
    
    # Number of assets
    n_assets = len(returns.columns)
    cov_matrix = returns.cov() * 252  # Annualized covariance
    
    # Calculate expected returns based on factor views
    factor_expected_returns = np.array([factor_views.get(factor, 0) for factor in factor_loadings.columns])
    asset_expected_returns = factor_loadings.values @ factor_expected_returns
    
    # Objective function: maximize utility (expected return - risk penalty)
    def objective(weights):
        portfolio_return = np.dot(weights, asset_expected_returns)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        utility = portfolio_return - 0.5 * risk_tolerance * portfolio_variance
        return -utility  # Minimize negative utility
    
    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
    bounds = tuple((0, 1) for _ in range(n_assets))  # Long-only constraints
    
    # Initial guess (equal weights)
    init_weights = np.ones(n_assets) / n_assets
    
    # Optimization
    result = minimize(
        objective,
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    # Extract optimal weights
    weights = result['x']
    
    # Calculate portfolio metrics
    expected_return = np.sum(returns.mean() * weights) * 252  # Annualized return
    expected_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # Annualized volatility
    
    # Calculate factor exposures
    factor_exposures = factor_loadings.values.T @ weights
    factor_exposure_dict = {factor: exposure for factor, exposure in zip(factor_loadings.columns, factor_exposures)}
    
    return {
        'weights': pd.Series(weights, index=returns.columns),
        'expected_return': expected_return,
        'expected_volatility': expected_volatility,
        'factor_exposures': factor_exposure_dict
    }

def multi_factor_optimization(returns, factors, target_return=None, max_factor_exposure=None):
    """
    Create an optimized portfolio with controlled factor exposures.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns for assets
    factors : pd.DataFrame
        Factor returns with the same index as asset returns
    target_return : float, optional
        Target portfolio return, by default None
    max_factor_exposure : float, optional
        Maximum absolute exposure to any factor, by default None
        
    Returns:
    --------
    dict
        Dictionary with optimal weights and portfolio metrics
    """
    # Extract factor loadings
    factor_loadings = extract_factors(returns, factors)
    
    # Number of assets and factors
    n_assets = len(returns.columns)
    n_factors = len(factor_loadings.columns)
    
    # Calculate covariance and expected returns
    cov_matrix = returns.cov() * 252  # Annualized covariance
    expected_returns = returns.mean() * 252  # Annualized returns
    
    # Objective function: minimize portfolio variance
    def objective(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
    
    # Add return constraint if specified
    if target_return is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda x: np.sum(expected_returns * x) - target_return
        })
    
    # Add factor exposure constraints if specified
    if max_factor_exposure is not None:
        for i in range(n_factors):
            factor_loadings_i = factor_loadings.iloc[:, i].values
            # Upper bound constraint
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, loadings=factor_loadings_i: max_factor_exposure - np.dot(loadings, x)
            })
            # Lower bound constraint
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, loadings=factor_loadings_i: np.dot(loadings, x) + max_factor_exposure
            })
    
    # Bounds for weights
    bounds = tuple((0, 1) for _ in range(n_assets))  # Long-only constraints
    
    # Initial guess (equal weights)
    init_weights = np.ones(n_assets) / n_assets
    
    # Optimization
    result = minimize(
        objective,
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    # Extract optimal weights
    weights = result['x']
    
    # Calculate portfolio metrics
    expected_return = np.sum(expected_returns * weights)
    expected_volatility = np.sqrt(objective(weights))
    
    # Calculate factor exposures
    factor_exposures = factor_loadings.values.T @ weights
    factor_exposure_dict = {factor: exposure for factor, exposure in zip(factor_loadings.columns, factor_exposures)}
    
    return {
        'weights': pd.Series(weights, index=returns.columns),
        'expected_return': expected_return,
        'expected_volatility': expected_volatility,
        'factor_exposures': factor_exposure_dict
    }

def create_factor_portfolios(returns, factors, factor_names=None):
    """
    Create pure factor portfolios that capture the return of each factor.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns for assets
    factors : pd.DataFrame
        Factor returns with the same index as asset returns
    factor_names : list, optional
        List of factor names to create portfolios for, by default None (all factors)
        
    Returns:
    --------
    dict
        Dictionary mapping factor names to portfolio weights
    """
    if factor_names is None:
        factor_names = factors.columns
    
    # Extract factor loadings
    factor_loadings = extract_factors(returns, factors)
    
    # Create dictionary to store factor portfolios
    factor_portfolios = {}
    
    # For each factor, create a pure factor portfolio
    for factor in factor_names:
        # Create target exposures: 1 for the current factor, 0 for all others
        target_exposures = {f: 1.0 if f == factor else 0.0 for f in factor_loadings.columns}
        
        # Create the factor-mimicking portfolio
        portfolio = factor_mimicking_portfolio(returns, factors, target_exposures)
        
        # Store the portfolio weights
        factor_portfolios[factor] = portfolio['weights']
    
    return factor_portfolios

def crypto_specific_factors(returns, market_returns=None):
    """
    Create cryptocurrency-specific factors for portfolio construction.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns for crypto assets
    market_returns : pd.Series, optional
        Market benchmark returns, by default None (will use equal-weighted portfolio)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with crypto-specific factor returns
    """
    # If market returns not provided, use equal-weighted portfolio
    if market_returns is None:
        market_returns = returns.mean(axis=1)
    
    # Initialize factors DataFrame
    factors = pd.DataFrame(index=returns.index)
    
    # Market factor
    factors['Market'] = market_returns
    
    # Size factor (small minus big market cap) - proxy with volatility
    volatilities = returns.rolling(30).std().mean()
    small_mask = volatilities > volatilities.median()
    big_mask = ~small_mask
    factors['Size'] = (returns.loc[:, small_mask].mean(axis=1) - 
                       returns.loc[:, big_mask].mean(axis=1))
    
    # Momentum factor (winners minus losers)
    lookback = 90  # 3 months
    momentum = returns.rolling(lookback).mean()
    winner_mask = momentum.iloc[-1] > momentum.iloc[-1].median()
    loser_mask = ~winner_mask
    factors['Momentum'] = (returns.loc[:, winner_mask].mean(axis=1) - 
                         returns.loc[:, loser_mask].mean(axis=1))
    
    # Volatility factor (low minus high volatility)
    vol_lookback = 30  # 1 month
    vol = returns.rolling(vol_lookback).std()
    low_vol_mask = vol.iloc[-1] < vol.iloc[-1].median()
    high_vol_mask = ~low_vol_mask
    factors['Volatility'] = (returns.loc[:, low_vol_mask].mean(axis=1) - 
                           returns.loc[:, high_vol_mask].mean(axis=1))
    
    # Liquidity factor (high minus low liquidity) - proxy with return autocorrelation
    def autocorr(x, lag=1):
        return np.corrcoef(x[:-lag], x[lag:])[0, 1] if len(x) > lag else 0
    
    auto_corr = returns.apply(lambda x: autocorr(x.dropna().values))
    high_liq_mask = auto_corr < auto_corr.median()
    low_liq_mask = ~high_liq_mask
    factors['Liquidity'] = (returns.loc[:, high_liq_mask].mean(axis=1) - 
                          returns.loc[:, low_liq_mask].mean(axis=1))
    
    # Remove any NaN values by forward filling
    factors = factors.fillna(method='ffill')
    
    return factors 