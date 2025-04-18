"""
Portfolio optimization techniques for cryptocurrency portfolios.
This module provides advanced optimization methods that go beyond standard
mean-variance optimization, tailored specifically for the unique characteristics
of cryptocurrency markets such as high volatility, fat tails, and extreme returns.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import cvxpy as cp
from scipy import stats

def max_sharpe_ratio(returns, risk_free_rate=0.01, constraints=None):
    """
    Maximize Sharpe ratio (return per unit of risk).
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns for assets
    risk_free_rate : float, optional
        Risk-free rate, by default 0.01
    constraints : dict, optional
        Additional constraints like max/min weights, by default None
        
    Returns:
    --------
    dict
        Dictionary with optimal weights and portfolio metrics
    """
    n_assets = len(returns.columns)
    
    # Calculate mean returns and covariance
    mean_returns = returns.mean() * 252  # Annualized returns
    cov_matrix = returns.cov() * 252     # Annualized covariance
    
    # Define the objective function to minimize (negative Sharpe ratio)
    def neg_sharpe(weights):
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(portfolio_return - risk_free_rate) / portfolio_volatility
    
    # Initial guess (equal weights)
    init_weights = np.ones(n_assets) / n_assets
    
    # Constraints
    constraint_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
    
    # Add additional constraints if provided
    if constraints:
        if 'max_weight' in constraints:
            for i in range(n_assets):
                constraint_list.append({
                    'type': 'ineq',
                    'fun': lambda x, idx=i: constraints['max_weight'] - x[idx]
                })
        if 'min_weight' in constraints:
            for i in range(n_assets):
                constraint_list.append({
                    'type': 'ineq',
                    'fun': lambda x, idx=i: x[idx] - constraints['min_weight']
                })
    
    # Long-only constraint
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Optimize
    result = minimize(
        neg_sharpe,
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraint_list
    )
    
    # Extract optimal weights
    weights = result['x']
    
    # Calculate portfolio metrics
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    return {
        'weights': pd.Series(weights, index=returns.columns),
        'expected_return': portfolio_return,
        'expected_volatility': portfolio_volatility,
        'sharpe_ratio': sharpe_ratio
    }

def min_cvar_portfolio(returns, alpha=0.05, target_return=None):
    """
    Minimize Conditional Value at Risk (CVaR) / Expected Shortfall.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns for assets
    alpha : float, optional
        Confidence level (e.g., 0.05 for 95% confidence), by default 0.05
    target_return : float, optional
        Target portfolio return, by default None
        
    Returns:
    --------
    dict
        Dictionary with optimal weights and portfolio metrics
    """
    n_assets = len(returns.columns)
    n_samples = len(returns)
    
    # Convert returns to numpy array
    R = returns.values
    
    # Define variables
    w = cp.Variable(n_assets)  # Portfolio weights
    VaR = cp.Variable()        # Value at Risk
    aux = cp.Variable(n_samples)  # Auxiliary variables for CVaR calculation
    
    # Expected returns
    mu = np.mean(R, axis=0)
    
    # Constraints
    constraints = [
        cp.sum(w) == 1,  # Weights sum to 1
        w >= 0           # Long-only constraint
    ]
    
    # Add target return constraint if specified
    if target_return is not None:
        constraints.append(mu @ w >= target_return)
    
    # CVaR calculation
    # For each scenario, aux[i] = max(0, -R[i]@w - VaR)
    for i in range(n_samples):
        constraints.append(aux[i] >= -R[i] @ w - VaR)
        constraints.append(aux[i] >= 0)
    
    # Objective: minimize CVaR
    objective = cp.Minimize(VaR + (1/(alpha*n_samples)) * cp.sum(aux))
    
    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    # Extract optimal weights
    weights = w.value
    
    # Calculate portfolio metrics
    portfolio_return = np.sum(mu * weights) * 252  # Annualized return
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # Annualized volatility
    
    # Calculate realized CVaR
    portfolio_returns = R @ weights
    var = np.percentile(portfolio_returns, alpha * 100)
    cvar = -np.mean(portfolio_returns[portfolio_returns <= var])
    
    return {
        'weights': pd.Series(weights, index=returns.columns),
        'expected_return': portfolio_return,
        'expected_volatility': portfolio_volatility,
        'var': -var * np.sqrt(252),  # Annualized and converted to positive value
        'cvar': cvar * np.sqrt(252)   # Annualized and converted to positive value
    }

def max_diversification_portfolio(returns):
    """
    Maximize portfolio diversification ratio.
    
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
    
    # Calculate asset volatilities and covariance
    vols = returns.std() * np.sqrt(252)  # Annualized volatilities
    cov_matrix = returns.cov() * 252     # Annualized covariance
    
    # Define the objective function to minimize (negative diversification ratio)
    def neg_div_ratio(weights):
        # Weighted sum of individual volatilities
        weighted_sum_vols = np.sum(weights * vols)
        # Portfolio volatility
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        # Diversification ratio = weighted sum of vols / portfolio vol
        div_ratio = weighted_sum_vols / portfolio_vol
        return -div_ratio
    
    # Initial guess (equal weights)
    init_weights = np.ones(n_assets) / n_assets
    
    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
    
    # Long-only constraint
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Optimize
    result = minimize(
        neg_div_ratio,
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    # Extract optimal weights
    weights = result['x']
    
    # Calculate portfolio metrics
    mean_returns = returns.mean() * 252  # Annualized returns
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Calculate diversification ratio
    weighted_sum_vols = np.sum(weights * vols)
    div_ratio = weighted_sum_vols / portfolio_volatility
    
    return {
        'weights': pd.Series(weights, index=returns.columns),
        'expected_return': portfolio_return,
        'expected_volatility': portfolio_volatility,
        'diversification_ratio': div_ratio
    }

def max_decorrelation_portfolio(returns):
    """
    Maximize portfolio decorrelation by minimizing average correlation.
    
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
    
    # Calculate correlation matrix
    corr_matrix = returns.corr().values
    
    # Define the objective function to minimize (average correlation)
    def avg_correlation(weights):
        weighted_corr_sum = 0
        weight_sum_squared = np.sum(weights ** 2)
        
        for i in range(n_assets):
            for j in range(n_assets):
                if i != j:
                    weighted_corr_sum += weights[i] * weights[j] * corr_matrix[i, j]
        
        # Normalize by sum of squared weights
        return weighted_corr_sum / (weight_sum_squared - np.sum(weights ** 2))
    
    # Initial guess (equal weights)
    init_weights = np.ones(n_assets) / n_assets
    
    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
    
    # Long-only constraint
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Optimize
    result = minimize(
        avg_correlation,
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    # Extract optimal weights
    weights = result['x']
    
    # Calculate portfolio metrics
    mean_returns = returns.mean() * 252  # Annualized returns
    cov_matrix = returns.cov() * 252     # Annualized covariance
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    return {
        'weights': pd.Series(weights, index=returns.columns),
        'expected_return': portfolio_return,
        'expected_volatility': portfolio_volatility,
        'avg_correlation': avg_correlation(weights)
    }

def max_entropy_portfolio(returns):
    """
    Maximize portfolio entropy (diversification measured by information theory).
    
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
    
    # Define the objective function to minimize (negative entropy)
    def neg_entropy(weights):
        # Filter out zero weights to avoid log(0)
        non_zero_weights = weights[weights > 1e-10]
        if len(non_zero_weights) == 0:
            return 0
        # Entropy = -sum(w_i * log(w_i))
        return np.sum(non_zero_weights * np.log(non_zero_weights))
    
    # Initial guess (equal weights)
    init_weights = np.ones(n_assets) / n_assets
    
    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
    
    # Long-only constraint
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Optimize
    result = minimize(
        neg_entropy,
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    # Extract optimal weights
    weights = result['x']
    
    # Calculate portfolio metrics
    mean_returns = returns.mean() * 252  # Annualized returns
    cov_matrix = returns.cov() * 252     # Annualized covariance
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Calculate entropy
    non_zero_weights = weights[weights > 1e-10]
    entropy = -np.sum(non_zero_weights * np.log(non_zero_weights))
    
    return {
        'weights': pd.Series(weights, index=returns.columns),
        'expected_return': portfolio_return,
        'expected_volatility': portfolio_volatility,
        'entropy': entropy
    }

def robust_optimization(returns, uncertainty=0.1):
    """
    Robust portfolio optimization that accounts for uncertainty in estimates.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns for assets
    uncertainty : float, optional
        Uncertainty level for returns and covariance, by default 0.1
        
    Returns:
    --------
    dict
        Dictionary with optimal weights and portfolio metrics
    """
    n_assets = len(returns.columns)
    
    # Calculate mean returns and covariance
    mean_returns = returns.mean() * 252  # Annualized returns
    cov_matrix = returns.cov() * 252     # Annualized covariance
    
    # Create worst-case estimates with uncertainty
    worst_returns = mean_returns - uncertainty * np.abs(mean_returns)
    worst_cov = cov_matrix * (1 + uncertainty)
    
    # Define variables
    w = cp.Variable(n_assets)  # Portfolio weights
    
    # Objective: maximize worst-case return with robust risk constraint
    objective = cp.Maximize(worst_returns @ w)
    
    # Constraints
    constraints = [
        cp.sum(w) == 1,  # Weights sum to 1
        w >= 0,          # Long-only constraint
        cp.quad_form(w, worst_cov) <= 0.1  # Limit risk (adjust the bound as needed)
    ]
    
    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    # Extract optimal weights
    weights = w.value
    
    # Calculate portfolio metrics using original estimates
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Calculate worst-case metrics
    worst_case_return = np.sum(worst_returns * weights)
    worst_case_volatility = np.sqrt(np.dot(weights.T, np.dot(worst_cov, weights)))
    
    return {
        'weights': pd.Series(weights, index=returns.columns),
        'expected_return': portfolio_return,
        'expected_volatility': portfolio_volatility,
        'worst_case_return': worst_case_return,
        'worst_case_volatility': worst_case_volatility
    }

def resampled_efficient_frontier(returns, num_portfolios=1000, risk_aversion=2):
    """
    Michaud's resampled efficient frontier method for more robust portfolios.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns for assets
    num_portfolios : int, optional
        Number of simulations, by default 1000
    risk_aversion : float, optional
        Risk aversion parameter, by default 2
        
    Returns:
    --------
    dict
        Dictionary with optimal weights and portfolio metrics
    """
    n_assets = len(returns.columns)
    n_samples = len(returns)
    
    # Calculate mean returns and covariance
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    
    # Mean-variance utility function
    def mv_utility(weights, returns, cov):
        port_return = np.sum(returns * weights)
        port_variance = np.dot(weights.T, np.dot(cov, weights))
        return port_return - 0.5 * risk_aversion * port_variance
    
    # Optimize function for a given mean and covariance
    def optimize_portfolio(mu, sigma):
        def neg_utility(weights):
            return -mv_utility(weights, mu, sigma)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        result = minimize(
            neg_utility,
            np.ones(n_assets) / n_assets,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result['x']
    
    # Generate simulated returns
    np.random.seed(42)  # For reproducibility
    all_weights = np.zeros((num_portfolios, n_assets))
    
    for i in range(num_portfolios):
        # Resample returns from a multivariate normal
        sim_returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_samples)
        
        # Calculate sample mean and covariance
        sim_mean = np.mean(sim_returns, axis=0)
        sim_cov = np.cov(sim_returns.T)
        
        # Optimize portfolio for this simulation
        all_weights[i] = optimize_portfolio(sim_mean, sim_cov)
    
    # Average the weights across simulations
    avg_weights = np.mean(all_weights, axis=0)
    
    # Normalize weights to sum to 1
    avg_weights = avg_weights / np.sum(avg_weights)
    
    # Calculate portfolio metrics using original estimates
    portfolio_return = np.sum(mean_returns * avg_weights) * 252  # Annualized
    portfolio_volatility = np.sqrt(np.dot(avg_weights.T, np.dot(cov_matrix, avg_weights))) * np.sqrt(252)  # Annualized
    
    return {
        'weights': pd.Series(avg_weights, index=returns.columns),
        'expected_return': portfolio_return,
        'expected_volatility': portfolio_volatility
    }

def custom_crypto_optimizer(returns, momentum_weight=0.3, volatility_weight=0.3, volume_data=None):
    """
    Custom optimization for crypto portfolios incorporating momentum and volatility.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns for assets
    momentum_weight : float, optional
        Weight given to momentum factor, by default 0.3
    volatility_weight : float, optional
        Weight given to volatility factor, by default 0.3
    volume_data : pd.DataFrame, optional
        Trading volume data for liquidity considerations, by default None
        
    Returns:
    --------
    dict
        Dictionary with optimal weights and portfolio metrics
    """
    n_assets = len(returns.columns)
    
    # Calculate standard metrics
    mean_returns = returns.mean() * 252  # Annualized returns
    cov_matrix = returns.cov() * 252     # Annualized covariance
    volatilities = returns.std() * np.sqrt(252)  # Annualized volatilities
    
    # Calculate momentum (e.g., last 3-month performance)
    momentum = returns.iloc[-60:].mean() * 252  # Using last ~3 months
    
    # Normalize metrics to 0-1 scale
    norm_returns = (mean_returns - mean_returns.min()) / (mean_returns.max() - mean_returns.min() + 1e-10)
    norm_momentum = (momentum - momentum.min()) / (momentum.max() - momentum.min() + 1e-10)
    
    # For volatility, lower is better
    inv_vol = 1 / (volatilities + 1e-10)  # Add small constant to avoid division by zero
    norm_inv_vol = (inv_vol - inv_vol.min()) / (inv_vol.max() - inv_vol.min() + 1e-10)
    
    # Calculate liquidity score if volume data is provided
    if volume_data is not None:
        avg_volume = volume_data.mean()
        norm_volume = (avg_volume - avg_volume.min()) / (avg_volume.max() - avg_volume.min() + 1e-10)
        # Combined score (adjust weights as needed)
        combined_score = (1 - momentum_weight - volatility_weight) * norm_returns + \
                         momentum_weight * norm_momentum + \
                         volatility_weight * norm_inv_vol + \
                         0.2 * norm_volume  # 20% weight to liquidity
    else:
        # Combined score without volume data
        remaining_weight = 1 - momentum_weight - volatility_weight
        combined_score = remaining_weight * norm_returns + \
                         momentum_weight * norm_momentum + \
                         volatility_weight * norm_inv_vol
    
    # Initial weights based on combined score
    weights = combined_score / combined_score.sum()
    
    # Ensure weights sum to 1
    weights = weights / weights.sum()
    
    # Calculate portfolio metrics
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility
    
    return {
        'weights': weights,
        'expected_return': portfolio_return,
        'expected_volatility': portfolio_volatility,
        'sharpe_ratio': sharpe_ratio
    } 