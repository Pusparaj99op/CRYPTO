import numpy as np
import pandas as pd
from scipy import stats

def calculate_nvt_ratio(network_value, transaction_volume, days=30):
    """
    Calculate Network Value to Transactions (NVT) Ratio
    
    Args:
        network_value (float): Market capitalization of the network
        transaction_volume (float): On-chain transaction volume
        days (int): Period in days for transaction volume
        
    Returns:
        float: NVT ratio
    """
    if transaction_volume == 0:
        return float('inf')
        
    # Annualize if period is not 365 days
    if days != 365:
        transaction_volume = transaction_volume * (365 / days)
        
    nvt_ratio = network_value / transaction_volume
    return nvt_ratio

def calculate_mvrv_ratio(market_cap, realized_cap):
    """
    Calculate Market Value to Realized Value (MVRV) Ratio
    
    Args:
        market_cap (float): Current market capitalization
        realized_cap (float): Realized capitalization (sum of UTXOs valued at price when last moved)
        
    Returns:
        float: MVRV ratio
    """
    if realized_cap == 0:
        return float('inf')
        
    mvrv_ratio = market_cap / realized_cap
    return mvrv_ratio

def calculate_stock_to_flow(current_supply, annual_production):
    """
    Calculate Stock-to-Flow Ratio
    
    Args:
        current_supply (float): Current circulating supply
        annual_production (float): Expected annual production
        
    Returns:
        float: Stock-to-Flow ratio
    """
    if annual_production == 0:
        return float('inf')
        
    stock_to_flow = current_supply / annual_production
    return stock_to_flow

def stock_to_flow_model(stock_to_flow, model_parameters={'a': 0.4, 'b': 3.3}):
    """
    Apply Stock-to-Flow model to estimate price
    
    Args:
        stock_to_flow (float): Stock-to-Flow ratio
        model_parameters (dict): Parameters for the model
        
    Returns:
        float: Estimated price
    """
    a = model_parameters.get('a', 0.4)
    b = model_parameters.get('b', 3.3)
    
    # S2F model: price = e^(a + b*ln(S2F))
    estimated_price = np.exp(a + b * np.log(stock_to_flow))
    return estimated_price

def calculate_metcalfe_value(active_users, price_per_user=None, current_market_cap=None):
    """
    Calculate network value based on Metcalfe's Law
    
    Args:
        active_users (int): Number of active users/addresses
        price_per_user (float, optional): Estimated price per user
        current_market_cap (float, optional): Current market cap for coefficient calculation
        
    Returns:
        float: Estimated network value
    """
    if price_per_user is not None:
        # Direct calculation with price per user
        network_value = price_per_user * (active_users ** 2)
    elif current_market_cap is not None:
        # Calculate coefficient from current market cap
        coefficient = current_market_cap / (active_users ** 2)
        network_value = coefficient * (active_users ** 2)
    else:
        return None
        
    return network_value

def calculate_pe_ratio(market_cap, earnings, days=30):
    """
    Calculate pseudo Price-to-Earnings ratio for blockchain networks
    
    Args:
        market_cap (float): Network market capitalization
        earnings (float): Network fee revenue or validator rewards
        days (int): Period in days for earnings
        
    Returns:
        float: P/E ratio
    """
    if earnings == 0:
        return float('inf')
        
    # Annualize if period is not 365 days
    if days != 365:
        annualized_earnings = earnings * (365 / days)
    else:
        annualized_earnings = earnings
        
    pe_ratio = market_cap / annualized_earnings
    return pe_ratio

def fair_value_regression(market_data, features, target='price', test_size=0.2, random_state=42):
    """
    Create regression model to estimate fair value based on fundamental metrics
    
    Args:
        market_data (pd.DataFrame): DataFrame with fundamental metrics and price
        features (list): List of feature columns to use
        target (str): Target variable column name
        test_size (float): Portion of data to use for testing
        random_state (int): Random state for train-test split
        
    Returns:
        dict: Regression model results and predictions
    """
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    
    X = market_data[features]
    y = market_data[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Create and fit model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_all = model.predict(X)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    
    results = {
        'model': model,
        'coefficients': dict(zip(features, model.coef_)),
        'intercept': model.intercept_,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'r2': r2,
        'predictions': pd.DataFrame({
            'actual': market_data[target],
            'predicted': y_pred_all,
            'residual': market_data[target] - y_pred_all
        })
    }
    
    return results
