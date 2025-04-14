import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def analyze_active_addresses(address_data, period_days=30):
    """
    Analyze active addresses over a time period
    
    Args:
        address_data (pd.DataFrame): DataFrame with 'date' and 'active_addresses' columns
        period_days (int): Period to analyze in days
        
    Returns:
        dict: Active address metrics
    """
    # Ensure data is sorted by date
    address_data = address_data.sort_values('date')
    
    # Calculate date range
    end_date = address_data['date'].max()
    start_date = end_date - timedelta(days=period_days)
    
    # Filter data for period
    period_data = address_data[address_data['date'] >= start_date]
    
    if period_data.empty:
        return {}
    
    # Calculate metrics
    daily_avg = period_data['active_addresses'].mean()
    daily_median = period_data['active_addresses'].median()
    max_addresses = period_data['active_addresses'].max()
    min_addresses = period_data['active_addresses'].min()
    
    # Calculate growth rate
    if len(period_data) > 1:
        first_day = period_data['active_addresses'].iloc[0]
        last_day = period_data['active_addresses'].iloc[-1]
        if first_day > 0:
            growth_rate = (last_day / first_day) - 1
        else:
            growth_rate = None
    else:
        growth_rate = None
    
    return {
        'daily_avg_active_addresses': daily_avg,
        'daily_median_active_addresses': daily_median,
        'max_active_addresses': max_addresses,
        'min_active_addresses': min_addresses,
        'growth_rate': growth_rate,
        'period_days': period_days
    }

def analyze_transaction_metrics(tx_data, period_days=30):
    """
    Analyze transaction metrics over a time period
    
    Args:
        tx_data (pd.DataFrame): DataFrame with date, tx_count, tx_volume, and tx_fees columns
        period_days (int): Period to analyze in days
        
    Returns:
        dict: Transaction metrics
    """
    # Ensure data is sorted by date
    tx_data = tx_data.sort_values('date')
    
    # Calculate date range
    end_date = tx_data['date'].max()
    start_date = end_date - timedelta(days=period_days)
    
    # Filter data for period
    period_data = tx_data[tx_data['date'] >= start_date]
    
    if period_data.empty:
        return {}
    
    # Calculate metrics
    metrics = {
        'daily_avg_tx_count': period_data['tx_count'].mean(),
        'daily_median_tx_count': period_data['tx_count'].median(),
        'total_tx_count': period_data['tx_count'].sum(),
        'max_daily_tx_count': period_data['tx_count'].max(),
        'period_days': period_days
    }
    
    # Add volume metrics if available
    if 'tx_volume' in period_data.columns:
        metrics.update({
            'daily_avg_volume': period_data['tx_volume'].mean(),
            'total_volume': period_data['tx_volume'].sum(),
            'max_daily_volume': period_data['tx_volume'].max()
        })
    
    # Add fee metrics if available
    if 'tx_fees' in period_data.columns:
        metrics.update({
            'daily_avg_fees': period_data['tx_fees'].mean(),
            'total_fees': period_data['tx_fees'].sum(),
            'max_daily_fees': period_data['tx_fees'].max()
        })
    
    return metrics

def calculate_network_usage(tx_count, estimated_capacity):
    """
    Calculate network usage as percentage of capacity
    
    Args:
        tx_count (float): Number of transactions
        estimated_capacity (float): Estimated maximum capacity
        
    Returns:
        float: Usage percentage
    """
    if estimated_capacity == 0:
        return None
    
    usage_pct = (tx_count / estimated_capacity) * 100
    return min(usage_pct, 100.0)  # Cap at 100%

def calculate_value_density(tx_volume, tx_count):
    """
    Calculate average value per transaction
    
    Args:
        tx_volume (float): Transaction volume in currency units
        tx_count (int): Number of transactions
        
    Returns:
        float: Average value per transaction
    """
    if tx_count == 0:
        return 0
    
    return tx_volume / tx_count

def analyze_network_growth(historical_data, metric_column, period_days=30):
    """
    Analyze network growth rate for a specific metric
    
    Args:
        historical_data (pd.DataFrame): DataFrame with 'date' and metric columns
        metric_column (str): Column name to analyze
        period_days (int): Period to analyze in days
        
    Returns:
        dict: Growth metrics
    """
    # Ensure data is sorted by date
    historical_data = historical_data.sort_values('date')
    
    # Calculate date range
    end_date = historical_data['date'].max()
    start_date = end_date - timedelta(days=period_days)
    
    # Filter data for period
    period_data = historical_data[historical_data['date'] >= start_date]
    
    if len(period_data) < 2:
        return {}
    
    # Calculate metrics
    first_value = period_data[metric_column].iloc[0]
    last_value = period_data[metric_column].iloc[-1]
    
    if first_value == 0:
        growth_rate = None
    else:
        growth_rate = (last_value / first_value) - 1
    
    # Calculate compound daily growth rate
    days_count = (period_data['date'].iloc[-1] - period_data['date'].iloc[0]).days
    if days_count > 0 and first_value > 0 and last_value > 0:
        cagr = ((last_value / first_value) ** (1 / days_count)) - 1
    else:
        cagr = None
    
    return {
        f'{metric_column}_growth_rate': growth_rate,
        f'{metric_column}_cagr': cagr,
        'start_value': first_value,
        'end_value': last_value,
        'period_days': period_days
    }

def calculate_network_value_to_transactions_ratio(market_cap, daily_transaction_volume, days=30):
    """
    Calculate Network Value to Transactions (NVT) Ratio
    
    Args:
        market_cap (float): Current market capitalization
        daily_transaction_volume (pd.Series or list): Daily transaction volumes
        days (int): Period in days for the calculation
        
    Returns:
        float: NVT ratio
    """
    if isinstance(daily_transaction_volume, pd.Series):
        avg_daily_volume = daily_transaction_volume.tail(days).mean()
    else:
        avg_daily_volume = sum(daily_transaction_volume[-days:]) / min(days, len(daily_transaction_volume))
    
    if avg_daily_volume == 0:
        return float('inf')
    
    nvt = market_cap / avg_daily_volume
    return nvt
